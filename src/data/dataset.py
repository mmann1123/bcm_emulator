"""BCMPixelDataset: pixel time-series from zarr store."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class BCMPixelDataset(Dataset):
    """Dataset that yields pixel time-series from a zarr store.

    Each sample is a single pixel's time-series over a window of `seq_len` months.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    pixel_indices : np.ndarray
        Array of (row, col) indices for valid pixels to include.
    time_slice : slice
        Temporal slice into the zarr arrays (e.g., slice(0, 480) for training).
    seq_len : int
        Number of months per training sample.
    normalize : bool
        Whether to apply z-score normalization.
    """

    def __init__(
        self,
        zarr_path: str,
        pixel_indices: np.ndarray,
        time_slice: slice,
        seq_len: int = 48,
        normalize: bool = True,
    ):
        store = zarr.open(zarr_path, mode="r")
        self.pixel_indices = pixel_indices  # (N, 2) array of (row, col)
        self.time_slice = time_slice
        self.seq_len = seq_len
        self.normalize = normalize

        # Get time dimension info
        times = np.array(store["meta/time"])
        self.times = times[time_slice]
        self.T = len(self.times)

        # Number of valid windows per pixel
        self.n_windows = max(1, self.T - seq_len + 1)
        self.n_pixels = len(pixel_indices)

        # Determine channel counts from zarr shape
        self.n_dyn = store["inputs/dynamic"].shape[1]  # 10
        n_static_total = store["inputs/static"].shape[0]  # 6
        self.n_static_cont = n_static_total - 1  # 5 (last channel is FVEG)

        # Load normalization stats
        if normalize:
            self.dyn_mean = np.array(store["norm/dynamic_mean"])  # (n_dyn,)
            self.dyn_std = np.array(store["norm/dynamic_std"])
            self.stat_mean = np.array(store["norm/static_mean"])[:self.n_static_cont]
            self.stat_std = np.array(store["norm/static_std"])[:self.n_static_cont]
            self.tgt_mean = np.array(store["norm/target_mean"])  # (4,)
            self.tgt_std = np.array(store["norm/target_std"])

        # Preload all data for selected pixels into RAM to avoid zarr I/O bottleneck
        rows = pixel_indices[:, 0]
        cols = pixel_indices[:, 1]

        # We need time range including 1 step before time_slice for teacher forcing
        t_start = max(0, time_slice.start - 1)
        t_end = time_slice.stop
        self._t_offset = time_slice.start - t_start  # 0 or 1

        logger.info(f"Preloading data for {len(pixel_indices)} pixels into RAM...")

        # Dynamic inputs: read each timestep once, extract all pixels -> (N_pixels, n_dyn, T_load)
        dynamic_zarr = store["inputs/dynamic"]  # (T, n_dyn, H, W)
        T_load = t_end - t_start
        self._dynamic = np.empty((self.n_pixels, self.n_dyn, T_load), dtype=np.float32)
        for t in range(T_load):
            t_abs = t_start + t
            data_t = np.array(dynamic_zarr[t_abs])  # (n_dyn, H, W)
            self._dynamic[:, :, t] = data_t[:, rows, cols].T.astype(np.float32)
            if (t + 1) % 100 == 0:
                logger.info(f"  Dynamic: {t+1}/{T_load} timesteps loaded")

        # Static inputs: continuous channels (0 to n_static_cont-1) + FVEG class ID (last)
        static_full = np.array(store["inputs/static"])  # (C, H, W)
        static_all = static_full[:, rows, cols].T.astype(np.float32)  # (N_pixels, C)
        self._static = static_all[:, :self.n_static_cont]  # (N_pixels, n_static_cont) — continuous static
        if static_full.shape[0] > self.n_static_cont:
            self._fveg_ids = static_all[:, self.n_static_cont].astype(np.int64)  # (N_pixels,)
        else:
            self._fveg_ids = np.zeros(self.n_pixels, dtype=np.int64)

        # Targets: (N_pixels, T_load, 4) for pet, pck, aet, cwd
        target_names = ["pet", "pck", "aet", "cwd"]
        self._targets = np.empty((self.n_pixels, 4, T_load), dtype=np.float32)
        for vi, var in enumerate(target_names):
            tgt_zarr = store[f"targets/{var}"]  # (T, H, W)
            for t in range(T_load):
                t_abs = t_start + t
                data_t = np.array(tgt_zarr[t_abs, :, :])  # (H, W)
                self._targets[:, vi, t] = data_t[rows, cols].astype(np.float32)

        logger.info(f"Preload complete. Dynamic: {self._dynamic.nbytes/1e6:.0f} MB, "
                     f"Targets: {self._targets.nbytes/1e6:.0f} MB")

    def __len__(self) -> int:
        return self.n_pixels * self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pixel_idx = idx // self.n_windows
        window_idx = idx % self.n_windows

        # Temporal indices into preloaded arrays
        t_start = self._t_offset + window_idx
        t_end = t_start + self.seq_len

        # Dynamic inputs: (n_dyn, seq_len) from preloaded array
        dynamic = self._dynamic[pixel_idx, :, t_start:t_end].copy()  # (n_dyn, seq_len)

        # Static inputs: (n_static_cont,) tiled to (n_static_cont, seq_len)
        static = self._static[pixel_idx]  # (n_static_cont,)
        n_sc = static.shape[0]
        static_tiled = np.broadcast_to(static[:, np.newaxis], (n_sc, self.seq_len)).copy()

        # Combine: (n_dyn + n_static_cont, seq_len)
        inputs = np.concatenate([dynamic, static_tiled], axis=0)

        # Targets: (4, seq_len) from preloaded array
        tgt_slice = self._targets[pixel_idx, :, t_start:t_end]  # (4, seq_len)
        targets = {
            "pet": tgt_slice[0].copy(),
            "pck": tgt_slice[1].copy(),
            "aet": tgt_slice[2].copy(),
            "cwd": tgt_slice[3].copy(),
        }

        # Ground truth PCK(t-1) and AET(t-1) for teacher forcing
        gt_pck_prev = np.zeros((1, self.seq_len), dtype=np.float32)
        gt_aet_prev = np.zeros((1, self.seq_len), dtype=np.float32)

        prev_start = t_start - 1
        if prev_start >= 0:
            gt_pck_prev[0] = self._targets[pixel_idx, 1, prev_start:prev_start + self.seq_len]
            gt_aet_prev[0] = self._targets[pixel_idx, 2, prev_start:prev_start + self.seq_len]
        else:
            gt_pck_prev[0, 1:] = self._targets[pixel_idx, 1, t_start:t_end - 1]
            gt_aet_prev[0, 1:] = self._targets[pixel_idx, 2, t_start:t_end - 1]

        # Normalize
        if self.normalize:
            # Dynamic channels (vectorized)
            n_d = self.n_dyn
            inputs[:n_d] = (inputs[:n_d] - self.dyn_mean[:, np.newaxis]) / self.dyn_std[:, np.newaxis]
            # Static channels (vectorized)
            inputs[n_d:] = (inputs[n_d:] - self.stat_mean[:, np.newaxis]) / self.stat_std[:, np.newaxis]
            # Targets
            for i, var in enumerate(["pet", "pck", "aet", "cwd"]):
                targets[var] = (targets[var] - self.tgt_mean[i]) / self.tgt_std[i]
            # GT prev (use pck and aet stats)
            gt_pck_prev = (gt_pck_prev - self.tgt_mean[1]) / self.tgt_std[1]
            gt_aet_prev = (gt_aet_prev - self.tgt_mean[2]) / self.tgt_std[2]

        # Convert to tensors
        result = {
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "targets": {
                var: torch.tensor(targets[var], dtype=torch.float32).unsqueeze(0)
                for var in ["pet", "pck", "aet", "cwd"]
            },
            "gt_pck_prev": torch.tensor(gt_pck_prev, dtype=torch.float32),
            "gt_aet_prev": torch.tensor(gt_aet_prev, dtype=torch.float32),
            "fveg_id": torch.tensor(self._fveg_ids[pixel_idx], dtype=torch.long),
        }
        return result


class EcoregionStratifiedSampler(Sampler):
    """Samples equal numbers of pixels from each L3 ecoregion per epoch.

    Parameters
    ----------
    pixel_indices : np.ndarray
        (N, 2) array of (row, col) for valid pixels.
    ecoregion_map : np.ndarray
        (H, W) integer array of ecoregion IDs.
    samples_per_epoch : int
        Total number of pixel samples per epoch.
    n_windows : int
        Number of temporal windows per pixel.
    """

    def __init__(
        self,
        pixel_indices: np.ndarray,
        ecoregion_map: np.ndarray,
        samples_per_epoch: int = 10000,
        n_windows: int = 1,
    ):
        self.n_windows = n_windows
        self.samples_per_epoch = samples_per_epoch

        # Get ecoregion for each pixel
        pixel_ecos = ecoregion_map[pixel_indices[:, 0], pixel_indices[:, 1]]

        # Group pixels by ecoregion (skip nodata)
        unique_ecos = np.unique(pixel_ecos)
        unique_ecos = unique_ecos[unique_ecos > 0]  # skip nodata/invalid

        self.bin_indices = []
        for eco_id in unique_ecos:
            mask = pixel_ecos == eco_id
            idx = np.where(mask)[0]
            if len(idx) > 0:
                self.bin_indices.append(idx)

        n_bins = len(self.bin_indices)
        self.samples_per_bin = samples_per_epoch // max(n_bins, 1)
        logger.info(f"EcoregionStratifiedSampler: {n_bins} ecoregions, "
                    f"{self.samples_per_bin} samples/region, "
                    f"{samples_per_epoch} total/epoch")

    def __iter__(self):
        indices = []
        for bin_idx in self.bin_indices:
            sampled = np.random.choice(
                bin_idx, size=self.samples_per_bin, replace=True
            )
            for px in sampled:
                window = np.random.randint(0, self.n_windows)
                indices.append(px * self.n_windows + window)

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch


# Keep old name as alias for backwards compatibility
ElevationStratifiedSampler = EcoregionStratifiedSampler
