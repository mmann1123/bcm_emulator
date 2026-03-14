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
        self.store = zarr.open(zarr_path, mode="r")
        self.pixel_indices = pixel_indices  # (N, 2) array of (row, col)
        self.time_slice = time_slice
        self.seq_len = seq_len
        self.normalize = normalize

        # Get time dimension info
        times = np.array(self.store["meta/time"])
        self.times = times[time_slice]
        self.T = len(self.times)

        # Number of valid windows per pixel
        self.n_windows = max(1, self.T - seq_len + 1)
        self.n_pixels = len(pixel_indices)

        # Load normalization stats
        if normalize:
            self.dyn_mean = np.array(self.store["norm/dynamic_mean"])  # (9,)
            self.dyn_std = np.array(self.store["norm/dynamic_std"])
            self.stat_mean = np.array(self.store["norm/static_mean"])  # (4,)
            self.stat_std = np.array(self.store["norm/static_std"])
            self.tgt_mean = np.array(self.store["norm/target_mean"])  # (4,)
            self.tgt_std = np.array(self.store["norm/target_std"])

        # Preload static inputs (small, fits in memory)
        self.static = np.array(self.store["inputs/static"])  # (4, H, W)

    def __len__(self) -> int:
        return self.n_pixels * self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pixel_idx = idx // self.n_windows
        window_idx = idx % self.n_windows

        row, col = self.pixel_indices[pixel_idx]

        # Temporal indices
        t_start = self.time_slice.start + window_idx
        t_end = t_start + self.seq_len

        # Dynamic inputs: (T_window, 9) -> (9, T_window)
        dynamic = np.array(
            self.store["inputs/dynamic"][t_start:t_end, :, row, col]
        )  # (seq_len, 9)
        dynamic = dynamic.T  # (9, seq_len)

        # Static inputs: (4,) tiled to (4, seq_len)
        static = self.static[:, row, col]  # (4,)
        static_tiled = np.tile(static[:, np.newaxis], (1, self.seq_len))  # (4, seq_len)

        # Combine: (13, seq_len)
        inputs = np.concatenate([dynamic, static_tiled], axis=0)

        # Targets
        targets = {}
        gt_pck_prev = np.zeros((1, self.seq_len), dtype=np.float32)
        gt_aet_prev = np.zeros((1, self.seq_len), dtype=np.float32)

        for i, var in enumerate(["pet", "pck", "aet", "cwd"]):
            t_data = np.array(
                self.store[f"targets/{var}"][t_start:t_end, row, col]
            )  # (seq_len,)
            targets[var] = t_data

        # Ground truth PCK(t-1) and AET(t-1) for teacher forcing
        # These are the actual target values shifted by 1
        if t_start > 0:
            pck_prev_data = np.array(
                self.store["targets/pck"][t_start - 1:t_end - 1, row, col]
            )
            aet_prev_data = np.array(
                self.store["targets/aet"][t_start - 1:t_end - 1, row, col]
            )
        else:
            pck_prev_data = np.zeros(self.seq_len, dtype=np.float32)
            pck_prev_data[1:] = np.array(
                self.store["targets/pck"][t_start:t_end - 1, row, col]
            )
            aet_prev_data = np.zeros(self.seq_len, dtype=np.float32)
            aet_prev_data[1:] = np.array(
                self.store["targets/aet"][t_start:t_end - 1, row, col]
            )

        gt_pck_prev[0] = pck_prev_data
        gt_aet_prev[0] = aet_prev_data

        # Normalize
        if self.normalize:
            # Dynamic channels
            for ch in range(9):
                inputs[ch] = (inputs[ch] - self.dyn_mean[ch]) / self.dyn_std[ch]
            # Static channels
            for ch in range(4):
                inputs[9 + ch] = (inputs[9 + ch] - self.stat_mean[ch]) / self.stat_std[ch]
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
        }
        return result


class ElevationStratifiedSampler(Sampler):
    """Samples equal numbers of pixels from each elevation band per epoch.

    Parameters
    ----------
    pixel_indices : np.ndarray
        (N, 2) array of (row, col) for valid pixels.
    elevations : np.ndarray
        (H, W) elevation array.
    n_bins : int
        Number of elevation bins.
    samples_per_epoch : int
        Total number of pixel samples per epoch.
    n_windows : int
        Number of temporal windows per pixel.
    """

    def __init__(
        self,
        pixel_indices: np.ndarray,
        elevations: np.ndarray,
        n_bins: int = 5,
        samples_per_epoch: int = 10000,
        n_windows: int = 1,
    ):
        self.pixel_indices = pixel_indices
        self.n_windows = n_windows
        self.samples_per_epoch = samples_per_epoch

        # Get elevation for each pixel
        pixel_elevs = elevations[pixel_indices[:, 0], pixel_indices[:, 1]]

        # Create elevation bins
        bin_edges = np.quantile(
            pixel_elevs[~np.isnan(pixel_elevs)],
            np.linspace(0, 1, n_bins + 1),
        )
        bin_edges[-1] += 1  # include max value

        # Assign pixels to bins
        self.bin_indices = []
        for i in range(n_bins):
            mask = (pixel_elevs >= bin_edges[i]) & (pixel_elevs < bin_edges[i + 1])
            self.bin_indices.append(np.where(mask)[0])

        self.samples_per_bin = samples_per_epoch // n_bins

    def __iter__(self):
        indices = []
        for bin_idx in self.bin_indices:
            if len(bin_idx) == 0:
                continue
            # Sample pixels from this bin
            sampled = np.random.choice(
                bin_idx, size=min(self.samples_per_bin, len(bin_idx)), replace=True
            )
            # Convert pixel indices to dataset indices (pixel_idx * n_windows + random window)
            for px in sampled:
                window = np.random.randint(0, self.n_windows)
                indices.append(px * self.n_windows + window)

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch
