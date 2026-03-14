"""Temporal train/val/test splits for BCM dataset."""

import logging
from typing import Dict, Tuple

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def get_time_splits(
    zarr_path: str,
    train_start: str = "1980-01",
    train_end: str = "2019-12",
    test_start: str = "2020-01",
    test_end: str = "2021-04",
) -> Dict[str, slice]:
    """Get temporal slices for train and test splits.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    train_start, train_end : str
        Training period as YYYY-MM.
    test_start, test_end : str
        Test period as YYYY-MM.

    Returns
    -------
    dict
        Keys 'train', 'test' with slice objects into the time dimension.
    """
    store = zarr.open(zarr_path, mode="r")
    times = list(np.array(store["meta/time"]))

    def ym_to_idx(ym: str) -> int:
        try:
            return times.index(ym)
        except ValueError:
            # Find closest
            for i, t in enumerate(times):
                if t >= ym:
                    return i
            return len(times)

    train_s = ym_to_idx(train_start)
    train_e = ym_to_idx(train_end) + 1
    test_s = ym_to_idx(test_start)
    test_e = ym_to_idx(test_end) + 1

    splits = {
        "train": slice(train_s, train_e),
        "test": slice(test_s, test_e),
    }

    logger.info(
        f"Splits: train={times[train_s]}..{times[train_e-1]} ({train_e-train_s} months), "
        f"test={times[test_s]}..{times[min(test_e-1, len(times)-1)]} ({test_e-test_s} months)"
    )

    return splits


def get_pixel_indices(zarr_path: str, subsample_frac: float = 1.0) -> np.ndarray:
    """Get valid pixel indices from the zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to zarr store.
    subsample_frac : float
        Fraction of valid pixels to use (for memory/speed).

    Returns
    -------
    np.ndarray
        (N, 2) array of (row, col) indices.
    """
    store = zarr.open(zarr_path, mode="r")
    valid_mask = np.array(store["meta/valid_mask"])
    rows, cols = np.where(valid_mask)
    indices = np.stack([rows, cols], axis=1)

    if subsample_frac < 1.0:
        n = int(len(indices) * subsample_frac)
        rng = np.random.RandomState(42)
        chosen = rng.choice(len(indices), size=n, replace=False)
        indices = indices[chosen]

    logger.info(f"Using {len(indices)} pixels ({subsample_frac*100:.0f}% of valid)")
    return indices
