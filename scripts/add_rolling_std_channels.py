"""Add rolling standard deviation channels to zarr store.

Computes and appends three new dynamic channels:
  - vpd_roll6_std:  6-month rolling std of VPD
  - srad_roll6_std: 6-month rolling std of solar radiation
  - tmax_roll3_std: 3-month rolling std of max temperature

These features were identified by panel_extremes_analysis.py as
disproportionately important for predicting AET/CWD extremes
(vpd_roll6_std: 73x ratio, srad_roll6_std: 6.8x, tmax_roll3_std: 10x)
but nearly invisible in overall feature importance. The TCN backbone
may not extract variance information effectively from raw sequences.

The zarr stores raw (unnormalized) values. Rolling std is computed on
raw values, then norm stats (mean/std of the rolling std) are stored
for BCMPixelDataset to z-normalize on-the-fly.

Usage:
    conda run -n deep_field python scripts/add_rolling_std_channels.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Features to add: (new_name, source_channel_name, window_size)
FEATURES = [
    ("vpd_roll6_std", "vpd", 6),
    ("srad_roll6_std", "srad", 6),
    ("tmax_roll3_std", "tmax", 3),
]


def compute_rolling_std(data: np.ndarray, window: int, valid_mask: np.ndarray) -> np.ndarray:
    """Compute rolling std along time axis for a (T, H, W) array.

    Uses a causal window (current + previous window-1 months) so there's
    no lookahead. First window-1 timesteps use partial windows (min_periods=1).

    Parameters
    ----------
    data : (T, H, W) raw values
    window : rolling window size in months
    valid_mask : (H, W) bool

    Returns
    -------
    (T, H, W) rolling std values (0.0 for invalid pixels and single-element windows)
    """
    T, H, W = data.shape
    result = np.zeros_like(data)

    # For efficiency, compute rolling std using cumulative sums
    # std = sqrt(E[x²] - E[x]²)
    # Process in chunks of columns to manage memory
    for col in range(W):
        valid_col = valid_mask[:, col]
        if not valid_col.any():
            continue

        col_data = data[:, valid_col, col]  # (T, n_valid)

        # Cumulative sums for online variance
        cumsum = np.cumsum(col_data, axis=0)      # (T, n_valid)
        cumsum2 = np.cumsum(col_data ** 2, axis=0)

        for t in range(T):
            t_start = max(0, t - window + 1)
            n = t - t_start + 1

            if n < 2:
                # Can't compute std with < 2 values
                continue

            if t_start == 0:
                s = cumsum[t]
                s2 = cumsum2[t]
            else:
                s = cumsum[t] - cumsum[t_start - 1]
                s2 = cumsum2[t] - cumsum2[t_start - 1]

            mean = s / n
            var = s2 / n - mean ** 2
            # Numerical stability: clamp negative variance
            var = np.maximum(var, 0.0)
            result[t, valid_col, col] = np.sqrt(var)

    result[:, ~valid_mask] = 0.0
    return result


def main():
    from src.utils.config import load_config

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    store = zarr.open(cfg.paths.zarr_store, mode="r+")

    T, C_dyn, H, W = store["inputs/dynamic"].shape
    dyn_names = list(np.array(store["norm/dynamic_names"]))
    logger.info(f"Zarr dynamic shape: T={T}, C={C_dyn}, H={H}, W={W}")
    logger.info(f"Existing channels: {dyn_names}")

    # Check none already added
    new_names = [f[0] for f in FEATURES]
    for name in new_names:
        if name in dyn_names:
            logger.error(f"Channel '{name}' already exists. Aborting.")
            sys.exit(1)

    # Build name→index mapping
    name_to_idx = {n: i for i, n in enumerate(dyn_names)}

    valid_mask = np.array(store["meta/valid_mask"])  # (H, W)

    # Compute each rolling std feature
    new_channels = []  # list of (T, H, W) arrays
    new_means = []
    new_stds = []

    for feat_name, src_name, window in FEATURES:
        src_idx = name_to_idx.get(src_name)
        if src_idx is None:
            logger.error(f"Source channel '{src_name}' not found in zarr. Available: {dyn_names}")
            sys.exit(1)

        logger.info(f"Computing {feat_name} (rolling std of {src_name}, window={window})...")
        src_data = np.array(store["inputs/dynamic"][:, src_idx, :, :])  # (T, H, W)

        logger.info(f"  {src_name} range (valid): {src_data[:, valid_mask].min():.2f} – "
                    f"{src_data[:, valid_mask].max():.2f}")

        roll_std = compute_rolling_std(src_data, window, valid_mask)
        del src_data

        # Stats for normalization
        valid_vals = roll_std[:, valid_mask]
        feat_mean = float(valid_vals.mean())
        feat_std = float(valid_vals.std())
        logger.info(f"  {feat_name}: mean={feat_mean:.4f}, std={feat_std:.4f}, "
                    f"max={valid_vals.max():.4f}")

        new_channels.append(roll_std)
        new_means.append(feat_mean)
        new_stds.append(feat_std)

    # --- Append all new channels to zarr ---
    n_new = len(new_channels)
    new_C = C_dyn + n_new
    old_chunks = store["inputs/dynamic"].chunks
    new_chunks = (old_chunks[0], new_C, old_chunks[2], old_chunks[3])

    logger.info(f"Appending {n_new} channels (C: {C_dyn} → {new_C})")
    logger.info(f"Chunks: {old_chunks} → {new_chunks}")

    # Load existing dynamic data
    logger.info("Loading existing dynamic array into memory...")
    old_dyn = np.array(store["inputs/dynamic"])  # (T, C_dyn, H, W)

    # Delete and recreate with expanded channel dim
    del store["inputs/dynamic"]
    new_dyn = store.create_array(
        "inputs/dynamic",
        shape=(T, new_C, H, W),
        chunks=new_chunks,
        dtype=np.float32,
    )

    # Write old channels
    new_dyn[:, :C_dyn, :, :] = old_dyn
    del old_dyn

    # Write new channels
    for i, (feat_name, roll_std) in enumerate(zip(new_names, new_channels)):
        ch_idx = C_dyn + i
        logger.info(f"  Writing {feat_name} at channel {ch_idx}...")
        new_dyn[:, ch_idx, :, :] = roll_std
    del new_channels

    logger.info("Dynamic array written.")

    # --- Update normalization stats ---
    dyn_means = np.array(store["norm/dynamic_mean"])
    dyn_stds = np.array(store["norm/dynamic_std"])
    updated_means = np.append(dyn_means, new_means).astype(np.float32)
    updated_stds = np.append(dyn_stds, new_stds).astype(np.float32)
    store.create_array("norm/dynamic_mean", data=updated_means, overwrite=True)
    store.create_array("norm/dynamic_std", data=updated_stds, overwrite=True)

    # Update channel names
    old_name_arr = np.array(store["norm/dynamic_names"])
    updated_names = np.append(old_name_arr, new_names)
    store.create_array("norm/dynamic_names", data=updated_names, overwrite=True)
    logger.info("Normalization stats and channel names updated.")

    # --- Verify ---
    logger.info("Verifying...")
    v_store = zarr.open(cfg.paths.zarr_store, mode="r")
    assert v_store["inputs/dynamic"].shape == (T, new_C, H, W), "Shape mismatch!"
    v_names = list(np.array(v_store["norm/dynamic_names"]))
    assert len(v_names) == new_C, f"Name count mismatch: {len(v_names)} vs {new_C}"
    assert len(np.array(v_store["norm/dynamic_mean"])) == new_C, "Norm means length mismatch!"
    logger.info(f"Verification passed. Channel names: {v_names}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info(f"Added {n_new} rolling std channels successfully.")
    logger.info(f"Dynamic channels: {C_dyn} → {new_C}")
    logger.info(f"New channels: {new_names}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(f"  1. Update config.yaml: model.backbone.in_channels: "
                f"{cfg.model.backbone.in_channels} → {cfg.model.backbone.in_channels + n_new}")
    logger.info(f"     (currently {new_C - 1} dynamic backbone + 13 static + 8 fveg = "
                f"{new_C - 1 + 13 + 8})")
    logger.info("  2. Train with updated config")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
