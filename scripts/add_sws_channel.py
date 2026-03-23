"""Precompute monthly soil water storage (SWS) and add to zarr store.

SWS[t] = clamp(SWS[t-1] + PPT[t] - PET_bcm[t], 0, AWC)

Uses:
  - AWC derived from zarr static channels: (FC - WP) * soil_depth * 1000 [mm]
    This is consistent with BCMv8's full soil column depth (mean ~2m),
    whereas POLARIS AWC integrates only 0-100cm.
  - BCMv8 PET targets from zarr (deterministic from climate forcing, not leakage)
  - PPT from zarr dynamic channel 0

Note: The zarr stores raw (unnormalized) values. The norm stats are used by
BCMPixelDataset for on-the-fly z-score normalization during training.
We read raw values directly and normalize only the SWS output channel.

Spin-up: 3 cycles through first 36 months to remove initialization bias.

Usage:
    conda run -n deep_field python scripts/add_sws_channel.py
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

SPINUP_YEARS = 3
PPT_CHANNEL = 0    # dynamic channel index for ppt
PET_TARGET = 0     # target index: pet=0, pck=1, aet=2, cwd=3

# Static channel indices (from preprocessing.py ordering)
FC_IDX = 9         # field_capacity
WP_IDX = 10        # wilting_point
SD_IDX = 7         # soil_depth


def compute_awc_from_zarr(store, valid_mask: np.ndarray) -> np.ndarray:
    """Compute AWC (mm) from zarr static channels: (FC - WP) × soil_depth × 1000.

    The zarr stores raw (unnormalized) values for both static and dynamic channels.
    Norm stats are stored separately for BCMPixelDataset to apply on-the-fly.

    Returns (H, W) array in mm, clipped to [0, inf].
    """
    static = np.array(store["inputs/static"])  # (C_static, H, W) — raw values

    fc = static[FC_IDX]    # field capacity, volumetric fraction
    wp = static[WP_IDX]    # wilting point, volumetric fraction
    sd = static[SD_IDX]    # soil depth, meters

    logger.info(f"Static channel stats (valid pixels):")
    logger.info(f"  FC[{FC_IDX}]: mean={fc[valid_mask].mean():.4f}")
    logger.info(f"  WP[{WP_IDX}]: mean={wp[valid_mask].mean():.4f}")
    logger.info(f"  SD[{SD_IDX}]: mean={sd[valid_mask].mean():.2f}m")

    awc = np.maximum(0.0, (fc - wp)) * np.maximum(0.0, sd) * 1000.0  # mm
    awc[~valid_mask] = 0.0

    logger.info(f"AWC (FC-WP)*depth*1000: min={awc[valid_mask].min():.1f}mm  "
                f"max={awc[valid_mask].max():.1f}mm  "
                f"mean={awc[valid_mask].mean():.1f}mm")
    return awc.astype(np.float32)


def compute_sws(ppt_raw, pet_raw, awc_mm, valid_mask, spinup_years=3):
    """Run simple bucket model: SWS[t] = clamp(SWS[t-1] + PPT[t] - PET[t], 0, AWC).

    Parameters
    ----------
    ppt_raw  : (T, H, W) mm/month — raw precipitation from zarr
    pet_raw  : (T, H, W) mm/month — raw BCMv8 PET targets from zarr
    awc_mm   : (H, W)    mm       — available water capacity
    valid_mask: (H, W)   bool

    Returns
    -------
    (T, H, W) SWS in mm (raw, unnormalized).
    """
    T, H, W = ppt_raw.shape
    sws_out = np.zeros((T, H, W), dtype=np.float32)
    sws_t = awc_mm * 0.5  # initialize at 50% capacity

    spinup_len = min(spinup_years * 12, T)
    logger.info(f"Running {spinup_years}-year spin-up ({spinup_len} months × 3 passes)...")
    for pass_num in range(3):
        for t in range(spinup_len):
            sws_t = np.clip(sws_t + ppt_raw[t] - pet_raw[t], 0.0, awc_mm)
        logger.info(f"  Spin-up pass {pass_num+1}/3 complete — "
                    f"SWS mean (valid): {sws_t[valid_mask].mean():.1f}mm")

    logger.info("Running full time series...")
    for t in range(T):
        sws_t = np.clip(sws_t + ppt_raw[t] - pet_raw[t], 0.0, awc_mm)
        sws_out[t] = sws_t
        if t % 120 == 0:
            logger.info(f"  t={t}/{T}  SWS mean: {sws_t[valid_mask].mean():.1f}mm")

    sws_out[:, ~valid_mask] = 0.0
    return sws_out


def main():
    from src.utils.config import load_config

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    store = zarr.open(cfg.paths.zarr_store, mode="r+")

    T, C_dyn, H, W = store["inputs/dynamic"].shape
    logger.info(f"Zarr dynamic shape: T={T}, C={C_dyn}, H={H}, W={W}")

    # Check SWS not already added
    dyn_names = list(np.array(store["norm/dynamic_names"]))
    logger.info(f"Existing dynamic channels: {dyn_names}")
    if "sws" in dyn_names:
        logger.error("SWS channel already exists in zarr. Aborting.")
        sys.exit(1)
    if C_dyn != 11:
        logger.warning(
            f"Dynamic array has {C_dyn} channels (expected 11). "
            f"Proceeding but verify channel ordering."
        )

    # --- Valid mask ---
    valid_mask = np.array(store["meta/valid_mask"])  # (H, W) bool

    # --- Compute AWC from zarr static channels ---
    awc_mm = compute_awc_from_zarr(store, valid_mask)

    # --- Load raw PPT from zarr (NOT normalized — zarr stores raw values) ---
    logger.info("Loading raw PPT from zarr (channel 0)...")
    ppt_raw = np.array(store["inputs/dynamic"][:, PPT_CHANNEL, :, :])  # (T, H, W)

    # --- Load raw BCMv8 PET targets ---
    logger.info("Loading raw BCMv8 PET targets...")
    pet_raw = np.array(store["targets/pet"])  # (T, H, W)

    # --- Sanity checks ---
    logger.info(f"PPT range (valid): {ppt_raw[:, valid_mask].min():.1f} – "
                f"{ppt_raw[:, valid_mask].max():.1f} mm/month")
    logger.info(f"PET range (valid): {pet_raw[:, valid_mask].min():.1f} – "
                f"{pet_raw[:, valid_mask].max():.1f} mm/month")

    if ppt_raw[:, valid_mask].max() > 2000:
        logger.warning("PPT max > 2000 mm/month — seems unreasonable. "
                       "Check if zarr stores z-normalized values.")
    if pet_raw[:, valid_mask].max() > 500:
        logger.warning("PET max > 500 mm/month — seems unreasonable. "
                       "Check if zarr stores z-normalized values.")

    # --- Compute SWS ---
    sws = compute_sws(ppt_raw, pet_raw, awc_mm, valid_mask, SPINUP_YEARS)
    del ppt_raw, pet_raw

    # --- Seasonal sanity check ---
    logger.info(f"SWS range (valid): {sws[:, valid_mask].min():.1f} – "
                f"{sws[:, valid_mask].max():.1f} mm")

    time_index = np.array(store["meta/time"]).astype(str)
    apr_months = [i for i, t in enumerate(time_index) if t.endswith("-04")]
    oct_months = [i for i, t in enumerate(time_index) if t.endswith("-10")]

    if apr_months and oct_months:
        apr_sws = sws[apr_months][:, valid_mask].mean()
        oct_sws = sws[oct_months][:, valid_mask].mean()
        logger.info(f"Seasonal check — April mean SWS: {apr_sws:.1f}mm  "
                    f"October mean SWS: {oct_sws:.1f}mm")
        if oct_sws >= apr_sws:
            logger.warning("October SWS >= April SWS — unexpected for Mediterranean CA! "
                           "Review inputs.")
        else:
            logger.info("Seasonal pattern correct (April > October).")

    # --- Compute norm stats for SWS (zarr stores raw; norm stats used by dataset) ---
    valid_vals = sws[:, valid_mask]
    sws_mean = float(valid_vals.mean())
    sws_std = float(valid_vals.std())
    logger.info(f"SWS normalization stats: mean={sws_mean:.2f}mm  std={sws_std:.2f}mm")

    # --- Write SWS as raw values into zarr (matching convention of other channels) ---
    # The zarr stores raw values; BCMPixelDataset applies z-score normalization on-the-fly
    new_C = C_dyn + 1
    old_chunks = store["inputs/dynamic"].chunks
    new_chunks = (old_chunks[0], new_C, old_chunks[2], old_chunks[3])
    logger.info(f"Appending SWS as channel {C_dyn} (new shape: T={T}, C={new_C}, H={H}, W={W})")
    logger.info(f"Chunks: {old_chunks} → {new_chunks}")

    # Load existing dynamic data before overwriting
    logger.info("Loading existing dynamic array into memory...")
    old_dyn = np.array(store["inputs/dynamic"])  # (T, C_dyn, H, W)

    # Delete old array and create new one with expanded channel dimension
    del store["inputs/dynamic"]
    new_dyn = store.create_array(
        "inputs/dynamic",
        shape=(T, new_C, H, W),
        chunks=new_chunks,
        dtype=np.float32,
    )
    new_dyn[:, :C_dyn, :, :] = old_dyn
    new_dyn[:, C_dyn, :, :] = sws  # raw values, not normalized
    del old_dyn, sws
    logger.info("Dynamic array written.")

    # --- Update normalization stats ---
    dyn_means = np.array(store["norm/dynamic_mean"])
    dyn_stds = np.array(store["norm/dynamic_std"])
    new_means = np.append(dyn_means, sws_mean).astype(np.float32)
    new_stds = np.append(dyn_stds, sws_std).astype(np.float32)
    store.create_array("norm/dynamic_mean", data=new_means, overwrite=True)
    store.create_array("norm/dynamic_std", data=new_stds, overwrite=True)

    # Update channel names
    old_names = np.array(store["norm/dynamic_names"])
    new_names = np.append(old_names, "sws")
    store.create_array("norm/dynamic_names", data=new_names, overwrite=True)
    logger.info("Normalization stats and channel names updated.")

    # --- Verify ---
    logger.info("Verifying...")
    v_store = zarr.open(cfg.paths.zarr_store, mode="r")
    assert v_store["inputs/dynamic"].shape == (T, new_C, H, W), "Shape mismatch!"
    v_names = list(np.array(v_store["norm/dynamic_names"]))
    assert v_names[-1] == "sws", f"Last channel name is {v_names[-1]}, expected 'sws'"
    assert len(np.array(v_store["norm/dynamic_mean"])) == new_C, "Norm means length mismatch!"
    logger.info(f"Verification passed. Channel names: {v_names}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SWS channel added successfully.")
    logger.info(f"Dynamic channels: {C_dyn} → {new_C}")
    logger.info(f"Channel names: {list(new_names)}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(f"  1. Update config.yaml: model.backbone.in_channels: "
                f"{cfg.model.backbone.in_channels} → {cfg.model.backbone.in_channels + 1}")
    logger.info("  2. Train: conda run -n deep_field python train.py "
                "--run-id v13-sws --notes 'Added SWS bucket model channel'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
