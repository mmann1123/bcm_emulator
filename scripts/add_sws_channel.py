"""Precompute monthly soil water storage (SWS) and add/overwrite in zarr store.

Stress-modulated bucket model (v14+):
    stress[t] = min(SWS[t-1] / AWC, 1.0)      # linear soil moisture stress
    AET_approx[t] = PET[t] * stress[t]          # actual ET decreases as soil dries
    SWS[t] = clamp(SWS[t-1] + PPT[t] - AET_approx[t], 0, AWC)

This replaces the original PPT-PET formulation which drained too aggressively
(PET >> AET in dry conditions), producing a degenerate distribution with 68%
of pixel-months at zero.

Uses:
  - AWC from POLARIS root-zone integration (0-100cm), loaded from awc_bcm.tif
    via config.yaml paths.awc_path. This replaces BCMv8 full-column AWC
    which used ~2m depth and understated drought stress.
  - BCMv8 PET targets from zarr (deterministic from climate forcing, not leakage)
  - PPT from zarr dynamic channel 0

Note: The zarr stores raw (unnormalized) values. The norm stats are used by
BCMPixelDataset for on-the-fly z-score normalization during training.
We read raw values directly and normalize only the SWS output channel.

Spin-up: 3 cycles through first 36 months to remove initialization bias.

Usage:
    # First run (append new channel):
    conda run -n deep_field python scripts/add_sws_channel.py

    # Overwrite existing channel 11 with fixed SWS:
    conda run -n deep_field python scripts/add_sws_channel.py --overwrite
"""

import argparse
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


def compute_awc_from_polaris(awc_path: str, bcm_profile: dict, valid_mask: np.ndarray) -> np.ndarray:
    """Load POLARIS root-zone AWC (0-100cm) from awc_bcm.tif.

    The raster is aligned to the BCM grid via reprojection if needed.

    Returns (H, W) array in mm, clipped to [0, inf].
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    h, w = bcm_profile["height"], bcm_profile["width"]

    with rasterio.open(awc_path) as src:
        if (
            src.crs == rasterio.CRS.from_string(str(bcm_profile["crs"]))
            and src.shape == (h, w)
        ):
            awc = src.read(1).astype(np.float32)
        else:
            awc = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=awc,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=bcm_profile["transform"],
                dst_crs=bcm_profile["crs"],
                resampling=Resampling.bilinear,
            )

    awc[awc == -9999.0] = 0.0
    awc[np.isnan(awc)] = 0.0
    awc = np.maximum(0.0, awc)
    awc[~valid_mask] = 0.0

    logger.info(f"POLARIS AWC: min={awc[valid_mask].min():.1f}mm  "
                f"max={awc[valid_mask].max():.1f}mm  "
                f"mean={awc[valid_mask].mean():.1f}mm")
    return awc.astype(np.float32)


def compute_sws(ppt_raw, pet_raw, awc_mm, valid_mask, spinup_years=3):
    """Stress-modulated bucket model matching BCMv8 linear stress formulation.

    At each timestep:
        stress = min(SWS[t-1] / AWC, 1.0)       # 0 (bone dry) to 1 (at capacity)
        AET_approx = PET[t] * stress              # drainage decreases as soil dries
        SWS[t] = clamp(SWS[t-1] + PPT[t] - AET_approx, 0, AWC)

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

    # Precompute safe AWC for division (avoid /0 where AWC=0)
    awc_safe = np.where(awc_mm > 0, awc_mm, 1.0)

    spinup_len = min(spinup_years * 12, T)
    logger.info(f"Running {spinup_years}-year spin-up ({spinup_len} months × 3 passes) "
                f"[stress-modulated drainage]...")
    for pass_num in range(3):
        for t in range(spinup_len):
            stress = np.minimum(sws_t / awc_safe, 1.0)
            aet_approx = pet_raw[t] * stress
            sws_t = np.clip(sws_t + ppt_raw[t] - aet_approx, 0.0, awc_mm)
        logger.info(f"  Spin-up pass {pass_num+1}/3 complete — "
                    f"SWS mean (valid): {sws_t[valid_mask].mean():.1f}mm")

    logger.info("Running full time series...")
    for t in range(T):
        stress = np.minimum(sws_t / awc_safe, 1.0)
        aet_approx = pet_raw[t] * stress
        sws_t = np.clip(sws_t + ppt_raw[t] - aet_approx, 0.0, awc_mm)
        sws_out[t] = sws_t
        if t % 120 == 0:
            logger.info(f"  t={t}/{T}  SWS mean: {sws_t[valid_mask].mean():.1f}mm")

    sws_out[:, ~valid_mask] = 0.0
    return sws_out


def main():
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Compute SWS and add/overwrite in zarr")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing SWS channel 11 in-place (no shape change)")
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    store = zarr.open_group(cfg.paths.zarr_store, mode="r+")

    T, C_dyn, H, W = store["inputs/dynamic"].shape
    logger.info(f"Zarr dynamic shape: T={T}, C={C_dyn}, H={H}, W={W}")

    dyn_names = list(np.array(store["norm/dynamic_names"]))
    logger.info(f"Existing dynamic channels: {dyn_names}")

    SWS_IDX = 11  # known channel index for SWS

    if args.overwrite:
        if "sws" not in dyn_names or dyn_names[SWS_IDX] != "sws":
            logger.error(f"--overwrite expects 'sws' at channel {SWS_IDX}, "
                         f"but found: {dyn_names}")
            sys.exit(1)
        logger.info(f"OVERWRITE mode: will replace channel {SWS_IDX} (sws) in-place")
    else:
        if "sws" in dyn_names:
            logger.error("SWS channel already exists in zarr. "
                         "Use --overwrite to replace it.")
            sys.exit(1)
        if C_dyn != 11:
            logger.warning(
                f"Dynamic array has {C_dyn} channels (expected 11). "
                f"Proceeding but verify channel ordering."
            )

    # --- Valid mask ---
    valid_mask = np.array(store["meta/valid_mask"])  # (H, W) bool

    # --- Load POLARIS root-zone AWC ---
    awc_path_str = getattr(cfg.paths, "awc_path", "")
    if not awc_path_str or not Path(awc_path_str).exists():
        logger.error(f"AWC raster not found at {awc_path_str}. "
                     "Set paths.awc_path in config.yaml.")
        sys.exit(1)

    from src.utils.io_helpers import get_bcm_reference_profile
    bcm_profile = get_bcm_reference_profile(cfg.paths.bcm_dir)
    awc_mm = compute_awc_from_polaris(awc_path_str, bcm_profile, valid_mask)

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

    # --- Compute SWS (stress-modulated) ---
    sws = compute_sws(ppt_raw, pet_raw, awc_mm, valid_mask, SPINUP_YEARS)
    del ppt_raw, pet_raw

    # --- Distribution diagnostics ---
    valid_vals = sws[:, valid_mask]
    pct_zero = float((valid_vals == 0).sum()) / valid_vals.size * 100
    pct_awc = float(
        np.isclose(valid_vals,
                   np.broadcast_to(awc_mm[valid_mask], valid_vals.shape),
                   atol=0.1).sum()
    ) / valid_vals.size * 100
    pct_mid = 100.0 - pct_zero - pct_awc
    logger.info(f"SWS distribution: {pct_zero:.1f}% at zero, "
                f"{pct_awc:.1f}% at AWC, {pct_mid:.1f}% in middle range")

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

    # --- Compute norm stats for SWS ---
    sws_mean = float(valid_vals.mean())
    sws_std = float(valid_vals.std())
    logger.info(f"SWS normalization stats: mean={sws_mean:.2f}mm  std={sws_std:.2f}mm")

    if args.overwrite:
        # --- Overwrite channel 11 in-place ---
        logger.info(f"Overwriting channel {SWS_IDX} in inputs/dynamic...")
        store["inputs/dynamic"][:, SWS_IDX, :, :] = sws
        del sws
        logger.info("Channel overwritten.")

        # Update norm stats for channel 11 only
        dyn_means = np.array(store["norm/dynamic_mean"])
        dyn_stds = np.array(store["norm/dynamic_std"])
        old_mean, old_std = dyn_means[SWS_IDX], dyn_stds[SWS_IDX]
        dyn_means[SWS_IDX] = sws_mean
        dyn_stds[SWS_IDX] = sws_std
        store.create_array("norm/dynamic_mean", data=dyn_means, overwrite=True)
        store.create_array("norm/dynamic_std", data=dyn_stds, overwrite=True)
        logger.info(f"Norm stats updated: mean {old_mean:.2f} → {sws_mean:.2f}, "
                    f"std {old_std:.2f} → {sws_std:.2f}")
    else:
        # --- Append SWS as new channel ---
        new_C = C_dyn + 1
        old_chunks = store["inputs/dynamic"].chunks
        new_chunks = (old_chunks[0], new_C, old_chunks[2], old_chunks[3])
        logger.info(f"Appending SWS as channel {C_dyn} "
                    f"(new shape: T={T}, C={new_C}, H={H}, W={W})")
        logger.info(f"Chunks: {old_chunks} → {new_chunks}")

        logger.info("Loading existing dynamic array into memory...")
        old_dyn = np.array(store["inputs/dynamic"])

        del store["inputs/dynamic"]
        new_dyn = store.create_array(
            "inputs/dynamic",
            shape=(T, new_C, H, W),
            chunks=new_chunks,
            dtype=np.float32,
        )
        new_dyn[:, :C_dyn, :, :] = old_dyn
        new_dyn[:, C_dyn, :, :] = sws
        del old_dyn, sws
        logger.info("Dynamic array written.")

        dyn_means = np.array(store["norm/dynamic_mean"])
        dyn_stds = np.array(store["norm/dynamic_std"])
        new_means = np.append(dyn_means, sws_mean).astype(np.float32)
        new_stds = np.append(dyn_stds, sws_std).astype(np.float32)
        store.create_array("norm/dynamic_mean", data=new_means, overwrite=True)
        store.create_array("norm/dynamic_std", data=new_stds, overwrite=True)

        old_names = np.array(store["norm/dynamic_names"])
        new_names = np.append(old_names, "sws")
        store.create_array("norm/dynamic_names", data=new_names, overwrite=True)
        logger.info("Normalization stats and channel names updated.")

    # --- Verify ---
    logger.info("Verifying...")
    v_store = zarr.open_group(cfg.paths.zarr_store, mode="r")
    expected_C = C_dyn if args.overwrite else C_dyn + 1
    assert v_store["inputs/dynamic"].shape == (T, expected_C, H, W), "Shape mismatch!"
    v_names = list(np.array(v_store["norm/dynamic_names"]))
    assert v_names[SWS_IDX] == "sws", f"Channel {SWS_IDX} is {v_names[SWS_IDX]}, expected 'sws'"
    assert len(np.array(v_store["norm/dynamic_mean"])) == expected_C, "Norm means length mismatch!"
    logger.info(f"Verification passed. Channel names: {v_names}")

    # --- Summary ---
    logger.info("=" * 60)
    if args.overwrite:
        logger.info("SWS channel OVERWRITTEN with stress-modulated drainage.")
        logger.info(f"Channel {SWS_IDX} updated in-place. No config changes needed.")
    else:
        logger.info("SWS channel added successfully.")
        logger.info(f"Dynamic channels: {C_dyn} → {expected_C}")
        logger.info(f"Channel names: {v_names}")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info(f"  1. Update config.yaml: model.backbone.in_channels: "
                    f"{cfg.model.backbone.in_channels} → "
                    f"{cfg.model.backbone.in_channels + 1}")
        logger.info("  2. Train: conda run -n deep_field python train.py "
                    "--run-id v14-sws-stress --notes 'Fixed SWS: "
                    "stress-modulated drainage'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
