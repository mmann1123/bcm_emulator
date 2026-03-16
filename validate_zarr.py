"""Validate zarr store after rebuild — check shapes, AWC values, windward index spatial patterns."""

import argparse
import logging
import sys

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate(zarr_path: str) -> bool:
    """Run all validation checks. Returns True if all pass."""
    store = zarr.open(zarr_path, mode="r")
    passed = True

    # --- 1. Static shape ---
    static = np.array(store["inputs/static"])
    expected_shape = (10, 1209, 941)
    if static.shape == expected_shape:
        logger.info(f"[PASS] Static shape: {static.shape}")
    else:
        logger.error(f"[FAIL] Static shape: {static.shape}, expected {expected_shape}")
        passed = False

    # --- 2. Dynamic shape ---
    dyn_shape = store["inputs/dynamic"].shape
    if dyn_shape[1] == 10:
        logger.info(f"[PASS] Dynamic channels: {dyn_shape[1]}")
    else:
        logger.error(f"[FAIL] Dynamic channels: {dyn_shape[1]}, expected 10")
        passed = False

    valid_mask = np.array(store["meta/valid_mask"])
    n_valid = valid_mask.sum()
    logger.info(f"  Valid pixels: {n_valid:,}")

    # --- 3. AWC values (channel 7) ---
    awc = static[7]
    awc_valid = awc[valid_mask]
    awc_nonzero = (awc_valid != 0).sum()
    awc_min, awc_max, awc_mean = awc_valid.min(), awc_valid.max(), awc_valid.mean()
    logger.info(f"  AWC (ch7): min={awc_min:.1f}, max={awc_max:.1f}, mean={awc_mean:.1f}, nonzero={awc_nonzero:,}/{n_valid:,}")
    if awc_nonzero == 0:
        logger.error("[FAIL] AWC channel is all zeros")
        passed = False
    elif awc_max < 10 or awc_max > 1000:
        logger.warning(f"[WARN] AWC max={awc_max:.1f} — expected range ~50-300mm, check units")
    else:
        logger.info("[PASS] AWC has reasonable values")

    # --- 4. Windward index (channel 8) — spatial sanity ---
    windward = static[8]
    wi_valid = windward[valid_mask]
    wi_min, wi_max, wi_mean = wi_valid.min(), wi_valid.max(), wi_valid.mean()
    wi_neg_frac = (wi_valid < 0).sum() / n_valid
    wi_pos_frac = (wi_valid > 0).sum() / n_valid
    logger.info(f"  Windward (ch8): min={wi_min:.1f}, max={wi_max:.1f}, mean={wi_mean:.1f}")
    logger.info(f"  Windward: {wi_neg_frac:.1%} negative (leeward), {wi_pos_frac:.1%} positive (windward)")

    if wi_min == 0 and wi_max == 0:
        logger.error("[FAIL] Windward index is all zeros")
        passed = False
    elif wi_neg_frac < 0.05 or wi_pos_frac < 0.05:
        logger.warning("[WARN] Windward index has very little spatial contrast")
    else:
        logger.info("[PASS] Windward index has spatial contrast")

    # Rough spatial check: western columns should have more positive (windward) values
    # than eastern columns (leeward / Central Valley)
    H, W = windward.shape
    west_third = windward[:, :W // 3]
    east_third = windward[:, 2 * W // 3:]
    west_mask = valid_mask[:, :W // 3]
    east_mask = valid_mask[:, 2 * W // 3:]
    if west_mask.sum() > 0 and east_mask.sum() > 0:
        west_mean = west_third[west_mask].mean()
        east_mean = east_third[east_mask].mean()
        logger.info(f"  West-third mean: {west_mean:.1f}, East-third mean: {east_mean:.1f}")
        if west_mean > east_mean:
            logger.info("[PASS] Western pixels more windward than eastern (expected)")
        else:
            logger.warning("[WARN] Eastern pixels more windward than western — unexpected")

    # --- 5. FVEG (channel 9) ---
    fveg = static[9]
    fveg_valid = fveg[valid_mask]
    n_classes = len(np.unique(fveg_valid[fveg_valid > 0]))
    logger.info(f"  FVEG (ch9): {n_classes} unique classes, max_id={fveg_valid.max():.0f}")
    if n_classes == 0:
        logger.warning("[WARN] FVEG channel has no non-zero classes")
    else:
        logger.info("[PASS] FVEG classes present")

    # --- 6. Normalization stats shape ---
    stat_mean = np.array(store["norm/static_mean"])
    stat_std = np.array(store["norm/static_std"])
    if len(stat_mean) == 10 and len(stat_std) == 10:
        logger.info(f"[PASS] Static norm stats: {len(stat_mean)} channels")
    else:
        logger.error(f"[FAIL] Static norm stats: {len(stat_mean)} channels, expected 10")
        passed = False

    # Check FVEG (ch9) has identity normalization
    if stat_mean[9] == 0.0 and stat_std[9] == 1.0:
        logger.info("[PASS] FVEG normalization is identity (mean=0, std=1)")
    else:
        logger.error(f"[FAIL] FVEG norm: mean={stat_mean[9]}, std={stat_std[9]} — expected (0, 1)")
        passed = False

    # --- 7. Print all static channel stats ---
    ch_names = ["elev", "topo_solar", "lat", "lon", "ksat", "sand", "clay", "awc", "windward", "fveg"]
    logger.info("\n  Static channel summary:")
    logger.info(f"  {'Channel':<12} {'Mean':>10} {'Std':>10} {'NormMean':>10} {'NormStd':>10}")
    for i, name in enumerate(ch_names):
        vals = static[i][valid_mask]
        logger.info(
            f"  {name:<12} {vals.mean():>10.2f} {vals.std():>10.2f} "
            f"{stat_mean[i]:>10.4f} {stat_std[i]:>10.4f}"
        )

    # --- Summary ---
    if passed:
        logger.info("\n=== ALL CHECKS PASSED ===")
    else:
        logger.error("\n=== SOME CHECKS FAILED ===")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate zarr store after rebuild")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--zarr", default=None, help="Override zarr path")
    args = parser.parse_args()

    if args.zarr:
        zarr_path = args.zarr
    else:
        from src.utils.config import load_config
        cfg = load_config(args.config)
        zarr_path = cfg.paths.zarr_store

    logger.info(f"Validating zarr store: {zarr_path}")
    ok = validate(zarr_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
