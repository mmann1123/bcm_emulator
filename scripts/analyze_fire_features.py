"""Regress candidate fire features against CWD extreme bias map.

Loads temporal summaries (mean, summer mean, max) of each fire feature and
regresses them against the per-pixel CWD extreme bias from v5-awc-windward
to identify which features best predict where CWD is under/over-estimated.

Usage:
    conda run -n deep_field python scripts/analyze_fire_features.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_feature_summary(feature_dir: Path, prefix: str, valid_mask: np.ndarray) -> dict:
    """Load all monthly rasters for a feature and compute temporal summaries.

    Returns dict with keys: 'mean', 'summer_mean' (Jun-Sep), 'max'.
    """
    import rasterio

    files = sorted(feature_dir.glob(f"{prefix}-*.tif"))
    if not files:
        logger.warning(f"No files found in {feature_dir} with prefix {prefix}")
        return None

    H, W = valid_mask.shape
    all_data = []
    summer_data = []

    for f in files:
        # Extract YYYYMM from filename
        ym = f.stem.split("-")[-1]
        month = int(ym[4:6])

        with rasterio.open(str(f)) as src:
            data = src.read(1).astype(np.float32)

        if data.shape != (H, W):
            logger.warning(f"Shape mismatch for {f}: {data.shape} vs ({H}, {W})")
            continue

        data[~valid_mask] = np.nan
        all_data.append(data)
        if month in (6, 7, 8, 9):
            summer_data.append(data)

    if not all_data:
        return None

    stack = np.stack(all_data, axis=0)
    result = {
        "mean": np.nanmean(stack, axis=0),
        "max": np.nanmax(stack, axis=0),
    }
    if summer_data:
        summer_stack = np.stack(summer_data, axis=0)
        result["summer_mean"] = np.nanmean(summer_stack, axis=0)

    return result


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.config import load_config
    from src.utils.io_helpers import get_valid_mask, write_raster, get_bcm_reference_profile

    config_path = PROJECT_ROOT / "config.yaml"
    cfg = load_config(str(config_path))
    bcm_profile = get_bcm_reference_profile(cfg.paths.bcm_dir)

    valid_mask = get_valid_mask(cfg.paths.bcm_dir)
    H, W = valid_mask.shape
    n_valid = valid_mask.sum()
    logger.info(f"Valid pixels: {n_valid:,} / {H * W:,}")

    # --- Load CWD extreme bias target ---
    target_path = PROJECT_ROOT / "snapshots" / "v5-awc-windward" / "spatial_maps" / "extreme_bias_cwd.tif"
    if not target_path.exists():
        logger.error(f"Target not found: {target_path}")
        logger.info("Available snapshots:")
        snap_dir = PROJECT_ROOT / "snapshots"
        if snap_dir.exists():
            for d in sorted(snap_dir.iterdir()):
                if d.is_dir():
                    logger.info(f"  {d.name}")
        sys.exit(1)

    import rasterio
    with rasterio.open(str(target_path)) as src:
        cwd_bias = src.read(1).astype(np.float32)
    cwd_bias[~valid_mask] = np.nan
    logger.info(f"CWD extreme bias: mean={np.nanmean(cwd_bias):.2f}, std={np.nanstd(cwd_bias):.2f}")

    # --- Load candidate features ---
    prism_monthly = Path(cfg.paths.prism_monthly_dir)

    features = {}
    feature_configs = [
        ("hdd30", prism_monthly / "hdd30", "hdd30"),
        ("hdd35", prism_monthly / "hdd35", "hdd35"),
        ("heat_stress", prism_monthly / "heat_stress", "heat_stress"),
        ("drought_code", prism_monthly / "drought_code", "drought_code"),
    ]

    for name, fdir, prefix in feature_configs:
        if fdir.exists():
            summary = load_feature_summary(fdir, prefix, valid_mask)
            if summary:
                features[name] = summary
                logger.info(f"Loaded {name}: {len(summary)} summaries")
        else:
            logger.warning(f"Feature directory not found: {fdir}")

    if not features:
        logger.error("No features loaded. Run fire_features step first.")
        sys.exit(1)

    # --- Univariate correlations ---
    target_flat = cwd_bias[valid_mask]
    valid_target = ~np.isnan(target_flat)

    print("\n" + "=" * 70)
    print("UNIVARIATE CORRELATIONS: Feature vs CWD Extreme Bias")
    print("=" * 70)
    print(f"{'Feature':<30} {'Metric':<15} {'Pearson r':>10} {'R²':>10}")
    print("-" * 70)

    all_candidates = {}  # (name, metric) -> flat array

    for feat_name, summaries in features.items():
        for metric_name, data in summaries.items():
            key = f"{feat_name}_{metric_name}"
            feat_flat = data[valid_mask]
            both_valid = valid_target & ~np.isnan(feat_flat)

            if both_valid.sum() < 100:
                continue

            t = target_flat[both_valid]
            f = feat_flat[both_valid]

            r = np.corrcoef(t, f)[0, 1]
            r2 = r ** 2

            print(f"{feat_name:<30} {metric_name:<15} {r:>10.4f} {r2:>10.4f}")
            all_candidates[key] = feat_flat

    # --- Top candidates ---
    print("\n" + "=" * 70)
    print("TOP CANDIDATES (by |r|)")
    print("=" * 70)

    correlations = []
    for key, feat_flat in all_candidates.items():
        both_valid = valid_target & ~np.isnan(feat_flat)
        if both_valid.sum() < 100:
            continue
        r = np.corrcoef(target_flat[both_valid], feat_flat[both_valid])[0, 1]
        correlations.append((key, r, r ** 2))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, (key, r, r2) in enumerate(correlations[:10]):
        print(f"  {i + 1}. {key:<40} r={r:+.4f}  R²={r2:.4f}")

    # --- Two-feature stepwise ---
    if len(all_candidates) >= 2:
        print("\n" + "=" * 70)
        print("TWO-FEATURE COMBINATIONS (OLS R²)")
        print("=" * 70)

        from itertools import combinations

        best_pair = None
        best_r2 = 0.0

        keys = list(all_candidates.keys())
        for k1, k2 in combinations(keys, 2):
            f1 = all_candidates[k1]
            f2 = all_candidates[k2]
            both_valid = valid_target & ~np.isnan(f1) & ~np.isnan(f2)
            if both_valid.sum() < 100:
                continue

            t = target_flat[both_valid]
            X = np.column_stack([f1[both_valid], f2[both_valid], np.ones(both_valid.sum())])

            # OLS: beta = (X'X)^-1 X'y
            try:
                beta = np.linalg.lstsq(X, t, rcond=None)[0]
                pred = X @ beta
                ss_res = np.sum((t - pred) ** 2)
                ss_tot = np.sum((t - t.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot
            except np.linalg.LinAlgError:
                continue

            if r2 > best_r2:
                best_r2 = r2
                best_pair = (k1, k2, r2)

        if best_pair:
            print(f"  Best pair: {best_pair[0]} + {best_pair[1]}")
            print(f"  Combined R² = {best_pair[2]:.4f}")

        # Show top 5 pairs
        pairs = []
        for k1, k2 in combinations(keys, 2):
            f1 = all_candidates[k1]
            f2 = all_candidates[k2]
            both_valid = valid_target & ~np.isnan(f1) & ~np.isnan(f2)
            if both_valid.sum() < 100:
                continue
            t = target_flat[both_valid]
            X = np.column_stack([f1[both_valid], f2[both_valid], np.ones(both_valid.sum())])
            try:
                beta = np.linalg.lstsq(X, t, rcond=None)[0]
                pred = X @ beta
                ss_res = np.sum((t - pred) ** 2)
                ss_tot = np.sum((t - t.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                pairs.append((k1, k2, r2))
            except np.linalg.LinAlgError:
                continue

        pairs.sort(key=lambda x: x[2], reverse=True)
        print("\n  Top 5 pairs:")
        for i, (k1, k2, r2) in enumerate(pairs[:5]):
            print(f"    {i + 1}. {k1} + {k2}  R²={r2:.4f}")

    # --- Save spatial correlation maps ---
    out_dir = PROJECT_ROOT / "outputs" / "fire_feature_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save top 3 features as spatial maps
    for key, r, r2 in correlations[:3]:
        feat_flat = all_candidates[key]
        feat_map = np.full((H, W), np.nan, dtype=np.float32)
        feat_map[valid_mask] = feat_flat
        out_path = out_dir / f"{key}.tif"
        write_raster(str(out_path), feat_map, bcm_profile)
        logger.info(f"Saved spatial map: {out_path}")

    logger.info(f"\nAnalysis outputs saved to {out_dir}")
    print("\n" + "=" * 70)
    print("NEXT STEP: Review results and select features for v9 model channels")
    print("=" * 70)


if __name__ == "__main__":
    main()
