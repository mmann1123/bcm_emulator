"""Evaluate fire models: Track A vs Track B comparison, metrics, spatial maps.

Usage:
    conda run -n deep_field python scripts/fire_model/03_evaluate.py
"""

import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/mmann1123/extra_space/bcm_emulator/outputs/fire_model")
PANEL_PATH = OUTPUT_DIR / "panel" / "fire_panel.parquet"
ECOREGION_TIF = "/home/mmann1123/extra_space/Regions/ca_eco_l3.tif"
ZARR_PATH = "/home/mmann1123/extra_space/bcm_emulator/data/bcm_dataset.zarr"

# BCM grid
H, W = 1209, 941
from rasterio.transform import Affine
TRANSFORM = Affine(1000.0, 0.0, -374495.8364, 0.0, -1000.0, 592636.6658)

# Features (must match 02_train_model.py)
COMMON_FEATURES = [
    "ppt", "tmin", "tmax", "vpd", "srad", "kbdi", "sws", "vpd_roll6_std",
    "month_sin", "month_cos", "fire_season",
    "elev", "aridity_index", "windward_index",
    "fveg_forest", "fveg_shrub", "fveg_herb",
    "tsf_years", "tsf_log",
]
TRACK_A_FEATURES = COMMON_FEATURES + [
    "cwd_anom_a", "aet_anom_a", "pet_anom_a", "cwd_cum3_anom_a", "cwd_cum6_anom_a",
]
TRACK_B_FEATURES = COMMON_FEATURES + [
    "cwd_anom_b", "aet_anom_b", "pet_anom_b", "cwd_cum3_anom_b", "cwd_cum6_anom_b",
]


def compute_metrics(y_true, y_prob):
    """Compute classification metrics. Returns dict or None if insufficient data."""
    if len(y_true) < 10 or y_true.sum() < 2 or y_true.sum() == len(y_true):
        return None
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }


def load_ecoregions():
    """Load ecoregion raster, resample to BCM grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ECOREGION_TIF) as src:
        eco_data = np.full((H, W), 0, dtype=np.int32)
        reproject(
            source=rasterio.band(src, 1),
            destination=eco_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=TRANSFORM,
            dst_crs="EPSG:3310",
            resampling=Resampling.nearest,
        )
    return eco_data


def evaluate_track(df_test, features, model_path, track_name):
    """Run evaluation for one track. Returns predictions and metrics."""
    logger.info(f"Evaluating {track_name}...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = df_test[features].values
    y_test = df_test["fire"].values
    y_prob = model.predict_proba(X_test)[:, 1]

    # Overall metrics
    overall = compute_metrics(y_test, y_prob)
    logger.info(f"  Overall: AUC={overall['roc_auc']:.4f}, AP={overall['avg_precision']:.4f}, "
                f"Brier={overall['brier_score']:.4f}")

    # Monthly metrics
    monthly = {}
    for m in range(1, 13):
        mask = df_test["month"].values == m
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                monthly[m] = met

    # Quarterly metrics
    quarterly = {}
    q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12],
             "fire_season": [6, 7, 8, 9, 10, 11]}
    for qname, months in q_map.items():
        mask = np.isin(df_test["month"].values, months)
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                quarterly[qname] = met

    # Water year metrics
    wy_metrics = {}
    years = df_test["year"].values
    months = df_test["month"].values
    wy = np.where(months >= 10, years + 1, years)
    for wy_val in sorted(np.unique(wy)):
        if wy_val < 2020 or wy_val > 2024:
            continue
        mask = wy == wy_val
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                wy_metrics[int(wy_val)] = met

    # Ecoregion metrics
    eco_metrics = {}
    try:
        eco_raster = load_ecoregions()
        eco_vals = eco_raster[df_test["row"].values, df_test["col"].values]
        for eco_id in np.unique(eco_vals):
            if eco_id == 0:
                continue
            mask = eco_vals == eco_id
            if mask.sum() > 0:
                met = compute_metrics(y_test[mask], y_prob[mask])
                if met and met["n_positive"] >= 100:
                    eco_metrics[int(eco_id)] = met
    except Exception as e:
        logger.warning(f"  Ecoregion metrics failed: {e}")

    results = {
        "overall": overall,
        "monthly": monthly,
        "quarterly": quarterly,
        "water_year": wy_metrics,
        "ecoregion": eco_metrics,
    }

    return y_prob, results


def save_track_metrics(results, track_dir):
    """Save metrics for one track."""
    track_dir.mkdir(parents=True, exist_ok=True)

    # Overall
    with open(track_dir / "metrics_overall.json", "w") as f:
        json.dump(results["overall"], f, indent=2)

    # Monthly
    if results["monthly"]:
        rows = [{"month": m, **v} for m, v in sorted(results["monthly"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_month.csv", index=False)

    # Quarterly
    if results["quarterly"]:
        rows = [{"quarter": q, **v} for q, v in results["quarterly"].items()]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_quarter.csv", index=False)

    # Water year
    if results["water_year"]:
        rows = [{"water_year": wy, **v} for wy, v in sorted(results["water_year"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_water_year.csv", index=False)

    # Ecoregion
    if results["ecoregion"]:
        rows = [{"ecoregion_id": eid, **v} for eid, v in sorted(results["ecoregion"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_ecoregion.csv", index=False)


def save_calibration_curve(y_true, y_prob, path, track_name, n_bins=10):
    """Save reliability diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_prob[mask].mean())
            bin_means.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_centers, bin_means, "o-", label=track_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {track_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_roc_comparison(y_true, prob_a, prob_b, path):
    """Save overlaid ROC curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr_a, tpr_a, _ = roc_curve(y_true, prob_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_b)
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_a, tpr_a, label=f"Track A (BCMv8) AUC={auc_a:.4f}", linewidth=2)
    ax.plot(fpr_b, tpr_b, label=f"Track B (Emulator) AUC={auc_b:.4f}", linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: BCMv8 vs Emulator")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_spatial_maps(df_test, prob_col, track_dir):
    """Save fire probability GeoTIFFs for each water year fire season."""
    import rasterio

    track_dir.mkdir(parents=True, exist_ok=True)
    months_arr = df_test["month"].values
    years_arr = df_test["year"].values
    wy_arr = np.where(months_arr >= 10, years_arr + 1, years_arr)
    fire_season_mask = np.isin(months_arr, [6, 7, 8, 9, 10, 11])

    profile = {
        "driver": "GTiff", "dtype": "float32", "width": W, "height": H,
        "count": 1, "crs": "EPSG:3310", "transform": TRANSFORM,
        "nodata": -9999.0, "compress": "lzw",
    }

    for wy_val in tqdm(range(2020, 2025), desc="Spatial maps"):
        mask = (wy_arr == wy_val) & fire_season_mask
        if mask.sum() == 0:
            continue

        # Accumulate probability per pixel
        prob_sum = np.full((H, W), 0.0, dtype=np.float64)
        prob_count = np.zeros((H, W), dtype=np.int32)

        rows = df_test["row"].values[mask]
        cols = df_test["col"].values[mask]
        probs = prob_col[mask]

        np.add.at(prob_sum, (rows, cols), probs)
        np.add.at(prob_count, (rows, cols), 1)

        prob_mean = np.where(prob_count > 0, prob_sum / prob_count, -9999.0).astype(np.float32)

        out_path = track_dir / f"fire_prob_WY{wy_val}_fire_season.tif"
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(prob_mean[np.newaxis, :])

    logger.info(f"  Spatial maps saved to {track_dir}")


def main():
    logger.info("Loading panel...")
    df = pd.read_parquet(PANEL_PATH)
    df_test = df[df["split"] == "test"].copy()
    logger.info(f"Test set: {len(df_test)} rows, {df_test['fire'].sum()} fires")

    eval_dir = OUTPUT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate both tracks
    model_a_path = OUTPUT_DIR / "model" / "trackA" / "lr_calibrated.pkl"
    model_b_path = OUTPUT_DIR / "model" / "trackB" / "lr_calibrated.pkl"

    prob_a, results_a = evaluate_track(df_test, TRACK_A_FEATURES, model_a_path, "Track A (BCMv8)")
    prob_b, results_b = evaluate_track(df_test, TRACK_B_FEATURES, model_b_path, "Track B (Emulator)")

    # Save per-track metrics
    save_track_metrics(results_a, eval_dir / "trackA")
    save_track_metrics(results_b, eval_dir / "trackB")

    # Save test predictions
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for track_name, probs in [("trackA", prob_a), ("trackB", prob_b)]:
        pred_df = df_test[["year", "month", "row", "col", "fire"]].copy()
        pred_df["prob"] = probs
        pred_df.to_parquet(pred_dir / f"test_predictions_{track_name}.parquet", index=False)

    # Calibration curves
    y_test = df_test["fire"].values
    save_calibration_curve(y_test, prob_a, eval_dir / "trackA" / "calibration_curve.png", "Track A (BCMv8)")
    save_calibration_curve(y_test, prob_b, eval_dir / "trackB" / "calibration_curve.png", "Track B (Emulator)")

    # ROC comparison
    save_roc_comparison(y_test, prob_a, prob_b, eval_dir / "roc_comparison.png")

    # Spatial maps
    maps_dir = OUTPUT_DIR / "spatial_maps"
    save_spatial_maps(df_test, prob_a, maps_dir / "trackA")
    save_spatial_maps(df_test, prob_b, maps_dir / "trackB")

    # ---- Comparison summary ----
    summary_rows = []

    def add_row(metric, val_a, val_b):
        summary_rows.append({
            "metric": metric,
            "trackA_bcmv8": round(val_a, 4) if val_a is not None else None,
            "trackB_emulator": round(val_b, 4) if val_b is not None else None,
            "delta": round(val_a - val_b, 4) if val_a is not None and val_b is not None else None,
        })

    add_row("Overall ROC-AUC", results_a["overall"]["roc_auc"], results_b["overall"]["roc_auc"])
    add_row("Overall Avg Precision", results_a["overall"]["avg_precision"], results_b["overall"]["avg_precision"])
    add_row("Overall Brier Score", results_a["overall"]["brier_score"], results_b["overall"]["brier_score"])

    if "fire_season" in results_a["quarterly"] and "fire_season" in results_b["quarterly"]:
        add_row("Fire Season ROC-AUC",
                results_a["quarterly"]["fire_season"]["roc_auc"],
                results_b["quarterly"]["fire_season"]["roc_auc"])

    for wy in range(2020, 2025):
        if wy in results_a["water_year"] and wy in results_b["water_year"]:
            add_row(f"WY{wy} ROC-AUC",
                    results_a["water_year"][wy]["roc_auc"],
                    results_b["water_year"][wy]["roc_auc"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "comparison_summary.csv", index=False)

    # Print comparison
    print("\n" + "=" * 72)
    print("FIRE MODEL: TRACK A (BCMv8) vs TRACK B (EMULATOR)")
    print("=" * 72)
    print(f"{'Metric':<30s} {'Track A':>10s} {'Track B':>10s} {'Delta':>10s}")
    print("-" * 72)
    for _, row in summary_df.iterrows():
        a_str = f"{row['trackA_bcmv8']:.4f}" if row['trackA_bcmv8'] is not None else "N/A"
        b_str = f"{row['trackB_emulator']:.4f}" if row['trackB_emulator'] is not None else "N/A"
        d_str = f"{row['delta']:+.4f}" if row['delta'] is not None else "N/A"
        print(f"{row['metric']:<30s} {a_str:>10s} {b_str:>10s} {d_str:>10s}")
    print("=" * 72)

    delta_auc = results_a["overall"]["roc_auc"] - results_b["overall"]["roc_auc"]
    if abs(delta_auc) < 0.02:
        print(f"\nDelta AUC = {delta_auc:+.4f} (< 0.02) — EMULATOR IS OPERATIONALLY VIABLE")
    else:
        print(f"\nDelta AUC = {delta_auc:+.4f} (>= 0.02) — emulator degrades fire skill")

    # ---- Manifest ----
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        pass

    # Load TSF coefficients
    tsf_coefs = {}
    for track in ["trackA", "trackB"]:
        coef_path = OUTPUT_DIR / "model" / track / "coefficients.csv"
        if coef_path.exists():
            cdf = pd.read_csv(coef_path)
            for feat in ["tsf_log", "tsf_years"]:
                row = cdf[cdf["feature"] == feat]
                if len(row):
                    tsf_coefs[f"{feat}_coef_{track}"] = float(row.iloc[0]["coefficient"])

    manifest = {
        "run_date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "frap_filter": {"state": "CA", "objective": 1, "min_acres": 300},
        "train_period": "1984-01 to 2016-12",
        "calib_period": "2017-01 to 2019-09",
        "test_period": "2019-10 to 2024-09",
        "neg_grid_spacing_km": 5,
        "random_seed": 42,
        "feature_cols": TRACK_A_FEATURES,
        "n_train_pos": int(df[df["split"] == "train"]["fire"].sum()),
        "n_train_neg": int((df["split"] == "train").sum() - df[df["split"] == "train"]["fire"].sum()),
        "n_calib_pos": int(df[df["split"] == "calib"]["fire"].sum()),
        "n_test_pos": int(df_test["fire"].sum()),
        "n_test_neg": int(len(df_test) - df_test["fire"].sum()),
        "trackA_overall_auc": results_a["overall"]["roc_auc"],
        "trackB_overall_auc": results_b["overall"]["roc_auc"],
        "auc_delta": delta_auc,
        **tsf_coefs,
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to {OUTPUT_DIR / 'manifest.json'}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
