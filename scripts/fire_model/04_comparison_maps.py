"""Generate side-by-side predicted vs actual fire maps and burned area comparison.

For each water year fire season:
  - Left panel: predicted fire probability (full grid)
  - Right panel: actual fire perimeters rasterized to same grid
  - Bar chart: actual vs predicted burned km² by water year

Threshold calibrated on calib period (2017-2019) to match observed burned area.

Usage:
    conda run -n deep_field python scripts/fire_model/04_comparison_maps.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/mmann1123/extra_space/bcm_emulator/outputs/fire_model")
ZARR_PATH = "/home/mmann1123/extra_space/bcm_emulator/data/bcm_dataset.zarr"
H, W = 1209, 941
FIRE_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]


def load_fire_raster_by_wy():
    """Load fire raster and compute per-WY fire season burned masks."""
    fire_raster = np.load(str(OUTPUT_DIR / "fire_raster.npy"), mmap_mode="r")
    store = zarr.open_group(ZARR_PATH, mode="r")
    time_index = np.array(store["meta/time"])
    valid_mask = np.array(store["meta/valid_mask"])

    wy_burned = {}
    for wy in range(2020, 2025):
        burned = np.zeros((H, W), dtype=bool)
        # Oct-Nov of prior year
        for m in [10, 11]:
            ym = f"{wy-1:04d}-{m:02d}"
            idx = np.searchsorted(time_index, ym)
            if idx < len(time_index) and time_index[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        # Jun-Sep of WY year
        for m in [6, 7, 8, 9]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index, ym)
            if idx < len(time_index) and time_index[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        burned &= valid_mask
        wy_burned[wy] = burned

    return wy_burned, valid_mask


def calibrate_threshold(panel_path, track="trackA"):
    """Find threshold on calib period where predicted area ≈ actual area."""
    df = pd.read_parquet(panel_path)
    calib = df[df["split"] == "calib"].copy()

    # Use full grid predictions if available, otherwise panel
    # For calibration, panel is fine since we're matching rates not areas
    y_true = calib["fire"].values

    # Load calib predictions from the model
    import pickle
    model_path = OUTPUT_DIR / "model" / track / "lr_calibrated.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    suffix = "_a" if track == "trackA" else "_b"
    common = [
        "ppt", "tmin", "tmax", "vpd", "srad", "kbdi", "sws", "vpd_roll6_std",
        "month_sin", "month_cos", "fire_season",
        "elev", "aridity_index", "windward_index",
        "fveg_forest", "fveg_shrub", "fveg_herb",
        "tsf_years", "tsf_log",
    ]
    hydro = [f"cwd_anom{suffix}", f"aet_anom{suffix}", f"pet_anom{suffix}",
             f"cwd_cum3_anom{suffix}", f"cwd_cum6_anom{suffix}"]
    features = common + hydro

    X_calib = calib[features].values
    y_prob = model.predict_proba(X_calib)[:, 1]

    actual_rate = y_true.mean()

    # Search for threshold that gives predicted rate closest to actual
    best_thresh = 0.05
    best_diff = float("inf")
    for t in np.arange(0.001, 0.20, 0.001):
        pred_rate = (y_prob >= t).mean()
        diff = abs(pred_rate - actual_rate)
        if diff < best_diff:
            best_diff = diff
            best_thresh = t

    pred_rate = (y_prob >= best_thresh).mean()
    logger.info(f"Calibrated threshold ({track}): {best_thresh:.3f} "
                f"(actual rate={actual_rate:.4f}, predicted rate={pred_rate:.4f})")
    return best_thresh


def make_comparison_figure(wy, prob_map, burned_mask, valid_mask, threshold, track_name, out_path):
    """Create side-by-side predicted vs actual figure for one water year."""
    # Mask invalid pixels
    prob_display = np.ma.masked_where(~valid_mask, prob_map)
    burned_display = np.ma.masked_where(~valid_mask, burned_mask.astype(float))
    predicted_burn = np.ma.masked_where(~valid_mask, (prob_map >= threshold).astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: Predicted probability
    im1 = axes[0].imshow(prob_display, cmap="YlOrRd", vmin=0, vmax=0.15, origin="upper")
    axes[0].set_title(f"Predicted Fire Probability\n{track_name} — WY{wy} Fire Season", fontsize=11)
    plt.colorbar(im1, ax=axes[0], shrink=0.6, label="Probability")

    # Panel 2: Predicted burned (thresholded) vs actual
    # Create RGB overlay: predicted=orange, actual=red, overlap=dark red
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    pred_bool = (prob_map >= threshold) & valid_mask
    actual_bool = burned_mask & valid_mask

    # Background: light gray for valid, white for invalid
    overlay[valid_mask] = [0.9, 0.9, 0.9]
    overlay[~valid_mask] = [1.0, 1.0, 1.0]

    # Predicted only: orange
    pred_only = pred_bool & ~actual_bool
    overlay[pred_only] = [1.0, 0.6, 0.0]

    # Actual only: blue
    actual_only = actual_bool & ~pred_bool
    overlay[actual_only] = [0.2, 0.4, 0.8]

    # Overlap: red
    both = pred_bool & actual_bool
    overlay[both] = [0.8, 0.0, 0.0]

    axes[1].imshow(overlay, origin="upper")
    axes[1].set_title(f"Predicted vs Actual Burned Area\nWY{wy} (threshold={threshold:.3f})", fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1.0, 0.6, 0.0), label=f"Predicted only ({pred_only.sum()} km²)"),
        Patch(facecolor=(0.2, 0.4, 0.8), label=f"Actual only ({actual_only.sum()} km²)"),
        Patch(facecolor=(0.8, 0.0, 0.0), label=f"Overlap ({both.sum()} km²)"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower left", fontsize=9)

    # Panel 3: Actual fires only
    actual_display = np.zeros((H, W, 3), dtype=np.float32)
    actual_display[valid_mask] = [0.9, 0.9, 0.9]
    actual_display[~valid_mask] = [1.0, 1.0, 1.0]
    actual_display[actual_bool] = [0.8, 0.0, 0.0]

    axes[2].imshow(actual_display, origin="upper")
    axes[2].set_title(f"Actual Fires\nWY{wy} ({actual_bool.sum()} km² burned)", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {out_path.name}")


def make_area_comparison_chart(area_data, threshold, out_path):
    """Bar chart of actual vs predicted burned area by water year."""
    fig, ax = plt.subplots(figsize=(10, 6))

    wys = sorted(area_data.keys())
    x = np.arange(len(wys))
    width = 0.25

    actual = [area_data[wy]["actual_km2"] for wy in wys]
    pred_a = [area_data[wy]["predicted_km2_trackA"] for wy in wys]
    pred_b = [area_data[wy]["predicted_km2_trackB"] for wy in wys]

    bars1 = ax.bar(x - width, actual, width, label="Actual Burned", color="#c0392b", alpha=0.85)
    bars2 = ax.bar(x, pred_a, width, label="Predicted (BCMv8)", color="#e67e22", alpha=0.85)
    bars3 = ax.bar(x + width, pred_b, width, label="Predicted (Emulator)", color="#2980b9", alpha=0.85)

    ax.set_xlabel("Water Year", fontsize=12)
    ax.set_ylabel("Burned Area (km²)", fontsize=12)
    ax.set_title(f"Actual vs Predicted Burned Area — Fire Season\n(threshold={threshold:.3f})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"WY{wy}" for wy in wys])
    ax.legend(fontsize=11)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
                        f"{int(height):,}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(max(actual), max(pred_a), max(pred_b)) * 1.15)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path.name}")


def main():
    maps_dir = OUTPUT_DIR / "spatial_maps"
    comp_dir = OUTPUT_DIR / "comparison"
    comp_dir.mkdir(exist_ok=True)

    # Calibrate threshold on calib period
    threshold = calibrate_threshold(OUTPUT_DIR / "panel" / "fire_panel.parquet", "trackA")

    # Load actual fire data by WY
    wy_burned, valid_mask = load_fire_raster_by_wy()

    area_data = {}

    for track, track_name in [("trackA", "Track A (BCMv8)"), ("trackB", "Track B (Emulator)")]:
        logger.info(f"Processing {track_name}...")

        for wy in range(2020, 2025):
            # Load predicted probability map
            tif_path = maps_dir / track / f"fire_prob_WY{wy}_fire_season.tif"
            if not tif_path.exists():
                logger.warning(f"  Missing: {tif_path}")
                continue

            with rasterio.open(str(tif_path)) as src:
                prob_map = src.read(1)

            prob_map[prob_map == -9999.0] = 0.0  # nodata → 0 for thresholding

            burned_mask = wy_burned[wy]
            actual_km2 = int(burned_mask.sum())
            predicted_km2 = int(((prob_map >= threshold) & valid_mask).sum())

            if wy not in area_data:
                area_data[wy] = {"actual_km2": actual_km2}
            area_data[wy][f"predicted_km2_{track}"] = predicted_km2

            logger.info(f"  WY{wy}: actual={actual_km2} km², predicted={predicted_km2} km²")

            # Side-by-side figure
            out_path = comp_dir / f"{track}_WY{wy}_comparison.png"
            make_comparison_figure(wy, prob_map, burned_mask, valid_mask,
                                  threshold, track_name, out_path)

    # Burned area comparison chart
    make_area_comparison_chart(area_data, threshold, comp_dir / "burned_area_comparison.png")

    # Save area data as CSV
    rows = []
    for wy in sorted(area_data.keys()):
        rows.append({"water_year": wy, **area_data[wy]})
    area_df = pd.DataFrame(rows)
    area_df.to_csv(comp_dir / "burned_area_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print(f"BURNED AREA COMPARISON (threshold={threshold:.3f})")
    print("=" * 60)
    print(f"{'WY':<8} {'Actual':>10} {'BCMv8':>10} {'Emulator':>10} {'Δ A':>8} {'Δ B':>8}")
    print("-" * 60)
    for wy in sorted(area_data.keys()):
        d = area_data[wy]
        a = d["actual_km2"]
        pa = d.get("predicted_km2_trackA", 0)
        pb = d.get("predicted_km2_trackB", 0)
        print(f"WY{wy}  {a:>10,} {pa:>10,} {pb:>10,} {pa-a:>+8,} {pb-a:>+8,}")
    print("=" * 60)

    logger.info(f"All outputs saved to {comp_dir}")


if __name__ == "__main__":
    main()
