"""Per-pixel NSE maps and extreme-bias maps as GeoTIFF and PNG visualization."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_nse_maps(
    observed: dict,
    predicted: dict,
    bcm_profile: dict,
    valid_mask: np.ndarray,
    output_dir: str,
) -> None:
    """Compute and save per-pixel NSE maps for all variables.

    Parameters
    ----------
    observed : dict
        Ground truth arrays keyed by variable, each (T, H, W).
    predicted : dict
        Predicted arrays keyed by variable, each (T, H, W).
    bcm_profile : dict
        BCM grid rasterio profile.
    valid_mask : np.ndarray
        Boolean mask of valid pixels (H, W).
    output_dir : str
        Output directory for GeoTIFFs and PNGs.
    """
    from ..evaluation.metrics import compute_pixel_nse
    from ..utils.io_helpers import write_raster

    out_path = Path(output_dir) / "spatial_maps"
    out_path.mkdir(parents=True, exist_ok=True)

    for var in ["pet", "pck", "aet", "cwd"]:
        if var not in observed or var not in predicted:
            continue

        nse_map = compute_pixel_nse(observed[var], predicted[var])
        nse_map[~valid_mask] = -9999.0

        # Save GeoTIFF
        tif_path = out_path / f"nse_{var}.tif"
        write_raster(str(tif_path), nse_map, bcm_profile)
        logger.info(f"Saved NSE map: {tif_path}")

        # Save PNG visualization
        _save_nse_png(nse_map, valid_mask, str(out_path / f"nse_{var}.png"), var)


def _save_nse_png(
    nse_map: np.ndarray,
    valid_mask: np.ndarray,
    out_path: str,
    var_name: str,
) -> None:
    """Save NSE map as a PNG with colorbar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        logger.warning("matplotlib not available, skipping PNG output")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Mask invalid pixels
    plot_data = nse_map.copy().astype(float)
    plot_data[~valid_mask] = np.nan

    # Use diverging colormap centered at 0
    vmin = max(-1.0, np.nanpercentile(plot_data, 2))
    vmax = 1.0
    # Ensure vmin < vcenter < vmax for TwoSlopeNorm
    if vmin >= 0:
        vmin = -0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.imshow(plot_data, cmap="RdYlGn", norm=norm)
    ax.set_title(f"Per-Pixel NSE: {var_name.upper()}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Stats
    valid_nse = plot_data[valid_mask]
    median_nse = np.nanmedian(valid_nse)
    pct_positive = np.nanmean(valid_nse > 0) * 100

    ax.text(
        0.02, 0.02,
        f"Median NSE: {median_nse:.3f}\n"
        f"NSE > 0: {pct_positive:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.colorbar(im, ax=ax, shrink=0.6, label="NSE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved NSE PNG: {out_path}")


def save_extreme_bias_maps(
    observed: dict,
    predicted: dict,
    bcm_profile: dict,
    valid_mask: np.ndarray,
    output_dir: str,
    quantile: float = 0.95,
) -> None:
    """Compute and save per-pixel extreme bias maps for AET and CWD.

    Parameters
    ----------
    observed : dict
        Ground truth arrays keyed by variable, each (T, H, W).
    predicted : dict
        Predicted arrays keyed by variable, each (T, H, W).
    bcm_profile : dict
        BCM grid rasterio profile.
    valid_mask : np.ndarray
        Boolean mask of valid pixels (H, W).
    output_dir : str
        Output directory for GeoTIFFs and PNGs.
    quantile : float
        Quantile threshold (default 0.95 for P95).
    """
    from ..evaluation.metrics import compute_pixel_extreme_bias
    from ..utils.io_helpers import write_raster

    out_path = Path(output_dir) / "spatial_maps"
    out_path.mkdir(parents=True, exist_ok=True)

    var_notes = {
        "cwd": "Negative bias = model underpredicts drought severity",
        "aet": "Negative bias = model underpredicts water consumption",
    }

    for var in ["aet", "cwd"]:
        if var not in observed or var not in predicted:
            continue

        bias_map = compute_pixel_extreme_bias(observed[var], predicted[var], quantile=quantile)
        bias_map[~valid_mask] = -9999.0

        # Save GeoTIFF
        tif_path = out_path / f"extreme_bias_{var}.tif"
        write_raster(str(tif_path), bias_map, bcm_profile)
        logger.info(f"Saved extreme bias map: {tif_path}")

        # Save PNG
        _save_extreme_bias_png(
            bias_map, valid_mask, observed[var], quantile,
            str(out_path / f"extreme_bias_{var}.png"),
            var, var_notes[var],
        )


def _save_extreme_bias_png(
    bias_map: np.ndarray,
    valid_mask: np.ndarray,
    observed_3d: np.ndarray,
    quantile: float,
    out_path: str,
    var_name: str,
    sign_note: str,
) -> None:
    """Save extreme bias map as a PNG with diverging colormap and stats."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        logger.warning("matplotlib not available, skipping PNG output")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    plot_data = bias_map.copy().astype(float)
    plot_data[~valid_mask] = np.nan

    valid_bias = plot_data[valid_mask & ~np.isnan(plot_data)]
    if len(valid_bias) == 0:
        plt.close()
        return

    # Symmetric limits centered at 0
    abs_max = max(abs(np.nanpercentile(valid_bias, 2)), abs(np.nanpercentile(valid_bias, 98)))
    if abs_max == 0:
        abs_max = 1.0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(plot_data, cmap="RdBu", norm=norm)

    pct_label = int(quantile * 100)
    ax.set_title(
        f"{var_name.upper()} Extreme Bias (obs > P{pct_label}): mean(pred - obs)",
        fontsize=13,
    )
    ax.text(
        0.5, 0.97,
        "Blue = model underpredicts extremes, Red = model overpredicts",
        transform=ax.transAxes, fontsize=10, ha="center", va="top",
        style="italic", color="gray",
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Stats box
    median_bias = np.nanmedian(valid_bias)
    pct_neg = np.nanmean(valid_bias < 0) * 100

    # Mean count of extreme events per pixel
    thresholds = np.nanpercentile(observed_3d, quantile * 100, axis=0)
    extreme_counts = np.sum(observed_3d >= thresholds[np.newaxis, :, :], axis=0)
    mean_extreme_count = np.nanmean(extreme_counts[valid_mask])

    ax.text(
        0.02, 0.02,
        f"Median bias: {median_bias:.2f} mm\n"
        f"Pixels underpredicting: {pct_neg:.1f}%\n"
        f"Mean extreme events/pixel: {mean_extreme_count:.1f}\n"
        f"{sign_note}",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.colorbar(im, ax=ax, shrink=0.6, label="Bias (mm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved extreme bias PNG: {out_path}")
