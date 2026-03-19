"""Compute fire-relevant dynamic features from daily PRISM tmax and ppt.

Features:
    1. HDD30 / HDD35: Heat Stress Degree Days above 30°C / 35°C base
    2. Sigmoid heat stress accumulator (smooth onset around 30°C)
    3. Drought Code (DC): Canadian FWI deep fuel moisture index (Van Wagner 1987)

All outputs are monthly BCM-grid rasters written to prism_monthly_dir.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Van Wagner (1987) Table 2: day-length adjustment factors by month.
# Index 0 = January, 11 = December.
# Rows: latitude bands [<=15, 15-25, 25-35, 35-45, 45-55, 55-65, 65-90] (°N)
# For S hemisphere, shift by 6 months — not needed for California.
DC_DAY_LENGTH_FACTORS = np.array(
    [
        [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5],  # <= 15°N
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],  # 15-25°N  (noqa: E501)
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],  # 25-35°N  (noqa: E501)
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],  # 35-45°N (noqa: E501)
        [13.9, 12.4, 10.9, 9.4, 8.1, 7.0, 7.2, 8.5, 10.2, 11.8, 13.3, 14.3],  # 45-55°N (noqa: E501)
        [16.7, 14.3, 11.9, 9.5, 7.4, 6.2, 6.7, 8.8, 11.2, 13.7, 16.0, 17.1],  # 55-65°N (noqa: E501)
        [20.6, 16.1, 12.4, 9.0, 6.1, 4.4, 5.2, 8.3, 12.0, 15.9, 19.7, 21.6],  # 65-90°N (noqa: E501)
    ],
    dtype=np.float32,
)

# Latitude band edges (degrees N)
DC_LAT_BANDS = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 90.0])


def _get_lat_band_index(lat: np.ndarray) -> np.ndarray:
    """Map latitude values to DC_DAY_LENGTH_FACTORS row indices."""
    idx = np.searchsorted(DC_LAT_BANDS, np.abs(lat))
    return np.clip(idx, 0, len(DC_LAT_BANDS) - 1)


def _get_day_length_factor(lat_grid: np.ndarray, month: int) -> np.ndarray:
    """Get day-length factor L_f for each pixel given latitude and month (1-indexed)."""
    band_idx = _get_lat_band_index(lat_grid)
    return DC_DAY_LENGTH_FACTORS[band_idx, month - 1]


def _build_lat_grid(bcm_profile: dict) -> np.ndarray:
    """Compute WGS84 latitude for each BCM grid pixel.

    Transforms BCM grid centroids (EPSG:3310) to WGS84 to get latitude.
    """
    from pyproj import Transformer

    H, W = bcm_profile["height"], bcm_profile["width"]
    transform = bcm_profile["transform"]

    # Pixel center coordinates in projected CRS
    cols = np.arange(W, dtype=np.float64) + 0.5
    rows = np.arange(H, dtype=np.float64) + 0.5
    x_coords = transform.c + cols * transform.a
    y_coords = transform.f + rows * transform.e

    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Transform to WGS84
    transformer = Transformer.from_crs(
        str(bcm_profile["crs"]), "EPSG:4326", always_xy=True
    )
    _, lat_grid = transformer.transform(x_grid.ravel(), y_grid.ravel())
    return lat_grid.reshape(H, W).astype(np.float32)


def _reproject_daily_to_bcm(tif_path: str, bcm_profile: dict) -> np.ndarray:
    """Read a daily PRISM raster and reproject to BCM grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    H, W = bcm_profile["height"], bcm_profile["width"]

    with rasterio.open(tif_path) as src:
        dst_data = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bcm_profile["transform"],
            dst_crs=bcm_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return dst_data


def compute_hdd(
    daily_tmax_dir: str,
    out_dir: str,
    bcm_profile: dict,
    bases: tuple = (30.0, 35.0),
) -> None:
    """Compute monthly Heat Stress Degree Days for each base temperature.

    HDD = Σ max(0, tmax_daily - base) for all days in month.

    Parameters
    ----------
    daily_tmax_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily tmax GeoTIFFs.
    out_dir : str
        Output directory (prism_monthly_dir). Creates hdd30/ and hdd35/ subdirs.
    bcm_profile : dict
        BCM grid rasterio profile for reprojection.
    bases : tuple of float
        Base temperatures in °C.
    """
    from ..utils.io_helpers import write_raster

    daily_base = Path(daily_tmax_dir)
    H, W = bcm_profile["height"], bcm_profile["width"]

    # Create output dirs for each base
    out_dirs = {}
    for base in bases:
        name = f"hdd{int(base)}"
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[base] = d

    for month_dir in sorted(daily_base.iterdir()):
        if not month_dir.is_dir():
            continue

        ym = month_dir.name  # e.g., "198101"

        # Check if all outputs already exist
        all_exist = all(
            (out_dirs[b] / f"hdd{int(b)}-{ym}.tif").exists() for b in bases
        )
        if all_exist:
            continue

        daily_files = sorted(month_dir.glob("*.tif"))
        if not daily_files:
            continue

        # Accumulate HDD for each base
        accumulators = {b: np.zeros((H, W), dtype=np.float32) for b in bases}

        for df in daily_files:
            tmax = _reproject_daily_to_bcm(str(df), bcm_profile)
            valid = ~np.isnan(tmax)
            for base in bases:
                excess = np.where(valid, np.maximum(0.0, tmax - base), 0.0)
                accumulators[base] += excess

        # Write outputs
        for base in bases:
            name = f"hdd{int(base)}"
            out_path = out_dirs[base] / f"{name}-{ym}.tif"
            write_raster(str(out_path), accumulators[base], bcm_profile)

        logger.info(f"Computed HDD for {ym}")


def compute_sigmoid_heat_stress(
    daily_tmax_dir: str,
    out_dir: str,
    bcm_profile: dict,
    base: float = 30.0,
    scale: float = 5.0,
) -> None:
    """Compute monthly sigmoid heat stress accumulator.

    heat_stress_daily = 1 / (1 + exp(-(tmax - base) / scale))
    Monthly value = sum over days in month.

    Near-zero below 25°C, rises steeply 30-40°C, saturates above ~40°C.

    Parameters
    ----------
    daily_tmax_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily tmax GeoTIFFs.
    out_dir : str
        Output directory (prism_monthly_dir). Creates heat_stress/ subdir.
    bcm_profile : dict
        BCM grid rasterio profile.
    base : float
        Center of the sigmoid (°C).
    scale : float
        Width of the sigmoid transition (°C).
    """
    from ..utils.io_helpers import write_raster

    daily_base = Path(daily_tmax_dir)
    hs_dir = Path(out_dir) / "heat_stress"
    hs_dir.mkdir(parents=True, exist_ok=True)

    H, W = bcm_profile["height"], bcm_profile["width"]

    for month_dir in sorted(daily_base.iterdir()):
        if not month_dir.is_dir():
            continue

        ym = month_dir.name
        out_path = hs_dir / f"heat_stress-{ym}.tif"
        if out_path.exists():
            continue

        daily_files = sorted(month_dir.glob("*.tif"))
        if not daily_files:
            continue

        accumulator = np.zeros((H, W), dtype=np.float32)

        for df in daily_files:
            tmax = _reproject_daily_to_bcm(str(df), bcm_profile)
            valid = ~np.isnan(tmax)
            exponent = np.clip(-(tmax - base) / scale, -50.0, 50.0)
            stress = np.where(
                valid,
                1.0 / (1.0 + np.exp(exponent)),
                0.0,
            )
            accumulator += stress

        write_raster(str(out_path), accumulator, bcm_profile)
        logger.info(f"Computed sigmoid heat stress for {ym}")


def compute_drought_code(
    daily_tmax_dir: str,
    daily_ppt_dir: str,
    out_dir: str,
    bcm_profile: dict,
) -> None:
    """Compute monthly mean Drought Code (DC) from Van Wagner (1987).

    DC is a sequential index tracking deep fuel moisture.  Each day depends on
    the previous day's DC, so we iterate forward through all days, vectorized
    across all pixels simultaneously.

    Monthly output = mean DC over all days in the month.

    Parameters
    ----------
    daily_tmax_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily tmax GeoTIFFs.
    daily_ppt_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily ppt GeoTIFFs.
    out_dir : str
        Output directory (prism_monthly_dir). Creates drought_code/ subdir.
    bcm_profile : dict
        BCM grid rasterio profile.
    """
    from ..utils.io_helpers import write_raster

    dc_dir = Path(out_dir) / "drought_code"
    dc_dir.mkdir(parents=True, exist_ok=True)

    H, W = bcm_profile["height"], bcm_profile["width"]

    # Build latitude grid for day-length factors
    lat_grid = _build_lat_grid(bcm_profile)

    # Collect all month directories (sorted chronologically)
    tmax_base = Path(daily_tmax_dir)
    ppt_base = Path(daily_ppt_dir)

    month_dirs = sorted(
        [d for d in tmax_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not month_dirs:
        logger.warning("No daily tmax directories found for Drought Code computation")
        return

    # Initialize DC state (start at 15 — spring startup value)
    dc_state = np.full((H, W), 15.0, dtype=np.float32)

    # Determine which months already exist
    existing_months = {f.stem.split("-")[1] for f in dc_dir.glob("drought_code-*.tif")}

    # We must process all months sequentially (DC depends on previous state),
    # but can skip writing outputs for months that already exist.
    # However, we still need to run the computation to maintain correct DC state.
    for month_dir in month_dirs:
        ym = month_dir.name  # e.g., "198101"
        month_int = int(ym[4:6])

        # Get matching ppt directory
        ppt_month_dir = ppt_base / ym
        if not ppt_month_dir.exists():
            logger.warning(f"No daily ppt directory for {ym}, skipping DC update")
            continue

        tmax_files = sorted(month_dir.glob("*.tif"))
        ppt_files = sorted(ppt_month_dir.glob("*.tif"))

        if not tmax_files:
            continue

        # Build lookup of ppt files by day
        ppt_by_day = {}
        for f in ppt_files:
            # Extract day from filename like PRISM_ppt_stable_4kmD2_19810101_bil.tif
            day_str = f.stem  # might contain date
            # Try to extract YYYYMMDD
            import re

            match = re.search(r"(\d{8})", f.stem)
            if match:
                ppt_by_day[match.group(1)] = f

        # Day-length factor for this month
        lf = _get_day_length_factor(lat_grid, month_int)

        # Accumulate DC values for monthly mean
        dc_sum = np.zeros((H, W), dtype=np.float64)
        day_count = 0

        for tmax_file in tmax_files:
            tmax = _reproject_daily_to_bcm(str(tmax_file), bcm_profile)

            # Find matching ppt file
            match = re.search(r"(\d{8})", tmax_file.stem)
            if match:
                date_key = match.group(1)
            else:
                # Fallback: use same index
                date_key = None

            if date_key and date_key in ppt_by_day:
                ppt = _reproject_daily_to_bcm(str(ppt_by_day[date_key]), bcm_profile)
            else:
                # If no matching ppt, assume 0 (dry day)
                ppt = np.zeros((H, W), dtype=np.float32)

            # Replace NaN with safe defaults
            tmax_safe = np.where(np.isnan(tmax), 0.0, tmax)
            ppt_safe = np.where(np.isnan(ppt), 0.0, np.maximum(0.0, ppt))

            # --- Drought Code update (Van Wagner 1987) ---

            # Potential evaporation
            pe = 0.36 * (tmax_safe + 2.8) + lf
            pe = np.maximum(0.0, pe)

            # Rain effect
            rain_mask = ppt_safe > 2.8
            if rain_mask.any():
                rw = 0.83 * ppt_safe - 1.27
                smi = 800.0 * np.exp(np.clip(-dc_state / 400.0, -50.0, 0.0))
                smi = np.maximum(smi, 1e-10)  # prevent division by zero
                dc_wet = dc_state - 400.0 * np.log(np.maximum(1.0 + 3.937 * rw / smi, 1e-10))
                dc_wet = np.maximum(0.0, dc_wet)
                dc_state = np.where(rain_mask, dc_wet, dc_state)

            # Drying
            dc_state = dc_state + 0.5 * pe
            dc_state = np.maximum(0.0, dc_state)

            dc_sum += dc_state
            day_count += 1

        # Write monthly mean DC
        if day_count > 0:
            dc_mean = (dc_sum / day_count).astype(np.float32)
            out_path = dc_dir / f"drought_code-{ym}.tif"
            if ym not in existing_months:
                write_raster(str(out_path), dc_mean, bcm_profile)
                logger.info(f"Computed Drought Code for {ym} (mean DC: {np.nanmean(dc_mean):.1f})")
            else:
                logger.debug(f"Drought Code for {ym} already exists, skipping write (state updated)")


def compute_all_fire_features(
    daily_tmax_dir: str,
    daily_ppt_dir: str,
    out_dir: str,
    bcm_profile: dict,
    hdd_bases: tuple = (30.0, 35.0),
    heat_stress_base: float = 30.0,
    heat_stress_scale: float = 5.0,
) -> None:
    """Compute all fire-relevant features.

    Parameters
    ----------
    daily_tmax_dir : str
        Path to tmax_daily/ directory.
    daily_ppt_dir : str
        Path to ppt_daily/ directory.
    out_dir : str
        Output directory (prism_monthly_dir).
    bcm_profile : dict
        BCM grid rasterio profile.
    hdd_bases : tuple
        Base temperatures for HDD computation.
    heat_stress_base : float
        Center of sigmoid heat stress function.
    heat_stress_scale : float
        Width of sigmoid transition.
    """
    logger.info("=== Computing HDD (Heat Stress Degree Days) ===")
    compute_hdd(daily_tmax_dir, out_dir, bcm_profile, bases=hdd_bases)

    logger.info("=== Computing sigmoid heat stress ===")
    compute_sigmoid_heat_stress(
        daily_tmax_dir, out_dir, bcm_profile,
        base=heat_stress_base, scale=heat_stress_scale,
    )

    logger.info("=== Computing Drought Code ===")
    compute_drought_code(daily_tmax_dir, daily_ppt_dir, out_dir, bcm_profile)

    logger.info("=== All fire features computed ===")
