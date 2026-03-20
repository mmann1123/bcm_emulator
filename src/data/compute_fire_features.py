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

# Correct DC day-length factors from cffdrs source (fl01 array).
# Applies to all latitudes > 20°N (all of California).
# Jan-Dec. Negative values in winter reduce PE, preventing unrealistic winter drying.
DC_LF_NORTH = np.array(
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
    dtype=np.float32,
)


def _get_day_length_factor(lat_grid: np.ndarray, month: int) -> np.ndarray:
    """Get DC day-length factor L_f for each pixel given month (1-indexed).

    All of California is > 20°N, so the single fl01 array applies everywhere.
    """
    return np.full_like(lat_grid, DC_LF_NORTH[month - 1])


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


def _overwinter_dc(
    dc_fall: np.ndarray,
    ppt_winter: np.ndarray,
    carry_over: float = 0.75,
    wetting_eff: float = 0.75,
    min_dc: float = 15.0,
) -> np.ndarray:
    """Lawson & Armitage (2008) / McElhinny et al. (2020) wDC overwintering formula.

    Computes spring starting DC from fall DC and total winter precipitation.
    Preserves inter-annual drought persistence — critical for CA fire seasons
    during multi-year droughts (2012-2016, 2020-2022).

    Parameters
    ----------
    dc_fall : np.ndarray
        DC at end of previous fire season (end of September).
    ppt_winter : np.ndarray
        Total precipitation (mm) accumulated over winter (Oct-Mar).
    carry_over : float
        Fraction of fall moisture deficit carried through winter.
        0.75 for CA (well-drained soils, occasional dry winters).
    wetting_eff : float
        Efficiency of winter precipitation at rewetting deep fuels.
        0.75 is the standard default for typical soils.
    min_dc : float
        Minimum DC floor (prevents negative values in very wet years).
    """
    Qf = 800.0 * np.exp(np.clip(-dc_fall / 400.0, -50.0, 0.0))  # moisture equiv of fall DC
    Qs = carry_over * Qf + 3.937 * wetting_eff * ppt_winter      # after winter wetting
    Qs = np.maximum(Qs, 1e-6)
    dc_start = 400.0 * np.log(800.0 / Qs)
    return np.maximum(dc_start, min_dc).astype(np.float32)


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

    # Overwintering state: save fall DC and accumulate winter ppt
    dc_fall = np.full((H, W), 15.0, dtype=np.float32)  # DC at end of September
    ppt_winter = np.zeros((H, W), dtype=np.float32)     # accumulated Oct-Mar ppt

    # Nodata mask — built from first valid tmax file; ocean/outside-CA pixels
    # are NaN in every PRISM file. These must stay NaN in DC state.
    nodata_mask = None  # set on first daily file read; True where always NaN

    # Process all months sequentially (DC depends on previous state)
    for month_dir in month_dirs:
        ym = month_dir.name  # e.g., "198101"
        month_int = int(ym[4:6])

        # Overwintering: at end of September, save fall DC for wDC formula
        # At start of October, begin accumulating winter precipitation
        if month_int == 10:
            dc_fall = dc_state.copy()
            ppt_winter = np.zeros((H, W), dtype=np.float32)

        # At start of April, apply Lawson & Armitage (2008) wDC formula
        # to compute physically-based spring DC from fall DC + winter ppt
        if month_int == 4:
            dc_state = _overwinter_dc(dc_fall, ppt_winter)
            if nodata_mask is not None:
                dc_state = np.where(nodata_mask, np.nan, dc_state)
            logger.debug(f"Overwintered DC for {ym}: mean={np.nanmean(dc_state):.1f}")

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

            # Build nodata mask from first file; clamp tmax to -2.8°C per cffdrs spec
            valid = ~np.isnan(tmax)
            if nodata_mask is None:
                nodata_mask = ~valid
                dc_state = np.where(nodata_mask, np.nan, dc_state)

            tmax_safe = np.where(valid, np.maximum(tmax, -2.8), -2.8)
            ppt_safe = np.where(~np.isnan(ppt), np.maximum(0.0, ppt), 0.0)

            # Accumulate winter ppt (Oct-Mar) for overwintering formula
            if month_int >= 10 or month_int <= 3:
                ppt_winter += ppt_safe

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

            # Restore NaN for ocean/nodata pixels
            if nodata_mask is not None:
                dc_state = np.where(nodata_mask, np.nan, dc_state)

            dc_sum += dc_state
            day_count += 1

        # Write monthly mean DC
        if day_count > 0:
            dc_mean = (dc_sum / day_count).astype(np.float32)
            out_path = dc_dir / f"drought_code-{ym}.tif"
            write_raster(str(out_path), dc_mean, bcm_profile)
            logger.info(f"Computed Drought Code for {ym} (mean DC: {np.nanmean(dc_mean):.1f})")


def compute_kbdi(
    daily_tmax_dir: str,
    daily_ppt_dir: str,
    out_dir: str,
    bcm_profile: dict,
    mean_annual_ppt_path: str,
) -> None:
    """Compute monthly mean KBDI (Keetch-Byram Drought Index).

    KBDI is a drought index designed for the continental US with a physical
    upper bound of 800. It tracks soil moisture deficit using daily tmax and
    precipitation, modulated by mean annual precipitation (climate normal).

    Values: 0 = saturated, 800 = extreme drought.

    Parameters
    ----------
    daily_tmax_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily tmax GeoTIFFs.
    daily_ppt_dir : str
        Directory containing monthly subdirs (YYYYMM/) of daily ppt GeoTIFFs.
    out_dir : str
        Output directory (prism_monthly_dir). Creates kbdi/ subdir.
    bcm_profile : dict
        BCM grid rasterio profile.
    mean_annual_ppt_path : str
        Path to mean annual precipitation raster (BCM ca_pptaveann.asc, in inches).
    """
    import re

    import rasterio
    from rasterio.warp import Resampling, reproject

    from ..utils.io_helpers import write_raster

    kbdi_dir = Path(out_dir) / "kbdi"
    kbdi_dir.mkdir(parents=True, exist_ok=True)

    H, W = bcm_profile["height"], bcm_profile["width"]

    # Load mean annual precipitation and reproject to BCM 1km grid
    # Source is BCM 270m grid (EPSG:3310, no CRS in file metadata), values in inches
    with rasterio.open(mean_annual_ppt_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_data[src_data == -9999.0] = np.nan
        src_data *= 25.4  # inches → mm

        mean_ann_ppt = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=mean_ann_ppt,
            src_transform=src.transform,
            src_crs="EPSG:3310",
            dst_transform=bcm_profile["transform"],
            dst_crs=bcm_profile["crs"],
            resampling=Resampling.bilinear,
        )

    logger.info(
        f"Loaded mean annual ppt: range {np.nanmin(mean_ann_ppt):.0f}-"
        f"{np.nanmax(mean_ann_ppt):.0f} mm/yr"
    )

    # Collect month directories
    tmax_base = Path(daily_tmax_dir)
    ppt_base = Path(daily_ppt_dir)

    month_dirs = sorted(
        [d for d in tmax_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not month_dirs:
        logger.warning("No daily tmax directories found for KBDI computation")
        return

    # Initialize KBDI state (start saturated)
    kbdi_state = np.zeros((H, W), dtype=np.float32)
    nodata_mask = None

    for month_dir in month_dirs:
        ym = month_dir.name

        # Get matching ppt directory
        ppt_month_dir = ppt_base / ym
        if not ppt_month_dir.exists():
            logger.warning(f"No daily ppt directory for {ym}, skipping KBDI update")
            continue

        tmax_files = sorted(month_dir.glob("*.tif"))
        if not tmax_files:
            continue

        # Build ppt file lookup by day
        ppt_by_day = {}
        for f in sorted(ppt_month_dir.glob("*.tif")):
            match = re.search(r"(\d{8})", f.stem)
            if match:
                ppt_by_day[match.group(1)] = f

        # Accumulate KBDI for monthly mean
        kbdi_sum = np.zeros((H, W), dtype=np.float64)
        day_count = 0

        for tmax_file in tmax_files:
            tmax = _reproject_daily_to_bcm(str(tmax_file), bcm_profile)

            # Find matching ppt
            match = re.search(r"(\d{8})", tmax_file.stem)
            date_key = match.group(1) if match else None

            if date_key and date_key in ppt_by_day:
                ppt = _reproject_daily_to_bcm(str(ppt_by_day[date_key]), bcm_profile)
            else:
                ppt = np.zeros((H, W), dtype=np.float32)

            # Build nodata mask from first file
            valid = ~np.isnan(tmax)
            if nodata_mask is None:
                nodata_mask = ~valid | np.isnan(mean_ann_ppt)
                kbdi_state = np.where(nodata_mask, np.nan, kbdi_state)

            tmax_safe = np.where(valid, tmax, 0.0)
            ppt_safe = np.where(~np.isnan(ppt), np.maximum(0.0, ppt), 0.0)

            # --- KBDI update (Keetch & Byram 1968) ---

            # Rainfall wetting: subtract 5.08mm canopy interception threshold
            net_rain = np.maximum(0.0, ppt_safe - 5.08)
            kbdi_state = np.maximum(0.0, kbdi_state - net_rain * 100.0 / 25.4)

            # Drying (ET) — only when tmax > 10°C (50°F)
            tmax_f = tmax_safe * 9.0 / 5.0 + 32.0  # °C → °F
            et = (
                (800.0 - kbdi_state)
                * (0.968 * np.exp(0.0486 * tmax_f) - 8.3)
                / (1.0 + 10.88 * np.exp(-0.0441 * mean_ann_ppt / 25.4))
            ) * 1e-3
            et = np.maximum(0.0, et)
            kbdi_state = np.where(tmax_safe > 10.0, kbdi_state + et, kbdi_state)
            kbdi_state = np.clip(kbdi_state, 0.0, 800.0)

            # Restore nodata
            if nodata_mask is not None:
                kbdi_state = np.where(nodata_mask, np.nan, kbdi_state)

            kbdi_sum += kbdi_state
            day_count += 1

        # Write monthly mean KBDI
        if day_count > 0:
            kbdi_mean = (kbdi_sum / day_count).astype(np.float32)
            out_path = kbdi_dir / f"kbdi-{ym}.tif"
            write_raster(str(out_path), kbdi_mean, bcm_profile)
            logger.info(f"Computed KBDI for {ym} (mean: {np.nanmean(kbdi_mean):.1f})")


def compute_all_fire_features(
    daily_tmax_dir: str,
    daily_ppt_dir: str,
    out_dir: str,
    bcm_profile: dict,
    mean_annual_ppt_path: str = None,
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
    mean_annual_ppt_path : str, optional
        Path to mean annual precipitation raster for KBDI.
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

    if mean_annual_ppt_path:
        logger.info("=== Computing KBDI ===")
        compute_kbdi(daily_tmax_dir, daily_ppt_dir, out_dir, bcm_profile, mean_annual_ppt_path)
    else:
        logger.warning("Skipping KBDI — mean_annual_ppt_path not provided")

    logger.info("=== All fire features computed ===")
