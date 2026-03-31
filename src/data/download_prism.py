"""Download PRISM monthly (ppt, tmin, tmax) and daily ppt data for CONUS at 800m.

New PRISM web service (2025+):
    https://services.nacse.org/prism/data/get/us/<res>/<element>/<date>
    - res: 800m (best available for scripted download)
    - element: ppt, tmin, tmax
    - date: YYYYMM (monthly), YYYYMMDD (daily)
    - Returns zip containing COG GeoTIFF + ancillary files
    - Rate limit: each file may only be downloaded twice per 24h period
"""

import logging
import os
import time
import zipfile
from pathlib import Path
from typing import List

import requests

logger = logging.getLogger(__name__)

# New PRISM web service URL patterns (4km CONUS)
PRISM_BASE = "https://services.nacse.org/prism/data/get/us/4km"
PRISM_MONTHLY_URL = PRISM_BASE + "/{variable}/{year}{month:02d}"
PRISM_DAILY_URL = PRISM_BASE + "/{variable}/{year}{month:02d}{day:02d}"

# Be polite to PRISM servers
REQUEST_DELAY_SEC = 0.5


def download_file(url: str, dest: str, timeout: int = 300) -> bool:
    """Download a file with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()

            # Sanity check: PRISM returns text/html for errors
            ct = resp.headers.get("Content-Type", "")
            if "text/html" in ct:
                logger.warning(f"Got HTML instead of data from {url}")
                return False

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except (requests.RequestException, IOError) as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(5)
    logger.error(f"Failed to download: {url}")
    return False


def extract_tif_from_zip(zip_path: str, out_dir: str) -> str:
    """Extract the GeoTIFF from a PRISM zip file.

    New PRISM service delivers COG GeoTIFFs (not BIL).
    """
    with zipfile.ZipFile(zip_path) as zf:
        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
        if not tif_names:
            raise FileNotFoundError(f"No .tif file in {zip_path}")
        tif_name = tif_names[0]
        zf.extract(tif_name, out_dir)

    tif_path = Path(out_dir) / tif_name
    os.remove(zip_path)
    return str(tif_path)


def download_prism_monthly(
    variable: str,
    year_start: int,
    year_end: int,
    out_dir: str,
    month_end: int = 12,
) -> List[str]:
    """Download PRISM monthly data for a variable at 800m.

    Parameters
    ----------
    variable : str
        One of 'ppt', 'tmin', 'tmax'.
    year_start, year_end : int
        Year range (inclusive).
    out_dir : str
        Output directory for GeoTIFFs.
    month_end : int
        Last month to download in year_end (1-12).
    """
    out_path = Path(out_dir) / variable
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Skip months beyond data range
            if year == year_end and month > month_end:
                break

            tif_pattern = f"*{year}{month:02d}*.tif"
            if list(out_path.glob(tif_pattern)):
                logger.debug(f"Already have {variable} {year}-{month:02d}")
                continue

            url = PRISM_MONTHLY_URL.format(variable=variable, year=year, month=month)
            zip_dest = str(out_path / f"{variable}_{year}{month:02d}.zip")

            logger.info(f"Downloading PRISM monthly {variable} {year}-{month:02d}")
            if download_file(url, zip_dest):
                try:
                    tif = extract_tif_from_zip(zip_dest, str(out_path))
                    downloaded.append(tif)
                except Exception as e:
                    logger.error(f"Failed to extract {zip_dest}: {e}")

            time.sleep(REQUEST_DELAY_SEC)

    logger.info(f"Downloaded {len(downloaded)} PRISM monthly {variable} files")
    return downloaded


def download_prism_daily_ppt(
    year_start: int,
    year_end: int,
    out_dir: str,
    month_end: int = 12,
) -> List[str]:
    """Download daily PRISM ppt at 800m for wet day derivation.

    WARNING: This is very large (~300GB+ at 800m for CONUS, 42 years).
    Downloads are stored as monthly subdirectories.
    """
    import calendar

    out_base = Path(out_dir) / "ppt_daily"
    out_base.mkdir(parents=True, exist_ok=True)
    downloaded = []

    # PRISM daily data only available from 1981 onward
    if year_start < 1981:
        logger.warning(
            f"PRISM daily data starts in 1981; adjusting year_start from {year_start} to 1981"
        )
        year_start = 1981

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            if year == year_end and month > month_end:
                break

            month_dir = out_base / f"{year}{month:02d}"
            month_dir.mkdir(exist_ok=True)

            # Check if already processed
            existing = list(month_dir.glob("*.tif"))
            days_in_month = calendar.monthrange(year, month)[1]
            if len(existing) >= days_in_month:
                logger.debug(f"Already have daily ppt {year}-{month:02d}")
                continue

            for day in range(1, days_in_month + 1):
                tif_pattern = f"*{year}{month:02d}{day:02d}*.tif"
                if list(month_dir.glob(tif_pattern)):
                    continue

                url = PRISM_DAILY_URL.format(
                    variable="ppt", year=year, month=month, day=day
                )
                zip_dest = str(month_dir / f"ppt_{year}{month:02d}{day:02d}.zip")

                if download_file(url, zip_dest):
                    try:
                        tif = extract_tif_from_zip(zip_dest, str(month_dir))
                        downloaded.append(tif)
                    except Exception as e:
                        logger.error(f"Failed to extract {zip_dest}: {e}")

                time.sleep(REQUEST_DELAY_SEC)

            logger.info(f"Downloaded daily ppt for {year}-{month:02d}")

    logger.info(f"Downloaded {len(downloaded)} daily ppt files total")
    return downloaded


def download_prism_daily_tmax(
    year_start: int,
    year_end: int,
    out_dir: str,
    month_end: int = 12,
) -> List[str]:
    """Download daily PRISM tmax at 4km for fire feature derivation.

    Downloads are stored as monthly subdirectories under tmax_daily/.
    """
    import calendar

    out_base = Path(out_dir) / "tmax_daily"
    out_base.mkdir(parents=True, exist_ok=True)
    downloaded = []

    # PRISM daily data only available from 1981 onward
    if year_start < 1981:
        logger.warning(
            f"PRISM daily data starts in 1981; adjusting year_start from {year_start} to 1981"
        )
        year_start = 1981

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            if year == year_end and month > month_end:
                break

            month_dir = out_base / f"{year}{month:02d}"
            month_dir.mkdir(exist_ok=True)

            # Check if already processed
            existing = list(month_dir.glob("*.tif"))
            days_in_month = calendar.monthrange(year, month)[1]
            if len(existing) >= days_in_month:
                logger.debug(f"Already have daily tmax {year}-{month:02d}")
                continue

            for day in range(1, days_in_month + 1):
                tif_pattern = f"*{year}{month:02d}{day:02d}*.tif"
                if list(month_dir.glob(tif_pattern)):
                    continue

                url = PRISM_DAILY_URL.format(
                    variable="tmax", year=year, month=month, day=day
                )
                zip_dest = str(month_dir / f"tmax_{year}{month:02d}{day:02d}.zip")

                if download_file(url, zip_dest):
                    try:
                        tif = extract_tif_from_zip(zip_dest, str(month_dir))
                        downloaded.append(tif)
                    except Exception as e:
                        logger.error(f"Failed to extract {zip_dest}: {e}")

                time.sleep(REQUEST_DELAY_SEC)

            logger.info(f"Downloaded daily tmax for {year}-{month:02d}")

    logger.info(f"Downloaded {len(downloaded)} daily tmax files total")
    return downloaded


def compute_wet_days_and_intensity(
    daily_ppt_dir: str,
    out_dir: str,
    bcm_profile: dict,
    threshold_mm: float = 1.0,
) -> None:
    """Compute monthly wet days and precipitation intensity from daily PRISM ppt.

    Parameters
    ----------
    daily_ppt_dir : str
        Directory containing monthly subdirectories of daily ppt GeoTIFFs.
    out_dir : str
        Output directory for wet_days and ppt_intensity monthly GeoTIFFs.
    bcm_profile : dict
        BCM grid rasterio profile for reprojection.
    threshold_mm : float
        Minimum daily ppt (mm) to count as a wet day.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    wet_dir = Path(out_dir) / "wet_days"
    int_dir = Path(out_dir) / "ppt_intensity"
    wet_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)

    daily_base = Path(daily_ppt_dir)
    for month_dir in sorted(daily_base.iterdir()):
        if not month_dir.is_dir():
            continue

        ym = month_dir.name  # e.g., "198001"
        wet_out = wet_dir / f"wet_days-{ym[:4]}{ym[4:]}.tif"
        int_out = int_dir / f"ppt_intensity-{ym[:4]}{ym[4:]}.tif"

        if wet_out.exists() and int_out.exists():
            continue

        daily_files = sorted(month_dir.glob("*.tif"))
        if not daily_files:
            continue

        # Read and reproject each daily file, accumulate
        h, w = bcm_profile["height"], bcm_profile["width"]
        wet_count = np.zeros((h, w), dtype=np.float32)
        ppt_total = np.zeros((h, w), dtype=np.float32)

        for df in daily_files:
            with rasterio.open(str(df)) as src:
                dst_data = np.full((h, w), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=bcm_profile["transform"],
                    dst_crs=bcm_profile["crs"],
                    resampling=Resampling.bilinear,
                )
                valid = ~np.isnan(dst_data)
                wet_count[valid & (dst_data >= threshold_mm)] += 1
                ppt_total[valid] += np.where(
                    dst_data[valid] >= threshold_mm, dst_data[valid], 0
                )

        # Intensity = total wet-day ppt / wet days (avoid div by zero)
        intensity = np.where(wet_count > 0, ppt_total / wet_count, 0.0)

        # Write outputs
        from ..utils.io_helpers import write_raster

        write_raster(str(wet_out), wet_count, bcm_profile)
        write_raster(str(int_out), intensity, bcm_profile)

        logger.info(f"Computed wet days/intensity for {ym}")
