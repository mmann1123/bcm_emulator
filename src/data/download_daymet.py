"""Download DAYMET daily srad and aggregate to monthly mean.

DAYMET data is accessed via NASA Earthdata (requires ~/.netrc credentials).
Uses direct THREDDS/OpenDAP download since pydaymet may have auth issues.
Falls back to downloading full-year NetCDF files from ORNL DAAC.
"""

import calendar
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# DAYMET THREDDS direct download URL (full year files)
DAYMET_THREDDS_URL = (
    "https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/"
    "daymet_v4_daily_na_srad_{year}.nc"
)


def download_daymet_srad(
    year_start: int,
    year_end: int,
    bbox: List[float],
    out_dir: str,
) -> List[str]:
    """Download DAYMET daily srad and aggregate to monthly mean.

    Tries pydaymet first; if that fails (auth issues), falls back to
    direct THREDDS file server download + subsetting with xarray.

    Parameters
    ----------
    year_start, year_end : int
        Year range (inclusive). DAYMET available 1980+.
    bbox : list
        [lon_min, lat_min, lon_max, lat_max] in WGS84.
    out_dir : str
        Output directory for monthly srad GeoTIFFs.
    """
    out_path = Path(out_dir) / "srad_monthly"
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for year in range(year_start, year_end + 1):
        existing = list(out_path.glob(f"srad-{year}*.tif"))
        if len(existing) >= 12:
            logger.debug(f"Already have DAYMET srad for {year}")
            continue

        logger.info(f"Downloading DAYMET srad for {year}")

        # Try pydaymet first
        success = _try_pydaymet(year, bbox, out_path, downloaded)
        if not success:
            # Fallback: direct THREDDS download
            success = _try_thredds_direct(year, bbox, out_path, downloaded)
        if not success:
            # Fallback 2: earthaccess authenticated download
            success = _try_earthaccess(year, bbox, out_path, downloaded)
        if not success:
            logger.error(f"All methods failed for DAYMET srad {year}")

    logger.info(f"Downloaded {len(downloaded)} monthly DAYMET srad files")
    return downloaded


def _try_pydaymet(year: int, bbox: list, out_path: Path, downloaded: list) -> bool:
    """Try downloading via pydaymet library."""
    try:
        import pydaymet

        daily = pydaymet.get_bygeom(
            geometry=tuple(bbox),
            dates=(f"{year}-01-01", f"{year}-12-31"),
            variables=["srad"],
            time_scale="daily",
        )

        monthly = daily["srad"].resample(time="MS").mean()

        for t in range(len(monthly.time)):
            month_data = monthly.isel(time=t)
            month_num = int(month_data.time.dt.month.values)
            out_file = out_path / f"srad-{year}{month_num:02d}.tif"
            if not out_file.exists():
                month_data.rio.to_raster(str(out_file))
                downloaded.append(str(out_file))

        return True
    except Exception as e:
        logger.warning(f"pydaymet failed for {year}: {e}")
        return False


def _try_thredds_direct(year: int, bbox: list, out_path: Path, downloaded: list) -> bool:
    """Download full year NetCDF from THREDDS and subset/aggregate locally."""
    import requests
    import xarray as xr

    url = DAYMET_THREDDS_URL.format(year=year)
    logger.info(f"  Trying direct THREDDS download: {url}")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Download with Earthdata auth from .netrc
        session = requests.Session()
        resp = session.get(url, timeout=600, stream=True)
        resp.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"  Downloaded {os.path.getsize(tmp_path)/1024/1024:.0f} MB")

        # Open and subset to CA bbox
        ds = xr.open_dataset(tmp_path)
        lon_min, lat_min, lon_max, lat_max = bbox

        # DAYMET uses Lambert Conformal Conic, so we need to subset by lat/lon
        if "lat" in ds.coords and "lon" in ds.coords:
            mask = (
                (ds.lat >= lat_min) & (ds.lat <= lat_max) &
                (ds.lon >= lon_min) & (ds.lon <= lon_max)
            )
            ds_sub = ds.where(mask, drop=True)
        else:
            ds_sub = ds

        # Aggregate to monthly mean
        monthly = ds_sub["srad"].resample(time="MS").mean()

        for t in range(len(monthly.time)):
            month_data = monthly.isel(time=t)
            month_num = int(month_data.time.dt.month.values)
            out_file = out_path / f"srad-{year}{month_num:02d}.tif"
            if not out_file.exists():
                month_data.rio.to_raster(str(out_file))
                downloaded.append(str(out_file))

        ds.close()
        return True

    except Exception as e:
        logger.error(f"THREDDS direct download failed for {year}: {e}")
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _try_earthaccess(year: int, bbox: list, out_path: Path, downloaded: list) -> bool:
    """Download DAYMET srad via earthaccess, subset locally after download.

    Downloads the full continental NetCDF (~18GB), subsets to CA bounding box,
    aggregates to monthly, then deletes the full file. Each year takes ~10-15 min
    on a fast connection.
    """
    try:
        import earthaccess
        import xarray as xr

        logger.info(f"  Trying earthaccess for DAYMET srad {year}")
        earthaccess.login()

        results = earthaccess.search_data(
            short_name="Daymet_Daily_V4R1_2129",
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
            count=20,
        )

        # Filter to srad .nc files only
        srad_results = [
            r for r in results
            if any("srad" in link and link.endswith(".nc") for link in r.data_links())
        ]

        if not srad_results:
            logger.warning(f"  earthaccess found no DAYMET srad granules for {year}")
            return False

        logger.info(f"  Downloading {len(srad_results)} srad granule(s)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            files = earthaccess.download(srad_results, tmpdir)
            logger.info(f"  Downloaded {len(files)} files")

            lon_min, lat_min, lon_max, lat_max = bbox

            for fpath in files:
                fpath = str(fpath)
                if not fpath.endswith(".nc"):
                    continue

                logger.info(f"  Opening and subsetting {Path(fpath).name}...")
                ds = xr.open_dataset(fpath, engine="h5netcdf")
                if "srad" not in ds:
                    ds.close()
                    continue

                # DAYMET uses Lambert Conformal Conic with 2D lat/lon arrays.
                # Build a mask on the 2D lat/lon and find bounding y/x indices
                # to slice efficiently before aggregating.
                lat = ds["lat"].values
                lon = ds["lon"].values
                mask = (
                    (lat >= lat_min) & (lat <= lat_max) &
                    (lon >= lon_min) & (lon <= lon_max)
                )

                # Find bounding index ranges from the 2D mask
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    logger.warning(f"  No data in bbox for {year}")
                    ds.close()
                    continue

                y_min, y_max = int(rows.min()), int(rows.max()) + 1
                x_min, x_max = int(cols.min()), int(cols.max()) + 1
                logger.info(f"  CA subset: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

                # Slice to bounding box then apply exact mask
                ds_sub = ds.isel(y=slice(y_min, y_max), x=slice(x_min, x_max))
                sub_mask = mask[y_min:y_max, x_min:x_max]
                ds_sub = ds_sub.where(sub_mask)

                logger.info(f"  Aggregating daily srad to monthly mean...")
                monthly = ds_sub["srad"].resample(time="MS").mean()

                for t in range(len(monthly.time)):
                    month_data = monthly.isel(time=t)
                    month_num = int(month_data.time.dt.month.values)
                    out_file = out_path / f"srad-{year}{month_num:02d}.tif"
                    if not out_file.exists():
                        month_data.rio.to_raster(str(out_file))
                        downloaded.append(str(out_file))

                ds.close()

        return len(list(out_path.glob(f"srad-{year}*.tif"))) > 0

    except Exception as e:
        logger.error(f"earthaccess failed for {year}: {e}")
        return False


def reproject_daymet_to_bcm(
    daymet_dir: str,
    out_dir: str,
    bcm_profile: dict,
) -> None:
    """Reproject all monthly DAYMET srad files to BCM grid."""
    from ..utils.io_helpers import reproject_to_bcm_grid

    src_dir = Path(daymet_dir) / "srad_monthly"
    dst_dir = Path(out_dir) / "srad_bcm"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_file in sorted(src_dir.glob("srad-*.tif")):
        dst_file = dst_dir / src_file.name
        if dst_file.exists():
            continue

        logger.info(f"Reprojecting {src_file.name} to BCM grid")
        reproject_to_bcm_grid(str(src_file), str(dst_file), bcm_profile)
