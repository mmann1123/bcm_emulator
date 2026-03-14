"""Download TerraClimate monthly srad and reproject to BCM grid.

TerraClimate provides monthly solar radiation at ~4.7km resolution (1958-2025).
Data is in EPSG:4326 (WGS84) with regular 1D lat/lon coords.
No authentication required — direct HTTP download.

Source: https://climate.northwestknowledge.net/TERRACLIMATE-DATA/
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import requests
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

TERRACLIMATE_URL = (
    "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/"
    "TerraClimate_srad_{year}.nc"
)


def download_srad(
    year_start: int,
    year_end: int,
    bbox: List[float],
    out_dir: str,
    bcm_profile: dict,
) -> List[str]:
    """Download TerraClimate monthly srad, reproject to BCM grid, save as GeoTIFF.

    Parameters
    ----------
    year_start, year_end : int
        Year range (inclusive).
    bbox : list
        [lon_min, lat_min, lon_max, lat_max] in WGS84.
    out_dir : str
        Base output directory (e.g. data/daymet). Files saved to out_dir/srad_bcm/.
    bcm_profile : dict
        Rasterio profile for BCM reference grid (EPSG:3310, 1km).

    Returns
    -------
    list of str
        Paths to newly created GeoTIFF files.
    """
    dst_dir = Path(out_dir) / "srad_bcm"
    dst_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    lon_min, lat_min, lon_max, lat_max = bbox

    for year in range(year_start, year_end + 1):
        # Check if all 12 months already exist
        existing = list(dst_dir.glob(f"srad-{year}??.tif"))
        if len(existing) >= 12:
            logger.debug(f"Already have TerraClimate srad for {year}")
            continue

        url = TERRACLIMATE_URL.format(year=year)
        logger.info(f"Downloading TerraClimate srad for {year}: {url}")

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Download annual NetCDF
            resp = requests.get(url, timeout=600, stream=True)
            resp.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            logger.info(f"  Downloaded {size_mb:.0f} MB")

            # Open and subset to CA bbox
            ds = xr.open_dataset(tmp_path)

            # TerraClimate has regular 1D lat/lon coords — simple slice
            # lat is descending (90 to -90), so slice accordingly
            ds_sub = ds.sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min, lon_max),
            )

            srad = ds_sub["srad"]  # (time, lat, lon)
            logger.info(
                f"  Subset shape: {srad.shape} "
                f"(time={srad.sizes['time']}, lat={srad.sizes['lat']}, lon={srad.sizes['lon']})"
            )

            # Process each month
            for t in range(srad.sizes["time"]):
                month_data = srad.isel(time=t)
                month_num = int(month_data.time.dt.month.values)
                out_file = dst_dir / f"srad-{year}{month_num:02d}.tif"

                if out_file.exists():
                    continue

                _reproject_month_to_bcm(month_data, out_file, bcm_profile)
                downloaded.append(str(out_file))

            ds.close()

        except Exception as e:
            logger.error(f"Failed to download/process TerraClimate srad for {year}: {e}")
            raise
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"TerraClimate srad: {len(downloaded)} new files created")
    return downloaded


def _reproject_month_to_bcm(
    month_data: xr.DataArray,
    out_path: Path,
    bcm_profile: dict,
) -> None:
    """Reproject a single monthly srad slice from EPSG:4326 to BCM grid."""
    values = month_data.values.astype(np.float32)

    # Handle NaN/fill values
    values = np.where(np.isnan(values), -9999.0, values)

    # Build source transform from lat/lon coords
    lats = month_data.lat.values
    lons = month_data.lon.values
    # TerraClimate pixel centers — compute bounds from cell edges
    lat_res = abs(lats[1] - lats[0]) if len(lats) > 1 else 1 / 24
    lon_res = abs(lons[1] - lons[0]) if len(lons) > 1 else 1 / 24
    src_transform = from_bounds(
        lons.min() - lon_res / 2,
        lats.min() - lat_res / 2,
        lons.max() + lon_res / 2,
        lats.max() + lat_res / 2,
        len(lons),
        len(lats),
    )

    # Destination profile
    dst_profile = bcm_profile.copy()
    dst_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")

    dst_data = np.full(
        (dst_profile["height"], dst_profile["width"]),
        -9999.0,
        dtype=np.float32,
    )

    reproject(
        source=values,
        destination=dst_data,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        src_nodata=-9999.0,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    with rasterio.open(str(out_path), "w", **dst_profile) as dst:
        dst.write(dst_data[np.newaxis, :])

    logger.debug(f"  Wrote {out_path.name}")
