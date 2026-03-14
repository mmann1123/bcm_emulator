"""Raster I/O helpers and BCM filename parsing."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling


def parse_bcm_filename(path: str) -> Tuple[str, int, int]:
    """Parse BCM filename like 'aet-194910.tif' -> (variable, year, month)."""
    name = Path(path).stem
    match = re.match(r"(\w+)-(\d{4})(\d{2})", name)
    if not match:
        raise ValueError(f"Cannot parse BCM filename: {path}")
    var = match.group(1)
    year = int(match.group(2))
    month = int(match.group(3))
    return var, year, month


def list_bcm_files(
    bcm_dir: str, variable: str, start_ym: Optional[str] = None, end_ym: Optional[str] = None
) -> List[Path]:
    """List BCM files for a variable, optionally filtered by date range.

    Parameters
    ----------
    bcm_dir : str
        Root BCM directory containing variable subdirectories.
    variable : str
        Variable name (aet, cwd, pck, pet, tmx).
    start_ym, end_ym : str, optional
        Date range as 'YYYY-MM' strings (inclusive).
    """
    var_dir = Path(bcm_dir) / variable
    files = sorted(var_dir.glob(f"{variable}-*.tif"))

    if start_ym or end_ym:
        filtered = []
        for f in files:
            _, year, month = parse_bcm_filename(f)
            ym = f"{year:04d}-{month:02d}"
            if start_ym and ym < start_ym:
                continue
            if end_ym and ym > end_ym:
                continue
            filtered.append(f)
        files = filtered

    return files


def read_raster(path: str, band: int = 1) -> Tuple[np.ndarray, dict]:
    """Read a single-band raster, return (array, profile)."""
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = dict(src.profile)
    return data, profile


def read_raster_as_masked(path: str, nodata: float = -9999.0) -> np.ndarray:
    """Read raster and return with nodata masked as NaN."""
    data, profile = read_raster(path)
    nd = profile.get("nodata", nodata)
    data = data.astype(np.float32)
    if nd is not None:
        data[data == nd] = np.nan
    return data


def get_bcm_reference_profile(bcm_dir: str) -> dict:
    """Get the rasterio profile from any BCM file to use as reference grid."""
    aet_dir = Path(bcm_dir) / "aet"
    ref_file = next(aet_dir.glob("aet-*.tif"))
    _, profile = read_raster(str(ref_file))
    return profile


def get_valid_mask(bcm_dir: str, nodata: float = -9999.0) -> np.ndarray:
    """Get boolean mask of valid (non-nodata) pixels from a BCM reference file."""
    aet_dir = Path(bcm_dir) / "aet"
    ref_file = next(aet_dir.glob("aet-*.tif"))
    data = read_raster_as_masked(str(ref_file))
    return ~np.isnan(data)


def reproject_to_bcm_grid(
    src_path: str,
    dst_path: str,
    bcm_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject a raster to match BCM grid (EPSG:3310, 1km)."""
    with rasterio.open(src_path) as src:
        dst_profile = bcm_profile.copy()
        dst_profile.update(dtype="float32", count=1, nodata=-9999.0)

        dst_data = np.full(
            (1, dst_profile["height"], dst_profile["width"]),
            dst_profile["nodata"],
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=resampling,
        )

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            dst.write(dst_data)


def write_raster(
    path: str,
    data: np.ndarray,
    profile: dict,
    nodata: float = -9999.0,
) -> None:
    """Write a 2D array as a single-band GeoTIFF."""
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    if data.ndim == 2:
        data = data[np.newaxis, :]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
