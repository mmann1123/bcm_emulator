"""Download POLARIS AWC (Available Water Storage) and reproject to BCM grid.

POLARIS provides 30m probabilistic soil property maps across CONUS.
We use aws0_100 (available water storage, 0-100cm depth, mm).

Tiles are organized by 1-degree lat/lon blocks.
URL pattern: http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{property}/{stat}/{depth}/{lat}_{lon}.tif

Source: Chaney et al. 2019, Water Resources Research
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

POLARIS_URL = (
    "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
    "{prop}/{stat}/{depth}/{lat}_{lon}.tif"
)


def download_awc(
    out_dir: str,
    bcm_profile: dict,
    bbox: List[float],
    polaris_prop: str = "aws0_100",
    polaris_stat: str = "mean",
    polaris_depth: str = "0_100",
) -> str:
    """Download POLARIS AWC tiles, mosaic, and reproject to BCM grid.

    Parameters
    ----------
    out_dir : str
        Output directory (e.g. data/awc). Final file: out_dir/awc_bcm.tif
    bcm_profile : dict
        Rasterio profile for BCM reference grid (EPSG:3310, 1km).
    bbox : list
        [lon_min, lat_min, lon_max, lat_max] in WGS84.
    polaris_prop : str
        POLARIS property name. Default 'aws0_100' (available water storage 0-100cm).
    polaris_stat : str
        Statistic type. Default 'mean'.
    polaris_depth : str
        Depth range. Default '0_100'.

    Returns
    -------
    str
        Path to the output BCM-grid AWC raster.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    final_path = out_path / "awc_bcm.tif"

    if final_path.exists():
        logger.info(f"AWC already exists: {final_path}")
        return str(final_path)

    tiles_dir = out_path / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox

    # POLARIS tiles are named by the SW corner integer lat/lon
    # lat tiles: floor(lat_min) to floor(lat_max)
    # lon tiles: floor(lon_min) to floor(lon_max)
    lat_start = int(np.floor(lat_min))
    lat_end = int(np.floor(lat_max))
    lon_start = int(np.floor(lon_min))
    lon_end = int(np.floor(lon_max))

    # Download tiles
    tile_paths = []
    n_tiles = (lat_end - lat_start + 1) * (lon_end - lon_start + 1)
    logger.info(f"Downloading POLARIS {polaris_prop} tiles: {n_tiles} tiles")

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tile_file = tiles_dir / f"{polaris_prop}_{lat}_{lon}.tif"

            if tile_file.exists():
                tile_paths.append(tile_file)
                continue

            url = POLARIS_URL.format(
                prop=polaris_prop,
                stat=polaris_stat,
                depth=polaris_depth,
                lat=lat,
                lon=lon,
            )

            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(tile_file, "wb") as f:
                    f.write(resp.content)
                tile_paths.append(tile_file)
                logger.debug(f"  Downloaded tile {lat}_{lon}")
            except requests.HTTPError as e:
                # Some tiles over ocean/outside CONUS will 404 — skip
                logger.warning(f"  Tile {lat}_{lon} not available: {e}")
            except Exception as e:
                logger.warning(f"  Failed to download tile {lat}_{lon}: {e}")

    if not tile_paths:
        raise RuntimeError("No POLARIS tiles downloaded — check bbox and URL")

    logger.info(f"Downloaded {len(tile_paths)} tiles, mosaicing...")

    # Mosaic tiles
    mosaic_path = out_path / "awc_mosaic.tif"
    _mosaic_tiles(tile_paths, mosaic_path)

    # Reproject to BCM grid
    logger.info("Reprojecting AWC mosaic to BCM grid...")
    _reproject_to_bcm(mosaic_path, final_path, bcm_profile)

    logger.info(f"AWC BCM raster written: {final_path}")
    return str(final_path)


def _mosaic_tiles(tile_paths: List[Path], out_path: Path) -> None:
    """Mosaic multiple GeoTIFF tiles into a single raster."""
    datasets = []
    for tp in tile_paths:
        try:
            ds = rasterio.open(tp)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"  Cannot open tile {tp}: {e}")

    if not datasets:
        raise RuntimeError("No valid tiles to mosaic")

    mosaic, mosaic_transform = merge(datasets)

    # Use profile from first dataset as template
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        compress="lzw",
    )

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    logger.info(f"  Mosaic: {mosaic.shape[1]}x{mosaic.shape[2]} pixels")


def _reproject_to_bcm(src_path: Path, dst_path: Path, bcm_profile: dict) -> None:
    """Reproject a raster to match BCM grid (EPSG:3310, 1km)."""
    with rasterio.open(str(src_path)) as src:
        dst_profile = bcm_profile.copy()
        dst_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")

        dst_data = np.full(
            (dst_profile["height"], dst_profile["width"]),
            -9999.0,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=-9999.0,
            resampling=Resampling.average,  # average for downsampling 30m -> 1km
        )

        with rasterio.open(str(dst_path), "w", **dst_profile) as dst:
            dst.write(dst_data[np.newaxis, :])
