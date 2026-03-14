"""Compute topographic potential solar radiation from DEM."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_topo_solar(
    dem_path: str,
    out_path: str,
    bcm_profile: dict,
) -> str:
    """Compute annual mean topographic solar radiation from a DEM.

    Uses slope and aspect to compute a relative solar radiation index
    based on the approach from Fu & Rich (2002). This is a simplified
    version that computes the cosine of the incidence angle integrated
    over the year.

    Parameters
    ----------
    dem_path : str
        Path to the DEM raster.
    out_path : str
        Output path for the topographic solar radiation raster.
    bcm_profile : dict
        BCM grid rasterio profile.

    Returns
    -------
    str
        Path to the output raster.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    # Read DEM and reproject to BCM grid if needed
    with rasterio.open(dem_path) as src:
        h, w = bcm_profile["height"], bcm_profile["width"]
        dem = np.full((h, w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bcm_profile["transform"],
            dst_crs=bcm_profile["crs"],
            resampling=Resampling.bilinear,
        )

    # Handle nodata
    nodata_mask = np.isnan(dem) | (dem < -500)
    dem[nodata_mask] = 0

    # Compute slope and aspect from DEM using numpy gradients
    # Cell size is 1000m (1km grid)
    cell_size = 1000.0
    dy, dx = np.gradient(dem, cell_size)

    slope = np.arctan(np.sqrt(dx**2 + dy**2))  # radians
    aspect = np.arctan2(-dx, dy)  # radians, north=0, clockwise

    # Get latitude from BCM grid coordinates
    transform = bcm_profile["transform"]
    rows = np.arange(h)
    # Y coordinates in EPSG:3310 (meters)
    y_coords = transform.f + rows * transform.e  # transform.e is negative

    # Convert EPSG:3310 Y to approximate latitude
    # EPSG:3310 is CA Albers, centered ~37°N
    # Rough conversion: lat ≈ 37 + (y - 0) / 111000
    lat_approx = 37.0 + y_coords / 111000.0
    lat_rad = np.deg2rad(lat_approx)[:, np.newaxis]  # (H, 1)
    lat_rad = np.broadcast_to(lat_rad, (h, w))

    # Compute annual mean solar radiation index
    # Integrate over 12 monthly solar declinations
    declinations = np.deg2rad([
        -23.0, -17.5, -8.0, 4.0, 14.5, 22.0,
        23.5, 18.0, 8.5, -3.0, -14.0, -22.0
    ])

    solar_sum = np.zeros((h, w), dtype=np.float64)

    for decl in declinations:
        # Solar incidence on sloped surface
        # cos(i) = cos(slope)*sin(lat)*sin(decl) + cos(slope)*cos(lat)*cos(decl)
        #        + sin(slope)*cos(aspect)*cos(lat)*sin(decl)
        #        - sin(slope)*cos(aspect)*sin(lat)*cos(decl)

        cos_i = (
            np.cos(slope) * np.sin(lat_rad) * np.sin(decl)
            + np.cos(slope) * np.cos(lat_rad) * np.cos(decl)
            + np.sin(slope) * np.cos(aspect) * np.cos(lat_rad) * np.sin(decl)
            - np.sin(slope) * np.cos(aspect) * np.sin(lat_rad) * np.cos(decl)
        )
        solar_sum += np.maximum(cos_i, 0)

    topo_solar = (solar_sum / len(declinations)).astype(np.float32)
    topo_solar[nodata_mask] = -9999.0

    # Write output
    from .io_helpers import write_raster
    write_raster(out_path, topo_solar, bcm_profile)

    logger.info(f"Computed topographic solar radiation: {out_path}")
    return out_path
