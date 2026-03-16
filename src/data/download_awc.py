"""Download POLARIS soil data and derive AWC, reprojected to BCM grid.

POLARIS provides 30m probabilistic soil property maps across CONUS.
AWC (Available Water Capacity) is derived as the depth-weighted integral of
(theta_s - theta_r) across 6 soil layers (0-100cm), converted to mm.

Tile naming: lat{S}{N}_lon{W}{E}.tif (1-degree blocks)
URL: http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{prop}/{stat}/{depth}/lat{S}{N}_lon{W}{E}.tif

Source: Chaney et al. 2019, Water Resources Research
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

POLARIS_BASE = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"

# Depth layers and their thickness in cm (for integrating AWC over 0-100cm)
DEPTH_LAYERS = [
    ("0_5", 5),
    ("5_15", 10),
    ("15_30", 15),
    ("30_60", 30),
    ("60_100", 40),
]
# Total: 5 + 10 + 15 + 30 + 40 = 100 cm


def download_awc(
    out_dir: str,
    bcm_profile: dict,
    bbox: List[float],
    polaris_prop: str = "aws0_100",
    polaris_stat: str = "mean",
    polaris_depth: str = "0_100",
) -> str:
    """Download POLARIS theta_s and theta_r, derive AWC, reproject to BCM grid.

    AWC = integral of (theta_s - theta_r) over 0-100cm depth, in mm.

    Parameters
    ----------
    out_dir : str
        Output directory (e.g. data/awc). Final file: out_dir/awc_bcm.tif
    bcm_profile : dict
        Rasterio profile for BCM reference grid (EPSG:3310, 1km).
    bbox : list
        [lon_min, lat_min, lon_max, lat_max] in WGS84.
    polaris_prop, polaris_stat, polaris_depth : str
        Ignored (kept for config compatibility). AWC is always derived from
        theta_s and theta_r across all depth layers.

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

    lon_min, lat_min, lon_max, lat_max = bbox
    lat_start = int(np.floor(lat_min))
    lat_end = int(np.ceil(lat_max))
    lon_start = int(np.floor(lon_min))
    lon_end = int(np.ceil(lon_max))

    # Download theta_s and theta_r for each depth layer, mosaic each
    awc_mosaic_path = out_path / "awc_mosaic.tif"
    _download_and_compute_awc(
        out_dir=out_path,
        lat_range=(lat_start, lat_end),
        lon_range=(lon_start, lon_end),
        stat="mean",
        awc_mosaic_path=awc_mosaic_path,
    )

    # Reproject to BCM grid
    logger.info("Reprojecting AWC mosaic to BCM grid...")
    _reproject_to_bcm(awc_mosaic_path, final_path, bcm_profile)

    logger.info(f"AWC BCM raster written: {final_path}")
    return str(final_path)


def _download_and_compute_awc(
    out_dir: Path,
    lat_range: Tuple[int, int],
    lon_range: Tuple[int, int],
    stat: str,
    awc_mosaic_path: Path,
) -> None:
    """Download theta_s and theta_r tiles per depth layer, compute AWC mosaic.

    AWC (mm) = sum over layers of (theta_s - theta_r) * thickness_cm * 10
    (theta in m3/m3 * cm * 10 = mm)
    """
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range

    # For each depth layer, download both theta_s and theta_r tiles,
    # mosaic each, then compute layer AWC contribution
    layer_awc_paths = []

    for depth_name, thickness_cm in DEPTH_LAYERS:
        logger.info(f"Processing depth layer {depth_name} ({thickness_cm} cm)...")

        for prop in ["theta_s", "theta_r"]:
            tiles_dir = out_dir / "tiles" / prop / depth_name
            tiles_dir.mkdir(parents=True, exist_ok=True)
            mosaic_path = out_dir / f"{prop}_{depth_name}_mosaic.tif"

            if mosaic_path.exists():
                logger.info(f"  {prop}/{depth_name} mosaic already exists")
                continue

            tile_paths = _download_tiles(
                prop=prop,
                stat=stat,
                depth=depth_name,
                lat_range=(lat_start, lat_end),
                lon_range=(lon_start, lon_end),
                tiles_dir=tiles_dir,
            )

            if not tile_paths:
                raise RuntimeError(
                    f"No tiles downloaded for {prop}/{depth_name} — check bbox and server"
                )

            _mosaic_tiles(tile_paths, mosaic_path)

        # Compute layer AWC = (theta_s - theta_r) * thickness_cm * 10 (mm)
        layer_path = out_dir / f"awc_layer_{depth_name}.tif"
        if not layer_path.exists():
            ts_path = out_dir / f"theta_s_{depth_name}_mosaic.tif"
            tr_path = out_dir / f"theta_r_{depth_name}_mosaic.tif"
            _compute_layer_awc(ts_path, tr_path, thickness_cm, layer_path)
        layer_awc_paths.append(layer_path)

    # Sum all layer AWC contributions into final mosaic
    logger.info("Summing AWC across depth layers...")
    _sum_layers(layer_awc_paths, awc_mosaic_path)


def _tile_url(prop: str, stat: str, depth: str, lat_s: int, lat_n: int, lon_w: int, lon_e: int) -> str:
    """Build POLARIS tile URL."""
    return (
        f"{POLARIS_BASE}/{prop}/{stat}/{depth}/"
        f"lat{lat_s}{lat_n}_lon{lon_w}{lon_e}.tif"
    )


def _download_tiles(
    prop: str,
    stat: str,
    depth: str,
    lat_range: Tuple[int, int],
    lon_range: Tuple[int, int],
    tiles_dir: Path,
) -> List[Path]:
    """Download all 1-degree tiles for a given property/stat/depth."""
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range
    tile_paths = []

    for lat_s in range(lat_start, lat_end):
        lat_n = lat_s + 1
        for lon_w in range(lon_start, lon_end):
            lon_e = lon_w + 1
            tile_file = tiles_dir / f"lat{lat_s}{lat_n}_lon{lon_w}{lon_e}.tif"

            if tile_file.exists():
                tile_paths.append(tile_file)
                continue

            url = _tile_url(prop, stat, depth, lat_s, lat_n, lon_w, lon_e)

            try:
                resp = requests.get(url, timeout=120, verify=False)
                resp.raise_for_status()
                with open(tile_file, "wb") as f:
                    f.write(resp.content)
                tile_paths.append(tile_file)
            except requests.HTTPError:
                # Ocean/outside CONUS tiles return 404 — expected
                logger.debug(f"  Tile {prop}/{depth} lat{lat_s}{lat_n}_lon{lon_w}{lon_e} not available (404)")
            except Exception as e:
                logger.warning(f"  Failed to download {prop}/{depth} lat{lat_s}{lat_n}_lon{lon_w}{lon_e}: {e}")

    logger.info(f"  {prop}/{depth}: {len(tile_paths)} tiles downloaded")
    return tile_paths


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


def _compute_layer_awc(
    theta_s_path: Path,
    theta_r_path: Path,
    thickness_cm: float,
    out_path: Path,
) -> None:
    """Compute AWC contribution for a single depth layer.

    AWC_layer (mm) = (theta_s - theta_r) * thickness_cm * 10
    theta_s, theta_r are in m3/m3.
    """
    with rasterio.open(str(theta_s_path)) as src_s:
        ts = src_s.read(1).astype(np.float32)
        profile = src_s.profile.copy()
        nodata_s = src_s.nodata

    with rasterio.open(str(theta_r_path)) as src_r:
        tr = src_r.read(1).astype(np.float32)
        nodata_r = src_r.nodata

    # Mask nodata
    valid = np.ones(ts.shape, dtype=bool)
    if nodata_s is not None:
        valid &= ts != nodata_s
    if nodata_r is not None:
        valid &= tr != nodata_r

    awc = np.where(valid, np.maximum(0, ts - tr) * thickness_cm * 10.0, -9999.0)

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(awc[np.newaxis, :])

    logger.info(f"  Layer AWC ({thickness_cm}cm): mean={awc[valid].mean():.1f} mm, "
                f"range=[{awc[valid].min():.1f}, {awc[valid].max():.1f}]")


def _sum_layers(layer_paths: List[Path], out_path: Path) -> None:
    """Sum AWC across depth layers into a single raster."""
    # All layers should be on the same grid (from mosaiced POLARIS tiles)
    with rasterio.open(str(layer_paths[0])) as src:
        total = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    valid = total != -9999.0

    for lp in layer_paths[1:]:
        with rasterio.open(str(lp)) as src:
            layer = src.read(1).astype(np.float32)
        layer_valid = layer != -9999.0
        # Only sum where both are valid
        both_valid = valid & layer_valid
        total = np.where(both_valid, total + layer, total)
        valid = valid & layer_valid

    total[~valid] = -9999.0

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(total[np.newaxis, :])

    logger.info(f"  Total AWC (0-100cm): mean={total[valid].mean():.1f} mm, "
                f"range=[{total[valid].min():.1f}, {total[valid].max():.1f}]")


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
