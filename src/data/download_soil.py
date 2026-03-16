"""Download POLARIS soil properties (ksat, sand, clay), reprojected to BCM grid.

POLARIS provides 30m probabilistic soil property maps across CONUS.
Each property is downloaded per depth layer, depth-weighted averaged over 0-100cm,
then reprojected to the BCM 1km grid (EPSG:3310).

Properties:
    ksat: Saturated hydraulic conductivity, stored as log10(µm/s) in POLARIS.
          Arithmetic mean of log10 values = geometric mean of raw values.
    sand: Sand fraction (%), arithmetic depth-weighted mean.
    clay: Clay fraction (%), arithmetic depth-weighted mean.

Tile naming: lat{S}{N}_lon{W}{E}.tif (1-degree blocks)
URL: http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{prop}/{stat}/{depth}/lat{S}{N}_lon{W}{E}.tif

Source: Chaney et al. 2019, Water Resources Research
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

POLARIS_BASE = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"

# Depth layers and their thickness in cm (for depth-weighted averaging over 0-100cm)
DEPTH_LAYERS = [
    ("0_5", 5),
    ("5_15", 10),
    ("15_30", 15),
    ("30_60", 30),
    ("60_100", 40),
]
# Total: 5 + 10 + 15 + 30 + 40 = 100 cm

# Properties to download: name -> output filename
SOIL_PROPERTIES = ["ksat", "sand", "clay"]


def download_soil_properties(
    out_dir: str,
    bcm_profile: dict,
    bbox: List[float],
) -> Dict[str, str]:
    """Download POLARIS ksat, sand, clay and reproject to BCM grid.

    For each property: download tiles per depth layer, mosaic, compute
    depth-weighted average across 0-100cm, then reproject to BCM grid.

    Parameters
    ----------
    out_dir : str
        Output directory (e.g. data/soil). Final files: {prop}_bcm.tif
    bcm_profile : dict
        Rasterio profile for BCM reference grid (EPSG:3310, 1km).
    bbox : list
        [lon_min, lat_min, lon_max, lat_max] in WGS84.

    Returns
    -------
    dict
        Mapping of property name to output BCM-grid raster path.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    lat_start = int(np.floor(lat_min))
    lat_end = int(np.ceil(lat_max))
    lon_start = int(np.floor(lon_min))
    lon_end = int(np.ceil(lon_max))

    results = {}

    for prop in SOIL_PROPERTIES:
        final_path = out_path / f"{prop}_bcm.tif"

        if final_path.exists():
            logger.info(f"{prop} already exists: {final_path}")
            results[prop] = str(final_path)
            continue

        logger.info(f"=== Processing soil property: {prop} ===")

        # Download tiles per depth layer, mosaic each, then depth-average
        avg_mosaic_path = out_path / f"{prop}_avg_mosaic.tif"
        _download_and_depth_average(
            prop=prop,
            out_dir=out_path,
            lat_range=(lat_start, lat_end),
            lon_range=(lon_start, lon_end),
            stat="mean",
            avg_mosaic_path=avg_mosaic_path,
        )

        # Reproject to BCM grid
        logger.info(f"Reprojecting {prop} mosaic to BCM grid...")
        _reproject_to_bcm(avg_mosaic_path, final_path, bcm_profile)

        logger.info(f"{prop} BCM raster written: {final_path}")
        results[prop] = str(final_path)

    return results


def _download_and_depth_average(
    prop: str,
    out_dir: Path,
    lat_range: Tuple[int, int],
    lon_range: Tuple[int, int],
    stat: str,
    avg_mosaic_path: Path,
) -> None:
    """Download tiles per depth layer, mosaic each, compute depth-weighted average."""
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range

    total_thickness = sum(t for _, t in DEPTH_LAYERS)  # 100 cm
    layer_mosaic_paths = []
    layer_weights = []

    for depth_name, thickness_cm in DEPTH_LAYERS:
        logger.info(f"Processing {prop} depth layer {depth_name} ({thickness_cm} cm)...")

        tiles_dir = out_dir / "tiles" / prop / depth_name
        tiles_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = out_dir / f"{prop}_{depth_name}_mosaic.tif"

        if mosaic_path.exists():
            logger.info(f"  {prop}/{depth_name} mosaic already exists")
        else:
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

        layer_mosaic_paths.append(mosaic_path)
        layer_weights.append(thickness_cm / total_thickness)

    # Compute depth-weighted average
    logger.info(f"Computing depth-weighted average for {prop}...")
    _depth_weighted_average(layer_mosaic_paths, layer_weights, avg_mosaic_path)


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
        BIGTIFF="YES",
    )

    # Validate mosaic has actual data (not all nodata)
    nodata_val = profile.get("nodata", -9999.0)
    if nodata_val is not None:
        n_valid = np.sum(mosaic[0] != nodata_val)
    else:
        n_valid = np.sum(~np.isnan(mosaic[0]))
    if n_valid == 0:
        for ds in datasets:
            ds.close()
        raise RuntimeError(f"Mosaic has zero valid pixels — likely corrupt tiles. "
                           f"Delete {out_path} and re-run.")

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    logger.info(f"  Mosaic: {mosaic.shape[1]}x{mosaic.shape[2]} pixels, {n_valid:,} valid")


def _depth_weighted_average(
    layer_paths: List[Path],
    weights: List[float],
    out_path: Path,
) -> None:
    """Compute depth-weighted arithmetic mean across layers.

    For ksat (already log10 in POLARIS), arithmetic mean of log10 values
    equals geometric mean of raw values — appropriate for conductivity.
    For sand/clay (%), this is a standard depth-weighted average.
    """
    with rasterio.open(str(layer_paths[0])) as src:
        first = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0

    valid = first != nodata
    result = np.where(valid, first * weights[0], 0.0).astype(np.float64)
    weight_sum = np.where(valid, weights[0], 0.0)

    for lp, w in zip(layer_paths[1:], weights[1:]):
        with rasterio.open(str(lp)) as src:
            layer = src.read(1).astype(np.float32)
        layer_valid = layer != nodata
        both_valid = valid & layer_valid
        # Accumulate weighted values where both are valid
        result = np.where(both_valid, result + layer * w, result)
        weight_sum = np.where(both_valid, weight_sum + w, weight_sum)
        # Update overall valid mask
        valid = valid & layer_valid

    # Normalize by actual weight sum (handles partial profiles)
    avg = np.where(valid & (weight_sum > 0), result / weight_sum, -9999.0).astype(np.float32)

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw", BIGTIFF="YES")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(avg[np.newaxis, :])

    if valid.sum() > 0:
        logger.info(f"  Depth-weighted avg: mean={avg[valid].mean():.3f}, "
                    f"range=[{avg[valid].min():.3f}, {avg[valid].max():.3f}]")
    else:
        logger.warning("  Depth-weighted avg: no valid pixels!")


def _reproject_to_bcm(src_path: Path, dst_path: Path, bcm_profile: dict) -> None:
    """Reproject a raster to match BCM grid (EPSG:3310, 1km)."""
    with rasterio.open(str(src_path)) as src:
        dst_profile = bcm_profile.copy()
        dst_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw", BIGTIFF="YES")

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
