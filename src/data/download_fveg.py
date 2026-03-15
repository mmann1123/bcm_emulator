"""Download and rasterize FRAP CWHR vegetation (FVEG) to BCM grid.

Source: CAL FIRE FRAP bulk download — fveg22_1 file geodatabase.
Produces fullveg and partveg (>=20 contiguous cells) rasters plus class mapping JSON.
"""

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Direct download URL for fveg22_1 file geodatabase from CAL FIRE FRAP
FVEG_GDB_URL = (
    "https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net"
    "/-/media/calfire-website/what-we-do/fire-resource-assessment-program---frap"
    "/gis-data/fveg221gdb.zip"
)

# The WHR type field to use for vegetation classification
WHR_FIELD = "WHRTYPE"


def download_fveg(
    out_dir: str,
    bcm_profile: dict,
    min_patch_cells: int = 20,
) -> Dict[str, str]:
    """Download FVEG geodatabase, rasterize to BCM grid, apply partveg filter.

    Parameters
    ----------
    out_dir : str
        Directory to save output rasters and class mapping.
    bcm_profile : dict
        BCM grid reference profile (crs, transform, height, width).
    min_patch_cells : int
        Minimum contiguous patch size for partveg filter (default 20).

    Returns
    -------
    dict
        Paths: 'fullveg', 'partveg', 'class_map'.
    """
    import geopandas as gpd
    import requests
    from rasterio.features import rasterize

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fullveg_path = str(out_path / "fveg_full.tif")
    partveg_path = str(out_path / "fveg_partveg.tif")
    classmap_path = str(out_path / "fveg_class_map.json")

    # --- Step 1: Download and extract geodatabase ---
    gdf = _download_and_read_gdb(out_path)
    logger.info(f"Loaded {len(gdf)} features")

    # --- Step 2: Identify WHR field ---
    field_name = _find_whr_field(gdf)
    logger.info(f"Using WHR field: {field_name} "
                f"({gdf[field_name].nunique()} unique types)")

    # --- Step 3: Reproject to BCM CRS (EPSG:3310) ---
    target_crs = str(bcm_profile["crs"])
    if gdf.crs is None or str(gdf.crs) != target_crs:
        logger.info(f"Reprojecting from {gdf.crs} to {target_crs}...")
        gdf = gdf.to_crs(target_crs)

    # --- Step 4: Build class mapping (string -> contiguous int, 0 = mixed/filtered) ---
    unique_types = sorted(gdf[field_name].dropna().unique())
    class_map = {name: idx + 1 for idx, name in enumerate(unique_types)}
    class_map_inv = {v: k for k, v in class_map.items()}
    class_map_inv[0] = "mixed_filtered"
    num_classes = len(unique_types) + 1  # +1 for class 0

    logger.info(f"Class mapping: {num_classes} classes "
                f"(0=mixed/filtered, 1-{len(unique_types)} = WHR types)")

    # --- Step 5: Rasterize to BCM grid ---
    H, W = bcm_profile["height"], bcm_profile["width"]
    transform = bcm_profile["transform"]

    gdf["class_id"] = gdf[field_name].map(class_map).fillna(0).astype(np.int32)
    # Drop rows with no geometry or class_id==0 before rasterizing
    gdf_valid = gdf[gdf["class_id"] > 0].dropna(subset=["geometry"])
    shapes = list(zip(gdf_valid.geometry, gdf_valid["class_id"]))

    logger.info(f"Rasterizing {len(shapes)} polygons to ({H}, {W}) grid...")
    fullveg = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    logger.info(f"Fullveg: {np.count_nonzero(fullveg)} classified pixels, "
                f"{(fullveg == 0).sum()} unclassified")

    # --- Step 6: Partveg filter (connected components >= min_patch_cells) ---
    logger.info(f"Applying partveg filter (min patch size: {min_patch_cells} cells)...")
    partveg = _apply_partveg_filter(fullveg, min_patch_cells)
    n_filtered = np.count_nonzero(fullveg) - np.count_nonzero(partveg)
    logger.info(f"Partveg: {np.count_nonzero(partveg)} pixels retained, "
                f"{n_filtered} fragmented pixels set to 0")

    # --- Step 7: Save outputs ---
    _save_raster(fullveg, fullveg_path, bcm_profile)
    _save_raster(partveg, partveg_path, bcm_profile)

    class_map_save = {
        "string_to_id": class_map,
        "id_to_string": {str(k): v for k, v in class_map_inv.items()},
        "num_classes": num_classes,
    }
    with open(classmap_path, "w") as f:
        json.dump(class_map_save, f, indent=2)

    logger.info(f"Saved: {fullveg_path}, {partveg_path}, {classmap_path}")

    return {"fullveg": fullveg_path, "partveg": partveg_path, "class_map": classmap_path}


def _download_and_read_gdb(out_dir: Path):
    """Download fveg .gdb zip, extract, and read into GeoDataFrame."""
    import geopandas as gpd
    import requests

    zip_path = out_dir / "fveg221gdb.zip"

    # Download if not already present
    if not zip_path.exists():
        logger.info(f"Downloading FVEG geodatabase (~147 MB)...")
        resp = requests.get(FVEG_GDB_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (20 * 1024 * 1024) < len(chunk):
                    logger.info(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB")
        logger.info(f"Download complete: {zip_path}")
    else:
        logger.info(f"Using cached download: {zip_path}")

    # Extract
    extract_dir = out_dir / "fveg_gdb"
    if not extract_dir.exists():
        logger.info("Extracting geodatabase...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Find the .gdb directory inside the extracted contents
    gdb_paths = list(extract_dir.rglob("*.gdb"))
    if not gdb_paths:
        raise FileNotFoundError(f"No .gdb found in {extract_dir}")
    gdb_path = gdb_paths[0]
    logger.info(f"Found geodatabase: {gdb_path}")

    # List layers
    layers = gpd.list_layers(gdb_path)
    logger.info(f"Available layers: {layers['name'].tolist()}")

    # Read the main layer (typically the first/only polygon layer)
    layer_name = layers["name"].iloc[0]
    logger.info(f"Reading layer '{layer_name}'...")
    gdf = gpd.read_file(gdb_path, layer=layer_name)

    return gdf


def _find_whr_field(gdf) -> str:
    """Find the best WHR type field in the GeoDataFrame."""
    candidates = [WHR_FIELD, "WHRTYPE", "WHR_TYPE", "WHR13NAME", "CWHR_TYPE"]
    cols = set(gdf.columns)

    for candidate in candidates:
        if candidate in cols:
            return candidate

    # Fallback: any column with WHR in the name
    for col in gdf.columns:
        if "WHR" in col.upper():
            return col

    raise RuntimeError(f"No WHR field found. Columns: {list(gdf.columns)}")


def _apply_partveg_filter(raster: np.ndarray, min_cells: int) -> np.ndarray:
    """Keep only contiguous patches of >= min_cells for each vegetation type."""
    from scipy.ndimage import label

    result = np.zeros_like(raster)
    unique_classes = np.unique(raster)
    unique_classes = unique_classes[unique_classes > 0]

    for cls_id in unique_classes:
        mask = raster == cls_id
        labeled, n_components = label(mask)
        # Use bincount for efficient component size filtering
        sizes = np.bincount(labeled.ravel())
        # sizes[0] is background, sizes[1:] are component sizes
        keep_ids = np.where(sizes >= min_cells)[0]
        keep_ids = keep_ids[keep_ids > 0]  # skip background
        if len(keep_ids) > 0:
            keep_mask = np.isin(labeled, keep_ids)
            result[keep_mask] = cls_id

    return result


def _save_raster(data: np.ndarray, path: str, bcm_profile: dict):
    """Save integer raster as GeoTIFF using BCM grid profile."""
    import rasterio

    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "width": bcm_profile["width"],
        "height": bcm_profile["height"],
        "count": 1,
        "crs": bcm_profile["crs"],
        "transform": bcm_profile["transform"],
        "nodata": 0,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
