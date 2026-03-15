"""Download and rasterize FRAP CWHR vegetation (FVEG) to BCM grid.

Source: CAL FIRE FRAP bulk download — fveg22_1 file geodatabase (raster).
URL: https://34c031f8-c9fd-4018-8c5a-4159cdff6b0d-cdn-endpoint.azureedge.net/
     -/media/calfire-website/what-we-do/fire-resource-assessment-program---frap/
     gis-data/fveg221gdb.zip

VAT (Value Attribute Table): Exported from QGIS as CSV. Each of the ~42,000 raw
pixel values maps to a WHRNUM (1-81) which identifies the CWHR habitat type.

Processing pipeline:
    1. Download fveg22_1.gdb zip (~147 MB) from CAL FIRE CDN
    2. Convert raster GDB → GeoTIFF via gdal_translate
       (rasterio's bundled GDAL cannot read ESRI raster GDBs)
    3. Reclassify raw pixel values → WHRNUM using the VAT CSV
    4. Save reclassified 30m WHRNUM raster as GeoTIFF
    5. Resample 30m → 1km BCM grid using majority (mode) via gdalwarp
    6. Remap WHRNUM values to contiguous class IDs (0 = unclassified)
    7. Apply partveg filter: discard contiguous patches < 20 cells per class
    8. Save fullveg raster, partveg raster, and class mapping JSON

Outputs (in out_dir):
    fveg_full.tif       — all pixels classified by WHRNUM (1km BCM grid)
    fveg_partveg.tif    — only patches >= 20 contiguous 1km cells retained
    fveg_class_map.json — bidirectional ID ↔ WHRTYPE name mapping
"""

import csv
import json
import logging
import subprocess
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


def download_fveg(
    out_dir: str,
    bcm_profile: dict,
    vat_csv_path: str = "",
    min_patch_cells: int = 20,
) -> Dict[str, str]:
    """Download FVEG raster, reclassify to WHRNUM, resample to BCM grid.

    Parameters
    ----------
    out_dir : str
        Directory to save output rasters and class mapping.
    bcm_profile : dict
        BCM grid reference profile (crs, transform, height, width).
    vat_csv_path : str
        Path to FVEG VAT CSV (exported from QGIS). Must have columns
        'Value' and 'WHRNUM'. If empty, looks for fveg_vat.csv in the
        parent fveg directory.
    min_patch_cells : int
        Minimum contiguous patch size for partveg filter (default 20).

    Returns
    -------
    dict
        Paths: 'fullveg', 'partveg', 'class_map'.
    """
    import rasterio

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Output paths
    fullveg_path = str(out_path / "fveg_full.tif")
    partveg_path = str(out_path / "fveg_partveg.tif")
    classmap_path = str(out_path / "fveg_class_map.json")

    # Intermediate paths
    zip_path = out_path / "fveg221gdb.zip"
    extract_dir = out_path / "fveg_gdb"
    raw_tif_path = out_path / "fveg22_1_raw.tif"
    whrnum_tif_path = out_path / "fveg22_1_whrnum_30m.tif"
    bcm_grid_tif_path = out_path / "fveg22_1_whrnum_1km.tif"

    # --- Step 1: Download geodatabase zip ---
    _download_zip(zip_path)

    # --- Step 2: Extract geodatabase ---
    gdb_path = _extract_gdb(zip_path, extract_dir)

    # --- Step 3: Convert raster GDB → GeoTIFF ---
    _gdal_translate(gdb_path, raw_tif_path)

    # --- Step 4: Load VAT and reclassify raw values → WHRNUM ---
    if not vat_csv_path:
        # Look in common locations
        for candidate in [
            out_path / "fveg_vat.csv",
            out_path.parent / "fveg_vat.csv",
            Path("/home/mmann1123/extra_space/fveg/fveg_vat.csv"),
        ]:
            if candidate.exists():
                vat_csv_path = str(candidate)
                break
    if not vat_csv_path or not Path(vat_csv_path).exists():
        raise FileNotFoundError(
            "VAT CSV not found. Export the raster attribute table from QGIS "
            "(Layer Properties → Attribute Tables) as CSV with at minimum "
            "'Value' and 'WHRNUM' columns."
        )

    value_to_whrnum, whrnum_info = _load_vat(vat_csv_path)
    _reclassify_to_whrnum(raw_tif_path, whrnum_tif_path, value_to_whrnum)

    # --- Step 5: Resample 30m → 1km BCM grid using majority (mode) ---
    _gdal_warp_majority(whrnum_tif_path, bcm_grid_tif_path, bcm_profile)

    # --- Step 6: Read resampled raster and build contiguous class IDs ---
    with rasterio.open(str(bcm_grid_tif_path)) as src:
        whrnum_data = src.read(1).astype(np.int32)

    # Build WHRNUM → contiguous class ID mapping
    unique_whrnum = sorted(int(x) for x in set(whrnum_data[whrnum_data > 0]))
    whrnum_to_classid = {wn: idx + 1 for idx, wn in enumerate(unique_whrnum)}
    num_classes = len(unique_whrnum) + 1  # +1 for class 0

    # Build class_id → name mapping from VAT info
    id_to_info = {0: {"whrnum": 0, "whr_code": "UNC", "name": "unclassified"}}
    for wn, cid in whrnum_to_classid.items():
        info = whrnum_info.get(wn, {"whr_code": f"WHR{wn}", "name": f"WHRNUM_{wn}"})
        id_to_info[cid] = {"whrnum": wn, **info}

    logger.info(f"Class mapping: {num_classes} classes "
                f"(0=unclassified, 1-{len(unique_whrnum)} CWHR types)")

    # Remap to contiguous IDs
    fullveg = np.zeros_like(whrnum_data, dtype=np.int32)
    for wn, cid in whrnum_to_classid.items():
        fullveg[whrnum_data == wn] = cid

    n_classified = np.count_nonzero(fullveg)
    logger.info(f"Fullveg: {n_classified} classified pixels, "
                f"{fullveg.size - n_classified} unclassified")

    # --- Step 7: Partveg filter ---
    logger.info(f"Applying partveg filter (min patch size: {min_patch_cells} cells)...")
    partveg = _apply_partveg_filter(fullveg, min_patch_cells)
    n_filtered = n_classified - np.count_nonzero(partveg)
    logger.info(f"Partveg: {np.count_nonzero(partveg)} pixels retained, "
                f"{n_filtered} fragmented pixels set to 0")

    # --- Step 8: Save outputs ---
    _save_raster(fullveg, fullveg_path, bcm_profile)
    _save_raster(partveg, partveg_path, bcm_profile)

    class_map_save = {
        "num_classes": num_classes,
        "id_to_info": {str(k): v for k, v in id_to_info.items()},
        "whrnum_to_classid": {str(k): int(v) for k, v in whrnum_to_classid.items()},
    }
    with open(classmap_path, "w") as f:
        json.dump(class_map_save, f, indent=2)

    logger.info(f"Saved: {fullveg_path}, {partveg_path}, {classmap_path}")

    return {"fullveg": fullveg_path, "partveg": partveg_path, "class_map": classmap_path}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_zip(zip_path: Path) -> None:
    """Download the FVEG geodatabase zip if not already cached."""
    if zip_path.exists():
        logger.info(f"Using cached download: {zip_path}")
        return

    import requests

    logger.info("Downloading FVEG geodatabase (~147 MB)...")
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


def _extract_gdb(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the zip and return the path to the .gdb directory."""
    if not extract_dir.exists():
        logger.info("Extracting geodatabase...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    gdb_paths = list(extract_dir.rglob("*.gdb"))
    if not gdb_paths:
        raise FileNotFoundError(f"No .gdb found in {extract_dir}")
    gdb_path = gdb_paths[0]
    logger.info(f"Found geodatabase: {gdb_path}")
    return gdb_path


def _gdal_translate(gdb_path: Path, out_tif: Path) -> None:
    """Convert raster GDB to GeoTIFF using system gdal_translate.

    The FVEG .gdb is a raster file geodatabase. rasterio's bundled GDAL
    cannot read ESRI raster GDBs, but the system-installed GDAL's
    gdal_translate handles it via the OpenFileGDB driver.
    """
    if out_tif.exists():
        logger.info(f"Raw GeoTIFF already exists: {out_tif}")
        return

    logger.info(f"Converting raster GDB → GeoTIFF: {gdb_path}")
    result = subprocess.run(
        [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            str(gdb_path),
            str(out_tif),
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\n{result.stderr}")
    logger.info(f"gdal_translate complete: {out_tif}")


def _load_vat(csv_path: str) -> tuple:
    """Load the FVEG Value Attribute Table from CSV.

    Returns
    -------
    value_to_whrnum : dict
        Mapping of raw pixel Value (int) → WHRNUM (int).
    whrnum_info : dict
        Mapping of WHRNUM → {whr_code, name} from the VAT.
    """
    value_to_whrnum = {}
    whrnum_info = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_val = int(row["Value"])
            whrnum = int(row["WHRNUM"])
            value_to_whrnum[raw_val] = whrnum

            if whrnum not in whrnum_info:
                whrnum_info[whrnum] = {
                    "whr_code": row.get("WHRTYPE", f"WHR{whrnum}"),
                    "name": row.get("WHRNAME", f"WHRNUM_{whrnum}"),
                }

    logger.info(f"Loaded VAT: {len(value_to_whrnum)} raw values → "
                f"{len(whrnum_info)} WHRNUM classes")
    return value_to_whrnum, whrnum_info


def _reclassify_to_whrnum(
    src_tif: Path, dst_tif: Path, value_to_whrnum: dict
) -> None:
    """Reclassify raw pixel values to WHRNUM at native 30m resolution.

    Reads the raw raster in chunks to avoid loading the full 39623x38094
    array into memory at once, applies the VAT lookup, and writes the
    reclassified raster.
    """
    import rasterio

    if dst_tif.exists():
        logger.info(f"WHRNUM raster already exists: {dst_tif}")
        return

    logger.info("Reclassifying raw values → WHRNUM at 30m...")

    # Build a numpy lookup array for fast reclassification
    max_val = max(value_to_whrnum.keys())
    lut = np.zeros(max_val + 1, dtype=np.uint8)
    for raw_val, whrnum in value_to_whrnum.items():
        lut[raw_val] = whrnum

    with rasterio.open(str(src_tif)) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", compress="lzw")
        H, W = src.height, src.width

        with rasterio.open(str(dst_tif), "w", **profile) as dst:
            # Process in horizontal strips to limit memory
            strip_height = 1024
            for y_start in range(0, H, strip_height):
                y_end = min(y_start + strip_height, H)
                window = rasterio.windows.Window(0, y_start, W, y_end - y_start)
                data = src.read(1, window=window).astype(np.int32)

                # Clip values to LUT range; out-of-range → 0
                valid = (data >= 0) & (data <= max_val)
                result = np.zeros_like(data, dtype=np.uint8)
                result[valid] = lut[data[valid]]

                dst.write(result, 1, window=window)

                if (y_start // strip_height) % 10 == 0:
                    logger.info(f"  Reclassified rows {y_start}-{y_end} / {H}")

    logger.info(f"Reclassification complete: {dst_tif}")


def _gdal_warp_majority(src_tif: Path, dst_tif: Path, bcm_profile: dict) -> None:
    """Resample raster to BCM 1km grid using majority (mode) resampling.

    Majority resampling picks the most frequent pixel value in each output
    cell, appropriate for categorical data like vegetation type codes.
    """
    if dst_tif.exists():
        logger.info(f"BCM-grid raster already exists: {dst_tif}")
        return

    t = bcm_profile["transform"]
    H, W = bcm_profile["height"], bcm_profile["width"]
    xmin = t.c
    ymax = t.f
    xmax = t.c + W * t.a
    ymin = t.f + H * t.e

    logger.info(f"Resampling 30m → 1km BCM grid ({W}x{H}) using majority...")
    result = subprocess.run(
        [
            "gdalwarp",
            "-t_srs", "EPSG:3310",
            "-te", str(xmin), str(ymin), str(xmax), str(ymax),
            "-ts", str(W), str(H),
            "-r", "mode",
            "-ot", "Byte",
            "-co", "COMPRESS=LZW",
            str(src_tif),
            str(dst_tif),
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")
    logger.info("gdalwarp majority resample complete")


def _apply_partveg_filter(raster: np.ndarray, min_cells: int) -> np.ndarray:
    """Keep only contiguous patches of >= min_cells for each vegetation type.

    Uses scipy connected-component labeling per class, then filters by
    component size using np.bincount for efficiency.
    """
    from scipy.ndimage import label

    result = np.zeros_like(raster)
    unique_classes = np.unique(raster)
    unique_classes = unique_classes[unique_classes > 0]

    for cls_id in unique_classes:
        mask = raster == cls_id
        labeled, n_components = label(mask)
        sizes = np.bincount(labeled.ravel())
        # sizes[0] is background; sizes[1:] are component sizes
        keep_ids = np.where(sizes >= min_cells)[0]
        keep_ids = keep_ids[keep_ids > 0]
        if len(keep_ids) > 0:
            keep_mask = np.isin(labeled, keep_ids)
            result[keep_mask] = cls_id

    return result


def _save_raster(data: np.ndarray, path: str, bcm_profile: dict) -> None:
    """Save integer raster as GeoTIFF aligned to BCM grid."""
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
