"""Download BCMv8 climate inputs from USGS ScienceBase.

ScienceBase item 5fb2d0a1d34eb413d5e0895a contains:
    - pet (potential evapotranspiration)
    - ppt (precipitation)
    - tmn (minimum temperature)
    - tmx (maximum temperature)

Files are ASCII Grid (.asc) format with month abbreviations in filenames
(e.g., pet2010jan.asc), organized as decade zips (e.g., pet_WY1980_89.zip).
Water year N covers Oct(N-1) to Sep(N).

We convert .asc to .tif on extraction since rasterio handles both.
Coverage: WY1896-2020 (through Sep 2020).
"""

import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

SCIENCEBASE_ITEM_ID = "5fb2d0a1d34eb413d5e0895a"
SCIENCEBASE_API = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM_ID}?format=json"

# PCK gap-fill: item containing pck decade zips (same format as climate inputs)
PCK_ITEM_ID = "5f29c62d82cef313ed9edb39"
PCK_API = f"https://www.sciencebase.gov/catalog/item/{PCK_ITEM_ID}?format=json"

# Variable name mapping: ScienceBase prefix -> our standard name
# PET not needed -- derived locally as AET + CWD
VAR_MAP = {
    "ppt": "ppt",
    "tmn": "tmin",
    "tmx": "tmax",
}


def get_file_urls(variable_prefix: str) -> List[Dict]:
    """Get zip file URLs for a variable from ScienceBase."""
    resp = requests.get(SCIENCEBASE_API, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    files = []
    for f in data.get("files", []):
        name = f.get("name", "")
        if name.startswith(f"{variable_prefix}_WY") and name.endswith(".zip"):
            files.append({"name": name, "url": f["url"]})

    return files


def _decade_overlaps(filename: str, var_prefix: str, start_ym: str, end_ym: str) -> bool:
    """Check if a decade zip overlaps with target date range."""
    match = re.match(rf"{var_prefix}_WY(\d{{4}})_(\d{{2}})\.zip", filename)
    if not match:
        return False

    wy_start = int(match.group(1))
    suffix = int(match.group(2))
    century = wy_start // 100 * 100
    wy_end = century + suffix
    if wy_end < wy_start:
        wy_end += 100

    # Water year N: Oct(N-1) to Sep(N)
    actual_start = f"{wy_start - 1:04d}-10"
    actual_end = f"{wy_end:04d}-09"

    return actual_end >= start_ym and actual_start <= end_ym


def download_variable(
    variable_prefix: str,
    out_dir: str,
    start_ym: str = "1980-01",
    end_ym: str = "2021-04",
    bcm_profile: Optional[dict] = None,
) -> List[str]:
    """Download a single variable from ScienceBase.

    Parameters
    ----------
    variable_prefix : str
        ScienceBase variable prefix: 'pet', 'ppt', 'tmn', 'tmx'.
    out_dir : str
        Output directory. Files saved as {our_name}-YYYYMM.tif.
    start_ym, end_ym : str
        Date range as 'YYYY-MM' (inclusive).
    bcm_profile : dict, optional
        BCM 1km grid profile for resampling from native 270m.
    """
    our_name = VAR_MAP.get(variable_prefix, variable_prefix)
    out_path = Path(out_dir) / our_name
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []

    zip_files = get_file_urls(variable_prefix)
    logger.info(f"Found {len(zip_files)} {variable_prefix} decade zips on ScienceBase")

    for finfo in zip_files:
        name = finfo["name"]
        url = finfo["url"]

        if not _decade_overlaps(name, variable_prefix, start_ym, end_ym):
            continue

        logger.info(f"Downloading {name}...")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            resp = requests.get(url, timeout=600, stream=True)
            resp.raise_for_status()
            total_size = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)
            logger.info(f"  Downloaded {total_size/1024/1024:.1f} MB")

            # Extract monthly files (.asc or .tif) and convert to GeoTIFF
            MONTH_ABBR = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }

            with zipfile.ZipFile(tmp_path) as zf:
                raster_names = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".tif", ".asc"))
                ]
                logger.info(f"  Contains {len(raster_names)} raster files")

                for raster_name in raster_names:
                    basename = Path(raster_name).stem.lower()

                    # Parse year + month from filename
                    # Pattern 1: pet2010jan (month abbreviation)
                    match = re.match(
                        rf"{variable_prefix}(\d{{4}})([a-z]{{3}})",
                        basename, re.IGNORECASE,
                    )
                    if match:
                        year = int(match.group(1))
                        month_str = match.group(2).lower()
                        month = MONTH_ABBR.get(month_str)
                        if month is None:
                            continue
                    else:
                        # Pattern 2: pet198001 (numeric month)
                        match = re.search(
                            rf"{variable_prefix}(\d{{6}})",
                            basename, re.IGNORECASE,
                        )
                        if not match:
                            continue
                        ym_compact = match.group(1)
                        year = int(ym_compact[:4])
                        month = int(ym_compact[4:])

                    if month < 1 or month > 12:
                        continue

                    ym = f"{year:04d}-{month:02d}"
                    ym_compact = f"{year:04d}{month:02d}"
                    if ym < start_ym or ym > end_ym:
                        continue

                    out_file = out_path / f"{our_name}-{ym_compact}.tif"
                    if out_file.exists():
                        continue

                    # Extract, assign CRS, resample to BCM 1km grid
                    _extract_and_convert(zf, raster_name, str(out_file), bcm_profile)

                    downloaded.append(str(out_file))

        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"Extracted {len(downloaded)} {our_name} monthly files from ScienceBase")
    return downloaded


def _extract_and_convert(
    zf: zipfile.ZipFile,
    member_name: str,
    out_tif_path: str,
    bcm_1km_profile: Optional[dict] = None,
) -> None:
    """Extract a raster from zip, assign CRS, resample to BCM 1km grid.

    ScienceBase BCMv8 files are 270m EPSG:3310 (CA Albers) with missing CRS tag.
    We assign CRS and resample to the 1km BCM grid.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with tempfile.TemporaryDirectory() as tmpdir:
        zf.extract(member_name, tmpdir)
        src_path = Path(tmpdir) / member_name

        with rasterio.open(str(src_path)) as src:
            src_profile = src.profile.copy()
            src_data = src.read(1).astype(np.float32)

            # Assign CRS if missing (BCMv8 native grid is EPSG:3310)
            src_crs = src.crs
            if src_crs is None:
                src_crs = rasterio.CRS.from_epsg(3310)

            if bcm_1km_profile is not None:
                # Resample from 270m to 1km
                h, w = bcm_1km_profile["height"], bcm_1km_profile["width"]
                dst_data = np.full((h, w), -9999.0, dtype=np.float32)

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=bcm_1km_profile["transform"],
                    dst_crs=rasterio.CRS.from_epsg(3310),
                    resampling=Resampling.bilinear,
                    src_nodata=-9999.0,
                    dst_nodata=-9999.0,
                )

                out_profile = bcm_1km_profile.copy()
                out_profile.update(
                    driver="GTiff", dtype="float32",
                    count=1, nodata=-9999.0, compress="lzw",
                )
                with rasterio.open(out_tif_path, "w", **out_profile) as dst:
                    dst.write(dst_data[np.newaxis, :])
            else:
                # Just convert format, assign CRS
                src_profile.update(
                    driver="GTiff", compress="lzw", crs=src_crs,
                )
                with rasterio.open(out_tif_path, "w", **src_profile) as dst:
                    dst.write(src_data[np.newaxis, :])


def download_all_from_sciencebase(
    out_dir: str,
    start_ym: str = "1980-01",
    end_ym: str = "2021-04",
    bcm_profile: Optional[dict] = None,
) -> Dict[str, List[str]]:
    """Download all BCMv8 climate inputs (pet, ppt, tmin, tmax) from ScienceBase.

    Parameters
    ----------
    out_dir : str
        Base output directory. Creates subdirectories per variable.
    start_ym, end_ym : str
        Date range as 'YYYY-MM' (inclusive).
    bcm_profile : dict, optional
        BCM 1km grid profile for resampling from native 270m.

    Returns
    -------
    dict
        {variable_name: [list of downloaded file paths]}
    """
    results = {}
    for sb_prefix, our_name in VAR_MAP.items():
        logger.info(f"=== Downloading {our_name} (ScienceBase: {sb_prefix}) ===")
        files = download_variable(sb_prefix, out_dir, start_ym, end_ym, bcm_profile)
        results[our_name] = files

    return results


def download_pck_gap(
    out_dir: str,
    start_ym: str = "2017-01",
    end_ym: str = "2020-09",
    bcm_profile: Optional[dict] = None,
) -> List[str]:
    """Download PCK gap-fill files from ScienceBase item 5f29c62d82cef313ed9edb39.

    Local BCM PCK data ends at 2016-12. This downloads the remaining months
    (2017-01 through 2020-09) from the ScienceBase PCK archive.

    Parameters
    ----------
    out_dir : str
        Output directory. Files saved in {out_dir}/pck/ as pck-YYYYMM.tif.
    start_ym, end_ym : str
        Date range for gap fill as 'YYYY-MM' (inclusive).
    bcm_profile : dict, optional
        BCM 1km grid profile for resampling from native 270m.
    """
    resp = requests.get(PCK_API, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    zip_files = []
    for f in data.get("files", []):
        name = f.get("name", "")
        if name.startswith("pck_WY") and name.endswith(".zip"):
            zip_files.append({"name": name, "url": f["url"]})

    logger.info(f"Found {len(zip_files)} PCK decade zips on ScienceBase (gap-fill item)")

    # Filter to only zips that overlap the gap period
    relevant = [
        z for z in zip_files
        if _decade_overlaps(z["name"], "pck", start_ym, end_ym)
    ]
    logger.info(f"  {len(relevant)} zips overlap gap period {start_ym} to {end_ym}")

    out_path = Path(out_dir) / "pck"
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded = []

    MONTH_ABBR = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    for finfo in relevant:
        name = finfo["name"]
        url = finfo["url"]

        logger.info(f"Downloading PCK gap-fill: {name}...")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            resp = requests.get(url, timeout=600, stream=True)
            resp.raise_for_status()
            total_size = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)
            logger.info(f"  Downloaded {total_size/1024/1024:.1f} MB")

            with zipfile.ZipFile(tmp_path) as zf:
                raster_names = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".tif", ".asc"))
                ]
                logger.info(f"  Contains {len(raster_names)} raster files")

                for raster_name in raster_names:
                    basename = Path(raster_name).stem.lower()

                    # Parse year + month: pck2017jan or pck201701
                    match = re.match(
                        r"pck(\d{4})([a-z]{3})",
                        basename, re.IGNORECASE,
                    )
                    if match:
                        year = int(match.group(1))
                        month = MONTH_ABBR.get(match.group(2).lower())
                        if month is None:
                            continue
                    else:
                        match = re.search(r"pck(\d{6})", basename, re.IGNORECASE)
                        if not match:
                            continue
                        ym_compact = match.group(1)
                        year = int(ym_compact[:4])
                        month = int(ym_compact[4:])

                    if month < 1 or month > 12:
                        continue

                    ym = f"{year:04d}-{month:02d}"
                    if ym < start_ym or ym > end_ym:
                        continue

                    ym_compact = f"{year:04d}{month:02d}"
                    out_file = out_path / f"pck-{ym_compact}.tif"
                    if out_file.exists():
                        continue

                    _extract_and_convert(zf, raster_name, str(out_file), bcm_profile)
                    downloaded.append(str(out_file))

        except Exception as e:
            logger.error(f"Failed to process PCK gap-fill {name}: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"Extracted {len(downloaded)} PCK gap-fill monthly files")
    return downloaded
