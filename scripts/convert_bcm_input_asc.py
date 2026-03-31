"""Convert BCM input .asc files (ppt, tmn, tmx) to 1km GeoTIFF in pet_sciencebase/.

BCM input files are 270m EPSG:3310 ASCII grids with filenames like:
    tmx2017jan.asc, ppt2021oct.asc, tmn2022mar.asc

This script resamples them to the BCM 1km grid and saves as
    data/pet_sciencebase/{var}/{var}-YYYYMM.tif
matching the ScienceBase naming convention.

Usage:
    conda run -n deep_field python scripts/convert_bcm_input_asc.py \
        --input-dir ~/Downloads/tmx_WY2010_20 --var tmx
    conda run -n deep_field python scripts/convert_bcm_input_asc.py \
        --input-dir ~/Downloads/ppt_WY2021_25 --var ppt
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io_helpers import get_bcm_reference_profile

# Variable name mapping: BCM input prefix -> our standard name
VAR_MAP = {
    "ppt": "ppt",
    "tmn": "tmin",
    "tmx": "tmax",
}

MONTH_ABBR = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def convert_asc_files(input_dir: str, var_prefix: str, out_base: str, bcm_profile: dict) -> int:
    """Convert .asc files from input_dir to 1km GeoTIFF.

    Parameters
    ----------
    input_dir : str
        Directory containing .asc files (e.g., tmx2017jan.asc).
    var_prefix : str
        BCM variable prefix: 'ppt', 'tmn', or 'tmx'.
    out_base : str
        Base output directory (pet_sciencebase_dir). Files go to {out_base}/{our_name}/.
    bcm_profile : dict
        BCM 1km grid rasterio profile.

    Returns
    -------
    int
        Number of files converted.
    """
    our_name = VAR_MAP.get(var_prefix)
    if our_name is None:
        print(f"ERROR: Unknown variable '{var_prefix}'. Must be one of: {list(VAR_MAP.keys())}")
        return 0

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 0

    out_dir = Path(out_base) / our_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up output profile
    h, w = bcm_profile["height"], bcm_profile["width"]
    out_profile = bcm_profile.copy()
    out_profile.update(
        driver="GTiff", dtype="float32",
        count=1, nodata=-9999.0, compress="lzw",
    )

    # Find all .asc files
    asc_files = sorted(input_path.glob("*.asc"))
    if not asc_files:
        print(f"No .asc files found in {input_dir}")
        return 0

    print(f"Found {len(asc_files)} .asc files in {input_dir}")
    print(f"Output: {out_dir}/")
    print(f"Variable: {var_prefix} -> {our_name}")
    print(f"BCM grid: {h}x{w}")

    converted = 0
    skipped_exist = 0
    skipped_parse = 0

    for asc_path in asc_files:
        basename = asc_path.stem.lower()

        # Parse filename: {prefix}{year}{month_abbr} e.g., tmx2017jan
        match = re.match(rf"{var_prefix}(\d{{4}})([a-z]{{3}})$", basename, re.IGNORECASE)
        if not match:
            # Try numeric month: {prefix}{YYYYMM}
            match = re.match(rf"{var_prefix}(\d{{4}})(\d{{2}})$", basename, re.IGNORECASE)
            if not match:
                skipped_parse += 1
                continue
            year = match.group(1)
            month = match.group(2)
        else:
            year = match.group(1)
            month_str = match.group(2).lower()
            month = MONTH_ABBR.get(month_str)
            if month is None:
                skipped_parse += 1
                continue

        out_file = out_dir / f"{our_name}-{year}{month}.tif"
        if out_file.exists():
            skipped_exist += 1
            continue

        # Read, assign CRS, resample to 1km
        with rasterio.open(str(asc_path)) as src:
            src_data = src.read(1).astype(np.float32)
            src_transform = src.transform
            src_nodata = src.nodata if src.nodata is not None else -9999.0

        dst_data = np.full((h, w), -9999.0, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=CRS.from_epsg(3310),
            dst_transform=bcm_profile["transform"],
            dst_crs=CRS.from_epsg(3310),
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=-9999.0,
        )

        with rasterio.open(str(out_file), "w", **out_profile) as dst:
            dst.write(dst_data[np.newaxis, :])

        valid = dst_data[dst_data != -9999]
        print(f"  {out_file.name}  valid={len(valid)}  range=[{valid.min():.1f}, {valid.max():.1f}]")
        converted += 1

    print(f"\nConverted: {converted}, Skipped (exist): {skipped_exist}, Skipped (unparseable): {skipped_parse}")

    # Show coverage
    all_files = sorted(out_dir.glob(f"{our_name}-*.tif"))
    if all_files:
        f0 = all_files[0].stem.split("-")[1]
        fN = all_files[-1].stem.split("-")[1]
        print(f"Total {our_name} files: {len(all_files)}, range: {f0} to {fN}")

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert BCM input .asc files (270m) to 1km GeoTIFF in pet_sciencebase/"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .asc files (e.g., ~/Downloads/tmx_WY2010_20)"
    )
    parser.add_argument(
        "--var", required=True, choices=["ppt", "tmn", "tmx"],
        help="BCM variable prefix"
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bcm_profile = get_bcm_reference_profile(cfg.paths.bcm_dir)

    convert_asc_files(
        input_dir=args.input_dir,
        var_prefix=args.var,
        out_base=cfg.paths.pet_sciencebase_dir,
        bcm_profile=bcm_profile,
    )


if __name__ == "__main__":
    main()
