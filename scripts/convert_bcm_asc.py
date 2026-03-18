"""Convert BCMv8 WY2021-2024 .asc files to 1km GeoTIFF matching existing grid."""

import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

BCM = Path("/home/mmann1123/extra_space/BCM_HIST_NoModel")
VARS = ["aet", "cwd", "pck", "pet"]
MA = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Reference grid from existing file
ref_path = BCM / "aet" / "aet-202104.tif"
with rasterio.open(ref_path) as ref:
    rp = ref.profile.copy()
    rt = ref.transform
    rc = ref.crs
    rh, rw = ref.shape

print(f"Reference grid: {rh}x{rw}, CRS={rc}, res=1000m")

rp.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="lzw")

converted = 0
skipped = 0

for var in VARS:
    out_dir = BCM / var
    out_dir.mkdir(exist_ok=True)
    wy_dirs = sorted([d for d in BCM.glob(f"{var}_WY*") if d.is_dir()])

    for wy_dir in wy_dirs:
        asc_files = sorted(wy_dir.glob("*.asc"))
        print(f"{wy_dir.name}: {len(asc_files)} files")

        for asc_path in asc_files:
            bn = asc_path.stem.lower()
            m = re.match(rf"({var})(\d{{4}})([a-z]{{3}})$", bn)
            if not m:
                print(f"  SKIP (unparseable): {asc_path.name}")
                continue

            year, month = m.group(2), MA.get(m.group(3))
            if not month:
                print(f"  SKIP (bad month): {asc_path.name}")
                continue

            out_path = out_dir / f"{var}-{year}{month}.tif"
            if out_path.exists():
                skipped += 1
                continue

            with rasterio.open(asc_path) as src:
                sd = src.read(1).astype(np.float32)
                st = src.transform
                sn = src.nodata if src.nodata is not None else -9999.0

            dd = np.full((rh, rw), -9999.0, dtype=np.float32)
            reproject(
                source=sd, destination=dd,
                src_transform=st, src_crs=CRS.from_epsg(3310),
                dst_transform=rt, dst_crs=rc,
                resampling=Resampling.bilinear,
                src_nodata=sn, dst_nodata=-9999.0,
            )

            with rasterio.open(out_path, "w", **rp) as dst:
                dst.write(dd[np.newaxis, :])

            v = dd[dd != -9999]
            print(f"  {out_path.name}  valid={len(v)}  range=[{v.min():.1f}, {v.max():.1f}]")
            converted += 1

print(f"\nConverted: {converted}, Skipped (exist): {skipped}")
print("\n=== Final coverage ===")
for var in VARS:
    files = sorted((BCM / var).glob(f"{var}-*.tif"))
    f0 = files[0].stem.split("-")[1] if files else "?"
    fN = files[-1].stem.split("-")[1] if files else "?"
    print(f"  {var}: {len(files)} files, {f0} to {fN}")
