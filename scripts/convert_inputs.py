"""Convert BCMv8 InputFiles .asc to 1km GeoTIFF matching BCM grid."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

BCM = Path("/home/mmann1123/extra_space/BCM_HIST_NoModel")
INPUT_DIR = BCM / "InputFiles"

# Reference grid from existing output file
ref_path = BCM / "aet" / "aet-202104.tif"
with rasterio.open(ref_path) as ref:
    rp = ref.profile.copy()
    rt = ref.transform
    rc = ref.crs
    rh, rw = ref.shape

print(f"Reference grid: {rh}x{rw}, CRS={rc}, res=1000m")

out_profile = rp.copy()
out_profile.update(driver="GTiff", count=1, nodata=-9999.0, compress="lzw")

converted = 0
skipped = 0

for asc_path in sorted(INPUT_DIR.glob("*.asc")):
    out_path = INPUT_DIR / f"{asc_path.stem}.tif"
    if out_path.exists():
        skipped += 1
        print(f"  SKIP (exists): {out_path.name}")
        continue

    with rasterio.open(asc_path) as src:
        sd = src.read(1)
        src_dtype = sd.dtype
        st = src.transform
        sn = src.nodata if src.nodata is not None else -9999.0

    # Determine appropriate dtype and resampling
    # Integer grids (veg class, geol id, basins, mask) use nearest neighbor
    is_categorical = any(k in asc_path.stem.lower() for k in
                         ["fullveg", "geolid", "huc8", "mask", "aridity"])
    resamp = Resampling.nearest if is_categorical else Resampling.bilinear

    # Use float32 for continuous, int32 for categorical
    if is_categorical:
        sd = sd.astype(np.int32)
        dd = np.full((rh, rw), -9999, dtype=np.int32)
        op = out_profile.copy()
        op.update(dtype="int32")
    else:
        sd = sd.astype(np.float32)
        dd = np.full((rh, rw), -9999.0, dtype=np.float32)
        op = out_profile.copy()
        op.update(dtype="float32")

    reproject(
        source=sd, destination=dd,
        src_transform=st, src_crs=CRS.from_epsg(3310),
        dst_transform=rt, dst_crs=rc,
        resampling=resamp,
        src_nodata=sn, dst_nodata=-9999.0 if not is_categorical else -9999,
    )

    with rasterio.open(out_path, "w", **op) as dst:
        dst.write(dd[np.newaxis, :])

    valid = dd[dd != -9999] if is_categorical else dd[dd != -9999.0]
    cat_label = " [categorical/nearest]" if is_categorical else ""
    print(f"  {out_path.name}  valid={len(valid)}  range=[{valid.min()}, {valid.max()}]{cat_label}")
    converted += 1

print(f"\nConverted: {converted}, Skipped: {skipped}")
print(f"Total .tif in InputFiles: {len(list(INPUT_DIR.glob('*.tif')))}")
