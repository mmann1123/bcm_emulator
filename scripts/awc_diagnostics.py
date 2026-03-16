"""Diagnose AWC data quality and spatial variability."""

import numpy as np
import zarr
import rasterio
import os

store = zarr.open("data/bcm_dataset.zarr", mode="r")
awc = np.array(store["inputs/static"][4])
mask = np.array(store["meta/valid_mask"])
elev = np.array(store["meta/pixel_elev"])
lat = np.array(store["inputs/static"][2])
lon = np.array(store["inputs/static"][3])

v = mask & (awc > 0)

# Spatial variability
cv = np.std(awc[v]) / np.mean(awc[v]) * 100
iqr = np.percentile(awc[v], 75) - np.percentile(awc[v], 25)
print("=== AWC spatial variability ===")
print(f"  CV: {cv:.1f}%")
print(f"  IQR: {iqr:.1f} mm (IQR/median = {iqr/np.median(awc[v])*100:.1f}%)")
print(f"  Correlation with elevation: {np.corrcoef(awc[v], elev[v])[0,1]:.3f}")
print(f"  Correlation with latitude:  {np.corrcoef(awc[v], lat[v])[0,1]:.3f}")
print(f"  Correlation with longitude: {np.corrcoef(awc[v], lon[v])[0,1]:.3f}")

# Per-layer contributions
print("\n=== Raw POLARIS layer contributions ===")
for f in sorted(os.listdir("data/awc")):
    if f.startswith("awc_layer_"):
        with rasterio.open(f"data/awc/{f}") as src:
            d = src.read(1)
            nodata = -9999.0
            vd = d[d != nodata]
            if len(vd) > 0:
                print(f"  {f}: median={np.median(vd):.1f}, "
                      f"5th={np.percentile(vd,5):.1f}, 95th={np.percentile(vd,95):.1f}")

# Check if layer 0_5 (the one that had corrupt mosaic) is present
layer_05 = "data/awc/awc_layer_0_5.tif"
if os.path.exists(layer_05):
    with rasterio.open(layer_05) as src:
        d = src.read(1)
        vd = d[d != -9999.0]
        print(f"\n  Layer 0_5 exists: {len(vd)} valid pixels, median={np.median(vd):.1f}" if len(vd) > 0 else f"\n  Layer 0_5 exists but has {len(vd)} valid pixels!")
else:
    print(f"\n  WARNING: {layer_05} MISSING — AWC is missing top 5cm contribution!")

# Expected vs observed
print("\n=== DIAGNOSIS: AWC quality ===")
print(f"Observed domain range: {awc[v].min():.0f} - {awc[v].max():.0f} mm")
print(f"Observed domain median: {np.median(awc[v]):.0f} mm")
print(f"")
print("POLARIS AWC = integral(theta_s - theta_r, 0-100cm)")
print("theta_s - theta_r represents TOTAL porosity minus residual,")
print("but does NOT account for:")
print("  - Rock fragment volume (skeletal desert soils can be 50-80% gravel)")
print("  - Soil depth < 100cm (desert soils often <30cm to bedrock)")
print("  - Cemented layers (caliche/petrocalcic horizons)")
print("")
print("Result: AWC is uniformly ~400mm everywhere, providing")
print("almost no useful spatial contrast between desert and valley soils.")
print("The model cannot learn water-limitation from a near-constant input.")
