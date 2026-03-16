"""Compare spatial NSE maps between two snapshots by region."""

import rasterio
import numpy as np
import zarr

snap_v2 = "snapshots/v2-fveg-srad-fix/spatial_maps"
snap_v3 = "snapshots/v3-vpd-awc/spatial_maps"

# Load supporting data
store = zarr.open("data/bcm_dataset.zarr", mode="r")
mask = np.array(store["meta/valid_mask"])
awc = np.array(store["inputs/static"][4])
elev = np.array(store["meta/pixel_elev"])
lat_norm = np.array(store["inputs/static"][2])  # northing (0=south, 1=north)
lon_norm = np.array(store["inputs/static"][3])  # easting (0=west, 1=east)

SEP = "=" * 60

for var in ["aet", "cwd", "pet", "pck"]:
    with rasterio.open(f"{snap_v2}/nse_{var}.tif") as src:
        v2 = src.read(1)
    with rasterio.open(f"{snap_v3}/nse_{var}.tif") as src:
        v3 = src.read(1)

    valid = mask & np.isfinite(v2) & np.isfinite(v3)
    diff = np.where(valid, v3 - v2, np.nan)

    d = diff[valid]
    improved = (d > 0).sum()
    degraded = (d < 0).sum()

    print(f"\n{SEP}")
    print(f"{var.upper()} NSE change (v3 - v2)")
    print(SEP)
    print(f"  Overall: median={np.nanmedian(d):+.4f}, mean={np.nanmean(d):+.4f}")
    print(f"  Improved: {improved}/{len(d)} ({100*improved/len(d):.1f}%)")
    print(f"  Degraded: {degraded}/{len(d)} ({100*degraded/len(d):.1f}%)")

    # SE California drylands: low lat (south), high lon (east)
    se_desert = valid & (lat_norm < 0.25) & (lon_norm > 0.6)
    if se_desert.sum() > 0:
        d_se = diff[se_desert]
        pct = 100 * (d_se > 0).sum() / len(d_se)
        print(f"  SE CA desert (n={se_desert.sum()}):")
        print(f"    median={np.nanmedian(d_se):+.4f}, mean={np.nanmean(d_se):+.4f}, improved={pct:.1f}%")
        # Also show absolute NSE in both versions
        print(f"    v2 NSE: median={np.nanmedian(v2[se_desert]):.4f}, v3 NSE: median={np.nanmedian(v3[se_desert]):.4f}")

    # Low AWC pixels (water-limited)
    low_awc = valid & (awc > 0) & (awc < 350)
    if low_awc.sum() > 0:
        d_la = diff[low_awc]
        pct = 100 * (d_la > 0).sum() / len(d_la)
        print(f"  Low AWC (<350mm, n={low_awc.sum()}):")
        print(f"    median={np.nanmedian(d_la):+.4f}, mean={np.nanmean(d_la):+.4f}, improved={pct:.1f}%")

    # High AWC pixels (energy-limited)
    high_awc = valid & (awc >= 350)
    if high_awc.sum() > 0:
        d_ha = diff[high_awc]
        pct = 100 * (d_ha > 0).sum() / len(d_ha)
        print(f"  High AWC (>=350mm, n={high_awc.sum()}):")
        print(f"    median={np.nanmedian(d_ha):+.4f}, mean={np.nanmean(d_ha):+.4f}, improved={pct:.1f}%")

    # NW coastal (fog/VPD target)
    nw_coast = valid & (lat_norm > 0.7) & (lon_norm < 0.3)
    if nw_coast.sum() > 0:
        d_nw = diff[nw_coast]
        pct = 100 * (d_nw > 0).sum() / len(d_nw)
        print(f"  NW coast (n={nw_coast.sum()}):")
        print(f"    median={np.nanmedian(d_nw):+.4f}, mean={np.nanmean(d_nw):+.4f}, improved={pct:.1f}%")
        print(f"    v2 NSE: median={np.nanmedian(v2[nw_coast]):.4f}, v3 NSE: median={np.nanmedian(v3[nw_coast]):.4f}")

    # Central Valley
    cv = valid & (elev < 200) & (lat_norm > 0.3) & (lat_norm < 0.7) & (lon_norm > 0.3) & (lon_norm < 0.7)
    if cv.sum() > 0:
        d_cv = diff[cv]
        pct = 100 * (d_cv > 0).sum() / len(d_cv)
        print(f"  Central Valley (n={cv.sum()}):")
        print(f"    median={np.nanmedian(d_cv):+.4f}, mean={np.nanmean(d_cv):+.4f}, improved={pct:.1f}%")

    # Worst v2 pixels (NSE < 0.5)
    poor_v2 = valid & (v2 < 0.5)
    if poor_v2.sum() > 0:
        d_poor = diff[poor_v2]
        pct = 100 * (d_poor > 0).sum() / len(d_poor)
        print(f"  Poor v2 pixels (NSE<0.5, n={poor_v2.sum()}):")
        print(f"    median={np.nanmedian(d_poor):+.4f}, mean={np.nanmean(d_poor):+.4f}, improved={pct:.1f}%")
        print(f"    v2 NSE: median={np.nanmedian(v2[poor_v2]):.4f}, v3 NSE: median={np.nanmedian(v3[poor_v2]):.4f}")

    # Pixels with no AWC data (awc=0) — control group
    no_awc = valid & (awc == 0)
    if no_awc.sum() > 0:
        d_na = diff[no_awc]
        pct = 100 * (d_na > 0).sum() / len(d_na)
        print(f"  No AWC data (n={no_awc.sum()}):")
        print(f"    median={np.nanmedian(d_na):+.4f}, mean={np.nanmean(d_na):+.4f}, improved={pct:.1f}%")
