"""Align all rasters to BCM grid, compute derived features, build zarr store."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def build_time_index(start_ym: str, end_ym: str) -> List[str]:
    """Build a list of YYYY-MM strings from start to end (inclusive)."""
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def compute_snow_frac(
    tmin: np.ndarray,
    tmax: np.ndarray,
    threshold: float = 0.0,
    transition_width: float = 2.0,
) -> np.ndarray:
    """Compute fraction of precipitation falling as snow.

    snow_frac = max(0, min(1, (threshold - tmean) / transition_width))
    """
    tmean = (tmin + tmax) / 2.0
    frac = np.clip((threshold - tmean) / transition_width, 0.0, 1.0)
    return frac


def compute_vpd(tmin: np.ndarray, tmax: np.ndarray) -> np.ndarray:
    """Compute Vapor Pressure Deficit from tmin and tmax using Tetens equation.

    Assumes dewpoint ~ tmin (standard meteorological approximation).

    Parameters
    ----------
    tmin, tmax : np.ndarray
        Temperature arrays in degrees Celsius.

    Returns
    -------
    np.ndarray
        VPD in kPa (clipped to >= 0).
    """
    e_sat = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    e_act = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    return np.maximum(0.0, e_sat - e_act)


def compute_windward_index(
    dem: np.ndarray, valid_mask: np.ndarray, window_km: int = 50
) -> np.ndarray:
    """Compute windward/leeward index from DEM.

    For each pixel, compute its elevation minus the mean elevation of terrain
    to its west within window_km. Positive = windward (exposed to prevailing
    westerly moisture), negative = leeward (rain shadow).
    """
    pixel_size_km = 1.0  # BCM grid is 1km
    window_px = int(window_km / pixel_size_km)

    windward = np.zeros_like(dem, dtype=np.float32)
    for col in range(dem.shape[1]):
        west_start = max(0, col - window_px)
        west_profile = dem[:, west_start : col + 1]
        windward[:, col] = dem[:, col] - np.nanmean(west_profile, axis=1)

    windward[~valid_mask] = 0.0
    return windward


def compute_sws(
    ppt_raw: np.ndarray,
    pet_raw: np.ndarray,
    awc_mm: np.ndarray,
    valid_mask: np.ndarray,
    spinup_years: int = 3,
) -> np.ndarray:
    """Stress-modulated bucket model matching BCMv8 linear stress formulation (v14+).

    At each timestep:
        stress = min(SWS[t-1] / AWC, 1.0)       # 0 (bone dry) to 1 (at capacity)
        AET_approx = PET[t] * stress              # drainage decreases as soil dries
        SWS[t] = clamp(SWS[t-1] + PPT[t] - AET_approx, 0, AWC)

    Parameters
    ----------
    ppt_raw  : (T, H, W) mm/month — raw precipitation
    pet_raw  : (T, H, W) mm/month — raw BCMv8 PET targets
    awc_mm   : (H, W)    mm       — available water capacity
    valid_mask: (H, W)   bool
    spinup_years : int — number of years for spin-up (cycled 3x)

    Returns
    -------
    (T, H, W) SWS in mm (raw, unnormalized).
    """
    T, H, W = ppt_raw.shape
    sws_out = np.zeros((T, H, W), dtype=np.float32)
    sws_t = awc_mm * 0.5  # initialize at 50% capacity

    # Precompute safe AWC for division (avoid /0 where AWC=0)
    awc_safe = np.where(awc_mm > 0, awc_mm, 1.0)

    spinup_len = min(spinup_years * 12, T)
    logger.info(f"SWS spin-up: {spinup_years} years ({spinup_len} months × 3 passes)")
    for pass_num in range(3):
        for t in range(spinup_len):
            stress = np.minimum(sws_t / awc_safe, 1.0)
            aet_approx = pet_raw[t] * stress
            sws_t = np.clip(sws_t + ppt_raw[t] - aet_approx, 0.0, awc_mm)
        logger.info(f"  Spin-up pass {pass_num+1}/3 — "
                    f"SWS mean (valid): {sws_t[valid_mask].mean():.1f}mm")

    logger.info("SWS: running full time series...")
    for t in range(T):
        stress = np.minimum(sws_t / awc_safe, 1.0)
        aet_approx = pet_raw[t] * stress
        sws_t = np.clip(sws_t + ppt_raw[t] - aet_approx, 0.0, awc_mm)
        sws_out[t] = sws_t
        if t % 120 == 0:
            logger.info(f"  t={t}/{T}  SWS mean: {sws_t[valid_mask].mean():.1f}mm")

    sws_out[:, ~valid_mask] = 0.0
    return sws_out


def compute_rolling_std(
    data: np.ndarray,
    window: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute causal rolling std along time axis for a (T, H, W) array.

    Uses current + previous (window-1) months. First window-1 timesteps
    use partial windows (min 2 values required for std).

    Parameters
    ----------
    data : (T, H, W) raw values
    window : rolling window size in months
    valid_mask : (H, W) bool

    Returns
    -------
    (T, H, W) rolling std values (0.0 for invalid pixels and single-element windows)
    """
    T, H, W = data.shape
    result = np.zeros_like(data)

    for col in range(W):
        valid_col = valid_mask[:, col]
        if not valid_col.any():
            continue

        col_data = data[:, valid_col, col]  # (T, n_valid)

        # Cumulative sums for online variance: std = sqrt(E[x²] - E[x]²)
        cumsum = np.cumsum(col_data, axis=0)
        cumsum2 = np.cumsum(col_data ** 2, axis=0)

        for t in range(T):
            t_start = max(0, t - window + 1)
            n = t - t_start + 1
            if n < 2:
                continue
            if t_start == 0:
                s = cumsum[t]
                s2 = cumsum2[t]
            else:
                s = cumsum[t] - cumsum[t_start - 1]
                s2 = cumsum2[t] - cumsum2[t_start - 1]
            mean = s / n
            var = np.maximum(0.0, s2 / n - mean ** 2)
            result[t, valid_col, col] = np.sqrt(var)

    result[:, ~valid_mask] = 0.0
    return result


def _process_single_timestep(
    t_idx: int,
    ym: str,
    sb_dir: Path,
    daymet_processed: Path,
    prism_processed: Path,
    bcm_dir: str,
    bcm_profile: dict,
    valid_mask: np.ndarray,
    snow_threshold: float,
    snow_transition: float,
) -> dict:
    """Read all files for one timestep and return numpy arrays.

    Does NOT write to zarr — the caller collects results and writes sequentially.
    """
    from ..utils.io_helpers import read_raster_as_masked

    H, W = bcm_profile["height"], bcm_profile["width"]
    year_str = ym[:4]
    month_str = ym[5:7]
    ym_compact = f"{year_str}{month_str}"

    dynamic_slice = np.zeros((11, H, W), dtype=np.float32)
    targets = {}
    missing = {}

    # --- ScienceBase climate inputs: ppt, tmin, tmax ---
    for ch_idx, our_name in enumerate(["ppt", "tmin", "tmax"]):
        fpath = _find_file(sb_dir / our_name, f"{our_name}-{ym_compact}.tif")
        if fpath:
            data = _read_bcm_grid_file(str(fpath))
            data[~valid_mask] = 0.0
            dynamic_slice[ch_idx] = data
        else:
            missing[our_name] = 1

    # --- Wet days and ppt intensity (from PRISM daily-derived) ---
    wet_path = _find_file(prism_processed / "wet_days", f"*{ym_compact}*.tif")
    if wet_path:
        dynamic_slice[3] = _read_and_align(str(wet_path), bcm_profile)
    else:
        missing["wet_days"] = 1

    int_path = _find_file(prism_processed / "ppt_intensity", f"*{ym_compact}*.tif")
    if int_path:
        dynamic_slice[4] = _read_and_align(str(int_path), bcm_profile)
    else:
        missing["ppt_intensity"] = 1

    # --- DAYMET srad ---
    srad_path = _find_file(daymet_processed / "srad_bcm", f"srad-{ym_compact}.tif")
    if srad_path:
        dynamic_slice[5] = _read_and_align(str(srad_path), bcm_profile)
    else:
        missing["srad"] = 1

    # --- Snow fraction (derived from tmin, tmax) ---
    tmin_data = dynamic_slice[1]
    tmax_data = dynamic_slice[2]
    snow_frac = compute_snow_frac(
        tmin_data, tmax_data,
        threshold=snow_threshold, transition_width=snow_transition,
    )
    dynamic_slice[6] = snow_frac

    # --- VPD (derived from tmin, tmax) ---
    vpd = compute_vpd(tmin_data, tmax_data)
    vpd[~valid_mask] = 0.0
    dynamic_slice[9] = vpd

    # --- KBDI (from PRISM daily-derived) ---
    kbdi_path = _find_file(prism_processed / "kbdi", f"*{ym_compact}*.tif")
    if kbdi_path:
        dynamic_slice[10] = _read_and_align(str(kbdi_path), bcm_profile)
    else:
        missing["kbdi"] = 1

    # --- Targets: aet, cwd from BCM local outputs ---
    aet_data = None
    cwd_data = None
    for var in ["aet", "cwd"]:
        fpath = _find_file(Path(bcm_dir) / var, f"{var}-{ym_compact}.tif")
        if fpath:
            data = read_raster_as_masked(str(fpath))
            data[np.isnan(data)] = 0.0
            targets[var] = data
            if var == "aet":
                aet_data = data
            else:
                cwd_data = data
        else:
            missing[var] = 1

    # --- Target: PCK from local BCM first, then ScienceBase gap-fill ---
    pck_fpath = _find_file(Path(bcm_dir) / "pck", f"pck-{ym_compact}.tif")
    if pck_fpath is None:
        pck_fpath = _find_file(sb_dir / "pck", f"pck-{ym_compact}.tif")
    if pck_fpath:
        pck_data = read_raster_as_masked(str(pck_fpath))
        pck_data[np.isnan(pck_data)] = 0.0
        targets["pck"] = pck_data
    else:
        missing["pck"] = 1

    # --- Target: PET derived as AET + CWD ---
    if aet_data is not None and cwd_data is not None:
        targets["pet"] = aet_data + cwd_data

    # Channels 7 (pck_prev) and 8 (aet_prev) left as zeros — filled in separate pass
    return {
        "t_idx": t_idx,
        "dynamic": dynamic_slice,
        "targets": targets,
        "missing": missing,
    }


def build_zarr_store(
    zarr_path: str,
    bcm_dir: str,
    sciencebase_dir: str,
    daymet_dir: str,
    prism_dir: str,
    topo_solar_path: str,
    elevation_path: str,
    fveg_dir: str = "",
    soil_dir: str = "",
    soil_depth_path: str = "",
    aridity_path: str = "",
    field_capacity_path: str = "",
    wilting_point_path: str = "",
    awc_path: str = "",
    bcm_profile: dict = None,
    time_range: Tuple[str, str] = ("1980-01", "2020-09"),
    snow_threshold: float = 0.0,
    snow_transition: float = 2.0,
) -> str:
    """Build zarr store with all inputs and targets aligned to BCM grid.

    Data sources:
        - ScienceBase: ppt, tmin, tmax (resampled 270m -> 1km BCM grid)
        - DAYMET: srad (reprojected to BCM grid)
        - PRISM daily-derived: wet_days, ppt_intensity (reprojected to BCM grid)
        - BCM outputs: aet, cwd, pck (targets, on BCM grid)
        - Local: elevation, topo_solar, lat, lon, ksat, sand, clay, soil_depth, aridity, FC, WP, SOM, windward_index (static)
        - POLARIS: root-zone AWC (0-100cm) for SWS bucket model

    Zarr structure:
        /inputs/dynamic    (T, 15, H, W)  - dynamic input channels (11 base + SWS + 3 rolling std)
        /inputs/static     (14, H, W)     - static input channels (no awc_total; FVEG at ch13)
        /targets/pet       (T, H, W)
        /targets/pck       (T, H, W)
        /targets/aet       (T, H, W)
        /targets/cwd       (T, H, W)
        /meta/time         (T,)           - YYYY-MM strings
        /meta/valid_mask   (H, W)         - boolean valid pixel mask
        /meta/pixel_elev   (H, W)         - elevation for stratification
    """
    from ..utils.io_helpers import (
        get_valid_mask,
        read_raster_as_masked,
    )

    time_index = build_time_index(*time_range)
    T = len(time_index)
    H, W = bcm_profile["height"], bcm_profile["width"]

    logger.info(f"Building zarr store: {T} timesteps, {H}x{W} grid")

    # Create zarr store
    store = zarr.open(zarr_path, mode="w")

    # Dynamic inputs: (T, 15, H, W)
    # Channels 0-10: ppt, tmin, tmax, wet_days, ppt_intensity, srad, snow_frac, pck_prev, aet_prev, vpd, kbdi
    # Channels 11-14: sws, vpd_roll6_std, srad_roll6_std, tmax_roll3_std (computed in post-processing)
    dynamic = store.zeros(
        name="inputs/dynamic", shape=(T, 15, H, W), chunks=(12, 15, H, W), dtype="float32"
    )

    # Static inputs: (14, H, W) -- elev, topo_solar, lat, lon, ksat, sand, clay, soil_depth, aridity, FC, WP, SOM, windward_index, fveg_class_id
    static = store.zeros(name="inputs/static", shape=(14, H, W), dtype="float32")

    # Targets: (T, H, W)
    for var in ["pet", "pck", "aet", "cwd"]:
        store.zeros(name=f"targets/{var}", shape=(T, H, W), chunks=(12, H, W), dtype="float32")

    # Time index
    store.create_array(name="meta/time", data=np.array(time_index, dtype="U7"))

    # Valid mask
    valid_mask = get_valid_mask(bcm_dir)
    store.create_array(name="meta/valid_mask", data=valid_mask)

    # Grid transform and CRS (for rasterization alignment)
    t = bcm_profile["transform"]
    store.create_array(name="meta/transform", data=np.array([t.a, t.b, t.c, t.d, t.e, t.f], dtype=np.float64))
    store.create_array(name="meta/crs", data=np.array([str(bcm_profile["crs"])]))

    # ---- Static inputs ----
    logger.info("Processing static inputs...")

    # Elevation
    elev = _read_and_align(elevation_path, bcm_profile)
    elev[~valid_mask] = 0.0
    static[0] = elev
    store.create_array(name="meta/pixel_elev", data=elev)

    # Topographic solar radiation
    topo = read_raster_as_masked(topo_solar_path)
    topo[np.isnan(topo)] = 0.0
    static[1] = topo

    # Normalized latitude and longitude from grid coordinates
    transform = bcm_profile["transform"]
    cols = np.arange(W)
    rows = np.arange(H)
    x_coords = transform.c + cols * transform.a
    y_coords = transform.f + rows * transform.e

    x_grid = np.broadcast_to(x_coords[np.newaxis, :], (H, W)).astype(np.float32)
    y_grid = np.broadcast_to(y_coords[:, np.newaxis], (H, W)).astype(np.float32)

    x_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min() + 1e-8)
    y_norm = (y_grid - y_grid.min()) / (y_grid.max() - y_grid.min() + 1e-8)

    static[2] = y_norm  # lat (northing)
    static[3] = x_norm  # lon (easting)

    # Soil properties — channels 4 (ksat), 5 (sand), 6 (clay)
    soil_path = Path(soil_dir) if soil_dir else None
    for ch_idx, prop_name in enumerate(["ksat", "sand", "clay"], start=4):
        prop_file = soil_path / f"{prop_name}_bcm.tif" if soil_path else None
        if prop_file and prop_file.exists():
            logger.info(f"Loading {prop_name} raster...")
            data = _read_and_align(str(prop_file), bcm_profile)
            data[~valid_mask] = 0.0
            static[ch_idx] = data
        else:
            logger.warning(f"{prop_name} not found; channel {ch_idx} will be zeros")

    # Soil depth — channel 7
    if soil_depth_path and Path(soil_depth_path).exists():
        logger.info("Loading soil depth raster...")
        data = _read_and_align(str(soil_depth_path), bcm_profile)
        data[~valid_mask] = 0.0
        static[7] = data
    else:
        logger.warning("Soil depth not found; channel 7 will be zeros")

    # Aridity index — channel 8
    if aridity_path and Path(aridity_path).exists():
        logger.info("Loading aridity index raster...")
        data = _read_and_align(str(aridity_path), bcm_profile)
        data[~valid_mask] = 0.0
        static[8] = data
    else:
        logger.warning("Aridity index not found; channel 8 will be zeros")

    # Field capacity — channel 9
    if field_capacity_path and Path(field_capacity_path).exists():
        logger.info("Loading field capacity raster...")
        data = _read_and_align(str(field_capacity_path), bcm_profile)
        data[~valid_mask] = 0.0
        static[9] = data
    else:
        logger.warning("Field capacity not found; channel 9 will be zeros")

    # Wilting point — channel 10
    if wilting_point_path and Path(wilting_point_path).exists():
        logger.info("Loading wilting point raster...")
        data = _read_and_align(str(wilting_point_path), bcm_profile)
        data[~valid_mask] = 0.0
        static[10] = data
    else:
        logger.warning("Wilting point not found; channel 10 will be zeros")

    # SOM (organic matter from POLARIS) — channel 11
    som_file = soil_path / "om_bcm.tif" if soil_path else None
    if som_file and som_file.exists():
        logger.info("Loading SOM (organic matter) raster...")
        data = _read_and_align(str(som_file), bcm_profile)
        data[~valid_mask] = 0.0
        static[11] = data
    else:
        logger.warning("SOM (om_bcm.tif) not found; channel 11 will be zeros")

    # Windward/leeward index (derived from DEM) -- channel 12
    logger.info("Computing windward index from DEM...")
    windward = compute_windward_index(elev, valid_mask)
    static[12] = windward

    # FVEG (CWHR vegetation class ID) -- channel 13 (categorical, must be last)
    import json as _json

    fveg_dir_path = Path(fveg_dir) if fveg_dir else None
    if fveg_dir_path and (fveg_dir_path / "fveg_partveg.tif").exists():
        logger.info("Loading FVEG partveg raster...")
        fveg_data = _read_bcm_grid_file(str(fveg_dir_path / "fveg_partveg.tif"))
        fveg_data[np.isnan(fveg_data)] = 0.0
        static[13] = fveg_data

        # Store FVEG metadata
        classmap_path = fveg_dir_path / "fveg_class_map.json"
        if classmap_path.exists():
            with open(classmap_path) as f:
                fveg_meta = _json.load(f)
            num_fveg_classes = fveg_meta["num_classes"]
            store.create_array(name="meta/fveg_num_classes", data=np.array([num_fveg_classes], dtype=np.int32))
            store.attrs["fveg_class_map"] = _json.dumps(fveg_meta["id_to_info"])
            logger.info(f"FVEG: {num_fveg_classes} classes stored")
    else:
        logger.warning("FVEG data not found; channel 13 will be zeros")

    # ---- Dynamic inputs and targets (parallel) ----
    sb_dir = Path(sciencebase_dir)
    daymet_processed = Path(daymet_dir)
    prism_processed = Path(prism_dir)

    n_workers = min(20, os.cpu_count() or 4)
    logger.info(
        f"Processing dynamic inputs and targets with {n_workers} threads..."
    )

    # Track missing data for summary
    missing_counts = {"ppt": 0, "tmin": 0, "tmax": 0, "wet_days": 0,
                      "ppt_intensity": 0, "srad": 0, "kbdi": 0,
                      "aet": 0, "cwd": 0, "pck": 0}

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_single_timestep,
                t_idx, ym, sb_dir, daymet_processed, prism_processed,
                bcm_dir, bcm_profile, valid_mask,
                snow_threshold, snow_transition,
            ): t_idx
            for t_idx, ym in enumerate(time_index)
        }
        for future in as_completed(futures):
            result = future.result()
            t = result["t_idx"]

            # Write to zarr in main thread (no concurrent chunk writes)
            dynamic[t, :11] = result["dynamic"]
            for var in ["pet", "pck", "aet", "cwd"]:
                if var in result["targets"]:
                    store[f"targets/{var}"][t] = result["targets"][var]

            # Accumulate missing counts
            for var, count in result["missing"].items():
                missing_counts[var] += count

            done += 1
            if done % 60 == 0:
                logger.info(f"Wrote {done}/{T} timesteps to zarr")

    # --- Log missing data summary ---
    for var, count in missing_counts.items():
        if count > 0:
            logger.warning(f"Missing {var}: {count}/{T} timesteps (filled with zeros)")

    # --- Fill pck_prev and aet_prev channels ---
    logger.info("Filling PCK(t-1) and AET(t-1) channels...")
    for t_idx in range(1, T):
        dynamic[t_idx, 7] = np.array(store["targets/pck"][t_idx - 1])
        dynamic[t_idx, 8] = np.array(store["targets/aet"][t_idx - 1])

    # --- Compute SWS (stress-modulated bucket model, v14+) ---
    logger.info("Computing SWS (stress-modulated bucket model)...")
    logger.info(f"Loading POLARIS root-zone AWC from {awc_path}...")
    awc_mm = _read_and_align(str(awc_path), bcm_profile)
    awc_mm[~valid_mask] = 0.0
    logger.info(f"POLARIS AWC: min={awc_mm[valid_mask].min():.1f}mm  "
                f"max={awc_mm[valid_mask].max():.1f}mm  "
                f"mean={awc_mm[valid_mask].mean():.1f}mm")

    ppt_raw = np.array(store["inputs/dynamic"][:, 0, :, :])  # channel 0 = ppt
    pet_raw = np.array(store["targets/pet"])
    sws = compute_sws(ppt_raw, pet_raw, awc_mm, valid_mask)
    del ppt_raw, pet_raw, awc_mm

    logger.info("Writing SWS to channel 11...")
    store["inputs/dynamic"][:, 11, :, :] = sws

    # SWS distribution diagnostics
    sws_valid = sws[:, valid_mask]
    pct_zero = float((sws_valid == 0).sum()) / sws_valid.size * 100
    logger.info(f"SWS: {pct_zero:.1f}% zeros, mean={sws_valid.mean():.1f}mm, "
                f"range=[{sws_valid.min():.1f}, {sws_valid.max():.1f}]mm")
    del sws, sws_valid

    # --- Compute rolling std channels ---
    rolling_features = [
        ("vpd_roll6_std", 9, 6, 12),    # vpd=ch9, window=6, dest=ch12
        ("srad_roll6_std", 5, 6, 13),   # srad=ch5, window=6, dest=ch13
        ("tmax_roll3_std", 2, 3, 14),   # tmax=ch2, window=3, dest=ch14
    ]
    for feat_name, src_idx, window, dest_idx in rolling_features:
        logger.info(f"Computing {feat_name} (source ch{src_idx}, window={window})...")
        src_data = np.array(store["inputs/dynamic"][:, src_idx, :, :])
        roll_std = compute_rolling_std(src_data, window, valid_mask)
        store["inputs/dynamic"][:, dest_idx, :, :] = roll_std
        valid_vals = roll_std[:, valid_mask]
        logger.info(f"  {feat_name}: mean={valid_vals.mean():.4f}, std={valid_vals.std():.4f}")
        del src_data, roll_std, valid_vals

    # --- Compute and store normalization stats ---
    logger.info("Computing normalization statistics...")
    _compute_norm_stats(store, valid_mask)

    logger.info(f"Zarr store built: {zarr_path}")
    return zarr_path


def _find_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find a file matching a glob pattern in a directory."""
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def _read_bcm_grid_file(path: str) -> np.ndarray:
    """Read a file already on the BCM grid (no reprojection needed)."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    nodata = -9999.0
    data[data == nodata] = np.nan
    data[np.isnan(data)] = 0.0
    return data


def _read_and_align(path: str, bcm_profile: dict) -> np.ndarray:
    """Read a raster and align to BCM grid, reprojecting if needed."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(path) as src:
        h, w = bcm_profile["height"], bcm_profile["width"]

        if (
            src.crs == rasterio.CRS.from_string(str(bcm_profile["crs"]))
            and src.shape == (h, w)
        ):
            data = src.read(1).astype(np.float32)
        else:
            data = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=bcm_profile["transform"],
                dst_crs=bcm_profile["crs"],
                resampling=Resampling.bilinear,
            )

    data[data == -9999.0] = 0.0
    data[np.isnan(data)] = 0.0
    return data


def _compute_norm_stats(store: zarr.Group, valid_mask: np.ndarray) -> None:
    """Compute per-channel mean and std for normalization."""
    dynamic = store["inputs/dynamic"]
    T = dynamic.shape[0]

    n_channels = 15
    channel_names = [
        "ppt", "tmin", "tmax", "wet_days", "ppt_intensity",
        "srad", "snow_frac", "pck_prev", "aet_prev", "vpd", "kbdi",
        "sws", "vpd_roll6_std", "srad_roll6_std", "tmax_roll3_std",
    ]

    n_valid = valid_mask.sum()
    chunk_size = 60

    running_sum = np.zeros(n_channels, dtype=np.float64)
    running_sq_sum = np.zeros(n_channels, dtype=np.float64)
    count = 0

    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        chunk = np.array(dynamic[t_start:t_end])
        for ch in range(n_channels):
            vals = chunk[:, ch][..., valid_mask]
            running_sum[ch] += vals.sum()
            running_sq_sum[ch] += (vals**2).sum()
        count += (t_end - t_start) * n_valid

    means = running_sum / count
    stds = np.sqrt(running_sq_sum / count - means**2)
    stds[stds < 1e-8] = 1.0

    # Static channels: normalize channels 0-12 (continuous: elev, topo_solar, lat, lon, ksat, sand, clay, soil_depth, aridity, FC, WP, SOM, windward_index)
    # Channel 13 (FVEG) gets identity (mean=0, std=1) — categorical, not z-scored
    static = np.array(store["inputs/static"])
    n_static = static.shape[0]  # 14
    static_means = np.zeros(n_static, dtype=np.float64)
    static_stds = np.ones(n_static, dtype=np.float64)
    for ch in range(min(13, n_static)):  # normalize continuous channels 0-12
        vals = static[ch][valid_mask]
        static_means[ch] = vals.mean()
        static_stds[ch] = vals.std()
        if static_stds[ch] < 1e-8:
            static_stds[ch] = 1.0
    # Channel 13 (FVEG): keep mean=0, std=1 (categorical, not z-scored)

    # Target stats
    target_names = ["pet", "pck", "aet", "cwd"]
    target_means = np.zeros(4, dtype=np.float64)
    target_stds = np.zeros(4, dtype=np.float64)

    for i, var in enumerate(target_names):
        target_data = store[f"targets/{var}"]
        r_sum, r_sq = 0.0, 0.0
        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk = np.array(target_data[t_start:t_end])
            vals = chunk[:, valid_mask]
            r_sum += vals.sum()
            r_sq += (vals**2).sum()
        cnt = T * n_valid
        target_means[i] = r_sum / cnt
        target_stds[i] = np.sqrt(r_sq / cnt - target_means[i]**2)
        if target_stds[i] < 1e-8:
            target_stds[i] = 1.0

    # Save stats
    store.create_array(name="norm/dynamic_mean", data=means.astype(np.float32), overwrite=True)
    store.create_array(name="norm/dynamic_std", data=stds.astype(np.float32), overwrite=True)
    store.create_array(name="norm/static_mean", data=static_means.astype(np.float32), overwrite=True)
    store.create_array(name="norm/static_std", data=static_stds.astype(np.float32), overwrite=True)
    store.create_array(name="norm/target_mean", data=target_means.astype(np.float32), overwrite=True)
    store.create_array(name="norm/target_std", data=target_stds.astype(np.float32), overwrite=True)
    store.create_array(name="norm/dynamic_names", data=np.array(channel_names, dtype="U20"), overwrite=True)
    store.create_array(name="norm/target_names", data=np.array(target_names, dtype="U20"), overwrite=True)

    logger.info(f"Normalization stats computed. Dynamic means: {means}")
