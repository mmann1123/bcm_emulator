"""Orchestrate data preparation: download from ScienceBase/PRISM/TerraClimate, compute topo solar, build zarr.

Data sources:
    - ScienceBase: ppt, tmin, tmax (monthly, 270m -> resampled to BCM 1km grid, WY1896-2020)
    - ScienceBase PCK gap-fill: pck (2017-01 to 2020-09, same format)
    - PRISM daily: ppt (for wet day / ppt intensity derivation)
    - TerraClimate: srad (monthly, ~4.7km, reprojected to BCM 1km grid)
    - SRTM: elevation (local), topo solar (derived)
    - BCM local: aet, cwd, pck (targets); PET derived as AET + CWD
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare BCM emulator data")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "sciencebase", "pck_gap", "prism_daily", "prism_daily_tmax", "srad", "daymet", "topo_solar", "fveg", "soil", "fire_features", "zarr"],
        help="Which steps to run",
    )
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config)
    steps = args.steps
    run_all = "all" in steps

    from src.utils.io_helpers import get_bcm_reference_profile
    bcm_profile = get_bcm_reference_profile(cfg.paths.bcm_dir)

    # Step 1: Download BCMv8 climate inputs from ScienceBase (ppt, tmin, tmax)
    if run_all or "sciencebase" in steps:
        logger.info("=== Downloading BCMv8 climate inputs from ScienceBase ===")
        from src.data.download_sciencebase import download_all_from_sciencebase

        results = download_all_from_sciencebase(
            out_dir=cfg.paths.pet_sciencebase_dir,
            start_ym=cfg.temporal.train_start,
            end_ym=cfg.temporal.test_end,
            bcm_profile=bcm_profile,
        )
        for var, files in results.items():
            logger.info(f"  {var}: {len(files)} files downloaded")

    # Step 2: Download PCK gap-fill from ScienceBase (2017-01 to 2020-09)
    if run_all or "pck_gap" in steps:
        logger.info("=== Downloading PCK gap-fill from ScienceBase ===")
        from src.data.download_sciencebase import download_pck_gap

        pck_files = download_pck_gap(
            out_dir=cfg.paths.pet_sciencebase_dir,
            start_ym="2017-01",
            end_ym=cfg.temporal.test_end,
            bcm_profile=bcm_profile,
        )
        logger.info(f"  PCK gap-fill: {len(pck_files)} files downloaded")

    # Step 3: Download PRISM daily ppt (for wet days + ppt intensity)
    if run_all or "prism_daily" in steps:
        logger.info("=== Downloading PRISM daily ppt ===")
        from src.data.download_prism import download_prism_daily_ppt, compute_wet_days_and_intensity

        download_prism_daily_ppt(
            year_start=int(cfg.temporal.train_start[:4]),
            year_end=int(cfg.temporal.test_end[:4]),
            out_dir=cfg.paths.prism_daily_dir,
        )

        logger.info("=== Computing wet days and ppt intensity ===")
        compute_wet_days_and_intensity(
            daily_ppt_dir=str(cfg.paths.prism_daily_dir) + "/ppt_daily",
            out_dir=cfg.paths.prism_monthly_dir,
            bcm_profile=bcm_profile,
        )

    # Step 3b: Download PRISM daily tmax (for fire features)
    if run_all or "prism_daily_tmax" in steps:
        logger.info("=== Downloading PRISM daily tmax ===")
        from src.data.download_prism import download_prism_daily_tmax

        download_prism_daily_tmax(
            year_start=int(cfg.temporal.train_start[:4]),
            year_end=int(cfg.temporal.test_end[:4]),
            out_dir=cfg.paths.prism_daily_dir,
        )

    # Step 4: Download srad (TerraClimate replaces DAYMET)
    if run_all or "srad" in steps or "daymet" in steps:
        logger.info("=== Downloading TerraClimate srad ===")
        from src.data.download_srad import download_srad

        srad_files = download_srad(
            year_start=int(cfg.temporal.train_start[:4]),
            year_end=int(cfg.temporal.test_end[:4]),
            bbox=cfg.download.ca_bbox,
            out_dir=cfg.paths.daymet_dir,
            bcm_profile=bcm_profile,
        )
        logger.info(f"  srad: {len(srad_files)} new files")

    # Step 5: Compute topographic solar radiation
    if run_all or "topo_solar" in steps:
        logger.info("=== Computing topographic solar radiation ===")
        from src.utils.topo_solar import compute_topo_solar

        compute_topo_solar(
            dem_path=cfg.paths.elevation_path,
            out_path=cfg.paths.topo_solar_path,
            bcm_profile=bcm_profile,
        )

    # Step 6: Download and rasterize FVEG (FRAP CWHR vegetation)
    if run_all or "fveg" in steps:
        logger.info("=== Downloading and rasterizing FVEG ===")
        from src.data.download_fveg import download_fveg

        fveg_results = download_fveg(
            out_dir=cfg.paths.fveg_dir,
            bcm_profile=bcm_profile,
            vat_csv_path=cfg.paths.fveg_vat_csv,
        )
        for key, path in fveg_results.items():
            logger.info(f"  {key}: {path}")

    # Step 7: Download POLARIS soil properties (ksat, sand, clay)
    if run_all or "soil" in steps:
        logger.info("=== Downloading POLARIS soil properties (ksat, sand, clay, om) ===")
        from src.data.download_soil import download_soil_properties

        soil_results = download_soil_properties(
            out_dir=cfg.paths.soil_dir,
            bcm_profile=bcm_profile,
            bbox=cfg.download.ca_bbox,
        )
        for prop, path in soil_results.items():
            logger.info(f"  {prop}: {path}")

    # Step 7b: Compute fire-relevant features from daily PRISM tmax + ppt
    if run_all or "fire_features" in steps:
        logger.info("=== Computing fire-relevant features ===")
        from src.data.compute_fire_features import compute_all_fire_features

        compute_all_fire_features(
            daily_tmax_dir=str(cfg.paths.prism_daily_dir) + "/tmax_daily",
            daily_ppt_dir=str(cfg.paths.prism_daily_dir) + "/ppt_daily",
            out_dir=cfg.paths.prism_monthly_dir,
            bcm_profile=bcm_profile,
            mean_annual_ppt_path=getattr(cfg.paths, "mean_annual_ppt_path", None),
        )

    # Step 8: Build zarr store
    if run_all or "zarr" in steps:
        zarr_path = cfg.paths.zarr_store
        # Auto-backup existing zarr before rebuild
        if Path(zarr_path).exists():
            from src.utils.snapshot import _zarr_fingerprint

            existing_fp = _zarr_fingerprint(zarr_path)[:16]
            backup_path = str(zarr_path).replace(".zarr", f"_{existing_fp}.zarr")
            if not Path(backup_path).exists():
                logger.info(f"Backing up existing zarr to {backup_path}")
                Path(zarr_path).rename(backup_path)
            else:
                logger.info(f"Backup already exists at {backup_path}, overwriting current zarr")

        logger.info("=== Building zarr store ===")
        from src.data.preprocessing import build_zarr_store

        build_zarr_store(
            zarr_path=cfg.paths.zarr_store,
            bcm_dir=cfg.paths.bcm_dir,
            sciencebase_dir=cfg.paths.pet_sciencebase_dir,
            daymet_dir=cfg.paths.daymet_dir,
            prism_dir=cfg.paths.prism_monthly_dir,
            topo_solar_path=cfg.paths.topo_solar_path,
            elevation_path=cfg.paths.elevation_path,
            fveg_dir=cfg.paths.fveg_dir,
            soil_dir=cfg.paths.soil_dir,
            soil_depth_path=getattr(cfg.paths, "soil_depth_path", ""),
            aridity_path=getattr(cfg.paths, "aridity_path", ""),
            field_capacity_path=getattr(cfg.paths, "field_capacity_path", ""),
            wilting_point_path=getattr(cfg.paths, "wilting_point_path", ""),
            awc_path=getattr(cfg.paths, "awc_path", ""),
            bcm_profile=bcm_profile,
            time_range=(cfg.temporal.train_start, cfg.temporal.test_end),
            snow_threshold=cfg.data.snow_threshold_celsius,
            snow_transition=cfg.data.snow_transition_width,
        )

    logger.info("=== Data preparation complete ===")


if __name__ == "__main__":
    main()
