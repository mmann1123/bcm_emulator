# Fire Probability Model Pipeline

## Purpose

Baseline logistic regression predicting monthly wildfire ignition probability at 1km pixel resolution across California. Compares two feature sources to quantify whether the BCM emulator can replace BCMv8 ground truth for fire prediction:

- **Track A (BCMv8):** Uses BCMv8 target values (CWD, AET, PET) as predictors — theoretical upper bound.
- **Track B (Emulator):** Uses v19a emulator predictions as predictors — operational configuration.

The AUC delta between tracks directly measures how much emulator error degrades downstream fire skill. Delta < 0.02 = operationally viable.

## Results

**Overall test period (Oct 2019 – Sep 2024):**

| Metric | Track A (BCMv8) | Track B (Emulator) | Delta |
|--------|----------------:|-------------------:|------:|
| ROC-AUC | 0.9139 | 0.9131 | +0.0008 |
| Avg Precision | 0.2345 | 0.2292 | +0.0053 |
| Brier Score | 0.0312 | 0.0311 | +0.0001 |
| Fire Season AUC | 0.8432 | 0.8454 | -0.0022 |

**Conclusion: Delta AUC = +0.0008 — the emulator is operationally viable for fire prediction.**

## Run Sequence

```bash
# Step 0: Export emulator predictions for test period (~5 min GPU)
conda run -n deep_field python scripts/fire_model/00_export_predictions.py

# Step 1: Build panel — rasterize FRAP, compute TSF, extract features (~5 min)
conda run -n deep_field python scripts/fire_model/01_build_panel.py

# Step 2: Train logistic regression for both tracks (~30 sec)
conda run -n deep_field python scripts/fire_model/02_train_model.py

# Step 3: Evaluate, compare, generate spatial maps (~5 sec)
conda run -n deep_field python scripts/fire_model/03_evaluate.py
```

Use `--force` to overwrite existing outputs.

## Data Sources

| Source | Path | Description |
|--------|------|-------------|
| BCM zarr | `data/bcm_dataset.zarr` | 537 months (1980-2024), 15 dynamic + 14 static channels, 4 targets |
| FRAP | `fire24_1.gdb` / `firep24_1` | 22,810 fire perimeters, filtered to 3,004 CA wildfires ≥300 acres |
| TSF raster | `TimeSinceFire_Raster/timeSinceFire_1983.tif` | Initial time-since-fire state (years, BCM grid) |
| Ecoregions | `ca_eco_l3.tif` | EPA Level 3 ecoregions for stratified evaluation |
| FVEG | `fveg_class_map.json` + `fveg_vat.csv` | Vegetation class → broad category mapping |

## Features

| Feature | Source | Description |
|---------|--------|-------------|
| cwd_anom | targets or predictions | CWD anomaly vs 1984-2016 pixel-month climatology |
| aet_anom | targets or predictions | AET anomaly |
| pet_anom | targets or predictions | PET anomaly |
| cwd_cum3_anom | derived | 3-month trailing CWD anomaly sum |
| cwd_cum6_anom | derived | 6-month trailing CWD anomaly sum |
| sws | dynamic ch11 | Soil water storage (mm) |
| kbdi | dynamic ch10 | Keetch-Byram Drought Index |
| vpd_roll6_std | dynamic ch12 | 6-month rolling VPD std |
| ppt, tmin, tmax, vpd, srad | dynamic | Climate inputs |
| month_sin, month_cos | derived | Seasonal cycle encoding |
| fire_season | derived | 1 if Jun-Nov |
| elev, aridity_index, windward_index | static | Terrain features |
| fveg_forest, fveg_shrub, fveg_herb | static | Vegetation one-hot dummies |
| tsf_years | TSF computation | Years since last fire (capped at 50) |
| tsf_log | TSF computation | log(1 + tsf_years) |

Track A and Track B share all features except the hydrology anomalies (cwd/aet/pet), which come from BCMv8 targets (Track A) or emulator predictions (Track B) during the test period. Training features use BCMv8 targets for both tracks.

## Panel Construction

- **Positives:** All pixel-months where a filtered FRAP fire perimeter touches the pixel (147,353 samples)
- **Negatives:** Spatially thinned grid (every 5th pixel, ~19K points), all non-fire months (9.4M samples)
- **Class imbalance:** Handled by `class_weight="balanced"` in logistic regression
- **Split:** Train (1984-2016), Calibration (2017-2019), Test (2019-2024)

## Time Since Last Fire

TSF is computed by forward-iterating through the monthly fire raster starting from a 1983 initial state. TSF at time t is recorded **before** processing that month's fires (causal ordering). Two features are derived:
- `tsf_years`: raw years since last fire (capped at 50)
- `tsf_log`: log(1 + years) — captures diminishing fuel accumulation

Terminal TSF state is saved for use in forward forecasting.

## Forward Forecasting

`src/fire_model/forecast.py` provides `FireProbabilityForecaster` for operational use:

```python
from src.fire_model.forecast import FireProbabilityForecaster

forecaster = FireProbabilityForecaster(
    model_path="outputs/fire_model/model/trackB/lr_calibrated.pkl",
    tsf_init_path="outputs/fire_model/tsf_state_2024-09.npy",
    climatology_path="outputs/fire_model/climatology_1984_2016.npz",
    zarr_path="data/bcm_dataset.zarr",
)

# Each month, pass BCM emulator outputs:
prob_map = forecaster.step(
    bcm_outputs={"cwd": cwd_arr, "aet": aet_arr, "pet": pet_arr,
                 "ppt": ppt_arr, "tmin": tmin_arr, "tmax": tmax_arr,
                 "vpd": vpd_arr, "srad": srad_arr, "kbdi": kbdi_arr,
                 "sws": sws_arr, "vpd_roll6_std": vpd_std_arr},
    month=10, year=2024,
)
```

In forecast mode, TSF accumulates without fire resets — this is a probability forecast, not a fire simulation.

## Output Structure

```
outputs/fire_model/
├── manifest.json                          — Run metadata, AUC scores, sample counts
├── predictions/{pet,pck,aet,cwd}.npy      — Emulator predictions (test period)
├── fire_raster.npy                        — Monthly fire binary raster (537 months)
├── tsf_state_*.npy                        — TSF state for forecasting
├── climatology_1984_2016.npz              — Monthly pixel climatologies
├── panel/fire_panel.parquet               — Full panel dataset (9.6M rows)
├── model/track{A,B}/lr_calibrated.pkl     — Trained models
├── model/track{A,B}/coefficients.csv      — Feature coefficients and odds ratios
├── evaluation/comparison_summary.csv      — Head-to-head comparison table
├── evaluation/track{A,B}/metrics_*.csv    — Metrics by month/quarter/WY/ecoregion
├── evaluation/track{A,B}/calibration_curve.png
├── evaluation/roc_comparison.png          — Overlaid ROC curves
├── predictions/test_predictions_*.parquet — Pixel-level test predictions
└── spatial_maps/track{A,B}/fire_prob_WY*_fire_season.tif
```

## Known Limitations

- **Monthly temporal resolution:** Fires are assigned to alarm month only. Sub-monthly timing is lost.
- **1km spatial resolution:** Fires < 300 acres (~1.2 km²) are excluded as sub-pixel.
- **Logistic regression:** Linear decision boundary. A random forest or gradient boosting model would likely improve AUC but the goal here is a clean baseline comparison, not maximum skill.
- **No ignition modeling:** The model predicts burn probability (whether a pixel burns), not ignition probability. Spread dynamics are implicit in the perimeter-based labels.
- **TSF initialization:** Pre-1984 fire history from `timeSinceFire_1983.tif` may have gaps in early records. Most pixels show ~50 years TSF at initialization, reflecting incomplete historical coverage rather than true fire absence.
