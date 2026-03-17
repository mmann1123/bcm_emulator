# BCM Emulator: Model Run Comparison

## Objective

This document compares all model versions (v1 through v6) with an emphasis on metrics that matter for **wildfire modeling**: accurate prediction of climatic water deficit (CWD) and actual evapotranspiration (AET) extremes. CWD is the primary driver of vegetation drought stress and fire danger in California; AET extremes reflect periods of rapid vegetation drying. Underpredicting these extremes means underestimating fire risk.

## Run Summary

| Run | Date | Description |
|-----|------|-------------|
| v1-baseline | 2026-03-15 | Baseline TCN, no FVEG embedding, MSE loss |
| v2-fveg-srad-fix | 2026-03-15 | Fixed srad bbox to 43.5N, added FVEG CWHR embedding (62 classes) |
| v3-vpd-awc | 2026-03-16 | Added VPD dynamic input + POLARIS AWC static input |
| v4-soil-props | 2026-03-16 | Replaced AWC with ksat+sand+clay from POLARIS |
| v5-awc-windward | 2026-03-16 | AWC restored, windward index added, higher AET/CWD loss weights |
| v5b-pet-reweight | 2026-03-16 | Same as v5 with PET weight 1.0 -> 1.5 to recover PET accuracy |
| v6-huber | 2026-03-17 | Huber loss (delta=1.35), uniform weights, AWC+windward features |
| v7-extreme-aware | pending | v6 + extreme-aware MSE penalty on AET (z>1.28, asym 1.5x, weight=2.0) |

## Global Performance Metrics

### NSE (Nash-Sutcliffe Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.852 | 0.862 | 0.790 | 0.872 |
| v2-fveg-srad-fix | 0.928 | 0.945 | 0.831 | 0.902 |
| v3-vpd-awc | 0.926 | 0.941 | 0.833 | 0.902 |
| v4-soil-props | 0.927 | 0.940 | 0.834 | 0.903 |
| v5-awc-windward | 0.862 | 0.944 | **0.851** | **0.915** |
| v5b-pet-reweight | 0.868 | 0.938 | 0.845 | 0.911 |
| v6-huber | **0.927** | **0.950** | 0.828 | 0.907 |

### KGE (Kling-Gupta Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.902 | 0.717 | 0.739 | 0.916 |
| v2-fveg-srad-fix | 0.947 | 0.883 | 0.752 | 0.928 |
| v3-vpd-awc | 0.946 | 0.873 | 0.755 | 0.931 |
| v4-soil-props | 0.944 | 0.863 | 0.756 | 0.932 |
| v5-awc-windward | 0.862 | 0.910 | **0.814** | 0.926 |
| v5b-pet-reweight | 0.871 | 0.811 | 0.824 | **0.937** |
| v6-huber | **0.945** | **0.886** | 0.740 | 0.929 |

### RMSE (mm/month) -- lower is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 23.2 | 19.5 | 13.8 | 20.8 |
| v2-fveg-srad-fix | 16.2 | 12.3 | 12.4 | 18.2 |
| v3-vpd-awc | 16.4 | 12.8 | 12.3 | 18.2 |
| v4-soil-props | 16.3 | 12.9 | 12.3 | 18.1 |
| v5-awc-windward | 22.4 | 12.5 | **11.6** | **17.0** |
| v5b-pet-reweight | 21.9 | 13.0 | 11.8 | 17.4 |
| v6-huber | **16.2** | **11.7** | 12.5 | 17.7 |

### Percent Bias (%) -- closer to 0 is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | -3.9 | 23.3 | 0.8 | -3.6 |
| v2-fveg-srad-fix | -0.7 | 9.4 | 6.5 | -2.4 |
| v3-vpd-awc | -1.0 | 10.3 | 5.4 | -2.2 |
| v4-soil-props | -0.8 | 11.6 | 6.1 | -2.3 |
| v5-awc-windward | -0.5 | 6.5 | 8.0 | -2.9 |
| v5b-pet-reweight | -1.1 | 15.7 | 3.6 | -1.8 |
| v6-huber | **-1.0** | **9.4** | 4.1 | **-1.7** |

## Extreme Value Performance (Wildfire-Critical)

Extreme metrics are only available for v5+ runs. These measure performance on samples above the P95 and P99 thresholds, which correspond to the hottest/driest months that drive fire danger.

### AET Extremes (P95)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | **25.7** | **-17.9** | **0.759** |
| v5b-pet-reweight | 26.9 | -18.4 | 0.754 |
| v6-huber | 31.6 | -26.6 | 0.754 |

### AET Extremes (P99)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | **31.8** | **-27.5** | **0.555** |
| v5b-pet-reweight | 31.7 | -26.9 | 0.562 |
| v6-huber | 40.5 | -37.7 | 0.561 |

### CWD Extremes (P95)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 8.9 | -2.7 | 0.797 |
| v5b-pet-reweight | **8.2** | **-1.4** | **0.798** |
| v6-huber | 8.8 | -0.8 | **0.801** |

### CWD Extremes (P99)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 5.7 | -2.6 | **0.692** |
| v5b-pet-reweight | **5.0** | -1.8 | 0.680 |
| v6-huber | 5.2 | **-0.3** | 0.685 |

## Analysis for Wildfire Modeling

### What matters for fire risk prediction

1. **CWD accuracy at extremes (P95+):** CWD is the strongest climate predictor of California wildfire. A -10mm bias at P95 translates directly to underestimated fire danger during the most critical months. CWD P95 bias ranges from -0.8mm (v6) to -2.7mm (v5), with RMSE ~8-9mm -- all reasonably good.

2. **AET accuracy at extremes:** AET underprediction means the model thinks plants are transpiring less than they actually are. Since CWD = PET - AET, AET underprediction actually *overpredicts* CWD in absolute terms (before denormalization). However, the z-score space bias matters for the loss function. AET P95 bias is a persistent problem: -17.9mm at best (v5), -26.6mm at worst (v6).

3. **Spatial fidelity in fire-prone regions:** Sierra Nevada foothills, Southern California chaparral, and North Coast ranges are where CWD extremes concentrate. Spatial NSE maps (in snapshots) should be checked for these regions specifically.

### Key trade-offs observed

**v6-huber is the best global model** with the highest PET NSE (0.927) and PCK NSE (0.950), and lowest CWD bias (-1.7%). However, it has the **worst AET extreme performance**: P95 bias of -26.6mm and P99 bias of -37.7mm. The Huber loss with delta=1.35 transitions to MAE (linear gradient) for large errors, which actually *reduces* the model's incentive to push predictions into the tails.

**v5-awc-windward is the best extreme-value model** with AET P95 bias of -17.9mm (9mm better than v6) and the highest AET NSE (0.851) and CWD NSE (0.915). The trade-off: PET NSE drops to 0.862 (vs 0.927), which cascades through the hierarchy.

**No model resolves the fundamental tension:** Huber loss stabilizes training and improves global metrics, but suppresses extreme-value gradients. Higher AET/CWD weights (v5) improve extremes but degrade PET.

### Why v7-extreme-aware should help

v7 adds a targeted MSE penalty *only* on AET samples where the target z-score exceeds 1.28 (~P90, capturing ~5-7% of samples). This restores the quadratic gradient for extreme values while keeping Huber's robustness for the bulk of the distribution. Asymmetric weighting (1.5x for underprediction) directly addresses the systematic negative bias.

**Expected outcome:** AET P95 bias should improve from -26.6mm toward the v5 level (-17.9mm) or better, while retaining v6's superior global PET/PCK accuracy. If CWD extremes don't improve enough, `extreme_vars: ["aet", "cwd"]` can be tried next.

### Remaining gaps for operational wildfire use

1. **Temporal resolution:** Monthly CWD smooths over intra-month drying events. Fire weather operates on daily-to-weekly scales. A downscaling step or daily BCM target would be needed.
2. **Fire season focus:** Current metrics average over all months. A fire-season-only evaluation (Jun-Nov) would better reflect operational accuracy.
3. **Spatial validation against fire perimeters:** Comparing CWD anomalies with MTBS/FRAP fire perimeters would validate whether the model's CWD patterns actually predict where fires occur.
4. **Forward climate scenarios:** The emulator's value proposition is running CMIP6-forced BCM thousands of times faster than the process model. Validation on out-of-sample climate extremes (2020-2025 held out) would build confidence.

## Version Progression Summary

```
v1  Baseline TCN (no FVEG) ............. AET NSE 0.790, CWD NSE 0.872
 |
v2  + FVEG embedding + srad fix ....... AET NSE 0.831, CWD NSE 0.902  (+5%)
 |
v3  + VPD + AWC ........................ AET NSE 0.833, CWD NSE 0.902  (~same)
 |
v4  ksat/sand/clay instead of AWC ...... AET NSE 0.834, CWD NSE 0.903  (~same)
 |
v5  AWC restored + windward index ...... AET NSE 0.851, CWD NSE 0.915  (+2%)
 |                                       AET P95 bias: -17.9mm (best)
v5b + PET reweight ..................... AET NSE 0.845, CWD NSE 0.911
 |
v6  Huber loss (uniform weights) ...... AET NSE 0.828, CWD NSE 0.907
 |                                       Best global PET/PCK but AET P95 bias: -26.6mm (worst)
v7  + Extreme-aware MSE penalty ....... PENDING
                                         Target: recover v5-level extremes + v6-level globals
```
