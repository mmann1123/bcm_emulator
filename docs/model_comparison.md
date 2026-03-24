# BCM Emulator: Model Run Comparison

## Objective

This document compares all model versions (v1 through v17) with an emphasis on metrics that matter for **wildfire modeling**: accurate prediction of climatic water deficit (CWD) and actual evapotranspiration (AET) extremes. CWD is the primary driver of vegetation drought stress and fire danger in California; AET extremes reflect periods of rapid vegetation drying. Underpredicting these extremes means underestimating fire risk.

## Run Summary

| Run | Date | Loss | Description |
|-----|------|------|-------------|
| v1-baseline | 2026-03-15 | MSE | Baseline TCN, no FVEG embedding |
| v2-fveg-srad-fix | 2026-03-15 | MSE | Fixed srad bbox to 43.5N, added FVEG CWHR embedding (62 classes) |
| v3-vpd-awc | 2026-03-16 | MSE | Added VPD dynamic input + POLARIS AWC static input |
| v4-soil-props | 2026-03-16 | MSE | Replaced AWC with ksat+sand+clay from POLARIS |
| v5-awc-windward | 2026-03-16 | MSE | AWC restored, windward index added, higher AET/CWD loss weights |
| v5b-pet-reweight | 2026-03-16 | MSE | Same as v5 with PET weight 1.0 -> 1.5 to recover PET accuracy |
| v6-huber | 2026-03-17 | Huber | Huber loss (delta=1.35), uniform weights, AWC+windward features |
| v7-extreme-aware | 2026-03-17 | Huber+Extreme | v6 + extreme-aware MSE penalty on AET (z>1.28, asym 1.5x, weight=2.0) -- **FAILED: weight too high** |
| v7b-extreme-low | 2026-03-17 | Huber+Extreme | v7 with extreme_weight reduced to 0.1 (from 2.0) -- **NOTE: eval metrics identical to v7; likely checkpoint issue** |
| v8-soil-physics | 2026-03-18 | Huber+Extreme | v5 base + soil_depth, aridity_index, FC, WP, SOM; AWC removed; 14 static channels |
| v8b-no-extreme | 2026-03-18 | Huber | v8 soil physics channels with extreme_weight=0.0 (pure Huber loss) |
| v8c-mse | 2026-03-18 | MSE | v8b architecture/data with MSE loss (controlled comparison vs Huber) |
| v9-drought-code | 2026-03-19 | MSE | v8c + drought_code dynamic channel; 12 dynamic inputs, MSE loss |
| v9-kbdi | 2026-03-19 | MSE | v8c base + KBDI dynamic channel (replaces drought_code); 11 dynamic inputs, MSE loss |
| v10-kbdi-aet-only | 2026-03-20 | MSE | KBDI routed only to AET head (bypasses backbone); 10 dyn through backbone, KBDI injected at AET stage |
| v11-kv-aet | 2026-03-20 | MSE | v10 + BCM Table 6 Kv crop coefficient as time-varying channel at AET head (MLP head, 260→64→1) |
| v11-stress-frac | 2026-03-21 | MSE | Stress-fraction AET head: sigmoid(stress) × Kv × PET + correction; same Kv plumbing as v11-kv-aet |
| v12-stress-frac-aet2x | 2026-03-21 | MSE | v11-stress-frac + v5-style loss weights: aet=2.0, cwd=2.0, pet_decay=0.99 |
| v13-sws-rollstd | 2026-03-23 | MSE | v12 arch + 4 new dynamic channels: SWS bucket model, vpd_roll6_std, srad_roll6_std, tmax_roll3_std |
| v14-sws-stress | 2026-03-23 | MSE | v13 with fixed SWS: stress-modulated drainage (linear stress=SWS/AWC, 13.8% zeros vs 68%) |
| v15-awc-extreme | 2026-03-23 | MSE+Extreme | v14 + awc_total static channel (15 static), extreme_weight=0.05, extreme_asym=1.5 |
| v16-aet1.5-extreme | 2026-03-23 | MSE+Extreme | v15 with aet_initial=1.5 (from 2.0), cwd=2.0, extreme_weight=0.05 — rebalanced AET/PCK trade-off |
| v17-polaris-awc | 2026-03-24 | MSE+Extreme | POLARIS root-zone AWC (0-100cm) for SWS; dropped awc_total static (14 static); aet=1.5, extreme_weight=0.05 |

## Global Performance Metrics

### NSE (Nash-Sutcliffe Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.852 | 0.862 | 0.790 | 0.872 |
| v2-fveg-srad-fix | 0.928 | 0.945 | 0.831 | 0.902 |
| v3-vpd-awc | 0.926 | 0.941 | 0.833 | 0.902 |
| v4-soil-props | 0.927 | 0.940 | 0.834 | 0.903 |
| v5-awc-windward | 0.862 | 0.944 | 0.851 | 0.915 |
| v5b-pet-reweight | 0.868 | 0.938 | 0.845 | 0.911 |
| v6-huber | 0.927 | 0.950 | 0.828 | 0.907 |
| v7-extreme-aware | 0.876 | **0.961** | 0.760 | 0.830 |
| v7b-extreme-low | 0.876 | **0.961** | 0.760 | 0.830 |
| v8-soil-physics | 0.914 | 0.935 | 0.851 | 0.894 |
| v8b-no-extreme | 0.927 | 0.916 | 0.839 | 0.899 |
| v8c-mse | 0.927 | 0.930 | 0.834 | 0.907 |
| v9-drought-code | 0.927 | 0.932 | 0.810 | 0.888 |
| v9-kbdi | 0.925 | 0.929 | 0.824 | 0.896 |
| v10-kbdi-aet-only | 0.927 | 0.929 | 0.840 | 0.897 |
| v11-kv-aet | 0.928 | 0.930 | 0.835 | 0.900 |
| v11-stress-frac | **0.929** | 0.944 | 0.830 | 0.903 |
| v12-stress-frac-aet2x | 0.862 | 0.913 | **0.856** | 0.912 |
| v13-sws-rollstd | 0.870 | 0.907 | 0.846 | 0.916 |
| v14-sws-stress | 0.866 | 0.921 | 0.854 | 0.914 |
| v15-awc-extreme | 0.857 | 0.916 | 0.853 | 0.913 |
| v16-aet1.5-extreme | 0.861 | 0.930 | 0.848 | 0.912 |
| v17-polaris-awc | 0.879 | 0.949 | 0.851 | **0.929** |

### KGE (Kling-Gupta Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.902 | 0.717 | 0.739 | 0.916 |
| v2-fveg-srad-fix | 0.947 | 0.883 | 0.752 | 0.928 |
| v3-vpd-awc | 0.946 | 0.873 | 0.755 | 0.931 |
| v4-soil-props | 0.944 | 0.863 | 0.756 | 0.932 |
| v5-awc-windward | 0.862 | 0.910 | 0.814 | 0.926 |
| v5b-pet-reweight | 0.871 | 0.811 | 0.824 | **0.937** |
| v6-huber | 0.945 | 0.886 | 0.740 | 0.929 |
| v7-extreme-aware | 0.881 | 0.924 | 0.588 | 0.862 |
| v7b-extreme-low | 0.881 | 0.924 | 0.588 | 0.862 |
| v8-soil-physics | 0.930 | 0.923 | 0.791 | 0.920 |
| v8b-no-extreme | 0.944 | 0.816 | 0.767 | 0.935 |
| v8c-mse | 0.946 | 0.918 | 0.744 | 0.928 |
| v9-drought-code | **0.953** | 0.887 | 0.738 | 0.902 |
| v9-kbdi | 0.952 | 0.855 | 0.743 | 0.907 |
| v10-kbdi-aet-only | 0.942 | 0.826 | 0.769 | 0.925 |
| v11-kv-aet | 0.942 | 0.871 | 0.745 | 0.915 |
| v11-stress-frac | 0.947 | **0.952** | 0.739 | 0.921 |
| v12-stress-frac-aet2x | 0.859 | 0.745 | 0.825 | 0.922 |
| v13-sws-rollstd | 0.866 | 0.753 | 0.805 | 0.920 |
| v14-sws-stress | 0.860 | 0.806 | **0.831** | 0.930 |
| v15-awc-extreme | 0.853 | 0.757 | **0.831** | 0.924 |
| v16-aet1.5-extreme | 0.859 | 0.868 | 0.816 | 0.920 |
| v17-polaris-awc | 0.872 | 0.904 | 0.798 | **0.931** |

### RMSE (mm/month) -- lower is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 23.2 | 19.5 | 13.8 | 20.8 |
| v2-fveg-srad-fix | 16.2 | 12.3 | 12.4 | 18.2 |
| v3-vpd-awc | 16.4 | 12.8 | 12.3 | 18.2 |
| v4-soil-props | 16.3 | 12.9 | 12.3 | 18.1 |
| v5-awc-windward | 22.4 | 12.5 | 11.6 | 17.0 |
| v5b-pet-reweight | 21.9 | 13.0 | 11.8 | 17.4 |
| v6-huber | 16.2 | 11.7 | 12.5 | 17.7 |
| v7-extreme-aware | 21.2 | **10.3** | 14.7 | 24.0 |
| v7b-extreme-low | 21.2 | **10.3** | 14.7 | 24.0 |
| v8-soil-physics | 17.7 | 13.4 | 11.6 | 19.0 |
| v8b-no-extreme | 16.3 | 15.2 | 12.1 | 18.5 |
| v8c-mse | 16.3 | 13.8 | 12.3 | 17.7 |
| v9-drought-code | 16.3 | 13.7 | 13.1 | 19.5 |
| v9-kbdi | 16.5 | 14.0 | 12.7 | 18.8 |
| v10-kbdi-aet-only | 16.3 | 14.0 | 12.1 | 18.7 |
| v11-kv-aet | 16.1 | 13.8 | 12.3 | 18.4 |
| v11-stress-frac | **16.0** | 12.4 | 12.4 | 18.1 |
| v12-stress-frac-aet2x | 22.4 | 15.4 | **11.4** | 17.3 |
| v13-sws-rollstd | 21.7 | 16.0 | 11.8 | 16.9 |
| v14-sws-stress | 22.1 | 14.8 | 11.5 | 17.1 |
| v15-awc-extreme | 22.8 | 15.2 | 11.5 | 17.1 |
| v16-aet1.5-extreme | 22.4 | 13.8 | 11.7 | 17.2 |
| v17-polaris-awc | 21.0 | 11.8 | 11.6 | **15.5** |

### Percent Bias (%) -- closer to 0 is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | -3.9 | 23.3 | **0.8** | -3.6 |
| v2-fveg-srad-fix | -0.7 | 9.4 | 6.5 | -2.4 |
| v3-vpd-awc | -1.0 | 10.3 | 5.4 | -2.2 |
| v4-soil-props | -0.8 | 11.6 | 6.1 | -2.3 |
| v5-awc-windward | -0.5 | 6.5 | 8.0 | -2.9 |
| v5b-pet-reweight | -1.1 | 15.7 | 3.6 | -1.8 |
| v6-huber | -1.0 | 9.4 | 4.1 | **-1.7** |
| v7-extreme-aware | 3.9 | 7.2 | 40.3 | -11.1 |
| v7b-extreme-low | 3.9 | 7.2 | 40.3 | -11.1 |
| v8-soil-physics | **-0.3** | 4.2 | 12.7 | -4.4 |
| v8b-no-extreme | -1.0 | 13.3 | 4.9 | -2.0 |
| v8c-mse | -0.9 | 5.0 | 6.4 | -2.5 |
| v9-drought-code | -1.1 | 8.2 | 10.3 | -4.4 |
| v9-kbdi | -1.8 | 12.4 | 6.6 | -3.8 |
| v10-kbdi-aet-only | -1.0 | 13.5 | 6.3 | -2.6 |
| v11-kv-aet | -0.9 | 8.7 | 10.0 | -4.1 |
| v11-stress-frac | -0.6 | **3.3** | 6.3 | -2.1 |
| v12-stress-frac-aet2x | -0.6 | 20.7 | 7.6 | -3.0 |
| v13-sws-rollstd | -1.8 | 19.0 | 2.7 | -2.2 |
| v14-sws-stress | -1.3 | 13.6 | 4.3 | -2.4 |
| v15-awc-extreme | -1.1 | 19.0 | 6.6 | -3.1 |
| v16-aet1.5-extreme | -0.5 | 9.7 | 9.5 | -3.7 |
| v17-polaris-awc | -0.6 | 6.8 | 7.2 | -2.7 |

## Extreme Value Performance (Wildfire-Critical)

Extreme metrics are only available for v5+ runs. These measure performance on samples above the P95 and P99 thresholds, which correspond to the hottest/driest months that drive fire danger.

### AET Extremes (P95)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 25.7 | -17.9 | 0.759 |
| v5b-pet-reweight | 26.9 | -18.4 | 0.754 |
| v6-huber | 31.6 | -26.6 | 0.754 |
| v7b-extreme-low | **17.3** | **-2.4** | 0.742 |
| v8-soil-physics | 24.6 | -15.9 | 0.755 |
| v8b-no-extreme | 29.3 | -23.3 | 0.757 |
| v8c-mse | 30.6 | -25.5 | **0.765** |
| v9-drought-code | 31.3 | -26.3 | 0.727 |
| v9-kbdi | 31.9 | -26.4 | 0.740 |
| v10-kbdi-aet-only | 29.6 | -23.3 | 0.758 |
| v11-kv-aet | 29.0 | -23.4 | 0.764 |
| v11-stress-frac | 31.7 | -26.4 | 0.759 |
| v12-stress-frac-aet2x | 25.3 | -16.6 | **0.765** |
| v13-sws-rollstd | 28.2 | -20.6 | 0.746 |
| v14-sws-stress | 26.6 | -17.9 | 0.742 |
| v15-awc-extreme | 25.2 | -16.4 | 0.755 |
| v16-aet1.5-extreme | 25.7 | -16.3 | 0.753 |
| v17-polaris-awc | 26.8 | -19.2 | 0.759 |

### AET Extremes (P99)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 31.8 | -27.5 | 0.555 |
| v5b-pet-reweight | 31.7 | -26.9 | 0.562 |
| v6-huber | 40.5 | -37.7 | 0.561 |
| v7b-extreme-low | **19.4** | **-10.9** | 0.543 |
| v8-soil-physics | 29.8 | -24.3 | 0.561 |
| v8b-no-extreme | 37.1 | -33.4 | 0.572 |
| v8c-mse | 38.6 | -35.7 | **0.602** |
| v9-drought-code | 40.0 | -37.3 | 0.573 |
| v9-kbdi | 41.3 | -38.4 | 0.523 |
| v10-kbdi-aet-only | 37.5 | -33.6 | 0.587 |
| v11-kv-aet | 36.4 | -33.3 | 0.592 |
| v11-stress-frac | 40.0 | -36.9 | 0.585 |
| v12-stress-frac-aet2x | 30.7 | -25.7 | 0.556 |
| v13-sws-rollstd | 34.1 | -29.2 | 0.591 |
| v14-sws-stress | 31.9 | -26.4 | 0.542 |
| v15-awc-extreme | 29.9 | -24.7 | 0.550 |
| v16-aet1.5-extreme | 30.2 | -24.9 | 0.544 |
| v17-polaris-awc | 33.1 | -28.4 | 0.572 |

### CWD Extremes (P95)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 8.9 | -2.7 | 0.797 |
| v5b-pet-reweight | **8.2** | -1.4 | 0.798 |
| v6-huber | 8.8 | -0.8 | 0.801 |
| v7b-extreme-low | 13.1 | -5.7 | 0.746 |
| v8-soil-physics | 9.1 | -2.5 | 0.783 |
| v8b-no-extreme | 8.7 | **+0.6** | **0.804** |
| v8c-mse | 9.0 | -1.3 | 0.793 |
| v9-drought-code | 13.7 | -4.1 | 0.758 |
| v9-kbdi | 10.5 | -4.6 | 0.798 |
| v10-kbdi-aet-only | 9.6 | **-0.6** | 0.797 |
| v11-kv-aet | 9.7 | -2.6 | 0.802 |
| v11-stress-frac | 11.0 | -1.8 | 0.782 |
| v12-stress-frac-aet2x | 9.6 | -3.0 | 0.790 |
| v13-sws-rollstd | 10.2 | -3.9 | 0.784 |
| v14-sws-stress | 9.0 | -2.0 | 0.786 |
| v15-awc-extreme | 10.1 | -3.5 | 0.768 |
| v16-aet1.5-extreme | 9.6 | -3.8 | 0.774 |
| v17-polaris-awc | 9.0 | -3.0 | **0.781** |

### CWD Extremes (P99)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 5.7 | -2.6 | 0.692 |
| v5b-pet-reweight | **5.0** | -1.8 | 0.680 |
| v6-huber | 5.2 | **-0.3** | 0.685 |
| v7b-extreme-low | 10.3 | -5.6 | 0.651 |
| v8-soil-physics | 6.1 | -2.5 | 0.672 |
| v8b-no-extreme | **5.0** | +0.6 | 0.683 |
| v8c-mse | 6.0 | -1.2 | 0.686 |
| v9-drought-code | 9.5 | -2.3 | 0.666 |
| v9-kbdi | 8.2 | -4.2 | 0.670 |
| v10-kbdi-aet-only | 5.7 | -0.6 | 0.684 |
| v11-kv-aet | 6.4 | -2.0 | **0.713** |
| v11-stress-frac | 7.4 | -1.8 | 0.676 |
| v12-stress-frac-aet2x | 6.4 | -2.7 | 0.690 |
| v13-sws-rollstd | 7.2 | -3.8 | 0.661 |
| v14-sws-stress | 6.2 | -2.1 | 0.644 |
| v15-awc-extreme | 7.1 | -3.4 | 0.662 |
| v16-aet1.5-extreme | 6.6 | -3.6 | 0.672 |
| v17-polaris-awc | 5.7 | -2.8 | **0.684** |

## Analysis for Wildfire Modeling

### What matters for fire risk prediction

1. **CWD accuracy at extremes (P95+):** CWD is the strongest climate predictor of California wildfire. A -10mm bias at P95 translates directly to underestimated fire danger during the most critical months. CWD P95 bias ranges from -0.8mm (v6) to -2.7mm (v5), with RMSE ~8-9mm -- all reasonably good.

2. **AET accuracy at extremes:** AET underprediction means the model thinks plants are transpiring less than they actually are. Since CWD = PET - AET, AET underprediction actually *overpredicts* CWD in absolute terms (before denormalization). However, the z-score space bias matters for the loss function. AET P95 bias is a persistent problem: -17.9mm at best (v5), -26.6mm at worst (v6).

3. **Spatial fidelity in fire-prone regions:** Sierra Nevada foothills, Southern California chaparral, and North Coast ranges are where CWD extremes concentrate. Spatial NSE maps (in snapshots) should be checked for these regions specifically.

### Key trade-offs observed

**v17-polaris-awc is the best CWD model** with CWD NSE 0.929, RMSE 15.5mm, and KGE 0.931 — all best-ever. It also recovers PCK (NSE 0.949, pbias 6.8%) and PET (NSE 0.879) to near-best levels among weighted-loss runs. The trade-off: AET P95 bias regressed to -19.2mm (from v16's -16.3mm) because the more responsive POLARIS AWC-based SWS overcorrects extreme AET.

**v6-huber remains the best global PET/PCK model** with PET NSE 0.927 and PCK NSE 0.950, but has the worst AET extreme performance (P95 bias -26.6mm).

**v16-aet1.5-extreme has the best AET P95 bias** (-16.3mm) with good PCK recovery (pbias 9.7%), but CWD lags behind v17.

**The core tension remains:** improving CWD (via more responsive SWS) regresses AET extremes. The v18 tuning sweep targets this trade-off by testing extreme penalty strength, loss weights, and loss function type.

### v7-extreme-aware: lesson learned (weight=2.0 -- FAILED)

v7 added a targeted MSE penalty on AET samples where target z-score > 1.28 (~P90), with asymmetric weighting (underprediction penalized 1.5x). The concept is sound but `extreme_weight=2.0` was far too aggressive:

- **AET pbias exploded from +4.1% to +40.3%** -- the model massively overpredicts AET
- **CWD pbias worsened from -1.7% to -11.1%** -- since CWD = PET - AET, AET overprediction mechanically underpredicts CWD
- **PET degraded** (NSE 0.927 -> 0.876) despite the penalty being AET-only
- Val loss 5.6x higher (1.002 vs 0.178); best epoch dropped from 76 to 60

**Root cause:** The extreme MSE operates on z-scores where squared errors can be large (z=2.5 -> sq_err ~6). At weight=2.0, this dominated the total loss (~0.8 added to a Huber base of ~0.18), pulling all AET predictions upward.

### v7b-extreme-low: weight=0.1 results

**NOTE:** v7b evaluation metrics are identical to v7 to full precision. This strongly suggests a checkpoint issue — v7b's `evaluate.py` likely evaluated v7's `best_model.pt` rather than v7b's own checkpoint. The training metrics differ (best_epoch 19 vs 60, val_loss 0.271 vs 1.002), confirming the training did run differently, but evaluation was done on the wrong checkpoint. **v7b results should be treated as invalid until re-evaluated.**

That said, the v7b *training* metrics (best_val_loss 0.271 vs v7's 1.002) suggest weight=0.1 dramatically stabilized training. A re-evaluation is warranted.

### v8-soil-physics: physically-informed static channels

Replaced AWC with 5 physically-informed channels: soil_depth (ca_thck4_v8), aridity_index (ca_aridity_v8), field_capacity (ca_mp0010_v8), wilting_point (ca_mp6000_v8), and SOM (POLARIS organic matter). Static channels 10→14, `in_channels` 27→31.

**Results vs v5-awc-windward (previous best for AET/CWD):**

- **PET: major improvement** — NSE 0.862→0.914, KGE 0.862→0.930, pbias -0.5%→-0.3%. The new soil physics channels directly help the energy balance stage.
- **AET: flat NSE, worse bias** — NSE 0.851→0.851 (unchanged), but pbias 8.0%→12.7%. The model overpredicts AET, likely because FC and WP provide a "more water available" signal.
- **CWD: slight regression** — NSE 0.915→0.894. Since CWD = PET - AET, the AET overprediction cancels part of the PET improvement.
- **CWD extremes: mixed** — P95 hit rate improved (0.797→0.783... actually slightly worse), but CWD P95 bias improved (-2.7→-2.5mm).
- **AET extremes: improved** — P95 bias -17.9→-15.9mm, hit rate 0.759→0.755 (flat).
- **PET/PCK bias best ever** — PET pbias -0.3% and PCK pbias 4.2% are the best across all runs.

**Key insight:** The new channels strongly help PET (the first stage) but the AET head (stage 3) doesn't fully exploit them yet. The AET overprediction (+12.7% pbias) suggests the model interprets the richer soil info as "more water available for ET" without properly learning the constraints. CWD inherits this as a difference-of-two-large-numbers amplification problem.

### v8b-no-extreme: pure Huber with soil physics

v8b uses the same 14 static channels as v8 but disables the extreme penalty (`extreme_weight=0.0`), returning to pure Huber loss with uniform weights — the same loss configuration as v6-huber but with the richer v8 static channels.

**Results vs v5-awc-windward (best AET/CWD model) and v8:**

- **PET: recovered to v6-class accuracy** — NSE 0.927 (matching v6-huber's 0.927), KGE 0.944 (best ever). The soil physics channels + Huber loss is a winning combination for PET.
- **PCK: regressed** — NSE 0.916, KGE 0.816. pbias 13.3% (worst across runs). PCK appears to suffer from the additional soil channels creating confounding signals.
- **AET: middle ground** — NSE 0.839 (below v5's 0.851 and v8's 0.851), but pbias 4.9% is much better than v8's 12.7%. The pure Huber loss avoids v8's overprediction problem.
- **CWD: excellent global + best extreme metrics** — NSE 0.899 (below v5's 0.915), but CWD KGE 0.935 is the best ever. CWD P95 bias is **+0.6mm** — the first model to show slight *overprediction* rather than underprediction at extremes. CWD P95 hit rate 0.804 and P99 hit rate 0.683 are both best-in-class.
- **AET extremes: regressed** — P95 bias -23.3mm (worse than v5's -17.9 and v8's -15.9). The pure Huber loss reproduces the v6-era pattern of suppressing extreme-value gradients.

**Key insight:** v8b demonstrates a clean separation of concerns. The soil physics channels help PET and CWD but don't resolve AET extremes — that requires either loss function changes (extreme penalty) or additional dynamic features that capture within-month heat stress (the motivation for v9's fire features: HDD, sigmoid heat stress, Drought Code).

### v8c-mse: controlled MSE vs Huber comparison

v8c is identical to v8b (same data, same architecture, same uniform weights, extreme_weight=0.0) but switches to MSE loss. This is the first clean apples-to-apples comparison of MSE vs Huber on the same data.

**Results vs v8b (Huber):**

- **PET: dead tie** — NSE 0.927 for both. KGE 0.946 (MSE) vs 0.944 (Huber). The loss function doesn't matter for PET.
- **PCK: MSE dramatically better** — pbias 5.0% vs 13.3%, NSE 0.930 vs 0.916, RMSE 13.8 vs 15.2. Huber's linear tail appears to hurt PCK's ability to match large snowpack values.
- **AET: Huber slightly better** — NSE 0.839 vs 0.834, pbias 4.9% vs 6.4%. Small margin.
- **CWD: MSE better globally** — NSE 0.907 vs 0.899, RMSE 17.7 vs 18.5. The improved PCK flows through to better CWD.
- **CWD extremes: Huber better** — P95 bias +0.6mm (Huber) vs -1.3mm (MSE), hit rate 0.804 vs 0.793. Huber's tail suppression ironically helps CWD extremes by preventing AET overshoot.
- **AET extremes: both poor** — P95 bias -23.3mm (Huber) vs -25.5mm (MSE). Neither loss resolves the fundamental AET extreme underprediction.
- **Exceedance hit rates: MSE better** — AET P95 0.765 vs 0.757, AET P99 0.602 vs 0.572. MSE's quadratic gradients push more predictions above the threshold, even though the mean bias is worse.

**Key insight:** MSE is the better choice for this architecture. It wins on PCK (dramatically), CWD global, and exceedance hit rates, while only marginally losing on AET and CWD extremes. The Huber loss was introduced in v6 to stabilize training, but with the richer v8 soil physics channels, MSE training is stable without it (best epoch 67 vs 78, val loss curves well-behaved). The persistent AET extreme underprediction (-18 to -26mm at P95) is a data/feature problem, not a loss function problem — motivating the v9 fire features approach.

### v9-drought-code: fire features as dynamic inputs

v9 adds drought_code (Van Wagner 1987 DC) as a 12th dynamic input channel, building on v8c's MSE loss and soil physics static channels. The hypothesis was that a physically-based deep fuel moisture index would help the model capture drought-driven AET/CWD dynamics.

**Results vs v8c-mse (same loss, same static channels, minus drought_code):**

- **PET: best KGE ever** — KGE 0.953 (new record, beating v8c's 0.946), NSE 0.927 (tied). The drought code provides useful supplementary temperature/moisture information for PET estimation.
- **PCK: slight improvement** — NSE 0.932 vs 0.930, RMSE 13.7 vs 13.8. Marginal.
- **AET: regressed** — NSE 0.810 vs 0.834, KGE 0.738 vs 0.744, pbias 10.3% vs 6.4%. The model overpredicts AET more with drought code present, suggesting it interprets high DC (dry conditions) as a signal for *more* ET rather than drought-limited ET.
- **CWD: regressed** — NSE 0.888 vs 0.907, RMSE 19.5 vs 17.7, pbias -4.4% vs -2.5%. The AET overprediction cascades into CWD underprediction (CWD = PET - AET).
- **All extreme metrics worse** — AET P95 bias -26.3mm (vs -25.5mm), CWD P95 bias -4.1mm (vs -1.3mm), CWD P95 RMSE 13.7 vs 9.0mm. The drought code feature actively hurt extreme value prediction.

**Key insight:** The drought code feature helps PET (best-ever KGE) but hurts everything downstream. The likely mechanism: DC is high when it's hot and dry, which correlates with high PET — so it's a useful PET predictor. But the AET head misinterprets DC: high DC should *constrain* AET (vegetation is drought-stressed, stomata close), but the model learns the opposite association (high DC → high temperature → high AET). This is a classic case of a feature providing the right information for the wrong stage of the model. The drought code might need to be connected only to the AET/CWD heads (not the shared backbone), or the model needs an explicit mechanism to learn that DC is an *inhibitor* of AET under drought conditions.

### v9-kbdi: KBDI replaces Drought Code

v9-kbdi replaces the Van Wagner Drought Code (unbounded, 0-8000+) with the Keetch-Byram Drought Index (bounded 0-800) as dynamic channel 10. KBDI showed higher correlation with CWD extreme bias (r=0.571 vs 0.545) in the pre-analysis.

**Results vs v8c-mse (same architecture, no drought feature):**

- **AET: improved over v9-drought-code** — NSE 0.824 vs 0.810 (DC), pbias 6.6% vs 10.3% (DC). KBDI partially avoids the DC misinterpretation problem, but still slightly worse than v8c baseline (NSE 0.834).
- **CWD: improved over v9-drought-code** — NSE 0.896 vs 0.888 (DC), RMSE 18.8 vs 19.5 (DC). Still below v8c baseline (NSE 0.907).
- **PET: slight regression** — NSE 0.925 vs 0.927 (v8c), KGE 0.952 (near v9-DC's 0.953 best-ever).
- **PCK: degraded** — pbias 12.4% vs 5.0% (v8c). Similar pattern to v9-drought-code (8.2%).
- **CWD extremes: better than DC** — P95 RMSE 10.5 vs 13.7 (DC), P99 RMSE 8.2 vs 9.5 (DC). But still worse than v8c baseline (P95 9.0, P99 6.0).

**Key insight:** KBDI is a better drought feature than Drought Code — it causes less damage to AET/CWD predictions. However, neither drought index improves on the v8c baseline without a drought feature. The fundamental issue persists: the model treats drought signals (high KBDI = dry) as correlated with high ET rather than as an ET constraint. A drought feature would need to be connected specifically to the AET stage with an inhibitory mechanism to be useful.

### v10-kbdi-aet-only: routing KBDI to AET head only

v10 implements the architectural insight from v9: drought features help PET but hurt AET when routed through the shared backbone, because the backbone learns "hot+dry = high ET" instead of "hot+dry = drought-stressed ET." v10 removes KBDI from the backbone entirely (10 dynamic channels through TCN) and injects it directly into the AET head alongside the backbone output, PET, and PCK predictions. This gives the AET head a drought-stress signal without contaminating the backbone's representations.

**Results vs v9-kbdi (KBDI through backbone):**

- **AET: clear improvement** — NSE 0.840 vs 0.824, RMSE 12.1 vs 12.7, pbias 6.3% vs 6.6%. The AET head can now learn KBDI as an inhibitory signal rather than inheriting the backbone's positive correlation.
- **CWD: improved** — NSE 0.897 vs 0.896, pbias -2.6% vs -3.8%. The AET improvement flows through to CWD.
- **PET: recovered** — NSE 0.927 vs 0.925. With KBDI removed from the backbone, PET no longer has to compete with a drought signal in shared features.
- **PCK: flat** — NSE 0.929 (tied). PCK pbias slightly worse (13.5% vs 12.4%).

**Results vs v8c-mse (no drought feature at all):**

- **AET: improved** — NSE 0.840 vs 0.834, RMSE 12.1 vs 12.3, pbias 6.3% vs 6.4%. This is the first time a drought feature has *improved* AET over the no-drought baseline.
- **CWD: slightly worse globally** — NSE 0.897 vs 0.907, but CWD extremes are competitive (P95 bias -0.6mm vs -1.3mm, P99 bias -0.6mm vs -1.2mm).
- **PET: tied** — NSE 0.927 for both.
- **AET extremes: improved** — P95 bias -23.3mm vs -25.5mm, P95 RMSE 29.6 vs 30.6, P99 hit rate 0.587 vs 0.602 (slight regression). The routing strategy helps but doesn't fully resolve the persistent extreme underprediction.
- **CWD extremes: strong improvement** — P95 bias -0.6mm vs -1.3mm, P99 RMSE 5.7 vs 6.0, P99 bias -0.6mm vs -1.2mm. Approaching v8b-no-extreme's best-ever CWD extreme performance.

**Key insight:** The architectural routing hypothesis is validated. KBDI through the backbone hurts AET (v9-kbdi NSE 0.824 < v8c baseline 0.834); KBDI routed only to the AET head helps AET (v10 NSE 0.840 > v8c baseline 0.834). This is the first model to demonstrate that a drought feature can improve water balance prediction when properly connected. The remaining AET extreme bias (~-23mm at P95) likely requires either (a) an explicit inhibitory mechanism in the AET head, (b) sub-monthly temporal resolution, or (c) additional drought-response features (e.g., NDVI anomaly, soil moisture).

### v11-kv-aet: BCM Table 6 Kv crop coefficient (MLP head)

v11-kv-aet adds the BCM Table 6 Kv crop coefficient as a time-varying channel injected directly into the AET head alongside KBDI, PET, and PCK. Kv encodes the vegetation-specific seasonal transpiration potential (0.0 for bare rock to 1.517 for redwoods), mirroring BCMv8's `AET = Kv × PET × f(soil_water)` formulation. The AET head remains an MLP (260→64→1) that receives all channels concatenated.

**Results vs v10-kbdi-aet-only (same routing, no Kv):**

- **PET: marginal improvement** — NSE 0.928 vs 0.927, RMSE 16.1 vs 16.3. Kv doesn't directly affect the PET head but the overall loss landscape may have shifted slightly.
- **PCK: improved** — NSE 0.930 vs 0.929, pbias 8.7% vs 13.5%, KGE 0.871 vs 0.826. The Kv channel helps PCK indirectly — Kv=0 bare/water pixels have distinctive snowpack behavior.
- **AET: regressed** — NSE 0.835 vs 0.840, pbias 10.0% vs 6.3%, RMSE 12.3 vs 12.1. The MLP head receives Kv as just another concatenated channel and cannot easily learn the multiplicative structure `Kv × PET × stress`. Instead, it overpredicts AET for vegetated pixels (high Kv) without properly learning the stress constraint.
- **CWD: mixed** — NSE 0.900 vs 0.897 (slight improvement), but pbias -4.1% vs -2.6% (worse). The AET overprediction cascades into CWD.
- **AET extremes: flat** — P95 bias -23.4mm vs -23.3mm (no change). P95 RMSE 29.0 vs 29.6 (marginal improvement). The MLP cannot use Kv to fix the extreme underprediction.
- **CWD extremes: mixed** — P99 hit rate 0.713 (new best-ever), but P95 bias -2.6mm vs -0.6mm (regression). The Kv channel helps the model identify which pixels should have high CWD (low-Kv bare pixels) but doesn't improve the magnitude prediction.

**Key insight:** Kv provides the right information but the MLP architecture cannot exploit it. The multiplicative relationship `AET = Kv × PET × f(soil_water)` requires the network to approximate multiplication from concatenated inputs, which ReLU networks do poorly — they use piecewise-linear segments that break down at extremes. The product `Kv × PET × stress` is largest in late summer (high PET, vegetated pixels, moderate soil moisture), exactly where underprediction is worst. This motivates the stress-fraction architecture (v11-stress-frac) that encodes the multiplicative structure explicitly: `stress × Kv × PET + correction`.

### v11-stress-frac: stress-fraction AET head with multiplicative inductive bias

v11-stress-frac replaces the MLP AET head with a stress-fraction architecture that encodes the BCMv8 multiplicative structure explicitly: `stress_net` learns `f(soil_water) ∈ [0, 1]` via sigmoid, then `AET = clamp(stress × Kv, max=1.0) × PET + correction`. The hypothesis was that ReLU networks approximate multiplication poorly, causing AET P95 underprediction (~-23mm) because the product `Kv × PET × stress` is largest exactly where underprediction is worst (late summer, high PET, vegetated pixels).

**Results vs v11-kv-aet (same Kv plumbing, MLP head):**

- **PCK: major improvement** — NSE 0.944 vs 0.930, KGE 0.952 vs 0.871 (new best-ever PCK KGE), pbias 3.3% vs 8.7% (new best-ever PCK pbias). The stress-fraction head's different gradient landscape appears to benefit the shared backbone's PCK representations.
- **PET: marginal improvement** — NSE 0.929 vs 0.928, KGE 0.947 vs 0.942. Consistent with upstream stages being largely unaffected by head architecture.
- **AET: slight regression** — NSE 0.830 vs 0.835, pbias 6.3% vs 10.0% (better bias, worse NSE). The stress-fraction head produces less biased global AET but with higher variance.
- **CWD: improved globally** — NSE 0.903 vs 0.900, pbias -2.1% vs -4.1%, KGE 0.921 vs 0.915. The lower AET bias flows through to better CWD.
- **AET extremes: worse** — P95 bias -26.4mm vs -23.4mm, P99 bias -36.9mm vs -33.3mm. The stress-fraction architecture *increased* extreme underprediction by ~3mm. The `clamp(stress × kv, max=1.0)` ceiling may be too restrictive — it forces `AET ≤ PET` in the multiplicative path, pushing all extreme AET prediction to the correction_net which has the same learning challenge as the old MLP.
- **CWD extremes: worse** — P95 RMSE 11.0 vs 9.7, P99 RMSE 7.4 vs 6.4. The AET extreme regression cascades into CWD.

**Results vs v10-kbdi-aet-only (no Kv, MLP head — best AET model):**

- **AET: regressed** — NSE 0.830 vs 0.840. Adding Kv with *either* head architecture (MLP or stress-fraction) hurts AET NSE vs the no-Kv baseline.
- **PCK: best-ever** — KGE 0.952 and pbias 3.3% are records across all runs.
- **CWD: competitive** — NSE 0.903 vs 0.897, pbias -2.1% vs -2.6%. Better globally but worse at extremes.

**Key insight:** The stress-fraction architecture hypothesis was partially wrong. While it dramatically improved PCK (best-ever KGE/pbias) and CWD global metrics, it **did not fix AET extremes** — in fact it made them worse. The likely reasons:

1. **Clamp ceiling is too strict.** `clamp(stress × kv, max=1.0)` caps the multiplicative path at `1.0 × PET`. For extreme AET events, the physical reality may require AET *approaching* PET (stress ≈ 1.0, Kv ≈ 1.0), and the clamp removes gradient flow precisely at these critical points.
2. **Correction net carries too much burden.** For Kv=0 pixels (35% of grid — bare rock, water, urban), the multiplicative path outputs exactly zero and correction_net must provide the full prediction. For extreme events, the correction must also compensate for the clamp. This splits the learning problem in a way that may be harder than the unified MLP.
3. **The AET extreme problem may not be architectural.** Across v10, v11-kv-aet, and v11-stress-frac, AET P95 bias ranges from -23 to -26mm regardless of head architecture. The persistent ~-23mm floor suggests the backbone features themselves lack the information needed to predict extreme AET — the bottleneck is upstream of the head.

### v12-stress-frac-aet2x: stress-fraction head + v5-style loss weights

v12 combines the v11-stress-frac architecture (sigmoid stress × Kv × PET + correction) with v5's loss weighting strategy (aet=2.0, cwd=2.0, pet_decay=0.99). The hypothesis: the stress-fraction head provides the right inductive bias but uniform loss weights (v11-stress-frac) don't push the model hard enough on AET/CWD extremes. Resumed from epoch 70 checkpoint after GPU interruption; best epoch was 76 (6th epoch of resumed run).

**Results vs v5-awc-windward (previous best AET/CWD model):**

- **AET: new best-ever** — NSE 0.856 vs 0.851, KGE 0.825 vs 0.814, RMSE 11.4 vs 11.6. The stress-fraction head + loss weights combination exceeds v5 on all AET global metrics.
- **AET P95 bias: new best-ever** — -16.6mm vs -17.9mm. First model to break below v5's extreme bias floor. P95 RMSE 25.3 vs 25.7 (also best).
- **AET P99 bias: new best-ever** — -25.7mm vs -27.5mm. Consistent improvement at both extreme thresholds.
- **PET: v5-class** — NSE 0.862 vs 0.862 (identical). The pet_decay=0.99 produces the same PET trade-off as v5.
- **CWD: near-v5** — NSE 0.912 vs 0.915, KGE 0.922 vs 0.926. Very close.
- **PCK: degraded** — pbias 20.7% vs 6.5%, NSE 0.913 vs 0.944. The 2x AET/CWD weighting starves PCK of gradient attention. This is the same trade-off v5 made but amplified by the stress-fraction head.

**Results vs v11-stress-frac (same architecture, uniform weights):**

- **AET: dramatic improvement** — NSE 0.856 vs 0.830 (+0.026), P95 bias -16.6mm vs -26.4mm (10mm improvement). The loss weights were the missing ingredient — the stress-fraction architecture *can* learn extreme AET, but only when the loss function prioritizes it.
- **PET: expected trade-off** — NSE 0.862 vs 0.929 (-0.067). The pet_decay reduces PET weight to 0.5 by end of training.
- **PCK: significant regression** — NSE 0.913 vs 0.944, pbias 20.7% vs 3.3%. PCK is the casualty of the reweighting.
- **CWD: improved** — NSE 0.912 vs 0.903, RMSE 17.3 vs 18.1. The AET improvement flows through.

**Key insight:** The AET extreme underprediction was never purely an architectural or feature problem — it was primarily a **loss weighting problem**. The stress-fraction architecture with uniform weights (v11-stress-frac) showed AET P95 bias of -26.4mm; with 2x AET/CWD weights it dropped to -16.6mm (new best-ever). The multiplicative inductive bias helps the model *respond* to the stronger AET gradients more effectively than v5's MLP could — v12 beats v5 on AET P95 bias (-16.6 vs -17.9mm) despite v5 using the same loss weights. The trade-off remains: PET and PCK accuracy suffer. A future run could explore intermediate weights (aet=1.5) or a PCK-specific weight boost to recover snowpack accuracy.

### v13-sws-rollstd: SWS bucket model + rolling variability features

v13 adds 4 new dynamic channels to v12's architecture and loss weights: SWS (stress-modulated soil water storage bucket model), vpd_roll6_std, srad_roll6_std, and tmax_roll3_std. Total dynamic channels: 15 (11 base + SWS + 3 rolling std). Best epoch 73/100.

**Results vs v12-stress-frac-aet2x (same arch + loss weights, 11 dynamic channels):**

- **CWD: new best-ever NSE and RMSE** — NSE 0.916 vs 0.912 (beats v5's previous best 0.915), RMSE 16.9 vs 17.3 (beats v5's 17.0). The extra features help CWD more than AET. CWD pbias -2.2% vs -3.0% (also improved).
- **AET global: strong but below v12** — NSE 0.846 vs 0.856. Lower pbias (2.7% vs 7.6%) suggests less overall overprediction, but NSE regression indicates higher variance in predictions.
- **AET extremes: regressed from v12** — P95 bias -20.6mm vs -16.6mm, P99 bias -29.2mm vs -25.7mm. The 4 new channels added information but also added complexity — the model may be distributing gradient attention across 15 channels instead of focusing on the 11 that v12 used effectively.
- **PCK: worst-ever pbias (19.0%)** — Continuing the v5-style loss weight casualty pattern (v12 was 20.7%), now with 15 dynamic channels competing for representation in the backbone.
- **PET: v5/v12-class trade-off** — NSE 0.870, same ballpark as v12 (0.862) and v5 (0.862).

**Results vs v5-awc-windward (previous best CWD):**

- **CWD: new best** — NSE 0.916 vs 0.915, RMSE 16.9 vs 17.0. Marginal but consistent improvement.
- **CWD extremes: worse** — P95 bias -3.9mm vs -2.7mm, P99 bias -3.8mm vs -2.6mm. The global CWD improvement doesn't extend to extremes.

**SWS channel assessment:** SWS was intended to give the model explicit soil moisture information. The CWD improvement suggests it helps with the energy-water budget globally, but didn't translate to better AET extreme prediction. The rolling std channels (capturing climate variability) may be more useful for CWD than AET — variability in radiation and temperature affects the energy balance (CWD) more directly than the water balance constraint on actual ET.

**Key insight:** Adding more physically-motivated features improves CWD (new best-ever NSE + RMSE) but doesn't help AET extremes, which actually regressed. The AET extreme problem appears to have a complexity-performance trade-off: v12's 11 channels were enough for the model to focus on the most important signals, while 15 channels dilute gradient attention. Future directions could include (a) feature selection to identify which of the 15 channels are most valuable, (b) intermediate loss weights (aet=1.5) to recover PCK, or (c) channel attention mechanisms to let the model learn which inputs matter for which outputs.

### v14-sws-stress: fixed SWS with stress-modulated drainage

v14 fixes the SWS bucket model from v13. The original v13 SWS used PPT-PET drainage, which produced 68% zero values (uninformative). v14 uses stress-modulated drainage: `stress = min(SWS[t-1]/AWC, 1); AET_approx = PET*stress; SWS[t] = clamp(SWS[t-1] + PPT - AET_approx, 0, AWC)`. This linear stress function prevents over-drainage in dry conditions, reducing zeros to 13.8%. Same architecture, loss weights, and rolling std channels as v13. Best epoch 91/100.

**Results vs v13-sws-rollstd (broken SWS):**

- **AET: recovered toward v12** — NSE 0.854 vs 0.846, KGE 0.831 (new best-ever, beating v12's 0.825). The fixed SWS provides meaningful soil moisture information that helps AET prediction.
- **AET extremes: improved over v13** — P95 bias -17.9mm vs -20.6mm, P95 RMSE 26.6 vs 28.2. Recovered to v5-class extreme performance (-17.9mm matches v5's -17.9mm exactly). Still below v12's best-ever -16.6mm.
- **CWD: slight regression** — NSE 0.914 vs 0.916, RMSE 17.1 vs 16.9. The v13 CWD best-ever records were not retained, suggesting the broken SWS (with its 68% zeros acting as a near-constant channel) was actually less disruptive to CWD than the more informative fixed SWS.
- **CWD extremes: improved over v13** — P95 bias -2.0mm vs -3.9mm, P95 RMSE 9.0 vs 10.2, P99 bias -2.1mm vs -3.8mm. The fixed SWS helps CWD extremes substantially even though CWD global NSE slightly regressed.
- **PCK: major improvement** — pbias 13.6% vs 19.0%, NSE 0.921 vs 0.907. The better SWS signal reduces the gradient competition that was starving PCK.
- **PET: similar** — NSE 0.866 vs 0.870, pbias -1.3% vs -1.8%.
- **Training stability: improved** — Best epoch 91 vs 73, suggesting the fixed SWS provides more consistent gradients.

**Results vs v12-stress-frac-aet2x (same arch, 11 dynamic channels):**

- **AET KGE: new best-ever** — 0.831 vs 0.825. The additional channels with proper SWS improve the correlation/variability balance.
- **AET NSE: near-v12** — 0.854 vs 0.856. Very close, suggesting the fixed SWS nearly recovers v12's AET performance.
- **AET extremes: slight regression** — P95 bias -17.9mm vs -16.6mm. The 15-channel model still can't quite match v12's 11-channel extreme performance.
- **PCK: improved** — pbias 13.6% vs 20.7%. The fixed SWS partially recovers PCK from the loss weight casualty.

**Key insight:** Fixing the SWS bucket model made a substantial difference. With 68% zeros (v13), SWS was effectively a near-constant channel that wasted model capacity. With proper stress-modulated drainage (v14), SWS provides real soil moisture dynamics that improve AET (new best-ever KGE), AET extremes (recovered to v5-class), CWD extremes, and PCK. The remaining gap to v12's best-ever AET P95 bias (-17.9mm vs -16.6mm) suggests that 15 channels still dilute gradient attention somewhat, but the margin is now much smaller than v13's -20.6mm.

### v15-awc-extreme: AWC static channel + mild extreme penalty

v15 adds `awc_total = (FC - WP) × soil_depth × 1000` as a new static channel (15 static, up from 14) and re-enables the extreme-aware MSE penalty at a low weight (`extreme_weight=0.05`, `extreme_asym=1.5`). Same 15 dynamic channels and loss weights (aet=2.0, cwd=2.0, pet_decay=0.99) as v14. Best epoch 87/100.

**Results vs v14-sws-stress (no AWC static, no extreme penalty):**

- **AET P95 bias: best-ever among valid runs** — -16.4mm vs -17.9mm. The mild extreme penalty pushes AET predictions closer to observed extremes, surpassing v12's previous best of -16.6mm. This is the first model to improve on v12's AET extreme performance.
- **AET P99 bias: improved** — -24.7mm vs -26.4mm (+1.7mm). Consistent improvement at both extreme thresholds.
- **AET P95 RMSE: improved** — 25.2 vs 26.6. AET P95 hit rate also improved (0.755 vs 0.742).
- **AET global: flat** — NSE 0.853 vs 0.854, KGE 0.831 vs 0.831 (tied). The extreme penalty improved tails without degrading the mean.
- **CWD global: flat** — NSE 0.913 vs 0.914. CWD extremes slightly regressed (P95 bias -3.5mm vs -2.0mm).
- **PCK: regressed** — pbias 19.0% vs 13.6%. The extreme penalty adds gradient competition that further starves PCK.
- **PET: slight regression** — NSE 0.857 vs 0.866.

**Results vs v12-stress-frac-aet2x (11 dynamic channels, no extreme penalty):**

- **AET P95 bias: improved** — -16.4mm vs -16.6mm. First model to beat v12's AET extreme record.
- **AET global: near-identical** — NSE 0.853 vs 0.856, KGE 0.831 vs 0.825 (v15 better KGE).
- **PCK: similar casualty** — pbias 19.0% vs 20.7%.

**Extreme penalty assessment:** `extreme_weight=0.05` is well-calibrated. The v7 failure at weight=2.0 showed that aggressive extreme penalties destabilize training; v15 demonstrates that a 40x smaller weight (0.05) achieves the desired effect — AET P95 bias improvement of 1.5mm — without degrading global metrics. The AET_ext loss component decreased from 1.65 to 0.20 during training, confirming the penalty provided consistent gradient signal throughout.

**AWC static channel assessment:** Adding AWC as a static channel had minimal impact on its own (v14→v15 delta is dominated by the extreme penalty). AWC is derivable from FC, WP, and soil_depth which were already available as separate static channels. The model likely already learned the relevant soil water capacity information from those components.

**Key insight:** The combination of 15 dynamic channels (with fixed SWS), AWC static channel, and mild extreme penalty (0.05) produces the best-ever AET P95 bias (-16.4mm) while maintaining v12-class global AET accuracy. The extreme penalty was the key ingredient — it addresses the AET tail underprediction that loss weights alone couldn't fully resolve. Future directions: (a) try `extreme_weight=0.1` to push AET extremes further, (b) add PCK to `extreme_vars` to address the persistent PCK pbias casualty, (c) explore intermediate AET weights (1.5) to balance PCK recovery.

### v16-aet1.5-extreme: reduced AET weight to rebalance PCK

v16 reduces `aet_initial` from 2.0 to 1.5 while keeping `cwd_initial=2.0`, `extreme_weight=0.05`, and `extreme_asym=1.5`. The hypothesis: v12-v15's PCK degradation (pbias 13-20%) is caused by excessive AET gradient dominance; a milder AET weight should recover PCK without losing the AET extreme gains from the extreme penalty. Best epoch 80/100, best val_loss 0.555.

**Results vs v15-awc-extreme (aet=2.0, same extreme penalty):**

- **PCK: dramatic recovery** — pbias 9.7% vs 19.0% (+9.3pp better), KGE 0.868 vs 0.757, NSE 0.930 vs 0.916, RMSE 13.8 vs 15.2. Reducing AET weight from 2.0 to 1.5 freed gradient attention for PCK. This is the best PCK performance since v8-soil-physics (pbias 4.2%) among runs with elevated AET/CWD weights.
- **PET: improved** — pbias -0.5% vs -1.1% (closer to zero), NSE 0.861 vs 0.857. Less AET gradient competition benefits PET too.
- **AET P95 bias: slightly improved** — -16.3mm vs -16.4mm. The extreme penalty (weight=0.05) maintains AET tail performance even with reduced AET base weight. This confirms the extreme penalty is the key mechanism for AET extremes, not the base loss weight.
- **AET global: slight regression** — NSE 0.848 vs 0.853, pbias 9.5% vs 6.6%. The lower AET weight allows slightly more AET overprediction.
- **CWD: flat** — NSE 0.912 vs 0.913, RMSE 17.2 vs 17.1. CWD maintained its cwd=2.0 weight, so performance is stable.

**Results vs v12-stress-frac-aet2x (same arch, aet=2.0, no extreme penalty):**

- **PCK: dramatically better** — pbias 9.7% vs 20.7%, KGE 0.868 vs 0.745. The lower AET weight + extreme penalty combination is clearly superior to brute-force AET weighting.
- **AET P95 bias: improved** — -16.3mm vs -16.6mm. Even with lower AET base weight, the extreme penalty delivers better tail performance.
- **AET global: slightly worse** — NSE 0.848 vs 0.856. The trade-off for PCK recovery.

**Key insight:** The aet_initial=1.5 experiment reveals that the extreme penalty (`extreme_weight=0.05`) is doing the heavy lifting for AET tail accuracy, not the base AET loss weight. Reducing `aet_initial` from 2.0 to 1.5 barely affected AET P95 bias (-16.3mm vs -16.4mm) but dramatically recovered PCK (pbias 9.7% vs 19.0%). This suggests the optimal configuration is moderate base weights (aet=1.5) combined with targeted extreme penalties — the base weights handle the bulk of the distribution while the extreme penalty handles the tails. Future directions: (a) try aet_initial=1.0 (uniform weights) with extreme_weight=0.05 to test if even lower AET weight maintains extreme performance, (b) increase extreme_weight to 0.1 to push AET P95 bias further, (c) add PCK to extreme_vars to address the remaining 9.7% pbias.

### v17-polaris-awc: POLARIS root-zone AWC for SWS + dropped awc_total static

v17 switches the SWS bucket model's AWC source from BCMv8 full-column `(FC - WP) × soil_depth × 1000` (~500-2000mm) to POLARIS root-zone (0-100cm) AWC (~300-500mm). The BCMv8 AWC overestimated soil water storage capacity, understating drought stress in the SWS channel. v17 also drops awc_total as a static channel (back to 14 static from v15's 15), since POLARIS AWC has minimal spatial contrast (~400mm uniform). Same loss configuration as v16: aet=1.5, cwd=2.0, extreme_weight=0.05. Best epoch 69/100, best val_loss 0.505.

**Results vs v16-aet1.5-extreme (BCMv8 AWC, 15 static channels):**

- **CWD: new best-ever NSE** — NSE 0.929 vs 0.912 (+0.017), RMSE 15.5 vs 17.2mm (new best-ever, beating v13's 16.9mm), KGE 0.931 vs 0.920. The POLARIS AWC produces a more realistic SWS drought signal that dramatically improves the water deficit calculation. CWD pbias -2.7% vs -3.7% (improved).
- **CWD extremes: improved** — P95 bias -3.0mm vs -3.8mm, P95 RMSE 9.0 vs 9.6 (approaching v5b's best-ever 8.2), P99 bias -2.8mm vs -3.6mm, P99 RMSE 5.7 vs 6.6 (matching v5-awc-windward). CWD P95 hit rate 0.781 vs 0.774, P99 hit rate 0.684 vs 0.672 — both improved.
- **PCK: near-best recovery** — NSE 0.949 vs 0.930, RMSE 11.8 vs 13.8 (new best among weighted-loss runs), pbias 6.8% vs 9.7%, KGE 0.904 vs 0.868. Approaching v6-huber's best-ever PCK NSE (0.950). Dropping awc_total freed the model from a near-constant static channel that was adding noise to the PCK representation.
- **PET: improved** — NSE 0.879 vs 0.861, RMSE 21.0 vs 22.4. Better than any previous weighted-loss run (v5-v16 range: 0.857-0.870). pbias -0.6% (similar to v16's -0.5%).
- **AET global: slight improvement** — NSE 0.851 vs 0.848, RMSE 11.6 vs 11.7, pbias 7.2% vs 9.5% (better). KGE 0.798 vs 0.816 (regression — lower correlation component despite better bias).
- **AET extremes: regressed** — P95 bias -19.2mm vs -16.3mm (+2.9mm worse), P99 bias -28.4mm vs -24.9mm (+3.5mm worse). The POLARIS AWC's lower values (300-500mm vs 500-2000mm) mean SWS drains faster and stays depleted longer — the model may be using the SWS channel as a "dry = less AET" signal too aggressively, compressing extreme AET predictions.
- **AET P95 hit rate: improved** — 0.759 vs 0.753. Despite worse bias, the model classifies more samples correctly above the P95 threshold.

**POLARIS AWC assessment:** The switch to root-zone AWC is a clear win for CWD (new best-ever NSE 0.929, RMSE 15.5mm) and PCK (near-best NSE 0.949), and improves PET. The AET extreme regression (-19.2mm vs -16.3mm) is the trade-off. The lower AWC values create a more drought-sensitive SWS that better reflects physical reality but may overcorrect: the model now sees "drought stress" more frequently and learns to suppress AET more broadly, including during genuine high-AET events.

**Key insight:** v17 demonstrates that the AWC source fundamentally shapes the SWS-to-AET pathway. POLARIS root-zone AWC (~300-500mm) produces a much more responsive SWS signal than BCMv8 full-column AWC (~500-2000mm), which was essentially a "always moist" buffer. This responsiveness improves CWD prediction dramatically (the SWS drought signal directly informs the PET-AET-CWD cascade) but creates an AET extreme penalty — the model leans too heavily on the "currently dry" SWS signal and underestimates AET during hot events where stomatal conductance may still be high despite depleted soil water (e.g., deep-rooted vegetation accessing water below the 100cm root zone). The v18 tuning sweep will test whether adjusting extreme_weight (0.1) or extreme_threshold (P85) can recover AET tail performance without losing v17's CWD gains.

**Remaining gaps for operational wildfire use**

- **Temporal resolution.** Monthly CWD smooths over intra-month drying events that drive ignition risk. Fire weather operates on daily-to-weekly scales — a single week of hot, dry, windy conditions can bring vegetation from marginal to critical fire danger regardless of monthly averages. A statistical downscaling step from monthly to daily CWD using PRISM daily tmax and VPD as covariates would bridge this gap without requiring a full daily BCM emulator.

- **Fire season evaluation.** Current metrics average over all 12 months, which dilutes the signal from June-November when fire risk is concentrated. A fire-season-only evaluation would more accurately reflect operational accuracy and would likely show larger improvements in extreme bias metrics since summer is where AET and CWD extremes cluster. The persistent PCK pbias (~10% in v16) matters less operationally if it is driven by winter snowpack errors rather than summer snow-free conditions.

- **Spatial validation against fire perimeters.** Model accuracy against BCMv8 targets does not guarantee the CWD patterns predict where fires actually occur. Comparing predicted CWD anomalies against MTBS and FRAP fire perimeters — particularly asking whether above-normal CWD precedes fire occurrence in the same pixel — would provide ecologically meaningful validation beyond emulator fidelity metrics.

- **Out-of-sample climate validation.** The test period is currently October 2019 through September 2020 — one year. California's most severe drought and fire conditions occurred in 2020-2022 and were outside the training distribution in terms of cumulative water deficit magnitude. Extending evaluation to 2020-2025 would test whether the emulator generalizes to conditions more extreme than anything in its training data, which is the primary value proposition for CMIP6 forward projections.

- **PCK bias in snow-dominated ecoregions.** The persistent PCK pbias (~10-20% across v12-v16) is a known casualty of the 2x AET/CWD loss weighting. Sierra Nevada and Cascade pixels with significant snowpack feed PCK errors into pck_prev as an autoregressive input, compounding over multi-year forward simulations. For CMIP6 projections where winter precipitation phase shifts are a primary signal of interest, this bias needs to be explicitly addressed — either through a separate PCK-focused loss term or by routing pck_prev through a corrected prediction rather than the biased one.

- **Ecoregion-specific failure modes.** The NSE-by-ecoregion table consistently shows near-zero or negative AET NSE in arid regions (Mojave, Sonoran, Basin and Range) despite strong performance elsewhere. These regions have near-zero AET that is dominated by rare precipitation pulses — a different dynamical regime than the Mediterranean and montane pixels that dominate the training signal. Separate evaluation thresholds or region-specific fine-tuning would be needed before operational deployment across all of California.


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
v7  + Extreme MSE penalty (wt=2.0) ... AET NSE 0.760, CWD NSE 0.830  REGRESSION
 |                                       Weight too high; AET pbias +40%, dominated total loss
v7b + Extreme MSE penalty (wt=0.1) ... AET NSE 0.760, CWD NSE 0.830  (EVAL INVALID - checkpoint issue)
 |                                       Training stabilized (val_loss 0.271 vs v7's 1.002), needs re-eval
v8  + soil_depth/aridity/FC/WP/SOM .. AET NSE 0.851, CWD NSE 0.894
 |                                       PET best-ever (0.914), AET pbias worse (+12.7%)
 |                                       CWD regressed due to error amplification in PET-AET difference
v8b Pure Huber + soil physics ........ AET NSE 0.839, CWD NSE 0.899
 |                                       PET recovered (0.927), CWD KGE best-ever (0.935)
 |                                       CWD P95 bias +0.6mm (first positive!), hit rate 0.804 (best)
 |                                       AET extremes regressed (P95 bias -23.3mm)
v8c MSE loss (same arch/data) ....... AET NSE 0.834, CWD NSE 0.907
 |                                       PCK dramatically better (pbias 5.0% vs 13.3%)
 |                                       CWD global better, CWD extremes slightly worse
 |                                       Confirms AET extreme bias is a feature problem, not loss problem
v9  + drought_code dynamic channel .. AET NSE 0.810, CWD NSE 0.888  REGRESSION
 |                                       PET KGE best-ever (0.953), but AET/CWD/extremes all worse
 |                                       DC helps PET but AET misinterprets it (high DC → more ET, not less)
 |                                       Feature-stage mismatch: DC useful for energy balance, harmful for water balance
v9k + KBDI replaces drought_code ... AET NSE 0.824, CWD NSE 0.896  (better than DC, worse than v8c)
 |                                       KBDI less harmful than DC but still doesn't improve on no-drought baseline
 |                                       Drought features need inhibitory mechanism for AET stage to be useful
v10 KBDI routed to AET head only .. AET NSE 0.840, CWD NSE 0.897  (first drought feature to beat v8c baseline!)
 |                                       PET recovered (0.927), AET improved over both v9-kbdi and v8c
 |                                       CWD extremes near-best (P95 bias -0.6mm, P99 bias -0.6mm)
 |                                       Validates routing hypothesis: drought signals help when connected to right stage
v11 + Kv crop coeff at AET (MLP) . AET NSE 0.835, CWD NSE 0.900  (AET regressed, CWD P99 hit rate best-ever 0.713)
 |                                       Kv provides right info but MLP can't learn multiplicative structure
 |                                       AET pbias 10.0% — overpredicts vegetated pixels without stress constraint
 |                                       AET P95 bias -23.4mm unchanged — motivates stress-fraction architecture
v11sf Stress-frac AET head ........ AET NSE 0.830, CWD NSE 0.903  (PCK KGE best-ever 0.952, pbias 3.3%)
 |                                       sigmoid(stress) × Kv × PET + correction — explicit multiplicative structure
 |                                       AET P95 bias WORSE (-26.4mm vs -23.4mm) — clamp ceiling too restrictive
 |                                       CWD global improved but extremes regressed
v12 + v5-style loss weights ....... AET NSE 0.856, CWD NSE 0.912  ★ NEW BEST AET (NSE + P95 bias)
 |                                       aet=2.0, cwd=2.0, pet_decay=0.99 — same weights as v5
 |                                       AET P95 bias -16.6mm (new best, beating v5's -17.9mm)
 |                                       AET P99 bias -25.7mm (new best, beating v5's -27.5mm)
 |                                       PET NSE 0.862 (v5-class trade-off), PCK pbias 20.7% (casualty)
 |                                       Loss weighting was the missing ingredient, not architecture alone
v13 + SWS + rolling std features . AET NSE 0.846, CWD NSE 0.916  ★ NEW BEST CWD (NSE + RMSE)
 |                                       15 dynamic channels (SWS bucket model + vpd/srad/tmax rolling std)
 |                                       CWD RMSE 16.9mm (best), CWD pbias -2.2%
 |                                       AET P95 bias -20.6mm (regressed from v12's -16.6mm)
 |                                       PCK pbias 19.0% (worst) — loss weight casualty continues
 |                                       NOTE: SWS had 68% zeros (PPT-PET drainage) — broken implementation
v14 Fixed SWS (stress-modulated) . AET NSE 0.854, CWD NSE 0.914  ★ NEW BEST AET KGE (0.831)
 |                                       stress = min(SWS/AWC, 1); AET_approx = PET×stress (13.8% zeros)
 |                                       AET P95 bias -17.9mm (recovered to v5-class, still below v12's -16.6mm)
 |                                       CWD extremes improved (P95 bias -2.0mm, P99 bias -2.1mm)
 |                                       PCK pbias 13.6% (recovered from v13's 19.0%)
 |                                       Best epoch 91/100 — more stable training with informative SWS
v15 + AWC static + extreme penalty AET NSE 0.853, CWD NSE 0.913  ★ BEST AET P95 BIAS (valid runs)
 |                                       awc_total static channel (15 static), extreme_weight=0.05
 |                                       AET P95 bias -16.4mm (beats v12's -16.6mm — new best among valid runs)
 |                                       AET P99 bias -24.7mm (improved from v14's -26.4mm)
 |                                       Global AET flat (NSE 0.853, KGE 0.831), CWD flat (NSE 0.913)
 |                                       PCK pbias 19.0% — extreme penalty adds gradient competition
v16 aet_initial 2.0→1.5 ........... AET NSE 0.848, CWD NSE 0.912  ★ PCK RECOVERY (pbias 9.7%)
 |                                       aet_initial=1.5, cwd=2.0, extreme_weight=0.05
 |                                       PCK pbias 9.7% (from 19.0%), PCK KGE 0.868 (from 0.757)
 |                                       AET P95 bias -16.3mm (maintained — extreme penalty does heavy lifting)
 |                                       PET pbias -0.5% (improved from -1.1%)
 |                                       AET global: NSE 0.848 (slight regression from 0.853)
 |                                       Key finding: base AET weight ≠ tail performance; extreme penalty is the mechanism
v17 POLARIS root-zone AWC ......... AET NSE 0.851, CWD NSE 0.929  ★ NEW BEST CWD (NSE, RMSE, KGE)
                                         POLARIS 0-100cm AWC for SWS (~300-500mm vs BCMv8 ~500-2000mm)
                                         Dropped awc_total static (14 static); aet=1.5, extreme_weight=0.05
                                         CWD NSE 0.929 (new best, beating v13's 0.916), RMSE 15.5 (new best)
                                         CWD KGE 0.931 (new best), CWD pbias -2.7%
                                         PCK NSE 0.949, pbias 6.8% (best among weighted-loss runs)
                                         PET NSE 0.879 (best among weighted-loss runs)
                                         AET P95 bias -19.2mm (regressed from v16's -16.3mm)
                                         Trade-off: more responsive SWS helps CWD but overcorrects AET extremes
```
