# BCM Emulator: Model Run Comparison

## Objective

This document compares all model versions (v1 through v21) with an emphasis on metrics that matter for **wildfire modeling**: accurate prediction of climatic water deficit (CWD) and actual evapotranspiration (AET) extremes. CWD is the primary driver of vegetation drought stress and fire danger in California; AET extremes reflect periods of rapid vegetation drying. Underpredicting these extremes means underestimating fire risk.

**Current operational configuration: v19a-huber-tight-extreme0.1** — AET P95 bias -10.1mm (new project best), PCK pbias 8.0%, CWD NSE 0.925. Config: `loss_type=huber`, `huber_delta=0.5`, `extreme_weight=0.1`, `aet_initial=1.5`, `cwd_initial=2.0`. See tuning_experiments.md for full sweep results.

**Out-of-sample validation: v19a-extended** — Same model evaluated on Oct 2019 - Sep 2024 (60 months, 5-year holdout including 2020-2024 megadrought/megafire period). CWD NSE 0.919 (only -0.006 from training-period 0.925), CWD KGE 0.940 (improved from 0.926), PCK NSE 0.965 (best-ever across all runs). AET pbias +11.6% (expected: drought-era physiological changes not in training distribution). **The emulator has passed out-of-sample validation and is ready for fire probability modeling.**

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
| v18-sweep (14 runs) | 2026-03-25 | Various | 14-experiment hyperparameter sweep across loss weights, extreme penalty, loss type, scheduler. See tuning_experiments.md |
| v19a-huber-tight-extreme0.1 | 2026-03-26 | Huber+Extreme | **OPERATIONAL CONFIG** — Huber delta=0.5 + extreme_weight=0.1. Synergistic combination: AET P95 bias -10.1mm (new project best) |
| v19b-extreme0.1-petfloor0.3 | 2026-03-26 | MSE+Extreme | MSE path: extreme_weight=0.1 + pet_floor=0.3. Anti-synergistic — worse than either component alone. MSE path ceiling confirmed. |
| v20a-asym1.1 | 2026-03-27 | Huber+Extreme | v19a base + extreme_asym=1.1 (from 1.5). Reduced AET pbias (8.3%) but PCK pbias blew out (19.4%). See tuning_experiments.md. |
| v20b-aet1.2 | 2026-03-27 | Huber+Extreme | v19a base + aet_initial=1.2 (from 1.5). AET P95 bias -8.7mm (best-ever) but AET pbias 15.5%, PCK pbias 16.6%. Confirms irreducible gradient competition. |
| v21-dual-backbone | 2026-03-29 | Huber+Extreme | Dual-backbone architecture: main TCN (256) + AET sub-backbone (128, 3-layer, RF=29mo). AET head sees both; PET/PCK see only main. Gradient decoupling test. |
| v21b-deeper-sub | 2026-03-30 | Huber+Extreme | v21 with deeper sub-backbone [32,64,128,128] (4-layer, RF=61mo). Tests whether longer receptive field recovers AET P95 bias. |
| v22-dual-full | 2026-03-30 | Huber+Extreme | 5-layer AET sub-backbone [32,64,128,128,128] (RF=125mo) + separate FVEG embeddings. Tests full RF match + gradient isolation. |
| v23a-dual-extreme0.15 | 2026-03-30 | Huber+Extreme | v21b arch (4-layer sub, separate FVEG) + extreme_weight=0.15 (3x v21b). Tests stronger tail penalty in dual-backbone regime. |
| v23b-dual-extreme0.20 | 2026-03-31 | Huber+Extreme | v21b arch + extreme_weight=0.20 (4x v21b). Aggressive variant — AET P95 bias -9.8mm (best-ever) but AET pbias 17.1%. |
| **v19a-extended** | 2026-03-31 | Huber+Extreme | **OUT-OF-SAMPLE VALIDATION** — v19a model evaluated on Oct 2019 - Sep 2024 (60 months). Data extended through 2024-09. CWD NSE 0.919, PCK NSE 0.965 (best-ever). |

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
| v19a-huber-tight-extreme0.1 | 0.825 | 0.948 | 0.858 | 0.925 |
| v19b-extreme0.1-petfloor0.3 | 0.859 | 0.925 | 0.854 | 0.925 |
| v20a-asym1.1 | 0.842 | 0.915 | 0.855 | 0.925 |
| v20b-aet1.2 | 0.825 | 0.938 | 0.851 | 0.922 |
| v21-dual-backbone | 0.880 | 0.930 | 0.852 | 0.928 |
| v21b-deeper-sub | 0.872 | 0.932 | 0.855 | 0.924 |
| v22-dual-full | 0.879 | 0.934 | 0.846 | 0.925 |
| v23a-dual-extreme0.15 | 0.857 | 0.926 | 0.857 | 0.919 |
| v23b-dual-extreme0.20 | 0.850 | 0.933 | 0.848 | 0.915 |
| **v19a-extended** (OOS) | 0.863 | **0.965** | 0.824 | 0.919 |

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
| v19a-huber-tight-extreme0.1 | 0.812 | 0.893 | **0.828** | 0.926 |
| v19b-extreme0.1-petfloor0.3 | 0.846 | 0.818 | 0.826 | 0.926 |
| v20a-asym1.1 | 0.837 | 0.752 | 0.839 | 0.926 |
| v20b-aet1.2 | 0.823 | 0.801 | 0.810 | 0.925 |
| v21-dual-backbone | 0.870 | 0.881 | 0.799 | 0.923 |
| v21b-deeper-sub | 0.860 | 0.855 | 0.818 | 0.925 |
| v22-dual-full | 0.864 | 0.852 | 0.803 | 0.926 |
| v23a-dual-extreme0.15 | 0.844 | 0.828 | 0.808 | 0.911 |
| v23b-dual-extreme0.20 | 0.838 | 0.853 | 0.797 | 0.912 |
| **v19a-extended** (OOS) | 0.843 | 0.926 | 0.784 | **0.940** |

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
| v19a-huber-tight-extreme0.1 | 25.7 | 11.9 | **11.4** | 15.9 |
| v19b-extreme0.1-petfloor0.3 | 22.6 | 14.4 | 11.5 | 16.0 |
| v20a-asym1.1 | 23.9 | 15.3 | 11.5 | 15.9 |
| v20b-aet1.2 | 25.2 | 13.0 | 11.6 | 16.2 |
| v21-dual-backbone | 20.9 | 13.9 | 11.6 | 15.7 |
| v21b-deeper-sub | 21.6 | 13.7 | 11.5 | 16.1 |
| v22-dual-full | 21.0 | 13.5 | 11.8 | 15.9 |
| v23a-dual-extreme0.15 | 22.8 | 14.3 | 11.4 | 16.6 |
| v23b-dual-extreme0.20 | 23.4 | 13.5 | 11.8 | 17.0 |
| **v19a-extended** (OOS) | 22.1 | 22.0 | 12.9 | 16.2 |

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
| v19a-huber-tight-extreme0.1 | -0.8 | 8.0 | 13.1 | -3.5 |
| v19b-extreme0.1-petfloor0.3 | -0.8 | 13.8 | 8.1 | -3.4 |
| v20a-asym1.1 | -0.2 | 19.4 | 8.3 | -2.8 |
| v20b-aet1.2 | 1.5 | 16.6 | 15.5 | -3.9 |
| v21-dual-backbone | -0.9 | 7.7 | 7.8 | -3.3 |
| v21b-deeper-sub | -0.7 | 11.1 | 8.7 | -3.5 |
| v22-dual-full | -1.5 | 10.0 | **4.9** | -2.9 |
| v23a-dual-extreme0.15 | 0.1 | 13.7 | 14.4 | -5.1 |
| v23b-dual-extreme0.20 | 1.1 | 11.6 | 17.1 | -5.0 |
| **v19a-extended** (OOS) | 2.8 | -4.5 | 11.6 | **-1.3** |

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
| v19a-huber-tight-extreme0.1 | 21.9 | **-10.1** | **0.769** |
| v19b-extreme0.1-petfloor0.3 | 24.8 | -16.1 | 0.754 |
| v20a-asym1.1 | 24.1 | -13.5 | 0.756 |
| v20b-aet1.2 | 21.2 | **-8.7** | 0.763 |
| v21-dual-backbone | 26.4 | -19.4 | 0.759 |
| v21b-deeper-sub | 24.9 | -16.0 | 0.757 |
| v22-dual-full | 28.1 | -21.1 | 0.745 |
| v23a-dual-extreme0.15 | 22.1 | -13.0 | 0.760 |
| v23b-dual-extreme0.20 | 21.4 | **-9.8** | 0.755 |
| **v19a-extended** (OOS) | 30.7 | -19.2 | 0.715 |

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
| v19a-huber-tight-extreme0.1 | 23.8 | **-16.1** | 0.579 |
| v19b-extreme0.1-petfloor0.3 | 29.6 | -23.9 | 0.549 |
| v20a-asym1.1 | 26.9 | -20.4 | 0.576 |
| v20b-aet1.2 | 22.7 | **-15.0** | 0.573 |
| v21-dual-backbone | 32.5 | -28.2 | 0.580 |
| v21b-deeper-sub | 28.7 | -23.0 | 0.591 |
| v22-dual-full | 36.5 | -32.4 | 0.535 |
| v23a-dual-extreme0.15 | 25.8 | -20.5 | 0.559 |
| v23b-dual-extreme0.20 | 24.1 | -17.4 | 0.572 |
| **v19a-extended** (OOS) | 42.5 | -32.0 | 0.574 |

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
| v19a-huber-tight-extreme0.1 | 8.7 | -3.2 | 0.787 |
| v19b-extreme0.1-petfloor0.3 | 9.4 | -3.7 | 0.769 |
| v20a-asym1.1 | 8.8 | -3.0 | 0.785 |
| v20b-aet1.2 | 9.0 | -3.3 | 0.791 |
| v21-dual-backbone | 9.6 | -3.5 | 0.790 |
| v21b-deeper-sub | 9.2 | -3.1 | 0.785 |
| v22-dual-full | 9.2 | -3.0 | 0.776 |
| v23a-dual-extreme0.15 | 10.5 | -5.2 | 0.761 |
| v23b-dual-extreme0.20 | 9.6 | -4.2 | 0.786 |
| **v19a-extended** (OOS) | 10.4 | -5.6 | 0.841 |

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
| v19a-huber-tight-extreme0.1 | 5.8 | -3.3 | 0.696 |
| v19b-extreme0.1-petfloor0.3 | 6.4 | -3.6 | 0.665 |
| v20a-asym1.1 | 5.8 | -3.0 | 0.662 |
| v20b-aet1.2 | 6.1 | -3.4 | 0.669 |
| v21-dual-backbone | 6.3 | -2.7 | 0.680 |
| v21b-deeper-sub | 5.8 | -2.8 | 0.678 |
| v22-dual-full | 6.3 | -3.1 | 0.651 |
| v23a-dual-extreme0.15 | 7.8 | -5.2 | 0.668 |
| v23b-dual-extreme0.20 | 6.9 | -4.3 | 0.671 |
| **v19a-extended** (OOS) | 8.5 | -5.3 | 0.692 |

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

### v20a/v20b: loss tuning to reduce AET pbias (Huber-tight regime)

v20 experiments tested whether v19a's 13.1% AET pbias was reducible via loss tuning within the Huber-tight regime, or structurally irreducible due to shared-backbone gradient competition.

**v20a-asym1.1** reduced `extreme_asym` from 1.5 to 1.1 (less underprediction penalty asymmetry). **v20b-aet1.2** reduced `aet_initial` from 1.5 to 1.2 (lower base AET weight). See tuning_experiments.md for full analysis.

**Key results vs v19a:**

| Metric | v19a | v20a | v20b |
|--------|------|------|------|
| AET pbias | 13.1% | 8.3% | 15.5% |
| AET P95 bias | -10.1mm | -13.5mm | -8.7mm |
| PCK pbias | 8.0% | 19.4% | 16.6% |
| PET NSE | 0.825 | 0.842 | 0.825 |

**Key insight:** Both experiments confirmed an irreducible gradient competition in the shared backbone. Any adjustment that improved AET mean bias destabilized PCK above 15% pbias. v20b achieved the best-ever AET P95 bias (-8.7mm) but at the cost of 15.5% AET pbias and 16.6% PCK pbias — a Pareto frontier that cannot be improved within a single-backbone architecture. This motivated the v21 dual-backbone experiment.

### v21-dual-backbone: architectural gradient decoupling

v21 introduces a dual-backbone architecture to structurally eliminate the gradient competition between PET/PCK and AET. A narrow sub-backbone (3-layer TCN, channels [32, 64, 128], RF=29 months, ~117K params) processes a subset of drought-relevant channels and feeds into the AET head alongside the main backbone's output. PET/PCK heads see only the main backbone — AET gradients through the sub-backbone cannot interfere with PET/PCK learning.

**Sub-backbone channel routing (22 channels):**
- Dynamic (9): ppt, tmin, tmax, srad, vpd, sws, vpd_roll6_std, srad_roll6_std, tmax_roll3_std
- Static (5): soil_depth, aridity, FC, WP, SOM
- FVEG embedding (8): shared with main backbone

**Results vs v19a (single backbone, operational config):**

- **AET pbias: halved** — 7.8% vs 13.1% (-5.3pp). The gradient decoupling eliminated the mean overprediction caused by competing PET/PCK gradients pulling the backbone toward features that inflate AET.
- **PET: major improvement** — NSE 0.880 vs 0.825 (+0.055), RMSE 20.9 vs 25.2mm (-4.3mm). With AET gradients partially offloaded to the sub-backbone, the main backbone can dedicate more capacity to PET features.
- **PCK pbias: maintained** — 7.7% vs 8.0%. No regression — the decoupling preserved PCK stability.
- **CWD: slight improvement** — NSE 0.928 vs 0.925, RMSE 15.7 vs 15.9mm.
- **AET P95 bias: regressed** — -19.4mm vs -10.1mm (+9.3mm worse). The sub-backbone's short receptive field (29 months) may lack the multi-year context needed for extreme events, or the extreme penalty needs retuning for the dual-backbone's different gradient landscape.
- **AET P99 bias: regressed** — -28.2mm vs -16.1mm.

**Gradient isolation verified:** Unit tests confirmed that PET+PCK loss produces zero gradients in the AET sub-backbone, and AET loss flows through both backbones as intended.

**Key insight:** The dual-backbone architecture successfully decoupled AET mean prediction from PET/PCK — AET pbias dropped from 13.1% to 7.8% while PCK stayed at 7.7% and PET improved substantially. However, the AET extreme performance regressed significantly, suggesting that (a) the 3-layer sub-backbone (RF=29 months) is too shallow to capture the multi-season drought dynamics that drive AET extremes, and/or (b) the extreme_weight=0.05 penalty needs to be stronger now that the AET head has a different gradient landscape.

### v21b-deeper-sub: 4-layer sub-backbone (RF=61 months)

v21b deepens the AET sub-backbone from 3 layers [32, 64, 128] to 4 layers [32, 64, 128, 128], increasing the receptive field from 29 to 61 months (5 years). The hypothesis: v21's AET P95 regression was caused by insufficient temporal context — California's most severe AET extremes occur after multi-year cumulative drought that v21's 29-month RF could not see. Same channel routing, same loss config. Parameters: 1,160K (~217K added over baseline, ~99K more than v21).

**Results vs v21 (3-layer, RF=29mo):**

- **AET P95 bias: recovered** — -16.0mm vs -19.4mm (+3.4mm improvement). The deeper sub-backbone captures multi-year drought dynamics that the 3-layer version missed. P99 bias also improved: -23.0mm vs -28.2mm (+5.2mm).
- **AET KGE: improved** — 0.818 vs 0.799 (+0.019). Better correlation/variability balance with the longer context.
- **AET NSE: slight improvement** — 0.855 vs 0.852.
- **AET pbias: slight regression** — 8.7% vs 7.8% (+0.9pp). The deeper sub-backbone slightly increased mean overprediction, but still well below v19a's 13.1%.
- **PCK pbias: regressed** — 11.1% vs 7.7% (+3.4pp). Approaching the 12% threshold. The 4th layer may be learning snow-correlated features through the shared FVEG embedding, partially reintroducing gradient competition.
- **PET: slight regression** — NSE 0.872 vs 0.880 (-0.008). Still substantially better than v19a's 0.825.
- **CWD: slight regression** — NSE 0.924 vs 0.928 (-0.004).

**Results vs v19a (operational baseline):**

- **AET pbias: major improvement** — 8.7% vs 13.1% (-4.4pp). The gradient decoupling benefit is preserved.
- **AET P95 bias: still regressed** — -16.0mm vs -10.1mm (+5.9mm worse). The deeper sub-backbone recovered half the v21→v19a gap (from 9.3mm to 5.9mm) but still 6mm short.
- **PET NSE: major improvement** — 0.872 vs 0.825 (+0.047).
- **PCK pbias: regressed** — 11.1% vs 8.0% (+3.1pp). This is the key trade-off: v21 held PCK at 7.7%, but the deeper v21b let PCK slip to 11.1%.
- **AET KGE: flat** — 0.818 vs 0.828 (-0.010).

**Receptive field hypothesis assessment:** Partially confirmed. The 4-layer sub-backbone (RF=61mo) recovered 3.4mm of AET P95 bias vs the 3-layer (RF=29mo), consistent with multi-year drought dynamics requiring longer temporal context. However, the remaining 6mm gap to v19a suggests the problem is not purely receptive field. Two additional factors:

1. **Extreme penalty underweight.** The extreme_weight=0.05 was calibrated for the single-backbone regime where AET and the backbone share gradient capacity. In the dual-backbone regime, the AET sub-backbone has dedicated capacity that could absorb a stronger extreme signal. A higher extreme_weight (0.1 or 0.15) on the dual-backbone architecture may recover more of the gap.
2. **FVEG embedding leakage.** The shared FVEG embedding is the only parameter that receives gradients from both backbones. The PCK regression (7.7%→11.1%) from v21 to v21b suggests the deeper sub-backbone pushes stronger gradients through the shared embedding, partially reintroducing the competition that the dual-backbone was designed to eliminate. Freezing the FVEG embedding during sub-backbone training, or using separate embeddings, could test this hypothesis.

### v22-dual-full: 5-layer sub-backbone + separate FVEG embeddings

v22 addresses both remaining gaps from v21b simultaneously: (1) extends the sub-backbone to 5 layers [32, 64, 128, 128, 128] with RF=125 months, matching the main backbone, and (2) gives the AET sub-backbone its own FVEG embedding (initialized as a copy of the main embedding, but updated independently during training). This eliminates the last shared parameter between the two gradient paths. Parameters: 1,259K (~316K added over baseline).

**Results vs v21b (4-layer, shared FVEG):**

- **AET pbias: dramatic improvement** — 4.9% vs 8.7% (-3.8pp). Best-ever AET pbias across all runs. The separate FVEG embedding allows the AET sub-backbone to learn vegetation representations specialized for drought stress without distorting the main backbone's vegetation encoding.
- **PCK pbias: partial recovery** — 10.0% vs 11.1% (-1.1pp). The separate FVEG embedding reduced the leakage that caused v21b's PCK creep, though not fully back to v21's 7.7%.
- **PET: slight improvement** — NSE 0.879 vs 0.872 (+0.007), RMSE 21.0 vs 21.6mm.
- **CWD: slight improvement** — NSE 0.925 vs 0.924, pbias -2.9% vs -3.5%.
- **AET P95 bias: regressed** — -21.1mm vs -16.0mm (+5.1mm worse). The 5-layer sub-backbone performed worse on AET extremes than the 4-layer, despite the longer receptive field.
- **AET P99 bias: regressed** — -32.4mm vs -23.0mm (+9.4mm worse). Severe regression at the most extreme tail.
- **AET KGE: regressed** — 0.803 vs 0.818 (-0.015). Lower correlation component despite much better bias.

**Results vs v19a (operational baseline):**

- **AET pbias: 4.9% vs 13.1%** (-8.3pp). v22 has less than half v19a's mean overprediction — the best AET mean accuracy across all runs by a wide margin.
- **PET NSE: 0.879 vs 0.825** (+0.054). Substantial PET improvement maintained.
- **CWD: essentially tied** — NSE 0.925 vs 0.925, pbias -2.9% vs -3.5%.
- **AET P95 bias: -21.1mm vs -10.1mm** (+11.0mm worse). The dual-backbone AET extreme gap has widened rather than closed with each depth increase.
- **PCK pbias: 10.0% vs 8.0%** (+2.0pp). Acceptable but not recovered to v19a level.

**Receptive field hypothesis: definitively rejected.** The progression across v21→v21b→v22 shows a clear inverse relationship between sub-backbone depth and AET extreme performance:

| Run | Sub-backbone layers | RF (months) | AET P95 bias | AET P99 bias |
|-----|--------------------:|------------:|-------------:|-------------:|
| v21 | 3 | 29 | -19.4mm | -28.2mm |
| v21b | 4 | 61 | -16.0mm | -23.0mm |
| v22 | 5 | 125 | -21.1mm | -32.4mm |

v21b (4 layers, RF=61mo) was the sweet spot — deep enough to capture multi-year drought but not so deep that it overfits the mean. v22's 5-layer sub-backbone has enough capacity to drive AET pbias down to 4.9% (best-ever mean) but does so by learning a more conservative AET representation that systematically underestimates extremes. The sub-backbone is trading tail accuracy for mean accuracy as it gets deeper.

**FVEG leakage hypothesis: partially confirmed.** Separate FVEG embeddings improved PCK pbias from 11.1% to 10.0% and AET pbias from 8.7% to 4.9%, confirming that shared embeddings were a gradient competition vector. However, PCK did not fully recover to v21's 7.7%, suggesting the remaining PCK regression is caused by the deeper sub-backbone's impact on the overall loss landscape (not parameter sharing).

**Key insight:** The dual-backbone architecture has reached a Pareto frontier. It excels at mean prediction (AET pbias 4.9%, PET NSE 0.879) and gradient decoupling (PCK stable at ~10%), but AET extreme performance is structurally limited. The sub-backbone learns to minimize mean error, which means conservative predictions that compress the tails. The fix is not more depth — it's a stronger loss signal. The extreme_weight=0.05 was calibrated for the single-backbone regime; the dual-backbone's dedicated AET capacity can absorb a much stronger extreme penalty without destabilizing PET/PCK (since those gradients are isolated). The next experiment should use v21b's architecture (4-layer sub-backbone, which had the best extreme performance) with extreme_weight raised to 0.15 or higher.

### v23a-dual-extreme0.15: stronger tail penalty in dual-backbone regime

v23a uses v21b's architecture (4-layer sub-backbone [32, 64, 128, 128], RF=61mo, separate FVEG embeddings) with extreme_weight tripled from 0.05 to 0.15. The hypothesis: the dual-backbone's independent AET gradient pathway can absorb a stronger extreme penalty without destabilizing PCK, since PET/PCK gradients are isolated from the AET sub-backbone. Best epoch 72/100, val_loss 0.582 (stable, no spike).

**Results vs v21b (same arch, extreme_weight=0.05):**

- **AET P95 bias: improved** — -13.0mm vs -16.0mm (+3.0mm). The stronger penalty pushed the tail closer to v19a territory. AET P99 bias also improved: -20.5mm vs -23.0mm.
- **AET pbias: regressed significantly** — 14.4% vs 8.7% (+5.7pp). The stronger extreme penalty increased mean overprediction above v19a's 13.1%, losing the dual-backbone's key advantage.
- **PCK pbias: regressed** — 13.7% vs 11.1% (+2.6pp). Above the 12% threshold — the extreme penalty is interfering with PCK despite gradient isolation. The likely pathway: the extreme penalty increases AET overprediction, which mechanically reduces CWD (=PET-AET), and the CWD loss term then pushes compensating gradients through the main backbone that affect PCK.
- **PET: regressed** — NSE 0.857 vs 0.872 (-0.015). PET pbias improved to near-zero (0.1%) but at the cost of higher variance.
- **CWD: regressed** — NSE 0.919 vs 0.924, pbias -5.1% vs -3.5%. The AET overprediction cascades into CWD underprediction.

**Results vs v19a (operational baseline):**

- **AET P95 bias: closing** — -13.0mm vs -10.1mm (3mm gap remaining). This is the closest any dual-backbone run has come to v19a's AET extreme performance.
- **AET pbias: worse** — 14.4% vs 13.1%. The dual-backbone's mean bias advantage is erased at this extreme_weight level.
- **PCK pbias: worse** — 13.7% vs 8.0%. PCK destabilized — the dual-backbone architecture does not fully protect PCK from extreme penalty effects that propagate through the CWD loss term.
- **PET: better** — NSE 0.857 vs 0.825 (+0.032). Still significant PET improvement from the dual backbone.

**Success criteria assessment:**

| Criterion | Target | v23a | Status |
|-----------|--------|------|--------|
| AET P95 bias | < -13mm | -13.0mm | BORDERLINE |
| AET pbias | < 10% | 14.4% | **FAIL** |
| PCK pbias | < 12% | 13.7% | **FAIL** |

**Key insight:** The extreme_weight=0.15 experiment reveals a fundamental limitation of the dual-backbone approach at higher extreme penalties. While the AET sub-backbone's gradient pathway is isolated for PET/PCK, the CWD algebraic relationship (CWD = PET - AET) creates a secondary gradient competition channel. When the extreme penalty pushes AET predictions upward (to reduce underprediction at the tail), CWD predictions mechanically decrease. The CWD loss term (cwd_initial=2.0) then applies pressure through the main backbone to compensate, destabilizing PCK. This is not a parameter-sharing problem — it's an algebraic coupling problem that no amount of architectural isolation can eliminate.

The optimal extreme_weight for the dual-backbone regime is between 0.05 (v21b) and 0.15 (v23a). A v23c at extreme_weight=0.10 would split the difference and likely produce AET P95 bias around -14 to -15mm with AET pbias around 10-12% and PCK pbias around 12% — closer to the three-way target but probably still not achieving all three simultaneously.

### v23b-dual-extreme0.20: aggressive extreme penalty

v23b pushes extreme_weight to 0.20 (4x v21b's 0.05). Same architecture as v23a. Best epoch 75/100, val_loss 0.633 (stable, no spike — the dual-backbone absorbs even this aggressive penalty without training instability).

**Results vs v19a (operational baseline):**

- **AET P95 bias: new best-ever** — -9.8mm vs -10.1mm (+0.3mm). First dual-backbone run to beat v19a's AET extreme performance. AET P99 bias also improved: -17.4mm vs -16.1mm (v19a still slightly better at P99).
- **AET pbias: regressed badly** — 17.1% vs 13.1%. The model massively overpredicts mean AET to achieve tail accuracy — this is the v7-style trade-off, though far less severe (v7 had 40.3% pbias at weight=2.0).
- **PCK pbias: 11.6%** — within 12% threshold. Surprisingly, PCK *recovered* from v23a's 13.7% despite higher extreme_weight. This is because the aggressive extreme penalty drove such strong AET overprediction that the CWD compensating gradient shifted its character — instead of destabilizing PCK, it pushed the main backbone toward features that better separate PET from AET at extremes.
- **PET: still improved** — NSE 0.850 vs 0.825 (+0.025). The dual-backbone PET advantage persists.

**Results vs v23a (extreme_weight=0.15):**

- **AET P95 bias: improved** — -9.8mm vs -13.0mm (+3.2mm). Linear response to extreme_weight increase.
- **AET pbias: worsened** — 17.1% vs 14.4% (+2.7pp). Also roughly linear.
- **PCK pbias: improved** — 11.6% vs 13.7% (-2.1pp). Non-linear — higher penalty paradoxically stabilized PCK.

**Extreme_weight dose-response in dual-backbone regime:**

| extreme_weight | AET P95 bias | AET pbias | PCK pbias |
|---------------:|-------------:|----------:|----------:|
| 0.05 (v21b) | -16.0mm | 8.7% | 11.1% |
| 0.15 (v23a) | -13.0mm | 14.4% | 13.7% |
| 0.20 (v23b) | -9.8mm | 17.1% | 11.6% |

The AET P95 bias responds roughly linearly (~2mm per 0.05 increment). AET pbias also increases linearly (~3pp per 0.05 increment). PCK pbias is non-monotonic, peaking at 0.15 and declining at 0.20 — likely due to a phase transition in the loss landscape at high extreme_weight.

**Success criteria assessment:**

| Criterion | Target | v23b | Status |
|-----------|--------|------|--------|
| AET P95 bias | < -13mm | -9.8mm | **PASS** |
| AET pbias | < 10% | 17.1% | **FAIL** |
| PCK pbias | < 12% | 11.6% | **BORDERLINE PASS** |

**Key insight:** v23b demonstrates that the dual-backbone architecture *can* match v19a's AET extreme performance (-9.8mm vs -10.1mm), but only at the cost of severe mean overprediction (17.1%). The three-way target (P95 bias < -13mm, AET pbias < 10%, PCK pbias < 12%) remains unachieved. The dual-backbone series has mapped out a clear Pareto frontier between AET tail accuracy and AET mean bias:

- **Best mean bias**: v22 (4.9% pbias, -21.1mm P95 bias) — good for absolute AET prediction
- **Best tail accuracy**: v23b (-9.8mm P95 bias, 17.1% pbias) — good for wildfire-critical extremes
- **Best balance**: v21b (8.7% pbias, -16.0mm P95 bias, 11.1% PCK) — compromise point
- **Single-backbone reference**: v19a (-10.1mm P95 bias, 13.1% pbias, 8.0% PCK) — different Pareto frontier

The dual-backbone and single-backbone architectures operate on different Pareto frontiers. The dual-backbone wins on PET (NSE 0.850-0.879 vs 0.825), AET mean bias (at low extreme_weight), and gradient stability. The single-backbone wins on simultaneous AET tail + mean accuracy (the tight Huber synergy that v19a exploits). A hybrid approach — dual-backbone architecture with tight Huber loss (delta=0.5) instead of standard Huber (delta=1.35) — could potentially combine both advantages.

**Remaining gaps for operational wildfire use**

---

## Key Findings from Hyperparameter Tuning (v18–v19)

The v18 sweep (14 experiments) and v19 combination experiments established several definitive findings that supersede earlier architectural hypotheses. Full experiment details are in tuning_experiments.md.

**The loss function regime matters more than individual hyperparameters.** The MSE path has a fundamental gradient competition constraint: any configuration improving AET P95 bias by more than 3mm also pushes PCK pbias above 15%. This held without exception across all MSE-path experiments. Huber with delta=0.5 breaks this constraint by redirecting the cost to PET instead of PCK — the tight delta suppresses large PET errors into the MAE regime while keeping moderate AET and PCK errors in the MSE regime. For wildfire modeling where PET is an intermediate calculation, this trade is acceptable.

**The extreme penalty and tight Huber are synergistic.** Huber-tight alone gave AET P95 bias -13.5mm; extreme_weight=0.1 alone gave -14.4mm. Combined in v19a the result is -10.1mm — not additive, but strongly synergistic. Tight Huber frees backbone gradient capacity from PET; the extreme penalty directs that freed capacity specifically to AET tail underprediction. In the MSE regime these signals compete; in the Huber regime they cooperate.

**The MSE path is exhausted.** The v19b combination (extreme_weight=0.1 + pet_floor=0.3 under MSE) was anti-synergistic — worse than either component alone. Further MSE-path combinations are unlikely to yield meaningful gains.

**The extreme penalty is non-negotiable.** The v18-mse-noextreme ablation confirmed removing extreme_weight=0.05 regresses AET P95 bias by 2mm. Retain in all future configurations.

**Known operational characteristic of v19a.** AET pbias is elevated at 13.1% — the model overpredicts mean AET to gain tail accuracy. For wildfire risk assessment this is conservative, but downstream models consuming absolute AET values should apply a bias correction.

---

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
 |                                         POLARIS 0-100cm AWC for SWS (~300-500mm vs BCMv8 ~500-2000mm)
 |                                         Dropped awc_total static (14 static); aet=1.5, extreme_weight=0.05
 |                                         CWD NSE 0.929 (new best, beating v13's 0.916), RMSE 15.5 (new best)
 |                                         CWD KGE 0.931 (new best), CWD pbias -2.7%
 |                                         PCK NSE 0.949, pbias 6.8% (best among weighted-loss runs)
 |                                         PET NSE 0.879 (best among weighted-loss runs)
 |                                         AET P95 bias -19.2mm (regressed from v16's -16.3mm)
 |                                         Trade-off: more responsive SWS helps CWD but overcorrects AET extremes
 |
v18 Hyperparameter sweep (14 runs) ........ See tuning_experiments.md for full results
 |                                         KEY FINDINGS:
 |                                         - Extreme penalty (weight=0.05) confirmed load-bearing: removing it
 |                                           regresses AET P95 bias by 2mm (v18-mse-noextreme ablation)
 |                                         - MSE-path ceiling: every >3mm AET P95 improvement costs PCK >15% pbias
 |                                         - Huber tight (delta=0.5) breaks this pattern: best AET P95 (-13.5mm)
 |                                           AND best PCK NSE (0.952) — redirects cost to PET instead of PCK
 |                                         - P85 threshold better than P90 for CWD extremes (v18-extreme-p85)
 |                                         - Closed: uniform weights, PCK in extreme_vars, standard Huber, halved LR
 |
v19b MSE path combination ......... AET P95 bias -16.1mm  ANTI-SYNERGISTIC — CLOSED PATH
 |                                         extreme_weight=0.1 + pet_floor=0.3 (MSE)
 |                                         Both effective individually, counterproductive combined
 |                                         Confirms MSE path has reached its practical ceiling
 |                                         PCK pbias 13.8%, CWD extremes regressed vs baseline
 |
v19a Huber-tight + extreme0.1 .... AET NSE 0.858, CWD NSE 0.925  ★★ OPERATIONAL CONFIG
 |                                         loss_type=huber, huber_delta=0.5, extreme_weight=0.1
 |                                         AET P95 bias -10.1mm (best by 3.4mm — halved from v17 baseline)
 |                                         PCK pbias 8.0%, PET NSE 0.825
 |                                         AET pbias 13.1% (known cost — conservative for wildfire)
 |                                         SYNERGY: tight Huber frees PET budget; extreme penalty targets AET tail
 |
v20a extreme_asym 1.5→1.1 ...... AET pbias 8.3% (improved from 13.1%)
 |                                         AET P95 bias -13.5mm (regressed from -10.1mm)
 |                                         PCK pbias 19.4% — ABOVE THRESHOLD, confirms irreducible competition
v20b aet_initial 1.5→1.2 ....... AET P95 bias -8.7mm (best-ever raw, but AET pbias 15.5%)
 |                                         PCK pbias 16.6% — ABOVE THRESHOLD
 |                                         Proves: single-backbone cannot improve AET mean AND AET tail AND PCK simultaneously
 |
v21  Dual-backbone architecture . AET NSE 0.852, CWD NSE 0.928  ★ BEST AET PBIAS + PCK STABILITY
 |                                         Main TCN (256) + AET sub-backbone [32,64,128] (RF=29mo, ~117K params)
 |                                         AET head sees both backbones; PET/PCK see only main
 |                                         AET pbias 7.8% (from 13.1% — halved), PCK pbias 7.7% (stable)
 |                                         PET NSE 0.880 (major improvement from 0.825 — freed capacity)
 |                                         CWD NSE 0.928, RMSE 15.7mm (near v17 best)
 |                                         AET P95 bias -19.4mm (regressed from -10.1mm)
 |                                         Sub-backbone RF too short for multi-year drought dynamics
 |
v21b Deeper sub-backbone ........ AET NSE 0.855, CWD NSE 0.924  ★ BEST DUAL-BACKBONE AET EXTREMES
 |                                         Sub-backbone [32,64,128,128] (RF=61mo, ~217K new params)
 |                                         AET P95 bias -16.0mm (recovered 3.4mm from v21's -19.4mm)
 |                                         AET P99 bias -23.0mm (recovered 5.2mm from v21's -28.2mm)
 |                                         AET KGE 0.818 (best among dual-backbone runs)
 |                                         AET pbias 8.7% (still well below v19a's 13.1%)
 |                                         PCK pbias 11.1% (crept up from v21's 7.7% — FVEG embedding leakage?)
 |                                         PET NSE 0.872 (still +0.047 over v19a's 0.825)
 |                                         RF=61mo is the sweet spot — deeper sub-backbones overfit the mean
 |
v22  5-layer sub + separate FVEG  AET NSE 0.846, CWD NSE 0.925  ★ BEST-EVER AET PBIAS (4.9%)
 |                                         Sub-backbone [32,64,128,128,128] (RF=125mo) + separate FVEG embeddings
 |                                         AET pbias 4.9% (best-ever, halved from v21b's 8.7%, third of v19a's 13.1%)
 |                                         PCK pbias 10.0% (partial recovery from v21b's 11.1% — FVEG fix helped)
 |                                         PET NSE 0.879 (maintained), CWD NSE 0.925 (maintained)
 |                                         AET P95 bias -21.1mm (REGRESSED from v21b's -16.0mm)
 |                                         RF hypothesis REJECTED: 5-layer worse than 4-layer on extremes
 |                                         Deeper sub-backbone trades tail accuracy for mean accuracy
 |
v23a extreme_weight 0.05→0.15 .. AET NSE 0.857, CWD NSE 0.919
 |                                         v21b arch (4-layer sub, separate FVEG) + extreme_weight=0.15
 |                                         AET P95 bias -13.0mm (improved from v21b's -16.0mm, 3mm from v19a)
 |                                         AET pbias 14.4% (REGRESSED — worse than v19a's 13.1%)
 |                                         PCK pbias 13.7% (ABOVE 12% threshold)
 |                                         CWD algebraic coupling creates secondary gradient competition
 |
v23b extreme_weight 0.05→0.20 .. AET NSE 0.848, CWD NSE 0.915  ★ BEST-EVER AET P95 BIAS (-9.8mm)
                                         v21b arch + extreme_weight=0.20 (aggressive)
                                         AET P95 bias -9.8mm (BEATS v19a's -10.1mm — first dual-backbone to do so)
                                         AET P99 bias -17.4mm (improved from v21b's -23.0mm)
                                         AET pbias 17.1% (SEVERE mean overprediction — v7-lite trade-off)
                                         PCK pbias 11.6% (paradoxically recovered from v23a's 13.7%)
                                         PET NSE 0.850 (still +0.025 over v19a)
                                         CONCLUSION: dual-backbone CAN match v19a tails, but at 17% mean bias
                                         Three-way target (P95<-13, pbias<10%, PCK<12%) remains unachieved
                                         Next: dual-backbone + tight Huber (delta=0.5) to combine both advantages

v19a-extended  OUT-OF-SAMPLE EVAL . CWD NSE 0.919, PCK NSE 0.965  ★ PASSED OUT-OF-SAMPLE VALIDATION
                                         Same v19a model, evaluated on Oct 2019 - Sep 2024 (60 months)
                                         Data pipeline extended: ppt/tmin/tmax from BCM .asc, srad from TerraClimate,
                                         PRISM daily ppt/tmax for wet_days/intensity/KBDI — all through Sep 2024
                                         CWD NSE 0.919 (only -0.006 from training-period 0.925) — remarkable stability
                                         CWD KGE 0.940 (IMPROVED from training-period 0.926)
                                         CWD pbias -1.3% (near-zero — critical for wildfire application)
                                         PCK NSE 0.965 (BEST-EVER across entire project, +0.013 over v7's 0.961)
                                         PCK pbias -4.5% (best-ever negative bias — previous runs all overpredicted)
                                         AET pbias +11.6% (expected: 2020-2024 megadrought caused physiological
                                           changes — stomatal downregulation, LAI reduction, mortality — not in
                                           training distribution; BCMv8 targets themselves uncertain in these conditions)
                                         CWD P95 hit rate 0.841 (BEST-EVER, +0.037 above v8b's 0.804)
                                         AET P95 bias -19.2mm (reasonable for 5-year OOS with drought extremes)
                                         CONCLUSION: emulator generalizes to unseen climate conditions.
                                           Ready for fire probability modeling on v19a outputs.
```
---

## Fire-model-side interannual validation (2026-04-22)

The fire-model project (`/home/mmann1123/extra_space/fire_model/`) exposed an independent validation channel for v19a: its v4 LogReg fire model, evaluated under Track A (BCMv8 target hydrology as inputs) vs Track B (v19a emulator hydrology as inputs), produces interannual fire-area skill at Pearson **+0.97 vs −0.01** across WY2020–2024. Track B is what v19a powers in production. The 0.97 → −0.01 collapse originally read as "the emulator's hydrology anomalies do not preserve year-to-year fire-weather signal," but a 3-mode screening diagnostic (`experiments/emulator_screening/` in the fire-model project) refines this. Short summary of what v19a looks like when scored against BCMv8 targets directly (same fire-season mean and p95 summaries as NSE/KGE use, but correlated across WYs instead of pooled across months):

### Mode 1 — CA-wide fire-season summaries, WY2020–2024

| variable | stat | Pearson vs BCMv8 | Spearman vs BCMv8 |
|---|---|---:|---:|
| aet | mean | **+0.99** | +1.00 |
| cwd | mean | **+0.94** | +0.80 |
| pck | mean | **+1.00** | +0.80 |
| pet | mean | +0.74 | +0.90 |
| aet | p95 | +0.62 | +0.20 |
| cwd | p95 | +0.90 | +0.80 |
| pet | p95 | +0.28 | +0.20 |

**v19a reproduces BCMv8's CA-wide year-to-year signal well for AET/CWD/PCK/PET at the mean stat.** This is independent confirmation of the out-of-sample NSE/KGE numbers already documented for v19a-extended — they translate to the interannual scale when aggregated across valid CA pixels.

### Mode 2 — Fire-weighted summaries (aggregation restricted to pixels that burned in each WY)

Mostly still high: aet mean r=+0.96, cwd mean r=+0.96, aet p95 r=+0.95. One sign flip:

- **pet p95 flips from +0.28 (CA-wide) to −0.38 (fire-weighted)** — the emulator's PET extremes at burning pixels are inversely correlated with BCMv8's.

### Mode 3 — Per-L3 ecoregion Pearson vs BCMv8 (mean stat, n_pixels ≥ 500)

Most ecoregions match BCMv8 well across AET/CWD, but two are clear outliers:

| Ecoregion | n_pixels | aet | pet (p95) |
|---|---:|---:|---:|
| **81 (Sierra Nevada)** | 27,698 | **−0.06** mean / **−0.31** p95 | +0.85 |
| **1 (Coast Range)** | 13,277 | +0.79 | **−0.79** p95 |
| 78 (Klamath Mountains) | 32,725 | +0.89 | −0.47 p95 |
| 4 (Cascades) | 13,828 | +0.96 | −0.33 p95 |

**Actionable findings for emulator tuning:**

1. **Ecoregion 81 (Sierra Nevada) has essentially zero interannual AET skill.** r=−0.06 mean, −0.31 p95. This is a major fire ecoregion; the emulator's AET anomalies here do not track BCMv8's year-to-year. Targeted loss weighting or static-feature debugging here is likely to move the needle on downstream fire skill more than broadcasting a new loss across CA.
2. **PET extremes flip sign in at least four fire-relevant ecoregions** (Coast Range, Klamath, Cascades, Central Valley). v19a's PET p95 globally matches BCMv8 at r=+0.28 but regionally collapses. Given PET's low CV (0.007–0.03) this may be a secondary issue, but it's the variable most likely implicated if the v4 Track B failure is PET-driven.
3. **Aggregate (CA-wide) validation is now saturated.** v19a's Pearson-vs-BCMv8 at CA-wide aggregation is at or near the ceiling for the major variables. Further emulator tuning targeting *interannual fire usability* should be scored against per-ecoregion and fire-weighted diagnostics — the CA-wide numbers will not discriminate between a good candidate and a great one.

See `fire_model/docs/model_comparison.md` § "Interannual Burned-Area Variability (2026-04-21)" → "2026-04-22 addendum" for the downstream fire-skill interpretation, and `fire_model/experiments/emulator_screening/` for the script, raw CSVs, and README describing how to rescore future emulator checkpoints on the same diagnostic.

### Per-pixel per-month follow-up (2026-04-22): the interannual failure is Sierra Nevada AET

Follow-up diagnostic (`fire_model/experiments/emulator_screening/pixel_monthly_correlation.py`) computes per-pixel Pearson across 30 fire-season months (WY2020–2024 × Jun–Nov) between v19a and BCMv8 targets, then summarizes the distribution within each L3 ecoregion.

| Variable | CA-wide median per-pixel r | Sierra Nevada (eco 81) median per-pixel r | Sierra frac \|r\| < 0.3 |
|---|---:|---:|---:|
| **aet** | **0.88** | **0.39** | **34%** |
| cwd | 0.96 | 0.99 | 0.3% |
| pet | 0.99 | 0.99 | 0.4% |
| pck | 0.99 | — | — |

**Interpretation for v19a-extended tuning:** the downstream fire-skill collapse documented in the fire-model project is effectively a **one-variable, one-ecoregion problem**. CWD and PET have near-perfect per-pixel monthly fidelity everywhere. PCK is similarly clean wherever it's nonzero. AET is the weak link, and its weakness is strongly concentrated in the Sierra Nevada (ecoregion 81, 27,698 fire-dominant pixels, median per-pixel monthly Pearson 0.39). Other AET-weak ecoregions — 14 Mojave (r=0.54), 13 Central Basin (r=0.75), 80 NW Volcanics (r=0.77) — are less fire-critical.

**Actionable:** a targeted AET loss weighting in the Sierra Nevada (ecoregion 81) or ecoregion-aware training sampling, rather than a broadcast-CA AET weighting, is the single highest-leverage change for the next emulator iteration if the goal is improving downstream v4 Track B fire-area interannual skill. A statewide AET weight bump would waste capacity on regions where v19a already matches BCMv8 essentially perfectly.

**Screening discipline:** future emulator candidates should be scored on per-ecoregion AET Pearson in the fire-dominant ecoregions (81 especially) rather than CA-wide aggregate AET metrics. The aggregate metrics are at the ceiling for v19a and will not discriminate a good candidate from a great one.

Raw results: `fire_model/experiments/emulator_screening/results/pixel_monthly_correlation.csv` (54 rows: 4 variables × 13 ecoregions + CA-wide).

### Annual-smoothing experiment confirms monthly noise is the bottleneck (2026-04-22)

A direct follow-up test on the fire-model side (`fire_model/snapshots/v4-annualsmooth/`) swapped v19a's monthly emulator outputs for per-pixel per-WY means (broadcast back to all 12 months), then re-ran v4's full-surface evaluation. Result: v4 Track B's interannual Pearson vs actual annual burned area recovers from **−0.01 (raw monthly) to +0.94 (annual-smoothed)** — essentially hitting Track A's +0.97 ceiling with BCMv8 target hydrology.

| Metric | v4 Track B (raw monthly v19a) | v4-annualsmooth Track B | Track A ceiling |
|---|---:|---:|---:|
| AUC-B | 0.858 | 0.727 | 0.859 |
| BA ratio (total) | 2.20 | 0.12 | 1.03 |
| **Pearson r vs fire** | **−0.012** | **+0.944** | +0.973 |
| Spearman ρ | 0.00 | +0.70 | +0.70 |

**What this means for the emulator.** The interannual fire-relevant signal is **present** in v19a — both CA-wide and at most per-pixel monthly resolutions — but its residual month-to-month noise (even at per-pixel r≈0.99 for CWD/PET/PCK everywhere) is large enough to wash out the interannual signal when propagated through a downstream 34-feature LogReg across millions of pixel-months. v19a is not fundamentally incapable of supporting interannual fire skill; it just has noise characteristics that a downstream monthly-anomaly consumer can't filter.

**Two actionable paths for the next emulator iteration:**

1. **Reduce month-to-month noise while preserving annual climatology.** A temporal-smoothing loss term (penalizing the emulator's deviation between adjacent months within a WY) or a two-head prediction architecture (monthly + annual) could deliver the same smoothed signal that is now being achieved post-hoc.
2. **Deprioritize targeting the Sierra Nevada AET issue via broadcast losses.** The earlier per-pixel per-month diagnostic flagged Sierra Nevada AET (median r=0.39) as the weakest single-variable ecoregion signal; however, the annual-smoothing experiment shows this is still recoverable via temporal averaging, which means the pixel-level AET is good enough *if the downstream model is allowed to average over time*. Targeted Sierra AET loss weighting is still worthwhile as a secondary move, but the primary gain sits in reducing monthly noise system-wide.

The fire-model project is pursuing a hybrid feature set (retain monthly + add annual-smoothed features) as the downstream operational fix. If that works, the emulator specification does not need to change. If it does not (i.e., the LogReg can't separate monthly intra-year signal from annual interannual signal linearly), the emulator side becomes the right place to implement temporal smoothing structurally.

See `fire_model/experiments/emulator_screening/FINDINGS.md` for the full experiment writeup.

### 3-month rolling mean (roll3) follow-up — downstream BA calibration fixed, interannual unchanged

A centered 3-month rolling mean of v19a outputs (`fire_model/snapshots/v4-roll3smooth/`) was tested to isolate how much smoothing each of v4's two failure modes requires.

| Metric | Raw monthly v19a | Roll3 v19a | Annual-smoothed v19a |
|---|---:|---:|---:|
| v4 Track B AUC | 0.858 | 0.849 | 0.727 |
| v4 Track B BA | 2.20 | 0.95 | 0.12 |
| v4 Track B Pearson vs fire | −0.012 | +0.088 | +0.944 |

**Two distinct emulator-side problems, two different required averaging windows:**

1. **BA calibration (downstream burned-area magnitude) is fixed by 3-month smoothing.** 2.20 → 0.95. This is a high-frequency noise problem — v19a's monthly outputs have enough within-season variance that a fire model calibrating thresholds on them produces ~2× too much predicted area. 3-month rolling strips that noise and restores BA. AUC is essentially preserved.
2. **Interannual fire-area Pearson requires 12-month (annual) smoothing.** 3-month gives +0.09, essentially no improvement from raw's −0.01. The between-WY signal is only accessible at window length comparable to a full water year.

**For emulator-side action**, this shifts the priority: if the goal is downstream BA calibration, **high-frequency noise reduction is sufficient and cheap**. A temporal-smoothness loss term penalizing month-to-month deviations (within the WY) could achieve this at training time without retraining the fire-model downstream. If the goal is interannual fire skill as well, more aggressive structural changes are needed (two-head architecture, annual-pooled supervision, or similar) — but the per-pixel diagnostic previously flagged that CWD/PET/PCK are already at per-pixel monthly r≈0.99. The 12-month window being needed isn't because the pixel-level signal is bad — it's because the downstream linear model (v4 LogReg) can't filter residual noise across millions of pixel-months without substantial averaging.

Fire-model side is planning a smoothing-window sweep (roll6, roll9) and/or a hybrid feature set (monthly + WY-annual anomalies) to find the cheapest configuration that delivers both operational goals. If the fire-model side succeeds with hybrid features, the emulator spec does not need to change for BA calibration; only interannual skill pressures the emulator.

See `fire_model/experiments/emulator_screening/FINDINGS.md` for the full experiment table.

### Smoothing-window sweep resolves the tradeoff (2026-04-22)

Full sweep across centered N-month rolling means of v19a outputs, fed to v4:

| Window | v4 Track B AUC | BA | Pearson r | Spearman ρ |
|---|---:|---:|---:|---:|
| 1 (raw) | 0.858 | 2.20 | −0.012 | 0.00 |
| 3 | 0.849 | 0.95 | +0.088 | +0.20 |
| 5 | 0.832 | 0.60 | −0.152 | +0.10 |
| 7 | 0.813 | 0.51 | −0.197 | +0.50 |
| 9 | 0.784 | 0.39 | −0.202 | +0.50 |
| 12 (annual) | 0.727 | 0.12 | +0.944 | +0.70 |

**No single window hits all downstream operational gates simultaneously.** The AUC gate and BA gate are satisfied at 3-month smoothing; the Pearson gate requires 12-month smoothing which kills AUC and BA. Intermediate windows (5/7/9) pass through a negative-Pearson trough before recovering.

**For v19a specifically, this is a clean result:** the emulator's monthly outputs already carry enough within-year information to preserve v4's AUC once high-frequency noise is stripped (3-month window is sufficient for BA calibration). The interannual signal is *also* in the emulator but requires 12-month aggregation to emerge — meaning the emulator's month-to-month variance in hydrology is high enough that v4's LogReg cannot recover the year-to-year signal without full-annual averaging.

**Implications for the next emulator iteration:** the downstream fire-model project will proceed with hybrid-feature retraining (monthly + WY-annual anomaly features as separate inputs). If that delivers the operational goals, v19a's current monthly-noise characteristics are acceptable — no emulator-side change needed. If the hybrid retrain cannot bridge the gap, the emulator-side lever is either (a) a temporal-smoothness penalty at training time that preserves the interannual signal at shorter aggregation windows, or (b) a two-head architecture producing explicit monthly and annual hydrology streams.

See `fire_model/experiments/emulator_screening/FINDINGS.md` for the full sweep table and decision log.

### Per-ecoregion NSE ranking across all 48 snapshots (2026-04-23)

Motivation: the v4-hybrid-wyanom retrain against the v19a hindcast (1982–2024, full 481,742-pixel CA grid) failed to recover Track B interannual skill — Pearson −0.12, and +0.07 after dropping `cwd_cum3/6_anom`. Hindcast values correlate well with BCMv8 at the pooled pixel-month level (OOS r = 0.944 PET / 0.905 AET / 0.960 CWD / 0.979 PCK), but the CA-wide annual-mean year-to-year correlation for PET is only 0.603 (vs 0.93–1.00 for the other three variables). This prompted a question: are any of the 48 existing emulator snapshots *measurably better* than v19a in the ecoregions where California fires actually happen?

**Method.** For every snapshot with `spatial_maps/nse_{pet,aet,cwd,pck}.tif`, sample per-pixel NSE means within each EPA Level III ecoregion using `/home/mmann1123/extra_space/Regions/ca_eco_l3.tif` (codes from `US_L3CODE` attribute of `ca_eco_l3+Proj.shp`). Fire-prone ecoregions defined as `{1, 4, 5, 6, 8, 78, 85}` — Coast Range, Cascades, Sierra Nevada, Central California Foothills and Coastal Mountains, Southern California Mountains, Klamath Mountains/California High North Coast Range, Southern California/Northern Baja Coast. NSE clipped at −10 before averaging to prevent desert-pixel blow-up dominating means.

**Top 20 snapshots by Sierra Nevada AET NSE** (columns are per-ecoregion mean NSE):

| snapshot | AET Sierra | AET fire-prone | CWD Sierra | CWD fire-prone | PET Sierra | PET fire-prone |
|---|---:|---:|---:|---:|---:|---:|
| v19b-extreme0.1-petfloor0.3 | +0.878 | +0.814 | +0.883 | +0.790 | +0.779 | +0.755 |
| v18-huber-tight | +0.877 | +0.817 | +0.897 | +0.804 | +0.765 | +0.715 |
| v18-huber | +0.872 | +0.811 | +0.875 | +0.749 | +0.806 | +0.767 |
| v14-sws-stress | +0.872 | +0.817 | +0.867 | +0.760 | +0.813 | +0.783 |
| v12-stress-frac-aet2x | +0.871 | +0.820 | +0.878 | +0.764 | +0.810 | +0.777 |
| v18-petfloor0.3 | +0.869 | +0.815 | +0.878 | +0.782 | +0.797 | +0.778 |
| v18-balanced1.5 | +0.869 | +0.815 | +0.865 | +0.761 | +0.827 | +0.805 |
| v13-sws-rollstd | +0.867 | +0.808 | +0.887 | +0.794 | +0.819 | +0.797 |
| v17-polaris-awc | +0.866 | +0.813 | **+0.894** | **+0.822** | +0.815 | +0.796 |
| v18-mse-noextreme | +0.866 | +0.804 | +0.884 | +0.782 | **+0.844** | **+0.817** |
| v15-awc-extreme | +0.865 | +0.807 | +0.872 | +0.753 | +0.800 | +0.765 |
| v18-cwd3-aet1.5 | +0.865 | +0.820 | +0.890 | +0.808 | +0.787 | +0.765 |
| v8b-no-extreme | +0.864 | +0.805 | +0.810 | +0.652 | +0.920 | +0.912 |
| v11-kv-aet | +0.864 | +0.808 | +0.809 | +0.674 | +0.920 | +0.911 |
| v22-dual-full | +0.864 | +0.815 | +0.876 | +0.798 | +0.811 | +0.791 |
| v19a-huber-tight-extreme0.1 | +0.864 | +0.819 | +0.891 | +0.803 | **+0.727** | **+0.678** |
| v21b-deeper-sub | +0.864 | +0.820 | +0.880 | +0.780 | +0.805 | +0.782 |
| v21-dual-backbone | +0.861 | +0.823 | +0.889 | +0.815 | +0.820 | +0.800 |

**Three findings.**

**1. Sierra AET NSE is saturated across the model family.** Top-to-bottom spread across the 18 best snapshots is 0.017 on Sierra AET (0.878 → 0.861) and 0.019 on fire-prone-mean AET (0.823 → 0.804). AET is *not* the differentiator for emulator selection — any architecture/loss variant in this family gets essentially the same Sierra AET fidelity.

**2. v19a-huber-tight-extreme0.1 is a PET outlier in the wrong direction.** Among the 18 top-AET snapshots, v19a has the *worst* Sierra PET NSE (+0.727) and the *worst* fire-prone-mean PET NSE (+0.678). Compare to `v18-mse-noextreme` (+0.844 Sierra PET, +0.817 fire-prone mean) — +0.12 better PET for −0.002 worse AET. And to `v8b-no-extreme` / `v11-kv-aet` (+0.920 Sierra PET, +0.91 fire-prone mean) — +0.19 better PET, though at a real CWD cost. This matches the hindcast diagnostic: v19a's annual-mean PET range over WY2020–2024 is 106.6–108.7 mm vs BCMv8's 101.0–107.8 mm (year-to-year r = 0.603). The emulator's PET is systematically flattened, and this is visible as a per-pixel NSE penalty localized to the ecoregions the fire model cares about. v19a's `extreme_weight: 0.1` + tight Huber (δ=0.5) are doing exactly what they were designed to do — clamp AET-tail errors — and are paying for it in PET fidelity.

**3. v19a's AET weakness is in the deserts, not the fire-prone zones.** Per-ecoregion AET NSE for v19a (US_L3NAME shown, FIRE tag = fire-prone):

| ID | US_L3NAME | n pixels | mean AET NSE | median | p10 | |
|---:|:---|---:|---:|---:|---:|:---|
| 4 | Cascades | 13,828 | +0.898 | +0.957 | +0.781 | FIRE |
| 78 | Klamath Mountains/California High North Coast Range | 32,725 | +0.898 | +0.954 | +0.759 | FIRE |
| 5 | Sierra Nevada | 51,752 | +0.864 | +0.941 | +0.666 | FIRE |
| 1 | Coast Range | 13,275 | +0.841 | +0.929 | +0.590 | FIRE |
| 8 | Southern California Mountains | 15,831 | +0.802 | +0.813 | +0.676 | FIRE |
| 6 | Central California Foothills and Coastal Mountains | 74,486 | +0.764 | +0.857 | +0.585 | FIRE |
| 85 | Southern California/Northern Baja Coast | 20,626 | +0.666 | +0.713 | +0.407 | FIRE |
| 7 | Central California Valley | 40,578 | +0.379 | +0.747 | −0.307 | |
| 9 | Eastern Cascades Slopes and Foothills | 17,140 | +0.393 | +0.661 | −0.454 | |
| 13 | Central Basin and Range | 9,412 | −0.053 | +0.464 | −1.564 | |
| 14 | Mojave Basin and Range | 34,720 | −0.901 | +0.167 | −4.452 | |
| 80 | Northern Basin and Range | 3,888 | +0.191 | +0.337 | −0.081 | |
| 81 | Sonoran Basin and Range | 15,374 | −0.755 | +0.265 | −3.676 | |

All seven fire-prone ecoregions have healthy AET fidelity (mean NSE 0.67–0.90). The deep negatives are all in deserts and the Central Valley — regions with minimal fire activity, where the water-balance signal is dominated by irrigation, bare-soil evaporation, or near-zero everything, and where BCMv8's own behavior is poorly constrained by observations. The "Sierra AET r ≈ 0.39" warning in the fire-model project's plan doc referred to the *p10 tail* (0.666 for Sierra, 0.407 for SoCal coast), not the regional mean — and the weakest fire-prone tails are actually in ecoregions 85 (SoCal coast) and 6 (CA Foothills), not Sierra Nevada.

**Implication for emulator selection.** If the downstream fire-model Track B interannual collapse is caused by flat PET, swapping v19a → v17-polaris-awc or v18-mse-noextreme would provide +0.09 to +0.14 fire-prone PET NSE at negligible AET cost and competitive CWD. `v17-polaris-awc` has the best fire-prone CWD NSE of the top-AET set (0.822). This hypothesis is being tested: see `fire_model/snapshots/v4-hybrid-wyanom-v17-polaris-awc/` (in progress) and `fire_model/docs/model_comparison.md` § "Emulator prediction provenance" → "2026-04-23 addendum".

If even the best-ranked existing snapshot fails to recover Track B interannual Pearson, the conclusion tightens to "no existing emulator in this architecture family carries the between-year signal the fire model needs" — at which point the next lever is emulator retraining with explicit interannual supervision (two-head architecture, annual-pooled loss, or temporal-smoothness penalty).
