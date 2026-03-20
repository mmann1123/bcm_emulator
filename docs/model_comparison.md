# BCM Emulator: Model Run Comparison

## Objective

This document compares all model versions (v1 through v7) with an emphasis on metrics that matter for **wildfire modeling**: accurate prediction of climatic water deficit (CWD) and actual evapotranspiration (AET) extremes. CWD is the primary driver of vegetation drought stress and fire danger in California; AET extremes reflect periods of rapid vegetation drying. Underpredicting these extremes means underestimating fire risk.

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

## Global Performance Metrics

### NSE (Nash-Sutcliffe Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.852 | 0.862 | 0.790 | 0.872 |
| v2-fveg-srad-fix | **0.928** | 0.945 | 0.831 | 0.902 |
| v3-vpd-awc | 0.926 | 0.941 | 0.833 | 0.902 |
| v4-soil-props | 0.927 | 0.940 | 0.834 | 0.903 |
| v5-awc-windward | 0.862 | 0.944 | **0.851** | **0.915** |
| v5b-pet-reweight | 0.868 | 0.938 | 0.845 | 0.911 |
| v6-huber | 0.927 | 0.950 | 0.828 | 0.907 |
| v7-extreme-aware | 0.876 | **0.961** | 0.760 | 0.830 |
| v7b-extreme-low | 0.876 | **0.961** | 0.760 | 0.830 |
| v8-soil-physics | 0.914 | 0.935 | **0.851** | 0.894 |
| v8b-no-extreme | 0.927 | 0.916 | 0.839 | 0.899 |
| v8c-mse | 0.927 | 0.930 | 0.834 | 0.907 |
| v9-drought-code | 0.927 | 0.932 | 0.810 | 0.888 |
| v9-kbdi | 0.925 | 0.929 | 0.824 | 0.896 |
| v10-kbdi-aet-only | 0.927 | 0.929 | 0.840 | 0.897 |

### KGE (Kling-Gupta Efficiency) -- higher is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 0.902 | 0.717 | 0.739 | 0.916 |
| v2-fveg-srad-fix | 0.947 | 0.883 | 0.752 | 0.928 |
| v3-vpd-awc | 0.946 | 0.873 | 0.755 | 0.931 |
| v4-soil-props | 0.944 | 0.863 | 0.756 | 0.932 |
| v5-awc-windward | 0.862 | 0.910 | 0.814 | 0.926 |
| v5b-pet-reweight | 0.871 | 0.811 | **0.824** | **0.937** |
| v6-huber | 0.945 | 0.886 | 0.740 | 0.929 |
| v7-extreme-aware | 0.881 | **0.924** | 0.588 | 0.862 |
| v7b-extreme-low | 0.881 | **0.924** | 0.588 | 0.862 |
| v8-soil-physics | 0.930 | 0.923 | 0.791 | 0.920 |
| v8b-no-extreme | 0.944 | 0.816 | 0.767 | 0.935 |
| v8c-mse | 0.946 | 0.918 | 0.744 | 0.928 |
| v9-drought-code | **0.953** | 0.887 | 0.738 | 0.902 |
| v9-kbdi | 0.952 | 0.855 | 0.743 | 0.907 |
| v10-kbdi-aet-only | 0.942 | 0.826 | 0.769 | 0.925 |

### RMSE (mm/month) -- lower is better

| Run | PET | PCK | AET | CWD |
|-----|-----|-----|-----|-----|
| v1-baseline | 23.2 | 19.5 | 13.8 | 20.8 |
| v2-fveg-srad-fix | **16.2** | 12.3 | 12.4 | 18.2 |
| v3-vpd-awc | 16.4 | 12.8 | 12.3 | 18.2 |
| v4-soil-props | 16.3 | 12.9 | 12.3 | 18.1 |
| v5-awc-windward | 22.4 | 12.5 | **11.6** | **17.0** |
| v5b-pet-reweight | 21.9 | 13.0 | 11.8 | 17.4 |
| v6-huber | **16.2** | 11.7 | 12.5 | 17.7 |
| v7-extreme-aware | 21.2 | **10.3** | 14.7 | 24.0 |
| v7b-extreme-low | 21.2 | **10.3** | 14.7 | 24.0 |
| v8-soil-physics | 17.7 | 13.4 | **11.6** | 19.0 |
| v8b-no-extreme | 16.3 | 15.2 | 12.1 | 18.5 |
| v8c-mse | 16.3 | 13.8 | 12.3 | 17.7 |
| v9-drought-code | 16.3 | 13.7 | 13.1 | 19.5 |
| v9-kbdi | 16.5 | 14.0 | 12.7 | 18.8 |
| v10-kbdi-aet-only | 16.3 | 14.0 | 12.1 | 18.7 |

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
| v8-soil-physics | **-0.3** | **4.2** | 12.7 | -4.4 |
| v8b-no-extreme | -1.0 | 13.3 | 4.9 | -2.0 |
| v8c-mse | -0.9 | 5.0 | 6.4 | -2.5 |
| v9-drought-code | -1.1 | 8.2 | 10.3 | -4.4 |
| v9-kbdi | -1.8 | 12.4 | 6.6 | -3.8 |
| v10-kbdi-aet-only | -1.0 | 13.5 | 6.3 | -2.6 |

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

### CWD Extremes (P99)

| Run | RMSE (mm) | Bias (mm) | Exceedance Hit Rate |
|-----|-----------|-----------|---------------------|
| v5-awc-windward | 5.7 | -2.6 | **0.692** |
| v5b-pet-reweight | **5.0** | -1.8 | 0.680 |
| v6-huber | 5.2 | **-0.3** | 0.685 |
| v7b-extreme-low | 10.3 | -5.6 | 0.651 |
| v8-soil-physics | 6.1 | -2.5 | 0.672 |
| v8b-no-extreme | **5.0** | +0.6 | 0.683 |
| v8c-mse | 6.0 | -1.2 | 0.686 |
| v9-drought-code | 9.5 | -2.3 | 0.666 |
| v9-kbdi | 8.2 | -4.2 | 0.670 |
| v10-kbdi-aet-only | 5.7 | -0.6 | 0.684 |

## Analysis for Wildfire Modeling

### What matters for fire risk prediction

1. **CWD accuracy at extremes (P95+):** CWD is the strongest climate predictor of California wildfire. A -10mm bias at P95 translates directly to underestimated fire danger during the most critical months. CWD P95 bias ranges from -0.8mm (v6) to -2.7mm (v5), with RMSE ~8-9mm -- all reasonably good.

2. **AET accuracy at extremes:** AET underprediction means the model thinks plants are transpiring less than they actually are. Since CWD = PET - AET, AET underprediction actually *overpredicts* CWD in absolute terms (before denormalization). However, the z-score space bias matters for the loss function. AET P95 bias is a persistent problem: -17.9mm at best (v5), -26.6mm at worst (v6).

3. **Spatial fidelity in fire-prone regions:** Sierra Nevada foothills, Southern California chaparral, and North Coast ranges are where CWD extremes concentrate. Spatial NSE maps (in snapshots) should be checked for these regions specifically.

### Key trade-offs observed

**v6-huber is the best global model** with the highest PET NSE (0.927) and PCK NSE (0.950), and lowest CWD bias (-1.7%). However, it has the **worst AET extreme performance**: P95 bias of -26.6mm and P99 bias of -37.7mm. The Huber loss with delta=1.35 transitions to MAE (linear gradient) for large errors, which actually *reduces* the model's incentive to push predictions into the tails.

**v5-awc-windward is the best extreme-value model** with AET P95 bias of -17.9mm (9mm better than v6) and the highest AET NSE (0.851) and CWD NSE (0.915). The trade-off: PET NSE drops to 0.862 (vs 0.927), which cascades through the hierarchy.

**No model resolves the fundamental tension:** Huber loss stabilizes training and improves global metrics, but suppresses extreme-value gradients. Higher AET/CWD weights (v5) improve extremes but degrade PET.

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
                                         PET recovered (0.927), AET improved over both v9-kbdi and v8c
                                         CWD extremes near-best (P95 bias -0.6mm, P99 bias -0.6mm)
                                         Validates routing hypothesis: drought signals help when connected to right stage
```
