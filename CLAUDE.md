# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the `deep_field` conda environment:

```bash
conda run -n deep_field python <script.py>
```

## Commands

```bash
# Data pipeline (individual steps or "all")
conda run -n deep_field python prepare_data.py --steps sciencebase pck_gap prism_daily srad topo_solar fveg soil zarr

# Training (always tag with --run-id and --notes)
conda run -n deep_field python train.py --run-id v3-vpd-awc --notes "Added VPD dynamic input + POLARIS AWC static input"

# Evaluation
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v3-vpd-awc

# Compare snapshots
conda run -n deep_field python -c "from src.utils.snapshot import compare_snapshots; compare_snapshots('v2-fveg-srad-fix', 'v3-vpd-awc', project_root='.')"
```

No test suite, linter, or CI configured. Evaluation metrics (NSE, KGE, RMSE) and spatial NSE maps serve as the validation mechanism.

## Architecture

**Three-stage hierarchical TCN** predicting monthly water balance variables on a 1km California grid (EPSG:3310, 1209x941 pixels).

### Data flow

```
prepare_data.py (downloads + zarr build)
    → data/bcm_dataset.zarr
        /inputs/dynamic   (T, 10, H, W)  - monthly climate
        /inputs/static    (10, H, W)     - terrain + soil + vegetation
        /targets/{pet,pck,aet,cwd}  (T, H, W)
        /norm/*           - per-channel z-score stats

train.py → BCMPixelDataset (pixel time-series from zarr)
    → BCMTrainer (AMP, cosine schedule, teacher forcing curriculum)
        → checkpoints/best_model.pt

evaluate.py → autoregressive inference (tf_ratio=0.0)
    → outputs/metrics.json, spatial_maps/nse_*.tif
    → snapshots/{run_id}/ (frozen config + model + metrics)
```

### Model (src/models/bcm_model.py)

```
Input: (B, 19, T) → _prepend_fveg → (B, 27, T)
  19 = 10 dynamic + 9 continuous static
  27 = 19 + 8 FVEG embedding

Stage 1: TCNBackbone - 5 causal dilated levels [64,128,128,256,256], dilations [1,2,4,8,16]
          Receptive field: 125 months. Output: (B, 256, T)

Stage 2: PET head (256→1, softplus) + PCK head (256→1, softplus)

Stage 3: AET head ([256+PET+PCK]=258 → 64 → 1)
          AET ≤ PET enforced post-denormalization, not in the head

CWD = PET - AET (algebraic, no parameters)
```

Teacher forcing: channels 7 (pck_prev) and 8 (aet_prev) are swapped between ground-truth and model predictions via `BCMEmulator.PCK_PREV_IDX` / `AET_PREV_IDX`. Curriculum: 100% GT for first half of training, then linear ramp to 100% predicted.

### Input channels

**Dynamic (10):** ppt, tmin, tmax, wet_days, ppt_intensity, srad, snow_frac, pck_prev, aet_prev, vpd

**Static (10):** elev, topo_solar, lat, lon, ksat, sand, clay, awc, windward_index, fveg_class_id
- Channels 0-8 are continuous (z-score normalized)
- Channel 9 (FVEG) is categorical (integer class ID, not normalized, fed through nn.Embedding)

### Dataset (src/data/dataset.py)

`BCMPixelDataset` preloads all selected pixels into RAM. Channel counts (`n_dyn`, `n_static_cont`) are read dynamically from zarr shape — no hardcoded values. The dataset outputs `(n_dyn + n_static_cont, seq_len)` tensors; FVEG IDs are passed separately.

`EcoregionStratifiedSampler` balances sampling across L3 ecoregions.

### Preprocessing (src/data/preprocessing.py)

`build_zarr_store()` is the central function: reads all rasters, aligns to BCM grid via `_read_and_align()`, computes derived features (snow_frac, vpd), fills lagged targets (pck_prev, aet_prev), and computes normalization stats. Key helper: `_read_and_align()` auto-detects CRS mismatch and reprojects with rasterio.

### Snapshot system (src/utils/snapshot.py)

Each `--run-id` creates `snapshots/{id}/` containing: manifest.json (git hash, metrics), config.yaml, best_model.pt, training_history.json, metrics.json, spatial_maps/. `compare_snapshots()` diffs metrics between runs.

## Configuration

All settings live in `config.yaml`. The `ConfigNamespace` loader (src/utils/config.py) converts nested YAML to attribute-access objects. Adding new config keys requires no code changes to the loader.

Key sections: `paths` (data locations), `grid` (EPSG:3310 reference), `temporal` (train/test split dates), `model.backbone.in_channels` (must match 10 dyn + 9 static + 8 fveg embed = 27), `training` (epochs, LR, loss weights, teacher forcing).

## Loss Function

`BCMMultiLoss` (src/training/losses.py) computes a composite loss:

```
Total = Σ w_var * Huber(var) + extreme_weight * MSE_extreme(extreme_vars)
```

- **Huber loss** (delta=1.35): MSE for errors < 1.35σ, MAE beyond. Prevents outlier-driven instability but reduces gradients on extreme-value samples.
- **Extreme-aware penalty** (v7+): Additive MSE term on samples where target z-score > `extreme_threshold` (1.28 ≈ P90). Asymmetric weighting penalizes underprediction 1.5x more than overprediction. Applied to AET only (CWD = PET - AET algebraically). Controlled by 4 config keys under `training.loss_weights`: `extreme_threshold`, `extreme_weight`, `extreme_vars`, `extreme_asym`. Set `extreme_weight: 0.0` to disable.

**Motivation:** v6-huber systematically underpredicts AET extremes (P95 bias = -26.6mm, 72.7% of pixels underpredicting) because Huber's MAE tail removes the quadratic gradient exactly where it's needed most.

## Current Status

Best overall model: **v6-huber** (PET NSE 0.927, PCK NSE 0.950, CWD NSE 0.907). Known weakness: AET/CWD extreme underprediction. **v7-extreme-aware** (not yet trained) adds the extreme-aware MSE penalty to address this. See `docs/model_comparison.md` for full run-by-run analysis.

## Development Practices

- All production data processing must be in project scripts (`src/data/`, `prepare_data.py`), not ad-hoc shell commands.
- Always tag training runs with `--run-id` and `--notes` for reproducibility.
- When adding new input channels: update zarr shape in `preprocessing.py`, add normalization in `_compute_norm_stats`, verify `dataset.py` reads counts dynamically, update `config.yaml` `in_channels`.
