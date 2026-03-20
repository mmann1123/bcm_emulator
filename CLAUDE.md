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
conda run -n deep_field python prepare_data.py --steps sciencebase pck_gap prism_daily prism_daily_tmax srad topo_solar fveg soil fire_features zarr

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
        /inputs/dynamic   (T, 11, H, W)  - monthly climate
        /inputs/static    (14, H, W)     - terrain + soil + vegetation
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
Input: (B, 23, T) + KBDI (B, 1, T)  → _prepend_fveg → (B, 31, T) → Backbone
  23 = 10 dynamic (excl. KBDI) + 13 continuous static
  31 = 23 + 8 FVEG embedding

Stage 1: TCNBackbone - 5 causal dilated levels [64,128,128,256,256], dilations [1,2,4,8,16]
          Receptive field: 125 months. Output: (B, 256, T)

Stage 2: PET head (256→1, softplus) + PCK head (256→1, softplus)  [no KBDI]

Stage 3: AET head ([256+PET+PCK+KBDI]=259 → 64 → 1)  [KBDI injected here]
          AET ≤ PET enforced post-denormalization, not in the head

CWD = PET - AET (algebraic, no parameters)
```

KBDI is excluded from the backbone to prevent the dominant "high KBDI = hot = high PET" encoding from polluting AET. Instead, KBDI is injected directly into the AET head where it acts as a drought-stress inhibitor.

Teacher forcing: channels 7 (pck_prev) and 8 (aet_prev) are swapped between ground-truth and model predictions via `BCMEmulator.PCK_PREV_IDX` / `AET_PREV_IDX`. Curriculum: 100% GT for first half of training, then linear ramp to 100% predicted.

### Input channels

**Dynamic (10 backbone + 1 KBDI routed to AET):** ppt, tmin, tmax, wet_days, ppt_intensity, srad, snow_frac, pck_prev, aet_prev, vpd | kbdi (AET-only)

**Static (14):** elev, topo_solar, lat, lon, ksat, sand, clay, soil_depth, aridity_index, field_capacity, wilting_point, SOM, windward_index, fveg_class_id
- Channels 0-12 are continuous (z-score normalized)
- Channel 13 (FVEG) is categorical (integer class ID, not normalized, fed through nn.Embedding)

### Dataset (src/data/dataset.py)

`BCMPixelDataset` preloads all selected pixels into RAM. Channel counts (`n_dyn`, `n_static_cont`) are read dynamically from zarr shape — no hardcoded values. The dataset outputs `(n_dyn + n_static_cont, seq_len)` tensors; FVEG IDs are passed separately.

`EcoregionStratifiedSampler` balances sampling across L3 ecoregions.

### Preprocessing (src/data/preprocessing.py)

`build_zarr_store()` is the central function: reads all rasters, aligns to BCM grid via `_read_and_align()`, computes derived features (snow_frac, vpd), fills lagged targets (pck_prev, aet_prev), and computes normalization stats. Key helper: `_read_and_align()` auto-detects CRS mismatch and reprojects with rasterio.

### Snapshot system (src/utils/snapshot.py)

Each `--run-id` creates `snapshots/{id}/` containing: manifest.json (git hash, metrics), config.yaml, best_model.pt, training_history.json, metrics.json, spatial_maps/. `compare_snapshots()` diffs metrics between runs.

## Configuration

All settings live in `config.yaml`. The `ConfigNamespace` loader (src/utils/config.py) converts nested YAML to attribute-access objects. Adding new config keys requires no code changes to the loader.

Key sections: `paths` (data locations), `grid` (EPSG:3310 reference), `temporal` (train/test split dates), `model.backbone.in_channels` (must match 10 dyn + 13 static + 8 fveg embed = 31; KBDI routed separately), `training` (epochs, LR, loss weights, teacher forcing).

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

## Model Build/Run/Evaluate Workflow

Every new model version MUST follow this exact sequence. Do not skip steps or reorder.

```bash
# 1. Download any new data (only needed if new data sources were added)
conda run -n deep_field python prepare_data.py --steps soil

# 2. Rebuild zarr store (ALWAYS required after changing static/dynamic channels)
conda run -n deep_field python prepare_data.py --steps zarr

# 3. Train with a descriptive run-id and notes explaining what changed
conda run -n deep_field python train.py --run-id v8-soil-physics \
    --notes "v5 base + soil_depth, aridity_index, FC, WP, SOM; AWC removed; 14 static channels"

# 4. Evaluate the trained model
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v8-soil-physics

# 5. Compare against the previous best run(s)
conda run -n deep_field python -c "from src.utils.snapshot import compare_snapshots; compare_snapshots('v5-awc-windward','v8-soil-physics', project_root='.')"
```

This is the canonical workflow. All five steps must be run in order for every new model version.

## Zarr Reproducibility

- **Auto-backup:** When `prepare_data.py --steps zarr` runs and an existing zarr store is found, it is automatically renamed to `bcm_dataset_{fingerprint}.zarr` (where fingerprint is derived from the zarr's normalization stats) before the new zarr is built. This preserves the exact data used by previous model versions.
- **Recreation:** Any previous zarr can be recreated from its snapshot config:
  ```bash
  conda run -n deep_field python prepare_data.py --config snapshots/{run_id}/config.yaml --steps zarr
  ```
- Source rasters are stable on disk; zarr is deterministically derived from config + rasters.
