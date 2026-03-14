# BCM Emulator

A deep learning emulator for the Basin Characterization Model (BCM), a hydrological simulation model for California. The emulator uses a temporal convolutional network (TCN) to predict key water balance variables — replacing expensive computational simulations with fast neural network inference.

## Predicted Variables

- **PET** — Potential Evapotranspiration
- **AET** — Actual Evapotranspiration
- **PCK** — Snowpack
- **CWD** — Climatic Water Deficit (algebraic: PET − AET)

## Project Structure

```
bcm_emulator/
├── config.yaml              # Master configuration
├── prepare_data.py          # Data download & preprocessing
├── train.py                 # Model training
├── evaluate.py              # Evaluation & inference
├── src/
│   ├── data/                # Datasets, splits, downloaders
│   │   ├── dataset.py       # BCMPixelDataset, ElevationStratifiedSampler
│   │   ├── splits.py        # Train/test temporal splits
│   │   ├── preprocessing.py # Zarr store construction
│   │   ├── download_prism.py
│   │   ├── download_sciencebase.py
│   │   ├── download_daymet.py
│   │   └── download_srad.py
│   ├── models/              # Neural network architecture
│   │   ├── bcm_model.py     # Main BCMEmulator model
│   │   ├── backbone.py      # 5-level dilated TCN backbone
│   │   ├── layers.py        # CausalConv1d, TemporalBlock
│   │   └── heads.py         # PET/PCK/AET output heads
│   ├── training/            # Training loop & losses
│   │   ├── trainer.py       # BCMTrainer
│   │   ├── losses.py        # Weighted multi-task MSE
│   │   └── teacher_forcing.py
│   ├── evaluation/          # Metrics & visualization
│   │   ├── metrics.py       # NSE, KGE, RMSE, percent bias
│   │   └── spatial_maps.py  # Per-pixel NSE maps (GeoTIFF)
│   └── utils/
│       ├── config.py        # YAML config loader
│       ├── io_helpers.py    # Raster I/O, BCM file parsing
│       └── topo_solar.py    # Topographic solar radiation
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Download climate inputs (PRISM, ScienceBase, TerraClimate) and build a normalized Zarr store:

```bash
python prepare_data.py --config config.yaml --steps all
```

Individual steps can be run separately: `sciencebase`, `pck_gap`, `prism_daily`, `srad`, `topo_solar`, `zarr`.

### 2. Train

```bash
python train.py --config config.yaml
```

Training uses:
- AdamW optimizer with cosine annealing + linear warmup
- Teacher forcing curriculum (ground-truth → autoregressive over 100 epochs)
- Elevation-stratified sampling across the California 1 km grid
- Mixed precision (AMP) with gradient clipping

Checkpoints are saved to `checkpoints/`.

### 3. Evaluate

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

Outputs:
- `outputs/metrics.json` — NSE, KGE, RMSE, percent bias per variable
- `outputs/acf_diagnostics.json` — residual autocorrelation (lags 1–12)
- `outputs/spatial_maps/nse_*.tif` — per-pixel NSE maps

## Model Architecture

- **Input**: 13 channels (9 dynamic climate + 4 static terrain features)
- **Backbone**: 5-level dilated TCN (channels: 64 → 128 → 128 → 256 → 256, kernel size 3, receptive field 125 months)
- **Heads**: PET and PCK use softplus activation (≥ 0); AET uses a sigmoid stress factor multiplied by PET to guarantee AET ≤ PET
- **CWD**: Computed algebraically as PET − AET (no learned parameters)

## Data Sources

| Source | Variables | Resolution |
|--------|-----------|------------|
| ScienceBase (BCMv8) | Tmin, Tmax, Precipitation, AET, CWD, PCK | 270 m → 1 km |
| PRISM | Daily precipitation → wet days, intensity | 4 km → 1 km |
| TerraClimate | Monthly solar radiation (srad) | ~4.7 km → 1 km |
| DEM-derived | Elevation, slope, aspect, topographic solar | 1 km |

## Configuration

All settings are in `config.yaml`, including:
- File paths for data sources and outputs
- Grid specification (EPSG:3310, 1209 × 941 pixels at 1 km)
- Train/test temporal splits (1980–2019 / 2019–2020)
- Model hyperparameters, loss weights, and training schedule

## License

See LICENSE file for details.
