# BCM Emulator - Claude Code Instructions

## Environment

Always use the `deep_field` conda environment for running scripts:

```bash
conda run -n deep_field python <script.py>
```

## Project Structure

- `prepare_data.py` — Data pipeline orchestration (download, preprocess, build zarr)
- `train.py` — Training entry point
- `evaluate.py` — Evaluation entry point
- `config.yaml` — All paths, hyperparameters, and settings
- `src/data/` — Dataset, preprocessing, download modules
- `src/models/` — Model architecture
- `src/training/` — Trainer, losses, teacher forcing
- `src/utils/` — Config, I/O helpers, topo solar
