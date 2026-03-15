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

## Development Practices

- One-off bash/python commands are fine for **testing and exploration**, but all production data processing steps must live in the project scripts (e.g., `src/data/`, `prepare_data.py`) so they are documented, reproducible, and can be written up later. Do not leave critical processing steps as ad-hoc shell commands.
