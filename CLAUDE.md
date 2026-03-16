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

## Training & Evaluation (v3-vpd-awc)

Run all commands with the `deep_field` conda environment. Always tag runs with `--run-id` and `--notes` for reproducibility.

```bash
# 1. Download AWC from POLARIS
conda run -n deep_field python prepare_data.py --steps awc

# 2. Rebuild zarr (VPD computed during build, AWC loaded as static)
conda run -n deep_field python prepare_data.py --steps zarr

# 3. Train v3
conda run -n deep_field python train.py --run-id v3-vpd-awc --notes "Added VPD dynamic input + POLARIS AWC static input"

# 4. Evaluate and compare
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v3-vpd-awc
conda run -n deep_field python -c "from src.utils.snapshot import compare_snapshots; compare_snapshots('v2-fveg-srad-fix', 'v3-vpd-awc', project_root='.')"
```
