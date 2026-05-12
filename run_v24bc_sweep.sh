#!/bin/bash
# Sequential v24b → eval → v24c → eval. Authorized 2026-05-02.
set -e
cd /home/mmann1123/extra_space/bcm_emulator
mkdir -p logs

echo "[chain] start v24b at $(date)"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python train.py \
    --config config_v24b_lambda3p0.yaml \
    --run-id v24b-annualpool-lambda3.0-v17-polaris-awc \
    --notes "v17-polaris-awc + annual-pooled MSE on PET+AET, lambda=3.0; 100 epochs from scratch" \
    > logs/v24b.log 2>&1
echo "[chain] v24b train done at $(date)"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --run-id v24b-annualpool-lambda3.0-v17-polaris-awc \
    > logs/v24b_eval.log 2>&1
echo "[chain] v24b eval done at $(date)"
cp -f checkpoints/best_model.pt snapshots/v24b-annualpool-lambda3.0-v17-polaris-awc/best_model.pt 2>/dev/null || true

echo "[chain] start v24c at $(date)"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python train.py \
    --config config_v24c_lambda10p0.yaml \
    --run-id v24c-annualpool-lambda10.0-v17-polaris-awc \
    --notes "v17-polaris-awc + annual-pooled MSE on PET+AET, lambda=10.0; 100 epochs from scratch" \
    > logs/v24c.log 2>&1
echo "[chain] v24c train done at $(date)"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --run-id v24c-annualpool-lambda10.0-v17-polaris-awc \
    > logs/v24c_eval.log 2>&1
echo "[chain] v24c eval done at $(date)"
cp -f checkpoints/best_model.pt snapshots/v24c-annualpool-lambda10.0-v17-polaris-awc/best_model.pt 2>/dev/null || true

echo "[chain] sweep complete at $(date)"
