#!/bin/bash
set -e
cd /home/mmann1123/extra_space/bcm_emulator
mkdir -p logs

echo "[hindcast] start v24a at $(date)"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python \
    scripts/fire_model/00_export_predictions_hindcast.py \
    --snapshot-dir snapshots/v24a-annualpool-lambda1.0-v17-polaris-awc \
    --output-dir /home/mmann1123/extra_space/fire_model/data/predictions_hindcast_v24a_lambda1p0 \
    > logs/v24a_hindcast.log 2>&1
echo "[hindcast] v24a done at $(date)"

echo "[hindcast] start v24c at $(date)"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n deep_field python \
    scripts/fire_model/00_export_predictions_hindcast.py \
    --snapshot-dir snapshots/v24c-annualpool-lambda10.0-v17-polaris-awc \
    --output-dir /home/mmann1123/extra_space/fire_model/data/predictions_hindcast_v24c_lambda10p0 \
    > logs/v24c_hindcast.log 2>&1
echo "[hindcast] v24c done at $(date)"
echo "[hindcast] all done at $(date)"
