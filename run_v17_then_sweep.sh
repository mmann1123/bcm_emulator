#!/usr/bin/env bash
# =============================================================================
# Wrapper: Wait for v17 zarr build → train v17 → evaluate v17 → launch sweep
# =============================================================================
# Usage:
#   nohup bash run_v17_then_sweep.sh > sweep.log 2>&1 &
#   tail -f sweep.log
#
# Assumes:
#   - prepare_data.py --steps zarr is currently running (building v17 zarr)
#   - conda environment: deep_field
#   - config.yaml is at v17 baseline state
# =============================================================================

set -euo pipefail

CONDA_ENV="deep_field"
PYTHON="conda run -n ${CONDA_ENV} python"
ZARR_PATH="/home/mmann1123/extra_space/bcm_emulator/data/bcm_dataset.zarr"
RUN_ID="v17-polaris-awc"
NOTES="POLARIS root-zone AWC (0-100cm) for SWS; dropped awc_total static (14 static); aet=1.5, extreme_weight=0.05"
POLL_INTERVAL=300  # 5 minutes

echo "============================================================"
echo "v17 → Sweep Pipeline"
echo "Started: $(date)"
echo "============================================================"

# ── Phase 1: Wait for zarr build to finish ─────────────────────────────────────
echo ""
echo "=== Phase 1: Waiting for zarr build (prepare_data.py) to finish ==="

while pgrep -f "prepare_data.py" > /dev/null 2>&1; do
    echo "  [$(date '+%H:%M:%S')] prepare_data.py still running... checking again in ${POLL_INTERVAL}s"
    sleep ${POLL_INTERVAL}
done

# Verify zarr actually exists (completed vs never started)
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: prepare_data.py finished but zarr not found at ${ZARR_PATH}"
    echo "The zarr build may have failed. Check logs and re-run:"
    echo "  conda run -n ${CONDA_ENV} python prepare_data.py --steps zarr"
    exit 1
fi

echo "  Zarr build complete: ${ZARR_PATH}"
echo "  Finished at: $(date)"

# ── Phase 2: Train v17 ────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Training ${RUN_ID} ==="

if [ -d "snapshots/${RUN_ID}" ]; then
    echo "  Skipping — snapshot already exists for ${RUN_ID}"
else
    echo "  Started: $(date)"
    ${PYTHON} train.py --run-id "${RUN_ID}" --notes "${NOTES}"

    TRAIN_EXIT=$?
    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
        exit 1
    fi
    echo "  Training complete: $(date)"
fi

# ── Phase 3: Evaluate v17 ─────────────────────────────────────────────────────
echo ""
echo "=== Phase 3: Evaluating ${RUN_ID} ==="

if [ -f "snapshots/${RUN_ID}/metrics.json" ]; then
    echo "  Skipping — metrics.json already exists for ${RUN_ID}"
else
    ${PYTHON} evaluate.py --checkpoint checkpoints/best_model.pt --run-id "${RUN_ID}"
    echo "  Evaluation complete: $(date)"
fi

# ── Phase 4: Launch tuning sweep ──────────────────────────────────────────────
echo ""
echo "=== Phase 4: Launching tuning sweep ==="
echo "  Started: $(date)"

bash run_tuning_sweep.sh all

echo ""
echo "============================================================"
echo "Full pipeline complete: $(date)"
echo "============================================================"
