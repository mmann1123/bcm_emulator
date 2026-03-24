#!/usr/bin/env bash
# =============================================================================
# BCM Emulator Hyperparameter Tuning Sweep — post v17-polaris-awc
# =============================================================================
# Usage: bash run_tuning_sweep.sh [group]
#   group: all | loss_weights | extreme | loss_type | scheduler (default: all)
#
# Each run:
#   1. Uses Python to surgically update config.yaml
#   2. Trains the model
#   3. Evaluates and snapshots
#   4. Restores config.yaml to the v17 baseline
#
# Prerequisites:
#   - v17-polaris-awc has finished training
#   - config.yaml is at the v17 baseline state (aet=1.5, extreme_weight=0.05, etc.)
#   - conda environment: deep_field
#
# The Python config-patcher at the bottom handles all YAML mutations safely.
# =============================================================================

set -euo pipefail

GROUP="${1:-all}"
CONDA_ENV="deep_field"
CONFIG="config.yaml"
BASELINE_BACKUP="config_v17_baseline.yaml"
PYTHON="conda run -n ${CONDA_ENV} python"
TRAIN="${PYTHON} train.py"
EVAL="${PYTHON} evaluate.py --checkpoint checkpoints/best_model.pt"
PATCH="${PYTHON} scripts/patch_config.py"

# ── Backup baseline config ────────────────────────────────────────────────────
echo "=== Backing up v17 baseline config ==="
cp "${CONFIG}" "${BASELINE_BACKUP}"

restore_config() {
    echo "--- Restoring v17 baseline config ---"
    cp "${BASELINE_BACKUP}" "${CONFIG}"
}

# Ensure config is always restored even on error
trap restore_config EXIT

# =============================================================================
# Helper: run one experiment
# =============================================================================
run_experiment() {
    local run_id="$1"
    local notes="$2"
    shift 2
    # Remaining args are key=value pairs for patch_config.py
    echo ""
    echo "============================================================"
    echo "STARTING: ${run_id}"
    echo "NOTES:    ${notes}"
    echo "============================================================"

    restore_config

    # Apply all config patches
    for patch in "$@"; do
        key="${patch%%=*}"
        val="${patch#*=}"
        ${PATCH} "${CONFIG}" "${key}" "${val}"
    done

    # Train
    ${TRAIN} --run-id "${run_id}" --notes "${notes}"

    # Evaluate
    ${EVAL} --run-id "${run_id}"

    echo "--- Completed: ${run_id} ---"
}

# =============================================================================
# GROUP 1: Loss weights
# Tests the AET/CWD weight interaction and pet_floor
# Baseline: aet=1.5, cwd=2.0, pet_initial=1.0, pet_decay=0.99, pet_floor=0.5
# =============================================================================
run_loss_weights() {
    echo ""
    echo "############################################################"
    echo "GROUP: Loss Weights"
    echo "############################################################"

    # 1a. Uniform weights — does the extreme penalty alone carry the tail?
    run_experiment \
        "v18-uniform-extreme" \
        "Uniform base weights (aet=1.0, cwd=1.0) + extreme_weight=0.05. Tests whether extreme penalty alone maintains tail performance without any AET upweighting." \
        "training.loss_weights.aet_initial=1.0" \
        "training.loss_weights.cwd_initial=1.0"

    # 1b. Higher CWD weight — does CWD extremes improve further?
    run_experiment \
        "v18-cwd3-aet1.5" \
        "cwd_initial=3.0, aet_initial=1.5. Tests whether pushing CWD weight further improves CWD extremes without hurting AET." \
        "training.loss_weights.cwd_initial=3.0"

    # 1c. Recover PET accuracy — lower pet_floor
    run_experiment \
        "v18-petfloor0.3" \
        "pet_floor=0.3 (from 0.5). Allows PET weight to decay further, testing whether backbone shifts more attention to AET/CWD without hurting global PET NSE critically." \
        "training.loss_weights.pet_floor=0.3"

    # 1d. Symmetric — aet=cwd=1.5, test balanced approach
    run_experiment \
        "v18-balanced1.5" \
        "aet=1.5, cwd=1.5, extreme_weight=0.05. Symmetric AET/CWD weighting — tests whether CWD regression from asymmetric weights can be recovered." \
        "training.loss_weights.cwd_initial=1.5"
}

# =============================================================================
# GROUP 2: Extreme penalty tuning
# Baseline: extreme_weight=0.05, extreme_asym=1.5, extreme_threshold=1.28 (P90)
# =============================================================================
run_extreme() {
    echo ""
    echo "############################################################"
    echo "GROUP: Extreme Penalty"
    echo "############################################################"

    # 2a. Double extreme weight
    run_experiment \
        "v18-extreme0.1" \
        "extreme_weight=0.1 (from 0.05). Tests whether doubling the tail penalty pushes AET P95 bias below -16mm without v7-style instability." \
        "training.loss_weights.extreme_weight=0.1"

    # 2b. Lower threshold — penalize more of the tail (P85 instead of P90)
    run_experiment \
        "v18-extreme-p85" \
        "extreme_threshold=1.04 (P85, from 1.28=P90). Wider tail penalty captures more of the underpredicted range." \
        "training.loss_weights.extreme_threshold=1.04"

    # 2c. Higher asymmetry — stronger underprediction penalty
    run_experiment \
        "v18-extreme-asym2" \
        "extreme_asym=2.0 (from 1.5). Stronger asymmetric penalty for underprediction vs overprediction of extremes." \
        "training.loss_weights.extreme_asym=2.0"

    # 2d. Add PCK to extreme vars — address persistent PCK pbias
    run_experiment \
        "v18-extreme-pck" \
        "extreme_vars=[aet,pck]. Applies extreme penalty to PCK as well, testing whether tail PCK errors (late-season snowpack) can be corrected alongside AET." \
        "training.loss_weights.extreme_vars=[aet,pck]"
}

# =============================================================================
# GROUP 3: Loss function type
# Baseline: loss_type=mse
# Huber is robust to outliers but softens the tail signal the extreme penalty needs
# Log-cosh is smooth everywhere, behaves like MSE near zero and MAE in tails
# Quantile loss directly optimizes a specific percentile
# =============================================================================
run_loss_type() {
    echo ""
    echo "############################################################"
    echo "GROUP: Loss Function Type"
    echo "############################################################"

    # 3a. Huber — the v6-v8 approach, now combined with 2x weights and extreme penalty
    run_experiment \
        "v18-huber" \
        "loss_type=huber, delta=1.35. Tests whether Huber's outlier robustness hurts or helps when combined with the v16 loss weight configuration. Previously tested without extreme penalty (v6-v8)." \
        "training.loss_type=huber"

    # 3b. Huber with tighter delta — more MSE-like, less MAE-like
    run_experiment \
        "v18-huber-tight" \
        "loss_type=huber, huber_delta=0.5. Tighter delta means MSE regime only for errors <0.5 sigma, MAE for larger. More aggressive outlier robustness than delta=1.35." \
        "training.loss_type=huber" \
        "training.loss_weights.huber_delta=0.5"

    # 3c. MSE with no extreme penalty — clean baseline to confirm extreme penalty value
    run_experiment \
        "v18-mse-noextreme" \
        "Pure MSE, extreme_weight=0.0, aet=1.5, cwd=2.0. Ablation: removes extreme penalty from v16 config to confirm its contribution to tail performance." \
        "training.loss_weights.extreme_weight=0.0"
}

# =============================================================================
# GROUP 4: Scheduler and learning rate
# Baseline: cosine scheduler, lr=0.001, warmup=5, epochs=100
# =============================================================================
run_scheduler() {
    echo ""
    echo "############################################################"
    echo "GROUP: Scheduler and Learning Rate"
    echo "############################################################"

    # 4a. Lower learning rate — given 2x loss weights effectively doubled gradient magnitude
    run_experiment \
        "v18-lr0.0005" \
        "learning_rate=0.0005 (from 0.001). Loss weights effectively double gradient magnitude for AET/CWD — lower LR may improve convergence stability and best epoch." \
        "training.learning_rate=0.0005"

    # 4b. Longer warmup — more stable early training with complex loss
    run_experiment \
        "v18-warmup10" \
        "warmup_epochs=10 (from 5). Longer warmup for the combined MSE+extreme loss to stabilize before full learning rate." \
        "training.warmup_epochs=10"

    # 4c. More epochs — check whether 100 is enough for current feature set
    run_experiment \
        "v18-epochs150" \
        "epochs=150 (from 100). Tests whether best epoch near 91 (v14) indicates underfitting given the richer 15-channel feature set." \
        "training.epochs=150"
}

# =============================================================================
# MAIN
# =============================================================================
echo "=== BCM Emulator Tuning Sweep ==="
echo "Baseline config: ${BASELINE_BACKUP}"
echo "Group: ${GROUP}"
echo ""

case "${GROUP}" in
    all)
        run_loss_weights
        run_extreme
        run_loss_type
        run_scheduler
        ;;
    loss_weights)
        run_loss_weights
        ;;
    extreme)
        run_extreme
        ;;
    loss_type)
        run_loss_type
        ;;
    scheduler)
        run_scheduler
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Valid: all | loss_weights | extreme | loss_type | scheduler"
        exit 1
        ;;
esac

echo ""
echo "=== Tuning sweep complete ==="

# ── Pairwise comparisons against v17 baseline ─────────────────────────────────
echo "=== Pairwise comparisons against v17 baseline ==="
BASELINE="v17-polaris-awc"
for rid in v18-uniform-extreme v18-cwd3-aet1.5 v18-petfloor0.3 v18-balanced1.5 \
           v18-extreme0.1 v18-extreme-p85 v18-extreme-asym2 v18-extreme-pck \
           v18-huber v18-huber-tight v18-mse-noextreme \
           v18-lr0.0005 v18-warmup10 v18-epochs150; do
    if [ -d "snapshots/${rid}" ]; then
        echo "--- ${rid} vs ${BASELINE} ---"
        ${PYTHON} -c "from src.utils.snapshot import compare_snapshots; compare_snapshots('${BASELINE}','${rid}', project_root='.')"
    fi
done
