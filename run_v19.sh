#!/bin/bash
set -e

cd /home/mmann1123/extra_space/bcm_emulator

echo "=========================================="
echo "v19a: huber-tight + extreme0.1"
echo "=========================================="

# Backup original config
cp config.yaml config.yaml.bak

# Apply v19a config changes
sed -i 's/^  loss_type: mse/  loss_type: huber/' config.yaml
sed -i 's/^    huber_delta: 1.35/    huber_delta: 0.5/' config.yaml
sed -i 's/^    extreme_weight: 0.05/    extreme_weight: 0.1/' config.yaml

echo "v19a config changes applied:"
grep -E '(loss_type|huber_delta|extreme_weight)' config.yaml

# Train v19a
conda run -n deep_field python train.py --run-id v19a-huber-tight-extreme0.1 \
  --notes "Huber delta=0.5 + extreme_weight=0.1. Combines sweep's best PCK/AET result (huber-tight) with stronger extreme penalty. Tests whether tight Huber and doubled extreme weight are synergistic for AET tails without PCK regression."

# Evaluate v19a
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v19a-huber-tight-extreme0.1

echo "=========================================="
echo "v19b: MSE + extreme0.1 + petfloor0.3"
echo "=========================================="

# Restore baseline config then apply v19b changes
cp config.yaml.bak config.yaml
sed -i 's/^    extreme_weight: 0.05/    extreme_weight: 0.1/' config.yaml
sed -i 's/^    pet_floor: 0.5/    pet_floor: 0.3/' config.yaml

echo "v19b config changes applied:"
grep -E '(loss_type|extreme_weight|pet_floor)' config.yaml

# Train v19b
conda run -n deep_field python train.py --run-id v19b-extreme0.1-petfloor0.3 \
  --notes "MSE path: extreme_weight=0.1 + pet_floor=0.3. Both independently improved AET P95 bias in sweep. Tests whether combined they push below -14mm without instability."

# Evaluate v19b
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v19b-extreme0.1-petfloor0.3

echo "=========================================="
echo "Comparing all runs"
echo "=========================================="

conda run -n deep_field python -c "
from src.utils.snapshot import compare_snapshots
compare_snapshots('v17-polaris-awc', 'v19a-huber-tight-extreme0.1', project_root='.')
compare_snapshots('v17-polaris-awc', 'v19b-extreme0.1-petfloor0.3', project_root='.')
compare_snapshots('v19a-huber-tight-extreme0.1', 'v19b-extreme0.1-petfloor0.3', project_root='.')
"

# Restore original config
cp config.yaml.bak config.yaml
rm config.yaml.bak

echo "=========================================="
echo "All v19 experiments complete. Config restored to v17 baseline."
echo "=========================================="
