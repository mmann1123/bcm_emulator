#!/bin/bash
set -e

cd /home/mmann1123/extra_space/bcm_emulator

echo "=========================================="
echo "v20a: asym1.1 (v19a base + extreme_asym=1.1)"
echo "=========================================="

# Backup original config
cp config.yaml config.yaml.bak

# Apply v19a base config + v20a change
sed -i 's/^  loss_type: mse/  loss_type: huber/' config.yaml
sed -i 's/^    huber_delta: 1.35/    huber_delta: 0.5/' config.yaml
sed -i 's/^    extreme_weight: 0.05/    extreme_weight: 0.1/' config.yaml
sed -i 's/^    extreme_asym: 1.5/    extreme_asym: 1.1/' config.yaml

echo "v20a config changes applied:"
grep -E '(loss_type|huber_delta|extreme_weight|extreme_asym|aet_initial)' config.yaml

# Train v20a
conda run -n deep_field python train.py --run-id v20a-asym1.1 \
  --notes "v19a config (huber delta=0.5, extreme_weight=0.1) with extreme_asym reduced from 1.5 to 1.1. Targets AET pbias reduction (currently 13.1%) while preserving tail gains. Hypothesis: asymmetry inflates mean bias; near-symmetric penalty retains tail correction with less global upward push."

# Evaluate v20a
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v20a-asym1.1

echo "=========================================="
echo "v20b: aet1.2 (v19a base + aet_initial=1.2)"
echo "=========================================="

# Restore baseline then apply v19a base + v20b change
cp config.yaml.bak config.yaml
sed -i 's/^  loss_type: mse/  loss_type: huber/' config.yaml
sed -i 's/^    huber_delta: 1.35/    huber_delta: 0.5/' config.yaml
sed -i 's/^    extreme_weight: 0.05/    extreme_weight: 0.1/' config.yaml
sed -i 's/^    aet_initial: 1.5/    aet_initial: 1.2/' config.yaml

echo "v20b config changes applied:"
grep -E '(loss_type|huber_delta|extreme_weight|extreme_asym|aet_initial)' config.yaml

# Train v20b
conda run -n deep_field python train.py --run-id v20b-aet1.2 \
  --notes "v19a config (huber delta=0.5, extreme_weight=0.1) with aet_initial reduced from 1.5 to 1.2. Tests whether extreme penalty now carries enough tail load that the base AET weight is partially redundant and contributing to the 13.1% pbias. Hypothesis: lower base weight reduces mean bias while extreme penalty preserves tail accuracy."

# Evaluate v20b
conda run -n deep_field python evaluate.py --checkpoint checkpoints/best_model.pt --run-id v20b-aet1.2

echo "=========================================="
echo "Comparing all runs"
echo "=========================================="

conda run -n deep_field python -c "
from src.utils.snapshot import compare_snapshots
compare_snapshots('v19a-huber-tight-extreme0.1', 'v20a-asym1.1', project_root='.')
compare_snapshots('v19a-huber-tight-extreme0.1', 'v20b-aet1.2', project_root='.')
compare_snapshots('v20a-asym1.1', 'v20b-aet1.2', project_root='.')
"

# Restore original config
cp config.yaml.bak config.yaml
rm config.yaml.bak

echo "=========================================="
echo "All v20 experiments complete. Config restored to v17 baseline."
echo "=========================================="
