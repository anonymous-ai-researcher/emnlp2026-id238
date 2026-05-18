#!/bin/bash
# Quick validation (~2 GPU-hours): agreement only, 2 seeds
set -euo pipefail

echo "=== Quick Validation ==="
python data/generate.py --phenomenon agreement --output_dir data/generated/
for ARCH in tf mamba; do
  python scripts/train_from_scratch.py --arch $ARCH --phenomenon agreement --seeds 0 1
done
echo "Quick validation complete."
