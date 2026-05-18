#!/bin/bash
# Full reproduction pipeline (~465 GPU-hours on A100)
set -euo pipefail

echo "============================================"
echo " Succinctness Predicts Syntax - Full Pipeline"
echo "============================================"

# Step 1: Generate all data
echo "[1/8] Generating data..."
python data/generate.py --all --output_dir data/generated/

# Step 2: Train from-scratch models (3 main + 2 baselines)
echo "[2/8] Training from-scratch models..."
for ARCH in tf lstm mamba gru rwkv; do
  for PHEN in agreement npi reflexive concord mc_npi cc_binding; do
    python scripts/train_from_scratch.py --arch $ARCH --phenomenon $PHEN
  done
done

# Step 3: DAS evaluation (main 3 architectures)
echo "[3/8] Running DAS evaluation..."
for PHEN in agreement npi reflexive concord mc_npi cc_binding; do
  python scripts/run_das.py --phenomenon $PHEN \
    --variables subj_num verb_form has_lic npi_ok ante_num refl_ok det_num conc_ok
done

# Step 4: Negative controls
echo "[4/8] Running negative controls..."
python scripts/run_controls.py --control all

# Step 5: Width scaling
echo "[5/8] Running width scaling..."
python scripts/run_width_scaling.py

# Step 6: Training dynamics
echo "[6/8] Tracking training dynamics..."
python scripts/run_dynamics.py

# Step 7: Pythia evaluation
echo "[7/8] Evaluating Pythia models..."
python scripts/run_pythia.py --eval_type both

# Step 8: Generate figures
echo "[8/8] Generating figures..."
python figures/generate_all.py

echo "Done! Results in results/, figures in figures/output/"
