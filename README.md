# Succinctness Predicts Syntax 🧬

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![pyvene](https://img.shields.io/badge/pyvene-0.1.2-orange.svg)](https://github.com/stanfordnlp/pyvene)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Venue](https://img.shields.io/badge/EMNLP-2026-purple.svg)](#)

> **Succinctness Predicts the Syntax Gap between Transformers and Recurrent Models**
>
> *Anonymous submission to EMNLP 2026 (ARR May 2026)*

---

## TL;DR

**Formal language theory can predict which neural architecture will succeed on which syntactic task, before running any experiment.** B-RASP program analysis shows Transformers encode syntactic computations in constant depth while sequential models face growing complexity with dependency distance. All 7 directional predictions are confirmed across 5 architectures, and causal intervention (DAS) verifies the Transformer implements the predicted algorithm (IIA ≥ 0.95) while Mamba does not (compositional IIA ≈ chance).

---

## Key Results at a Glance

| Finding | Evidence |
|:--------|:---------|
| 🎯 7/7 directional predictions confirmed | TF > 96% vs Mamba < 55% at max complexity |
| 📐 Gap resists 46× parameter scaling | Mamba@d_h=1024 still below TF@d_h=64 |
| 🔬 Causal alignment confirmed in TF | IIA ≥ 0.95 for all 8 prescribed variables |
| ❌ Mamba fails causally despite high accuracy | Compositional IIA ≈ chance (0.53 -- 0.54) |
| 📈 Assembly follows dependency order | Atomic before compositional in 20/20 seeds |
| 🤖 Patterns transfer to pre-trained models | Pythia 1.4B: IIA up to 0.94 |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 12.1
- 1× GPU with ≥ 16 GB VRAM (A100 recommended for full reproduction)

### Setup

```bash
git clone https://github.com/anonymous/succinctness-syntax.git
cd succinctness-syntax

conda create -n succinctness python=3.10 -y
conda activate succinctness
pip install -r requirements.txt
pip install -e .
```

### Verify

```bash
python -c "from src.brasp.programs import PROGRAMS; print(f'{len(PROGRAMS)} phenomena loaded')"
python -c "import pyvene; print('pyvene OK')"
python -c "from mamba_ssm import Mamba; print('Mamba OK')"
```

---

## Quick Start (~30 min, 1 GPU)

Reproduce the core finding (Transformer vs Mamba on agreement):

```bash
# 1. Generate data
python data/generate.py --phenomenon agreement

# 2. Train both architectures (2 seeds)
python scripts/train_from_scratch.py --arch tf    --phenomenon agreement --seeds 0 1
python scripts/train_from_scratch.py --arch mamba --phenomenon agreement --seeds 0 1

# 3. Run DAS
python scripts/run_das.py --phenomenon agreement --variables subj_num verb_form

# 4. View results
python scripts/summarize.py --experiment agreement
```

---

## Repository Structure

```
succinctness-syntax/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/                          # YAML experiment configurations
│   ├── train_config.yaml               # Training hyperparameters
│   ├── das_config.yaml                 # DAS evaluation settings
│   ├── width_scaling.yaml              # Width scaling grid
│   └── pythia_config.yaml              # Pythia evaluation + BLiMP paradigms
│
├── data/                             # Data generation
│   ├── alphabet.py                     # 11-symbol token alphabet
│   ├── generate.py                     # CFG-based sentence generator (6 phenomena)
│   ├── counterfactual.py               # DAS counterfactual pair construction
│   └── grammars/                       # Per-phenomenon EBNF grammars
│
├── src/                              # Source code
│   ├── models/                         # 5 architecture implementations
│   │   ├── transformer.py                # 2-layer causal Transformer
│   │   ├── lstm.py                       # 2-layer LSTM
│   │   ├── mamba_model.py                # 2-layer Mamba (selective SSM)
│   │   ├── gru.py                        # 2-layer GRU
│   │   └── rwkv.py                       # 2-layer RWKV-4
│   ├── brasp/                          # Formal language theory
│   │   ├── programs.py                   # All 6 B-RASP programs (|P|=2 each)
│   │   └── scm.py                        # SCM compilation from B-RASP
│   ├── das/                            # Causal intervention
│   │   ├── alignment.py                  # Orthogonal DAS (Cayley parameterization)
│   │   ├── intervention.py               # Interchange interventions via pyvene
│   │   └── controls.py                   # 7 families of negative controls
│   ├── training/                       # Training
│   │   ├── trainer.py                    # AdamW loop with grokking monitoring
│   │   └── dynamics.py                   # IIA trajectory tracking
│   └── evaluation/                     # Evaluation
│       ├── behavioral.py                 # Per-distance accuracy
│       ├── iia.py                        # IIA computation (pyvene wrapper)
│       ├── pythia_eval.py                # Pythia BLiMP + DAS
│       └── metrics.py                    # MCC, Cohen's d, bootstrap CI
│
├── scripts/                          # Experiment entry points
│   ├── run_all.sh                      # Full reproduction (~465 GPU-hours)
│   ├── run_quick.sh                    # Quick validation (~2 GPU-hours)
│   ├── train_from_scratch.py           # Train any architecture on any phenomenon
│   ├── run_das.py                      # DAS evaluation with rank/layer search
│   ├── run_controls.py                 # All 7 negative control families
│   ├── run_width_scaling.py            # d_h ∈ {32, 64, 128, 256, 512, 1024}
│   ├── run_pythia.py                   # Pythia BLiMP + DAS evaluation
│   └── run_dynamics.py                 # IIA trajectories during training
│
├── figures/                          # Figure generation
│   └── generate_all.py                 # Reproduce all paper figures
│
└── tests/                            # Unit tests
    ├── test_brasp.py                   # B-RASP program correctness
    └── test_scm.py                     # SCM counterfactual verification
```

---

## Experimental Pipeline

### 1. Data Generation

```bash
python data/generate.py --all --output_dir data/generated/
```

| Phenomenon | Structural parameter | Range | Train | Test |
|:-----------|:---------------------|:------|------:|-----:|
| Agreement | attractor count *d* | 0 -- 8 | 50K | 10K |
| NPI licensing | attractor count *d* | 0 -- 8 | 50K | 10K |
| Reflexive binding | attractor count *d* | 0 -- 8 | 50K | 10K |
| Det-noun concord | modifier count *d* | 0 -- 1 | 20K | 4K |
| Multi-clause NPI | clause depth *c* | 1 -- 3 | 30K | 6K |
| Cross-cl. binding | attractor count *d* | 2 -- 8 | 40K | 8K |

### 2. Training

```bash
python scripts/train_from_scratch.py \
    --arch tf \
    --phenomenon agreement \
    --d_h 128 --n_layers 2 \
    --lr 1e-3 --weight_decay 1.0 \
    --batch_size 512 --steps 40000 \
    --seeds 0 1 2 3 4
```

| Hyperparameter | Value | Notes |
|:---------------|:------|:------|
| Optimizer | AdamW | β₁ = 0.9, β₂ = 0.999 |
| Learning rate | 10⁻³ | Robust in [5 × 10⁻⁴, 5 × 10⁻³] |
| Weight decay | 1.0 | **Critical for grokking** |
| Batch size | 512 | |
| Steps | 40,000 | All architectures memorize by ~15K |
| Seeds | {0, 1, 2, 3, 4} | Fixed |

### 3. DAS Evaluation

```bash
python scripts/run_das.py \
    --archs tf lstm mamba \
    --phenomenon agreement \
    --variables subj_num verb_form \
    --ranks 1 2 4 8 \
    --das_steps 2000 --das_lr 0.01 --das_inits 3
```

### 4. Negative Controls

```bash
python scripts/run_controls.py --control all
```

| # | Control family | Tests for | Expected IIA |
|:-:|:---------------|:----------|:-------------|
| 1 | Wrong-SCM | Phenomenon specificity | ≈ 0.50 |
| 2 | Wrong-Boolean | Function specificity | ≈ 0.50 |
| 3 | Random-init | Training necessity | ≈ 0.50 |
| 4 | Shuffled-label | Supervision necessity | ≈ 0.50 |
| 5 | Cross-phenomenon | Variable-level specificity | ≈ 0.50 (off-diagonal) |
| 6 | Alternative-SCM | Leftmost vs nearest | Below prescribed |
| 7 | Unconstrained | Orthogonal constraint necessity | High even on random nets |

### 5. Width Scaling

```bash
python scripts/run_width_scaling.py \
    --archs tf lstm mamba \
    --d_h_values 32 64 128 256 512 1024
```

### 6. Pythia Evaluation

```bash
python scripts/run_pythia.py --models 160m 410m 1.4b --eval_type both
```

### 7. Training Dynamics

```bash
python scripts/run_dynamics.py --archs tf mamba --n_checkpoints 15
```

---

## Hardware Requirements

| Experiment | GPU-hours | Minimum hardware |
|:-----------|----------:|:-----------------|
| Quick validation | ~2 | 1× consumer GPU (8 GB) |
| Core results (Tables 1 -- 4) | ~75 | 1× A100 (40 GB) |
| Full reproduction (all appendices) | ~465 | 4× A100 (~12 days) |

### Compute Breakdown

| Experiment group | GPU-hours |
|:-----------------|----------:|
| From-scratch training (5 arch × 6 phen × 5 seeds) | 75 |
| Width scaling (3 arch × 6 widths × 6 phen × 5 seeds) | 150 |
| DAS evaluation + controls | 55 |
| Training dynamics (2 arch × 15 ckpts × 5 seeds) | 40 |
| Ablation studies (14 experiments) | 80 |
| Pythia DAS | 15 |
| Depth ablation + 2× param control | 30 |
| **Total** | **~465** |

### Software Versions

| Package | Version | Purpose |
|:--------|:--------|:--------|
| Python | 3.10.12 | Runtime |
| PyTorch | 2.1.2 | Training |
| CUDA | 12.1 | GPU acceleration |
| pyvene | 0.1.2 | Causal interventions |
| mamba-ssm | 1.2.0 | Mamba architecture |
| transformers | 4.36.0 | Pythia models |

---

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Test B-RASP program correctness
pytest tests/test_brasp.py -v

# Test SCM counterfactual computation
pytest tests/test_scm.py -v
```

---

## Citation

```
This paper is currently under anonymous review.
Citation information will be provided upon acceptance.
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

| Component | License |
|:----------|:--------|
| Code | MIT |
| Generated data | CC-BY 4.0 |
| Pythia models | Apache 2.0 (EleutherAI) |
| BLiMP data | CC-BY 4.0 |

## Acknowledgments

We thank the developers of [pyvene](https://github.com/stanfordnlp/pyvene), [mamba-ssm](https://github.com/state-spaces/mamba), and [Pythia](https://github.com/EleutherAI/pythia) for their open-source contributions.
