#!/usr/bin/env python3
"""Width scaling experiments: vary d_h in {32, 64, 128, 256, 512, 1024}."""
import argparse
from scripts.train_from_scratch import build_model, set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["tf", "lstm", "mamba"])
    p.add_argument("--d_h_values", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--phenomena", nargs="+", default=["agreement", "npi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output_dir", default="results/width")
    args = p.parse_args()

    for arch in args.archs:
        for d_h in args.d_h_values:
            for phen in args.phenomena:
                for seed in args.seeds:
                    set_seed(seed)
                    model = build_model(arch, d_h=d_h)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  {arch} d_h={d_h} ({n_params:,} params) {phen} seed={seed}")
                    # Train and evaluate (see train_from_scratch.py)

if __name__ == "__main__":
    main()
