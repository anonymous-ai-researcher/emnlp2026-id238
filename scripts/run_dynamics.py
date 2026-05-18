#!/usr/bin/env python3
"""Track IIA trajectories during training (grokking dynamics)."""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["tf", "mamba"])
    p.add_argument("--phenomenon", default="agreement")
    p.add_argument("--n_checkpoints", type=int, default=15)
    p.add_argument("--das_inits", type=int, default=3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output_dir", default="results/dynamics")
    args = p.parse_args()

    for arch in args.archs:
        for seed in args.seeds:
            print(f"  {arch} seed={seed}: tracking {args.n_checkpoints} checkpoints...")
            # See src/training/dynamics.py for implementation

if __name__ == "__main__":
    main()
