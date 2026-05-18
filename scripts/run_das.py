#!/usr/bin/env python3
"""Run DAS evaluation on trained models."""
import argparse, json, torch
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["tf", "lstm", "mamba"])
    p.add_argument("--phenomenon", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--das_lr", type=float, default=0.01)
    p.add_argument("--das_steps", type=int, default=2000)
    p.add_argument("--das_batch", type=int, default=256)
    p.add_argument("--das_inits", type=int, default=3)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--output_dir", default="results/das")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for arch in args.archs:
        results[arch] = {}
        for var in args.variables:
            best_iia = 0.0
            for rank in args.ranks:
                for init in range(args.das_inits):
                    # Load model, build DAS, train, evaluate
                    # (see src/das/alignment.py and src/evaluation/iia.py)
                    pass
            results[arch][var] = best_iia
            print(f"  {arch}/{var}: IIA = {best_iia:.3f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.phenomenon}_iia.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
