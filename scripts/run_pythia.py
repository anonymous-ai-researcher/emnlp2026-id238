#!/usr/bin/env python3
"""Evaluate Pythia models (160M, 410M, 1.4B) on BLiMP and DAS."""
import argparse
from src.evaluation.pythia_eval import load_pythia, eval_blimp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["160m", "410m", "1.4b"])
    p.add_argument("--eval_type", choices=["blimp", "das", "both"], default="both")
    p.add_argument("--output_dir", default="results/pythia")
    args = p.parse_args()

    for size in args.models:
        print(f"\n=== Pythia {size} ===")
        if args.eval_type in ("blimp", "both"):
            model, tokenizer = load_pythia(size)
            # Evaluate BLiMP paradigms
        if args.eval_type in ("das", "both"):
            # Run DAS on Pythia hidden states
            pass

if __name__ == "__main__":
    main()
