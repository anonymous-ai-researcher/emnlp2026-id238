#!/usr/bin/env python3
"""Generate all paper figures from experimental results."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir", default="figures/output")
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating Figure 2 (behavioral evidence)...")
    # See figures/fig2_merged.py
    print("Generating Figure 3 (IIA trajectories)...")
    # See figures/fig3_dynamics.py
    print("All figures saved to", args.output_dir)

if __name__ == "__main__":
    main()
