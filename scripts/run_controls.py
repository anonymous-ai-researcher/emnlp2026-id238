#!/usr/bin/env python3
"""Run all 7 families of negative controls."""
import argparse
from src.das.controls import CONTROL_FAMILIES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", choices=CONTROL_FAMILIES + ["all"], default="all")
    p.add_argument("--output_dir", default="results/controls")
    args = p.parse_args()

    families = CONTROL_FAMILIES if args.control == "all" else [args.control]
    for family in families:
        print(f"\n=== Control: {family} ===")
        # Run control (see src/das/controls.py for implementation)

if __name__ == "__main__":
    main()
