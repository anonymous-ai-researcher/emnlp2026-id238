#!/usr/bin/env python3
"""Train from-scratch models on syntactic tasks."""
import argparse, json, os, random, torch
import numpy as np
from pathlib import Path

ARCHS = {"tf": "TransformerClassifier", "lstm": "LSTMClassifier",
         "mamba": "MambaClassifier", "gru": "GRUClassifier", "rwkv": "RWKVClassifier"}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(arch, d_h=128, **kwargs):
    if arch == "tf":
        from src.models.transformer import TransformerClassifier
        return TransformerClassifier(d_h=d_h, **kwargs)
    elif arch == "lstm":
        from src.models.lstm import LSTMClassifier
        return LSTMClassifier(d_h=d_h, **kwargs)
    elif arch == "mamba":
        from src.models.mamba_model import MambaClassifier
        return MambaClassifier(d_h=d_h, **kwargs)
    elif arch == "gru":
        from src.models.gru import GRUClassifier
        return GRUClassifier(d_h=d_h, **kwargs)
    elif arch == "rwkv":
        from src.models.rwkv import RWKVClassifier
        return RWKVClassifier(d_h=d_h, **kwargs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHS.keys()))
    p.add_argument("--phenomenon", required=True)
    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--data_dir", default="data/generated")
    p.add_argument("--output_dir", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in args.seeds:
        print(f"\n=== {args.arch} / {args.phenomenon} / seed={seed} ===")
        set_seed(seed)
        model = build_model(args.arch, d_h=args.d_h, n_layers=args.n_layers)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        # Training loop (see src/training/trainer.py for full implementation)
        print(f"  Training for {args.steps} steps...")
        out_dir = Path(args.output_dir) / args.phenomenon / args.arch / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # ... training code ...
        torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"  Saved to {out_dir}")

if __name__ == "__main__":
    main()
