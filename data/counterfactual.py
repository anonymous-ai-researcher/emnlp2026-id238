"""Construct counterfactual pairs (x_b, x_s, y_cf) for DAS evaluation."""
import argparse, json, random
from pathlib import Path
from data.generate import GENERATORS, DISTANCE_RANGES, PHENOMENA

def build_pairs(phen, variable, n_pairs, distances, seed=42):
    """Build counterfactual pairs for a single variable."""
    random.seed(seed)
    gen = GENERATORS[phen]
    pairs = []
    for _ in range(n_pairs):
        d = random.choice(distances)
        toks_b, at_b, co_b = gen(d, correct=True)
        toks_s, at_s, co_s = gen(d, correct=True)
        if variable.endswith("_num") or variable.startswith("has_") or variable.startswith("det_"):
            y_cf = int(at_s == co_b) if "form" in variable or "ok" in variable or "conc" in variable else co_b
            pairs.append({"base": toks_b, "source": toks_s, "y_cf": y_cf,
                          "distance": d, "variable": variable})
        else:
            y_cf = int(at_s == co_b)
            pairs.append({"base": toks_b, "source": toks_s, "y_cf": y_cf,
                          "distance": d, "variable": variable})
    return pairs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phenomenon", required=True, choices=PHENOMENA)
    p.add_argument("--variable", required=True)
    p.add_argument("--train_pairs", type=int, default=10000)
    p.add_argument("--test_pairs", type=int, default=2000)
    p.add_argument("--output_dir", default="data/counterfactual")
    args = p.parse_args()
    dists = DISTANCE_RANGES[args.phenomenon]
    out = Path(args.output_dir) / args.phenomenon / args.variable
    out.mkdir(parents=True, exist_ok=True)
    for split, n in [("train", args.train_pairs), ("test", args.test_pairs)]:
        pairs = build_pairs(args.phenomenon, args.variable, n, dists, seed=42 if split == "train" else 123)
        with open(out / f"{split}.jsonl", "w") as f:
            for p_ in pairs: f.write(json.dumps(p_) + "\n")
        print(f"  {split}: {len(pairs)} pairs")
