"""CFG-based sentence generator for all 6 syntactic phenomena."""
import argparse, json, random
from pathlib import Path

PHENOMENA = ["agreement", "npi", "reflexive", "concord", "mc_npi", "cc_binding"]
DISTANCE_RANGES = {
    "agreement": [0, 1, 2, 4, 6, 8], "npi": [0, 1, 2, 4, 6, 8],
    "reflexive": [0, 1, 2, 4, 6, 8], "concord": [0, 1],
    "mc_npi": [1, 2, 3], "cc_binding": [2, 4, 6, 8],
}

def _flip(tok):
    mapping = {"n_sg":"n_pl","n_pl":"n_sg","v_sg":"v_pl","v_pl":"v_sg",
               "det_sg":"det_pl","det_pl":"det_sg","refl_sg":"refl_pl","refl_pl":"refl_sg"}
    return mapping.get(tok, tok)

def _rand_noun(): return random.choice(["n_sg", "n_pl"])
def _attractors(d): return sum([["other", _rand_noun()] for _ in range(d)], [])

def gen_agreement(d, correct=True):
    subj = _rand_noun()
    verb = ("v_pl" if subj == "n_pl" else "v_sg")
    if not correct: verb = _flip(verb)
    return [subj] + _attractors(d) + [verb], int(subj == "n_pl"), int(verb == "v_pl")

def gen_npi(d, correct=True):
    has_lic = random.choice([True, False])
    prefix = ["lic"] if has_lic else ["other"]
    filler = _attractors(d)
    npi_tok = "npi"
    label = 1 if has_lic else 0
    if not correct: label = 1 - label
    return prefix + filler + [npi_tok, "other"], int(has_lic), label

def gen_reflexive(d, correct=True):
    ante = _rand_noun()
    refl = ("refl_pl" if ante == "n_pl" else "refl_sg")
    if not correct: refl = _flip(refl)
    return [ante] + _attractors(d) + ["other", refl], int(ante == "n_pl"), int(refl == "refl_pl")

def gen_concord(d, correct=True):
    det = random.choice(["det_sg", "det_pl"])
    noun = ("n_pl" if det == "det_pl" else "n_sg")
    if not correct: noun = _flip(noun)
    return [det] + ["other"] * d + [noun, "other"], int(det == "det_pl"), int(noun == "n_pl")

def gen_mc_npi(depth, correct=True):
    has_lic = random.choice([True, False])
    toks = ["lic"] if has_lic else ["other"]
    for _ in range(depth):
        toks.extend(["other", _rand_noun(), "other"])
    toks.extend(["npi", "other"])
    label = 1 if has_lic else 0
    if not correct: label = 1 - label
    return toks, int(has_lic), label

def gen_cc_binding(d, correct=True):
    ante = _rand_noun()
    refl = ("refl_pl" if ante == "n_pl" else "refl_sg")
    if not correct: refl = _flip(refl)
    return [_rand_noun(), "other", "other", ante] + _attractors(d) + ["other", refl], \
           int(ante == "n_pl"), int(refl == "refl_pl")

GENERATORS = dict(agreement=gen_agreement, npi=gen_npi, reflexive=gen_reflexive,
                  concord=gen_concord, mc_npi=gen_mc_npi, cc_binding=gen_cc_binding)

def generate_dataset(phen, sizes, out_dir, seed=42):
    random.seed(seed)
    gen = GENERATORS[phen]; dists = DISTANCE_RANGES[phen]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for split, total in sizes.items():
        samples = []; per_d = total // len(dists)
        for d in dists:
            for _ in range(per_d // 2):
                for correct in [True, False]:
                    toks, atomic_val, comp_val = gen(d, correct)
                    samples.append({"tokens": toks, "label": int(correct), "distance": d,
                                    "atomic": atomic_val, "comp": comp_val})
        random.shuffle(samples)
        with open(f"{out_dir}/{split}.jsonl", "w") as f:
            for s in samples: f.write(json.dumps(s) + "\n")
        print(f"  {phen}/{split}: {len(samples)} samples")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phenomenon", choices=PHENOMENA + ["all"], default="all")
    p.add_argument("--output_dir", default="data/generated")
    p.add_argument("--train_size", type=int, default=50000)
    p.add_argument("--test_size", type=int, default=10000)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    phens = PHENOMENA if args.all or args.phenomenon == "all" else [args.phenomenon]
    splits = {"train": args.train_size, "val": args.train_size // 10, "test": args.test_size}
    for phen in phens:
        print(f"Generating {phen}...")
        generate_dataset(phen, splits, f"{args.output_dir}/{phen}")
