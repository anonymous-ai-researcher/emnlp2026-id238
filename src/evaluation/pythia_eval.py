"""Evaluation on pre-trained Pythia models (160M, 410M, 1.4B)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PYTHIA_MODELS = {
    "160m": "EleutherAI/pythia-160m", "410m": "EleutherAI/pythia-410m",
    "1.4b": "EleutherAI/pythia-1.4b",
}

def load_pythia(size="160m", device="cuda"):
    name = PYTHIA_MODELS[size]
    model = AutoModelForCausalLM.from_pretrained(name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def eval_blimp(model, tokenizer, paradigm_file, device="cuda"):
    """Evaluate BLiMP paradigm: accuracy = P(good) > P(bad)."""
    import json
    correct, total = 0, 0
    with open(paradigm_file) as f:
        for line in f:
            item = json.loads(line)
            good_ids = tokenizer(item["sentence_good"], return_tensors="pt").input_ids.to(device)
            bad_ids = tokenizer(item["sentence_bad"], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                good_ll = model(good_ids, labels=good_ids).loss.item() * -good_ids.size(1)
                bad_ll = model(bad_ids, labels=bad_ids).loss.item() * -bad_ids.size(1)
            correct += int(good_ll > bad_ll)
            total += 1
    return correct / total if total > 0 else 0.0
