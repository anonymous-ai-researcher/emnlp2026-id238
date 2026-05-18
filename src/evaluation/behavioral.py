"""Behavioral accuracy evaluation per distance level."""
import torch
from collections import defaultdict

def evaluate_per_distance(model, test_loader, device="cuda"):
    """Evaluate accuracy broken down by attractor distance."""
    model.eval()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    with torch.no_grad():
        for batch in test_loader:
            x = batch["tokens"].to(device)
            y = batch["label"].to(device)
            d = batch["distance"]
            preds = model(x).argmax(dim=-1)
            for i in range(len(y)):
                dist = d[i].item()
                stats[dist]["total"] += 1
                stats[dist]["correct"] += int(preds[i] == y[i])
    return {d: v["correct"] / v["total"] for d, v in sorted(stats.items())}

def evaluate_overall(model, test_loader, device="cuda"):
    """Overall accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch["tokens"].to(device), batch["label"].to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
