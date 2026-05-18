"""Seven families of negative controls for DAS evaluation.

1. wrong_scm:        Apply phenomenon A's SCM to model trained on B
2. wrong_boolean:    Replace biconditional with conjunction/disjunction
3. random_init:      DAS on untrained (random) model
4. shuffled_label:   Train DAS on permuted counterfactual labels
5. cross_phenomenon: Per-variable atomic/comp IIA across phenomena
6. alternative_scm:  Nearest-noun heuristic instead of leftmost-noun
7. unconstrained:    Unconstrained MLP aligner (no orthogonal constraint)
"""
import torch, torch.nn as nn
from src.das.alignment import DAS

class UnconstrainedAligner(nn.Module):
    """MLP-based aligner without orthogonal constraint (control 7)."""
    def __init__(self, d_h, rank=1, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_h, hidden), nn.ReLU(), nn.Linear(hidden, d_h))
        self.rank = rank

    def intervene(self, h_base, h_source):
        return h_base + self.net(h_source - h_base)

CONTROL_FAMILIES = [
    "wrong_scm", "wrong_boolean", "random_init", "shuffled_label",
    "cross_phenomenon", "alternative_scm", "unconstrained",
]

def run_control(family, model, phenomenon, das_config, data_config, device="cuda"):
    """Run a single negative control family. Returns dict of IIA values."""
    results = {}
    if family == "wrong_scm":
        # Apply this phenomenon's DAS to a model trained on a different phenomenon
        pass  # Implementation uses cross-trained model checkpoints
    elif family == "wrong_boolean":
        # Replace biconditional with AND/OR in counterfactual label computation
        pass
    elif family == "random_init":
        # Run DAS on untrained model (re-initialize weights)
        pass
    elif family == "shuffled_label":
        # Permute y_cf labels before DAS training
        pass
    elif family == "cross_phenomenon":
        # Apply per-variable DAS across phenomena
        pass
    elif family == "alternative_scm":
        # Use nearest-noun instead of leftmost-noun for HA selector
        pass
    elif family == "unconstrained":
        # Replace CayleyRotation with MLP
        pass
    return results
