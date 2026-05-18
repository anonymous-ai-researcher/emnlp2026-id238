"""Distributed Alignment Search with Cayley-parameterized orthogonal rotations."""
import torch, torch.nn as nn

class CayleyRotation(nn.Module):
    """Orthogonal matrix via Cayley transform of a skew-symmetric matrix."""
    def __init__(self, d_h):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_h, d_h) * 0.01)
        self._d_h = d_h

    def forward(self):
        S = self.A - self.A.T  # skew-symmetric
        I = torch.eye(self._d_h, device=S.device)
        return torch.linalg.solve(I + S, I - S)  # Cayley: (I+S)^{-1}(I-S)

class DAS(nn.Module):
    """Distributed Alignment Search for a single causal variable."""
    def __init__(self, d_h, rank=1):
        super().__init__()
        self.rotation = CayleyRotation(d_h)
        self.rank = rank

    def intervene(self, h_base, h_source):
        """Interchange intervention: replace first `rank` rotated dims."""
        R = self.rotation()  # (d_h, d_h) orthogonal
        R_r = R[:, :self.rank]  # (d_h, rank)
        proj_b = R_r @ (R_r.T @ h_base.unsqueeze(-1))  # project base
        proj_s = R_r @ (R_r.T @ h_source.unsqueeze(-1))  # project source
        return h_base + (proj_s - proj_b).squeeze(-1)

    def forward(self, model, x_base, x_source, layer, position):
        """Run model with interchange intervention at (layer, position)."""
        # Get base and source hidden states
        with torch.no_grad():
            _, h_base_all = model(x_base, return_hidden=True)
            _, h_source_all = model(x_source, return_hidden=True)
        h_b = h_base_all[:, position]
        h_s = h_source_all[:, position]
        h_int = self.intervene(h_b, h_s)
        # Re-run from intervention point (simplified: inject at last layer)
        h_modified = h_base_all.clone()
        h_modified[:, position] = h_int
        logits = model.head(h_modified[:, -1])
        return logits

def compute_iia(model, das, dataloader, layer, position, device="cuda"):
    """Compute interchange intervention accuracy on held-out data."""
    correct, total = 0, 0
    das.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_b, x_s, y_cf = batch["base"].to(device), batch["source"].to(device), batch["y_cf"].to(device)
            logits = das(model, x_b, x_s, layer, position)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_cf).sum().item()
            total += y_cf.size(0)
    return correct / total if total > 0 else 0.0
