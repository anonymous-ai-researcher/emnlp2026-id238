"""Two-layer Mamba (selective SSM) classifier."""
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class MambaClassifier(nn.Module):
    def __init__(self, vocab_size=11, d_embed=32, d_h=128, n_layers=2,
                 ssm_state=16, n_classes=2):
        super().__init__()
        assert Mamba is not None, "Install mamba-ssm: pip install mamba-ssm"
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.proj = nn.Linear(d_embed, d_h) if d_embed != d_h else nn.Identity()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_h), Mamba(d_model=d_h, d_state=ssm_state, d_conv=4, expand=2))
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_h)
        self.head = nn.Linear(d_h, n_classes)
        self._d_h = d_h

    def forward(self, x, return_hidden=False):
        h = self.proj(self.embed(x))
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm(h)
        logits = self.head(h[:, -1])
        return (logits, h) if return_hidden else logits
