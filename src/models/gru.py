"""Two-layer GRU classifier."""
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=11, d_embed=32, d_h=128, n_layers=2,
                 dropout=0.0, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.proj = nn.Linear(d_embed, d_h) if d_embed != d_h else nn.Identity()
        self.gru = nn.GRU(d_h, d_h, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(d_h, n_classes)
        self._d_h = d_h

    def forward(self, x, return_hidden=False):
        h = self.proj(self.embed(x))
        h, _ = self.gru(h)
        logits = self.head(h[:, -1])
        return (logits, h) if return_hidden else logits
