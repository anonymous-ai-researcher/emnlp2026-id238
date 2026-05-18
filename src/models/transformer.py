"""Two-layer causal Transformer with pre-norm and sinusoidal PE."""
import math, torch, torch.nn as nn

class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=11, d_embed=32, d_h=128, n_heads=4, n_layers=2,
                 d_ff=512, dropout=0.0, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.proj = nn.Linear(d_embed, d_h) if d_embed != d_h else nn.Identity()
        self.pe = SinusoidalPE(d_h)
        layer = nn.TransformerEncoderLayer(d_model=d_h, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="relu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_h, n_classes)
        self._d_h = d_h
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight) if m.weight.dim() > 1 else None

    def forward(self, x, return_hidden=False):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.pe(self.proj(self.embed(x)))
        h = self.encoder(h, mask=mask, is_causal=True)
        logits = self.head(h[:, -1])
        return (logits, h) if return_hidden else logits
