"""Two-layer RWKV-4 classifier with linear attention."""
import torch, torch.nn as nn

class RWKV_TimeMix(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.W_r = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)
        self.time_decay = nn.Parameter(torch.randn(d_h) * 0.1)

    def forward(self, x):
        B, T, C = x.shape
        r, k, v = torch.sigmoid(self.W_r(x)), self.W_k(x), self.W_v(x)
        out = torch.zeros_like(x)
        state = torch.zeros(B, C, C, device=x.device)
        for t in range(T):
            kv = torch.einsum("bi,bj->bij", k[:, t], v[:, t])
            state = state * torch.exp(-torch.abs(self.time_decay)).unsqueeze(-1) + kv
            out[:, t] = torch.einsum("bi,bij->bj", r[:, t], state)
        return self.W_o(out)

class RWKV_ChannelMix(nn.Module):
    def __init__(self, d_h, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_h * 4
        self.W_k = nn.Linear(d_h, d_ff, bias=False)
        self.W_v = nn.Linear(d_ff, d_h, bias=False)
        self.W_r = nn.Linear(d_h, d_h, bias=False)
    def forward(self, x):
        return torch.sigmoid(self.W_r(x)) * self.W_v(torch.relu(self.W_k(x)) ** 2)

class RWKVClassifier(nn.Module):
    def __init__(self, vocab_size=11, d_embed=32, d_h=128, n_layers=2,
                 d_ff=512, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.proj = nn.Linear(d_embed, d_h) if d_embed != d_h else nn.Identity()
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "ln1": nn.LayerNorm(d_h), "time_mix": RWKV_TimeMix(d_h),
            "ln2": nn.LayerNorm(d_h), "channel_mix": RWKV_ChannelMix(d_h, d_ff),
        }) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_h)
        self.head = nn.Linear(d_h, n_classes)
        self._d_h = d_h

    def forward(self, x, return_hidden=False):
        h = self.proj(self.embed(x))
        for b in self.blocks:
            h = h + b["time_mix"](b["ln1"](h))
            h = h + b["channel_mix"](b["ln2"](h))
        h = self.ln_out(h)
        logits = self.head(h[:, -1])
        return (logits, h) if return_hidden else logits
