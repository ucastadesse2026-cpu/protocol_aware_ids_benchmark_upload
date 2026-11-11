"""Transformer encoder IDS: each scalar feature becomes a token (value embedding + learned feature-position embedding)."""
from __future__ import annotations

from models.torch_base import TorchIDS, nn, require_torch, torch


def build_transformer(d: int, cfg: dict, seed: int) -> TorchIDS:
    require_torch()
    p = cfg.get("models", {}).get("transformer", {})
    dm, heads, ff, nl = p.get("d_model", 64), p.get("heads", 4), p.get("ff", 128), p.get("layers", 3)

    class Net(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.embed = nn.Linear(1, dm)
            self.pos = nn.Parameter(torch.zeros(1, d_in, dm))
            enc = nn.TransformerEncoderLayer(dm, heads, ff, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, nl, enable_nested_tensor=False)
            self.head = nn.Linear(dm, 2)

        def forward(self, x):
            h = self.enc(self.embed(x.unsqueeze(-1)) + self.pos)
            return self.head(h.mean(dim=1))

    return TorchIDS("transformer", Net, d, lr=p.get("lr", 5e-4), batch_size=p.get("batch_size", 256),
                    epochs=p.get("epochs", 50), seed=seed)
