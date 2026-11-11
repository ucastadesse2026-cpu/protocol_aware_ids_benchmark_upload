"""1-D CNN over the feature axis (3 conv blocks 32/64/128, kernel 3, global pooling, dropout)."""
from __future__ import annotations

from models.torch_base import TorchIDS, nn, require_torch


def build_cnn(d: int, cfg: dict, seed: int) -> TorchIDS:
    require_torch()
    p = cfg.get("models", {}).get("cnn", {})
    filters, drop = p.get("filters", [32, 64, 128]), p.get("dropout", 0.3)

    class Net(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            layers, c = [], 1
            for f in filters:
                layers += [nn.Conv1d(c, f, 3, padding=1), nn.ReLU()]
                c = f
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(c, 2))

        def forward(self, x):
            return self.head(self.pool(self.conv(x.unsqueeze(1))).squeeze(-1))

    return TorchIDS("cnn", Net, d, lr=p.get("lr", 1e-3), batch_size=p.get("batch_size", 256),
                    epochs=p.get("epochs", 50), seed=seed)
