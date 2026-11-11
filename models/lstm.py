"""LSTM IDS. DEVIATION FROM THE PAPER DRAFT: the d scalar features of one flow/window are treated as a
length-d token sequence (no temporal windows of consecutive flows), because terrestrial datasets are
flow-level. Document this in the paper."""
from __future__ import annotations

from models.torch_base import TorchIDS, nn, require_torch


def build_lstm(d: int, cfg: dict, seed: int) -> TorchIDS:
    require_torch()
    p = cfg.get("models", {}).get("lstm", {})
    hidden, layers, drop = p.get("hidden", 128), p.get("layers", 2), p.get("dropout", 0.3)

    class Net(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.rnn = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))

        def forward(self, x):
            out, _ = self.rnn(x.unsqueeze(-1))
            return self.head(out[:, -1])

    return TorchIDS("lstm", Net, d, lr=p.get("lr", 1e-3), batch_size=p.get("batch_size", 256),
                    epochs=p.get("epochs", 50), seed=seed)
