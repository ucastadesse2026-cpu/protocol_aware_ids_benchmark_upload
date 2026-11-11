"""Shared PyTorch training wrapper exposing the repo-wide model interface (see models/numpy_mlp.py).

NOTE: this module needs PyTorch. It was syntax-checked but could NOT be executed in the authoring sandbox
(no torch installed); run tests/test_torch_models.py on your machine before trusting results.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import os
    _threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(_threads)
    try:
        torch.set_num_interop_threads(_threads)
    except RuntimeError:
        pass
    HAS_TORCH = True
except ImportError:                                          # pragma: no cover
    torch = None
    nn = None
    HAS_TORCH = False


def require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this model. `pip install torch` or use configs/smoke.yaml")


class TorchIDS:
    differentiable = True

    def __init__(self, name: str, net_factory: Callable[[int], "nn.Module"], d: int, lr: float = 1e-3,
                 batch_size: int = 256, epochs: int = 50, seed: int = 0, patience: int = 5,
                 class_weight: Optional[str] = "balanced", device: str = "cpu"):
        require_torch()
        self.name, self.d, self.lr, self.batch_size, self.epochs = name, d, lr, batch_size, epochs
        self.seed, self.patience, self.class_weight, self.device = seed, patience, class_weight, device
        torch.manual_seed(seed)
        self.net = net_factory(d).to(device)
        self.gen = torch.Generator().manual_seed(seed)
        self._opt = None
        self._cw = None

    # ---------------------------------------------------------------- helpers
    def _t(self, X) -> "torch.Tensor":
        return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

    def _class_weights(self, y: np.ndarray) -> "torch.Tensor":
        if self.class_weight != "balanced":
            return torch.ones(2, device=self.device)
        cnt = np.bincount(y, minlength=2).astype(float) + 1.0
        return torch.as_tensor(len(y) / (2.0 * cnt), dtype=torch.float32, device=self.device)

    def _run_epochs(self, X: np.ndarray, y: np.ndarray, epochs: int, reset_optimizer: bool) -> None:
        if self._opt is None or reset_optimizer:
            self._opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss(weight=self._class_weights(np.asarray(y, dtype=int)))
        Xt, yt = self._t(X), torch.as_tensor(np.asarray(y, dtype=int), device=self.device)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=self.gen)
            for s in range(0, len(Xt), self.batch_size):
                idx = perm[s:s + self.batch_size]
                self._opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                self._opt.step()

    # ---------------------------------------------------------------- interface
    def partial_fit(self, X, y, epochs: int = 1, reset_optimizer: bool = False) -> "TorchIDS":
        self._run_epochs(X, y, epochs, reset_optimizer)
        return self

    def fit(self, X, y, X_val=None, y_val=None) -> "TorchIDS":
        if X_val is None:
            self._run_epochs(X, y, self.epochs, True)
            return self
        best, best_state, bad = np.inf, None, 0
        self._opt = None
        for ep in range(self.epochs):
            self._run_epochs(X, y, 1, ep == 0)
            vl = self._val_loss(X_val, y_val)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad() if HAS_TORCH else (lambda f: f)
    def _val_loss(self, X, y) -> float:
        self.net.eval()
        return float(nn.CrossEntropyLoss()(self.net(self._t(X)), torch.as_tensor(np.asarray(y, dtype=int))))

    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(self._t(X)), dim=1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss_gradient(self, X, y) -> np.ndarray:
        self.net.eval()
        x = self._t(X).clone().requires_grad_(True)
        loss = nn.CrossEntropyLoss(reduction="sum")(self.net(x), torch.as_tensor(np.asarray(y, dtype=int), device=self.device))
        grad, = torch.autograd.grad(loss, x)
        return grad.detach().cpu().numpy()

    def get_weights(self) -> List[np.ndarray]:
        return [v.detach().cpu().numpy().astype(np.float64) for v in self.net.state_dict().values()]

    def set_weights(self, ws: List[np.ndarray]) -> None:
        sd = self.net.state_dict()
        new = {k: torch.as_tensor(w, dtype=v.dtype) for (k, v), w in zip(sd.items(), ws)}
        self.net.load_state_dict(new)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def size_bytes(self) -> int:
        return int(sum(p.numel() * 4 for p in self.net.parameters()))      # float32 parameters
