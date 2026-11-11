"""Pure-NumPy MLP with analytic input gradients.

Purpose: (i) a dependency-free differentiable model so the attack/defense/FL logic can be tested without
PyTorch, (ii) a cheap surrogate for transfer attacks. It is NOT one of the six IDS families of the paper.

Model interface shared by all models in this repo:
    fit(X, y, X_val=None, y_val=None) / partial_fit(X, y, epochs=1, reset_optimizer=False)
    predict_proba(X) -> (n, 2)     predict(X) -> (n,)
    loss_gradient(X, y) -> (n, d)  [differentiable models only]
    get_weights() -> list[np.ndarray] / set_weights(list)
    n_params, size_bytes
"""
from __future__ import annotations

import pickle
from typing import List, Optional, Sequence

import numpy as np


class NumpyMLP:
    name = "numpy_mlp"
    differentiable = True

    def __init__(self, d: int, hidden: Sequence[int] = (64, 32), lr: float = 5e-3, batch_size: int = 256,
                 epochs: int = 20, seed: int = 0, class_weight: Optional[str] = "balanced", l2: float = 1e-5):
        self.d, self.hidden, self.lr, self.batch_size = d, tuple(hidden), lr, batch_size
        self.epochs, self.seed, self.class_weight, self.l2 = epochs, seed, class_weight, l2
        self.rng = np.random.default_rng(seed)
        dims = [d, *self.hidden, 2]
        self.W = [self.rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self._reset_adam()

    # -- optimisation state
    def _reset_adam(self) -> None:
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, X: np.ndarray):
        acts = [np.asarray(X, dtype=float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0.0) if i < len(self.W) - 1 else z)
        return acts

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self._forward(X)[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _backprop(self, acts, delta):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (acts[i] > 0)
        return gW, gb, delta        # delta now = dLoss/dInput (per-sample, un-averaged)

    def loss_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of per-sample cross-entropy w.r.t. the input features."""
        acts = self._forward(X)
        p = self._softmax(acts[-1])
        p[np.arange(len(y)), np.asarray(y, dtype=int)] -= 1.0
        return self._backprop(acts, p)[2]

    # -- training
    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, reset_optimizer: bool = False) -> "NumpyMLP":
        if reset_optimizer:
            self._reset_adam()
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        cw = np.ones(2)
        if self.class_weight == "balanced":
            cnt = np.bincount(y, minlength=2).astype(float) + 1.0
            cw = len(y) / (2.0 * cnt)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            for s in range(0, len(X), self.batch_size):
                idx = order[s:s + self.batch_size]
                acts = self._forward(X[idx])
                p = self._softmax(acts[-1])
                p[np.arange(len(idx)), y[idx]] -= 1.0
                p *= cw[y[idx]][:, None] / len(idx)
                gW, gb, _ = self._backprop(acts, p)
                self.t += 1
                for i in range(len(self.W)):
                    gW[i] = gW[i] + self.l2 * self.W[i]
                    for (m, v, par, g) in ((self.mW, self.vW, self.W, gW), (self.mb, self.vb, self.b, gb)):
                        m[i] = b1 * m[i] + (1 - b1) * g[i]
                        v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                        mh, vh = m[i] / (1 - b1 ** self.t), v[i] / (1 - b2 ** self.t)
                        par[i] = par[i] - self.lr * mh / (np.sqrt(vh) + eps)
        return self

    def fit(self, X, y, X_val=None, y_val=None) -> "NumpyMLP":
        return self.partial_fit(X, y, epochs=self.epochs)

    # -- weights / size
    def get_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]

    def set_weights(self, ws: List[np.ndarray]) -> None:
        n = len(self.W)
        self.W = [np.array(w, dtype=float) for w in ws[:n]]
        self.b = [np.array(b, dtype=float) for b in ws[n:]]

    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    def size_bytes(self) -> int:
        return int(sum(w.astype(np.float32).nbytes for w in self.W + self.b))
