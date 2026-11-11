"""GAN-augmented IDS: a WGAN-GP is trained on the minority class of the TRAINING split only; synthetic
minority samples balance the data, then a CNN classifier is trained on real + synthetic data.

NOTE: needs PyTorch; could not be executed in the authoring sandbox (see models/torch_base.py)."""
from __future__ import annotations

import numpy as np

from models.cnn import build_cnn
from models.torch_base import nn, require_torch, torch


class GANAugmentedIDS:
    name = "gan_augmented"
    differentiable = True

    def __init__(self, d: int, cfg: dict, seed: int):
        require_torch()
        p = cfg.get("models", {}).get("gan_augmented", {})
        self.d, self.seed = d, seed
        self.z_dim, self.iters = p.get("z_dim", 32), p.get("gan_iters", 2000)
        self.n_critic, self.gp_lambda, self.gan_batch = 5, 10.0, 128
        self.clf = build_cnn(d, cfg, seed)
        self.G = nn.Sequential(nn.Linear(self.z_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d))
        self.D = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def _train_gan(self, Xmin: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        real = torch.as_tensor(Xmin, dtype=torch.float32)
        og = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        od = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.9))
        for _ in range(self.iters):
            for _ in range(self.n_critic):
                xr = real[torch.randint(0, len(real), (self.gan_batch,))]
                xf = self.G(torch.randn(self.gan_batch, self.z_dim)).detach()
                eps = torch.rand(self.gan_batch, 1)
                xh = (eps * xr + (1 - eps) * xf).requires_grad_(True)
                g, = torch.autograd.grad(self.D(xh).sum(), xh, create_graph=True)
                gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
                ld = self.D(xf).mean() - self.D(xr).mean() + self.gp_lambda * gp
                od.zero_grad(); ld.backward(); od.step()
            lg = -self.D(self.G(torch.randn(self.gan_batch, self.z_dim))).mean()
            og.zero_grad(); lg.backward(); og.step()

    def fit(self, X, y, X_val=None, y_val=None) -> "GANAugmentedIDS":
        X, y = np.asarray(X), np.asarray(y, dtype=int)
        minority = int(np.argmin(np.bincount(y, minlength=2)))
        n_need = int(np.bincount(y, minlength=2).max() - np.bincount(y, minlength=2).min())
        if n_need > 0 and (y == minority).sum() >= 10:
            self._train_gan(X[y == minority])
            with torch.no_grad():
                Xs = self.G(torch.randn(n_need, self.z_dim)).numpy()
            X, y = np.vstack([X, Xs]), np.concatenate([y, np.full(n_need, minority)])
        self.clf.fit(X, y, X_val, y_val)
        return self

    # delegate the rest of the interface to the classifier
    def partial_fit(self, X, y, epochs: int = 1, reset_optimizer: bool = False):
        self.clf.partial_fit(X, y, epochs, reset_optimizer); return self
    def predict_proba(self, X): return self.clf.predict_proba(X)
    def predict(self, X): return self.clf.predict(X)
    def loss_gradient(self, X, y): return self.clf.loss_gradient(X, y)
    def get_weights(self): return self.clf.get_weights()
    def set_weights(self, w): self.clf.set_weights(w)
    @property
    def n_params(self): return self.clf.n_params           # deployed model = classifier only
    def size_bytes(self): return self.clf.size_bytes()
