"""Standard adversarial training (Madry et al.-style PGD AT) in scaled feature space.

Threat model consistent with the benchmark: only attack-class rows are perturbed (evasion). Each epoch a random
``mix`` fraction of attack-class rows is replaced by PGD examples generated against the CURRENT model.
Non-differentiable models (Random Forest) cannot be adversarially trained by gradient; they receive
adversarial *augmentation* with transfer examples from a differentiable surrogate (one-shot).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from attacks.standard_attacks import pgd


def _mixed_epoch_data(grad_model, X, y, eps, steps, mix, rng, clip) -> np.ndarray:
    Xt = X.copy()
    pos = np.where(y == 1)[0]
    if len(pos):
        sel = pos[rng.random(len(pos)) < mix]
        if len(sel):
            Xt[sel] = pgd(grad_model, X[sel], y[sel], eps, steps, None, True, int(rng.integers(1 << 30)), clip)
    return Xt


def adversarial_training(model, X: np.ndarray, y: np.ndarray, eps: float, epochs: int, steps: int = 5,
                         mix: float = 0.5, seed: int = 0, surrogate=None, clip=(0.0, 1.0),
                         extra: Optional[Tuple[np.ndarray, np.ndarray]] = None, X_val=None, y_val=None):
    """Train ``model`` with PGD-AT (``extra`` = static extra training rows, e.g. a PAA set, appended every epoch)."""
    rng = np.random.default_rng(seed)
    ex, ey = (extra if extra is not None else (np.zeros((0, X.shape[1])), np.zeros(0, dtype=int)))
    if getattr(model, "differentiable", False):
        # warm start with one clean epoch so that the first adversarial examples are meaningful
        model.partial_fit(np.vstack([X, ex]), np.concatenate([y, ey]), epochs=1, reset_optimizer=True)
        for _ in range(max(epochs - 1, 0)):
            Xt = _mixed_epoch_data(model, X, y, eps, steps, mix, rng, clip)
            model.partial_fit(np.vstack([Xt, ex]), np.concatenate([y, ey]), epochs=1)
        return model
    if surrogate is None:
        raise ValueError("non-differentiable model: adversarial augmentation needs a differentiable surrogate")
    Xt = _mixed_epoch_data(surrogate, X, y, eps, steps, mix, rng, clip)
    Xall = np.vstack([X, Xt[y == 1], ex])
    yall = np.concatenate([y, y[y == 1], ey])
    model.fit(Xall, yall, X_val, y_val)
    return model
