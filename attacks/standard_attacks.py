"""Standard (unconstrained) l_inf feature-space attacks: FGSM, PGD, and an APGD-CE (AutoAttack-style) attack via ART.

Threat model used throughout the benchmark: the adversary perturbs only attack-class samples (evasion);
benign samples are left untouched. Inputs are min-max scaled features; ``eps`` is in scaled units.
``clip=(0, 1)`` keeps terrestrial examples inside the data box; pass ``clip=None`` for the satellite domain
(the unconstrained adversary is not told which values are realisable there).

AutoAttack: only ART's APGD-CE component is used (Square is image-only in ART; APGD-DLR needs more than two classes).
This is NOT the full reference AutoAttack of Croce & Hein; report it as "APGD-CE (ART)".
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

Clip = Optional[Tuple[float, float]]


def _clip(X: np.ndarray, clip: Clip) -> np.ndarray:
    return X if clip is None else np.clip(X, clip[0], clip[1])


def fgsm(grad_model, X: np.ndarray, y: np.ndarray, eps: float, clip: Clip = (0.0, 1.0)) -> np.ndarray:
    """x' = x + eps * sign(grad_x CE(f(x), y))."""
    return _clip(X + eps * np.sign(grad_model.loss_gradient(X, y)), clip)


def pgd(grad_model, X: np.ndarray, y: np.ndarray, eps: float, steps: int = 10, step_size: Optional[float] = None,
        random_start: bool = True, seed: int = 0, clip: Clip = (0.0, 1.0)) -> np.ndarray:
    """l_inf PGD with optional random start (Madry et al.)."""
    rng = np.random.default_rng(seed)
    alpha = step_size if step_size is not None else eps / 4.0
    Xa = X + (rng.uniform(-eps, eps, X.shape) if random_start else 0.0)
    Xa = _clip(np.clip(Xa, X - eps, X + eps), clip)
    for _ in range(steps):
        Xa = Xa + alpha * np.sign(grad_model.loss_gradient(Xa, y))
        Xa = _clip(np.clip(Xa, X - eps, X + eps), clip)
    return Xa


def perturb_positives(attack: Callable[[np.ndarray, np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply ``attack`` only to rows with y == 1; benign rows are returned unchanged."""
    Xa = X.copy()
    m = np.asarray(y) == 1
    if m.any():
        Xa[m] = attack(X[m], y[m])
    return Xa


def run_standard_attack(name: str, grad_model, X, y, eps: float, clip: Clip, seed: int = 0, pgd_steps: int = 10,
                        torch_model=None) -> np.ndarray:
    """Dispatch by attack name; ``grad_model`` supplies gradients (the target itself or a transfer surrogate)."""
    if name == "fgsm":
        return perturb_positives(lambda a, b: fgsm(grad_model, a, b, eps, clip), X, y)
    if name == "pgd":
        return perturb_positives(lambda a, b: pgd(grad_model, a, b, eps, pgd_steps, None, True, seed, clip), X, y)
    if name == "autoattack":
        return perturb_positives(lambda a, b: autoattack_art(torch_model or grad_model, a, b, eps, clip), X, y)
    raise ValueError(f"unknown standard attack {name!r}")


def autoattack_art(model, X: np.ndarray, y: np.ndarray, eps: float, clip: Clip = (0.0, 1.0), max_iter: int = 50) -> np.ndarray:
    """AutoAttack-style attack through ART: ONLY the APGD component with cross-entropy loss.

    Why not the full ensemble: ART's Square attack accepts image inputs only (it raises on tabular data) and the
    APGD-DLR loss is undefined for two classes. This is therefore ONE component of the reference AutoAttack suite
    of Croce & Hein, not the whole suite; describe it as "APGD-CE (ART)" in the paper.
    Needs torch + adversarial-robustness-toolbox and a ``TorchIDS`` (or a wrapper exposing ``.clf``).
    """
    try:
        import torch.nn as nn
        from art.attacks.evasion import AutoProjectedGradientDescent
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        raise ImportError("autoattack needs `torch` and `adversarial-robustness-toolbox`") from exc
    base = getattr(model, "clf", model)
    est = PyTorchClassifier(model=base.net, loss=nn.CrossEntropyLoss(), input_shape=(base.d,), nb_classes=2,
                            clip_values=clip, device_type="cpu")
    apgd = AutoProjectedGradientDescent(est, norm=np.inf, eps=eps, eps_step=eps / 4, max_iter=max_iter, targeted=False,
                                        nb_random_init=1, batch_size=256, loss_type="cross_entropy")
    return apgd.generate(X.astype(np.float32), np.eye(2, dtype=np.float32)[np.asarray(y, dtype=int)]).astype(np.float64)
