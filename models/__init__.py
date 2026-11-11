"""Model registry. Torch-based models are imported lazily so that numpy/RF configs run without PyTorch."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

CENTRAL_MODELS = ("cnn", "lstm", "transformer", "random_forest", "gan_augmented", "numpy_mlp")
FAMILIES = ("cnn", "lstm", "transformer", "random_forest", "fedavg", "gan_augmented")


def build_model(name: str, d: int, cfg: Dict[str, Any], seed: int):
    """Build an untrained centralised model. (FedAvg is assembled in experiments.common.train_model.)"""
    mc = cfg.get("models", {})
    if name == "numpy_mlp":
        from models.numpy_mlp import NumpyMLP
        p = mc.get("numpy_mlp", {})
        return NumpyMLP(d, hidden=tuple(p.get("hidden", (64, 32))), lr=p.get("lr", 5e-3),
                        epochs=p.get("epochs", 20), seed=seed)
    if name == "random_forest":
        from models.random_forest import RandomForestIDS
        p = mc.get("random_forest", {})
        return RandomForestIDS(d, n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth"), seed=seed)
    if name == "cnn":
        from models.cnn import build_cnn
        return build_cnn(d, cfg, seed)
    if name == "lstm":
        from models.lstm import build_lstm
        return build_lstm(d, cfg, seed)
    if name == "transformer":
        from models.transformer import build_transformer
        return build_transformer(d, cfg, seed)
    if name == "gan_augmented":
        from models.gan_augmented import GANAugmentedIDS
        return GANAugmentedIDS(d, cfg, seed)
    raise ValueError(f"unknown model {name!r}")
