"""Protocol-aware adversarial augmentation (PAA).

Builds an augmentation set from the SATELLITE TRAINING split: windows are drawn up to ``lam * n_terrestrial``
rows; a fraction ``adv_fraction`` of the attack-class windows is replaced by CCSDS-conformant adversarial
versions (pgd-ccsds against a differentiable surrogate; every example passed the C1-C8 validator).

CONFOUND CONTROL: PAA also exposes the model to satellite-domain windows, so any gain may come from domain
exposure rather than adversarial content. ``clean_control=True`` builds the same set WITHOUT perturbation;
report both. The augmentation set is generated once (no per-epoch refresh).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from attacks.ccsds_conformant import CcsdsAttackConfig, attack_windows
from attacks.validator import MissionProfile
from data.ccsds_parser import Window
from data.feature_schema import FeaturePipeline


def build_paa_set(windows_train: Sequence[Window], surrogate, pipeline: FeaturePipeline, profile: MissionProfile,
                  atk_cfg: CcsdsAttackConfig, n_terrestrial: int, lam: float = 0.25, adv_fraction: float = 0.5,
                  seed: int = 0, clean_control: bool = False, max_attack_windows: int = 500
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = min(int(lam * n_terrestrial), len(windows_train))
    idx = rng.choice(len(windows_train), size=n, replace=False)
    chosen = [windows_train[i] for i in idx]
    t0 = time.perf_counter()
    info: Dict[str, Any] = dict(n_windows=n, n_adv=0, feasibility_rate=float("nan"), clean_control=clean_control)
    if not clean_control:
        att = [k for k, w in enumerate(chosen) if w.label == 1]
        pert = [k for k in att if rng.random() < adv_fraction][:max_attack_windows]
        if pert:
            adv, stats = attack_windows("pgd_ccsds", [chosen[k] for k in pert], surrogate, pipeline, profile, atk_cfg)
            for k, w in zip(pert, adv):
                chosen[k] = w
            info["n_adv"] = len(pert)
            info["feasibility_rate"] = float(np.mean([s["feasible"] for s in stats]))
    info["gen_time_s"] = time.perf_counter() - t0
    X = pipeline.windows(chosen) if chosen else np.zeros((0, len(pipeline.scaler.lo)))
    y = np.array([w.label for w in chosen], dtype=int)
    return X, y, info
