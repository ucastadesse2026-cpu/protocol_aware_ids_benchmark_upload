"""SYNTHETIC flow records with UNSW-NB15-style column names (dur, spkts, dpkts, sbytes, dbytes, label).

NOT REAL DATA: used only by the smoke configuration and tests when the public datasets are unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_unsw_like(n: int, seed: int, attack_fraction: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < attack_fraction).astype(int)
    dur = np.where(y == 1, rng.lognormal(-2.2, 1.0, n), rng.lognormal(-0.5, 1.2, n))
    spk = np.where(y == 1, rng.poisson(30, n), rng.poisson(10, n)) + 1
    dpk = np.where(y == 1, rng.poisson(3, n), rng.poisson(12, n)) + 1
    sb = spk * np.where(y == 1, rng.lognormal(4.3, 0.5, n), rng.lognormal(5.6, 0.7, n))
    db = dpk * np.where(y == 1, rng.lognormal(4.0, 0.6, n), rng.lognormal(6.0, 0.8, n))
    return pd.DataFrame({"dur": dur, "spkts": spk, "dpkts": dpk, "sbytes": sb, "dbytes": db, "label": y})
