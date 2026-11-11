"""Transferability score tau = [Acc(S,ccsds | defended) - Acc(S,ccsds | base)] / [Acc(T,std | defended) - Acc(T,std | base)].

Computed over paired seeds with a bootstrap CI. tau is reported as not applicable (applicable=False) when the
bootstrap CI of the denominator (terrestrial robustness gain) includes zero OR the mean gain is below ``min_gain``
(default 0.01 = 1 percentage point), because tau is unbounded when the denominator is near zero.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def transferability_score(def_S: Sequence[float], base_S: Sequence[float], def_T: Sequence[float],
                          base_T: Sequence[float], n_boot: int = 2000, seed: int = 0,
                          min_gain: float = 0.01) -> Dict[str, float]:
    num = np.asarray(def_S, float) - np.asarray(base_S, float)
    den = np.asarray(def_T, float) - np.asarray(base_T, float)
    n = len(num)
    rng = np.random.default_rng(seed)
    taus, dens = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d = den[i].mean()
        dens.append(d)
        taus.append(num[i].mean() / d if abs(d) > 1e-12 else np.nan)
    lo_d, hi_d = np.percentile(dens, [2.5, 97.5])
    applicable = bool(n >= 2 and (lo_d > 0 or hi_d < 0) and abs(den.mean()) >= min_gain)
    d_mean = den.mean()
    tau = float(num.mean() / d_mean) if abs(d_mean) > 1e-12 else float("nan")
    lo, hi = np.nanpercentile(taus, [2.5, 97.5]) if np.isfinite(taus).any() else (np.nan, np.nan)
    return dict(tau=tau if applicable else float("nan"), tau_raw=tau, ci_low=float(lo), ci_high=float(hi), applicable=applicable)
