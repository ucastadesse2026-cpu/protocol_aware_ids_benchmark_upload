"""Decomposed robustness gap  G = G_domain + G_protocol  (percentage points).

    G          = Acc(T, std) - Acc(S, ccsds)
    G_domain   = Acc(T, std) - Acc(S, std)      same unconstrained attack, different domain
    G_protocol = Acc(S, std) - Acc(S, ccsds)    effect of restricting the attack to valid CCSDS packets
H-blind: G_protocol > 0 (conformant attacks hurt more); H-weak: G_protocol < 0 (constraints weaken the attack).
Inputs are accuracies as fractions in [0, 1]; outputs are in percentage points.
"""
from __future__ import annotations

from typing import Dict


def decompose_gap(acc_T_std: float, acc_S_std: float, acc_S_ccsds: float) -> Dict[str, float]:
    g_dom = 100.0 * (acc_T_std - acc_S_std)
    g_pro = 100.0 * (acc_S_std - acc_S_ccsds)
    g = 100.0 * (acc_T_std - acc_S_ccsds)
    assert abs(g - (g_dom + g_pro)) < 1e-9
    return dict(G=g, G_domain=g_dom, G_protocol=g_pro,
                G_pct=100.0 * g / (100.0 * acc_T_std) if acc_T_std else float("nan"))


def clean_cross_domain_drop(acc_T_clean: float, acc_S_clean: float) -> float:
    """Control: accuracy lost by moving to satellite traffic with NO attacker (percentage points)."""
    return 100.0 * (acc_T_clean - acc_S_clean)
