"""CCSDS-conformant (problem-space) evasion attacks: pgd-ccsds, transfer-ccsds, greedy-ccsds.

The adversary edits PACKETS, never feature vectors:
  * editable packets: only those of ``editable_types`` (default: TC, i.e. what a ground-side attacker emits)
  * mutable fields: trailing padding length (``pad``), inter-packet gaps (timing), and (greedy only) the APID
    within the allocated set; dependent fields (sequence counts, lengths, PEC) are repaired automatically
  * immutable fields (C8): packet types, opcodes, parameter bytes, window start time, label
  * a perturbed window counts as an adversarial example only if it passes the full C1-C8 validator;
    otherwise the attack is recorded as INFEASIBLE and the original window is kept (attack failed)

pgd-ccsds (white-box): iterate x-space gradient -> chain through a finite-difference Jacobian of the
feature map phi w.r.t. the edit parameters theta=(pad, log gap scale) -> signed step on theta -> clip to the
legal box -> re-realise packets. Budget B restricts the number of edited packets (top-B by gradient score).
transfer-ccsds: pgd-ccsds on a differentiable surrogate, evaluated on the target.
greedy-ccsds (black-box): query-only greedy search over discrete valid edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from attacks.validator import MissionProfile, validate_window
from data.ccsds_parser import SEQ_MOD, SUBSEC_RES, TC, TM, Window
from data.feature_schema import FeaturePipeline


@dataclass
class CcsdsAttackConfig:
    budget_packets: int = 8
    max_pad: int = 64
    gap_scale_min: float = 0.5
    gap_scale_max: float = 2.0
    steps: int = 15
    step_pad: float = 8.0
    step_log_gap: float = 0.15
    fd_pad: float = 4.0
    fd_log_gap: float = 0.1
    greedy_iters: int = 30
    greedy_candidates: int = 24
    pad_choices: Tuple[int, ...] = (4, 8, 16, 32)
    gap_choices: Tuple[float, ...] = (0.5, 0.7, 1.4, 2.0)
    editable_types: Tuple[int, ...] = (TC,)
    seed: int = 0

    @staticmethod
    def from_dict(d: Optional[dict]) -> "CcsdsAttackConfig":
        c = CcsdsAttackConfig()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, tuple(v) if isinstance(getattr(c, k), tuple) else v)
        return c


@dataclass
class AttackResult:
    window: Window
    success: bool            # target predicts benign on the (feasible) adversarial window
    feasible: bool           # passed C1-C8 (no new violations)
    n_edits: int
    queries: int
    p_before: float
    p_after: float
    violations: List[str] = field(default_factory=list)


class ConformantAttacker:
    def __init__(self, pipeline: FeaturePipeline, profile: MissionProfile, cfg: CcsdsAttackConfig):
        self.pipe, self.prof, self.cfg = pipeline, profile, cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------ realisation
    def _caps(self, w: Window) -> np.ndarray:
        lim = np.where(w.ptype == TC, self.prof.max_user_data[TC], self.prof.max_user_data[TM])
        return np.clip(np.minimum(self.cfg.max_pad, lim - w.base_plen), 0, None)

    def _editable(self, w: Window) -> np.ndarray:
        return np.where(np.isin(w.ptype, self.cfg.editable_types))[0]

    def realize(self, w0: Window, pad: np.ndarray, logg: np.ndarray, caps: np.ndarray) -> Window:
        """Build the packet window described by edit parameters (pad bytes, log gap scales)."""
        w = w0.copy()
        w.pad = np.clip(np.rint(pad), 0, caps).astype(np.int64)
        lo, hi = np.log(self.cfg.gap_scale_min), np.log(self.cfg.gap_scale_max)
        gaps = np.diff(w0.ts) * np.exp(np.clip(logg[1:], lo, hi))
        gaps = np.minimum(np.round(gaps * SUBSEC_RES) / SUBSEC_RES, self.prof.max_gap_s)
        w.ts = w0.ts[0] + np.concatenate([[0.0], np.cumsum(gaps)])
        return w

    @staticmethod
    def count_edits(w0: Window, w1: Window) -> int:
        """Edit distance = number of (packet, field) pairs changed (pad, timing gap, APID)."""
        g0, g1 = np.diff(w0.ts), np.diff(w1.ts)
        return int((w0.pad != w1.pad).sum() + (np.abs(g0 - g1) > 1e-9).sum() + (w0.apid != w1.apid).sum())

    def _feasible(self, w0: Window, w1: Window) -> Tuple[bool, List[str]]:
        before = {(v.code, v.message) for v in validate_window(w0, self.prof)}
        after = validate_window(w1, self.prof, original=w0)
        new = [str(v) for v in after if (v.code, v.message) not in before]
        return (not new), new

    # ------------------------------------------------------------------ pgd-ccsds
    def pgd(self, model, w0: Window, target=None) -> AttackResult:
        """White-box on ``model`` (needs loss_gradient). If ``target`` is given, the final window is scored on it
        (transfer-ccsds); otherwise on ``model`` itself."""
        c, W = self.cfg, len(w0)
        scorer = target if target is not None else model
        caps = self._caps(w0)
        edit_idx = self._editable(w0)
        p0 = float(scorer.predict_proba(self.pipe.window(w0)[None])[0, 1])
        if len(edit_idx) == 0:
            return AttackResult(w0, p0 < 0.5, True, 0, 0, p0, p0)
        pad = w0.pad.astype(float).copy()
        logg = np.zeros(W)
        lo, hi = np.log(c.gap_scale_min), np.log(c.gap_scale_max)
        coords = [("pad", i) for i in edit_idx] + [("gap", i) for i in edit_idx if i >= 1]
        active: Optional[List[Tuple[str, int]]] = None
        queries = 0
        best, best_loss = w0, -np.log(max(float(model.predict_proba(self.pipe.window(w0)[None])[0, 1]), 1e-12))
        for _ in range(c.steps):
            cur = self.realize(w0, pad, logg, caps)
            F0 = self.pipe.window(cur)
            g_x = model.loss_gradient(F0[None], np.array([1]))[0]
            queries += 1
            use = coords if active is None else active
            cols = []
            for kind, i in use:
                p2, l2 = pad.copy(), logg.copy()
                if kind == "pad":
                    p2[i] += c.fd_pad; d = c.fd_pad
                else:
                    l2[i] += c.fd_log_gap; d = c.fd_log_gap
                cols.append((self.pipe.window(self.realize(w0, p2, l2, caps)) - F0) / d)
            g_theta = np.stack(cols, axis=1).T @ g_x
            if active is None:                              # choose the B most influential packets once
                score: Dict[int, float] = {}
                for (kind, i), g in zip(use, g_theta):
                    s = abs(g) * (c.step_pad if kind == "pad" else c.step_log_gap)
                    score[i] = max(score.get(i, 0.0), s)
                top = set(sorted(score, key=score.get, reverse=True)[:c.budget_packets])
                keep = [(k, (kind, i)) for k, (kind, i) in enumerate(use) if i in top]
                active = [u for _, u in keep]
                g_theta = g_theta[[k for k, _ in keep]]
                use = active
            for (kind, i), g in zip(use, g_theta):
                if kind == "pad":
                    pad[i] = np.clip(pad[i] + c.step_pad * np.sign(g), 0, caps[i])
                else:
                    logg[i] = np.clip(logg[i] + c.step_log_gap * np.sign(g), lo, hi)
            w1 = self.realize(w0, pad, logg, caps)
            p = float(model.predict_proba(self.pipe.window(w1)[None])[0, 1])
            loss = -np.log(max(p, 1e-12))
            if loss > best_loss:
                best, best_loss = w1, loss
            if p < 0.5:
                break
        feas, viol = self._feasible(w0, best)
        final = best if feas else w0
        p_after = float(scorer.predict_proba(self.pipe.window(final)[None])[0, 1])
        return AttackResult(final, feas and p_after < 0.5, feas, self.count_edits(w0, final), queries, p0, p_after, viol)

    # ------------------------------------------------------------------ greedy-ccsds
    def _repair_seq(self, w: Window) -> None:
        last: Dict[int, int] = {}
        for i in range(len(w)):
            a = int(w.apid[i])
            if a in last:
                w.seqc[i] = (last[a] + 1) % SEQ_MOD
            last[a] = int(w.seqc[i])

    def _random_edit(self, cur: Window, i: int, caps: np.ndarray) -> Window:
        c = self.cfg
        w = cur.copy()
        field = self.rng.choice(["pad", "gap", "apid"])
        if field == "pad":
            delta = int(self.rng.choice(c.pad_choices)) * int(self.rng.choice([-1, 1]))
            w.pad[i] = int(np.clip(w.pad[i] + delta, 0, caps[i]))
        elif field == "gap" and i >= 1:
            old = w.ts[i] - w.ts[i - 1]
            new = np.round(min(old * float(self.rng.choice(c.gap_choices)), self.prof.max_gap_s) * SUBSEC_RES) / SUBSEC_RES
            w.ts[i:] += new - old
        elif field == "apid":
            w.apid[i] = int(self.rng.choice(sorted(self.prof.allowed_apids(int(w.ptype[i])))))
            self._repair_seq(w)
        return w

    def greedy(self, model, w0: Window) -> AttackResult:
        """Black-box: queries ``model.predict_proba`` only."""
        c = self.cfg
        caps = self._caps(w0)
        edit_idx = list(self._editable(w0))
        cur, p = w0, float(model.predict_proba(self.pipe.window(w0)[None])[0, 1])
        p0, queries, edited = p, 1, set()
        for _ in range(c.greedy_iters):
            if p < 0.5 or not edit_idx:
                break
            pool = sorted(edited) if len(edited) >= c.budget_packets else edit_idx
            cands, owners = [], []
            for _ in range(c.greedy_candidates):
                i = int(self.rng.choice(pool))
                cands.append(self._random_edit(cur, i, caps)); owners.append(i)
            probs = model.predict_proba(np.stack([self.pipe.window(x) for x in cands]))[:, 1]
            queries += len(cands)
            k = int(np.argmin(probs))
            if probs[k] < p - 1e-9:
                feas, _ = self._feasible(w0, cands[k])
                if feas:
                    cur, p = cands[k], float(probs[k]); edited.add(owners[k])
        feas, viol = self._feasible(w0, cur)
        final = cur if feas else w0
        p_after = float(model.predict_proba(self.pipe.window(final)[None])[0, 1])
        return AttackResult(final, feas and p_after < 0.5, feas, self.count_edits(w0, final), queries, p0, p_after, viol)


# ----------------------------------------------------------------------------------------
def attack_windows(kind: str, windows: Sequence[Window], target, pipeline: FeaturePipeline, profile: MissionProfile,
                   cfg: CcsdsAttackConfig, surrogate=None, max_windows: Optional[int] = None):
    """Attack every attack-class window (benign windows pass through unchanged).

    kind in {'pgd_ccsds', 'transfer_ccsds', 'greedy_ccsds'}. Returns (adv_windows, per-window stats list).
    """
    atk = ConformantAttacker(pipeline, profile, cfg)
    out, stats = [], []
    n_att = 0
    for w in windows:
        if w.label != 1 or (max_windows is not None and n_att >= max_windows):
            out.append(w); continue
        n_att += 1
        if kind == "pgd_ccsds":
            r = atk.pgd(target, w)
        elif kind == "transfer_ccsds":
            if surrogate is None:
                raise ValueError("transfer_ccsds needs a surrogate model")
            r = atk.pgd(surrogate, w, target=target)
        elif kind == "greedy_ccsds":
            r = atk.greedy(target, w)
        else:
            raise ValueError(f"unknown conformant attack {kind!r}")
        out.append(r.window)
        stats.append(dict(success=r.success, feasible=r.feasible, edits=r.n_edits, queries=r.queries,
                          p_before=r.p_before, p_after=r.p_after))
    return out, stats
