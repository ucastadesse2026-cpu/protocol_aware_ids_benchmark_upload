"""Common feature schema Fc shared by terrestrial flow datasets and CCSDS packet windows.

Views
-----
core      4 features computable from ALL three terrestrial datasets (incl. NSL-KDD):
          log duration, log fwd bytes, log bwd bytes, fwd byte ratio
extended  core + packet-count based features (UNSW-NB15, CICIDS2017 only; not NSL-KDD)
sat_aug   extended + CCSDS-specific features (only computable on satellite windows; used for sensitivity
          analyses with satellite-trained models, never for terrestrial-trained baselines)

Direction mapping (an ASSUMPTION of this benchmark): telecommand (TC, ground->space) = "forward",
telemetry (TM, space->ground) = "backward".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from data.ccsds_parser import TC, Window

FC_CORE = ["log_duration", "log_fwd_bytes", "log_bwd_bytes", "fwd_byte_ratio"]
FC_EXT = ["log_fwd_pkts", "log_bwd_pkts", "mean_pkt_len", "log_pkt_rate"]
FC_SAT = ["apid_entropy", "seq_gap_mean", "seq_gap_max", "n_unique_apid"]
VIEWS = {"core": FC_CORE, "extended": FC_CORE + FC_EXT, "sat_aug": FC_CORE + FC_EXT + FC_SAT}

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]


def view_names(view: str) -> List[str]:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {list(VIEWS)}")
    return list(VIEWS[view])


def fc_from_natural(dur, fwd_bytes, bwd_bytes, fwd_pkts=None, bwd_pkts=None, view: str = "core") -> np.ndarray:
    """Build Fc (core/extended) from natural-unit arrays (seconds, bytes, packets). Shared by both domains."""
    dur = np.maximum(np.asarray(dur, dtype=float), 0.0)
    fb = np.maximum(np.asarray(fwd_bytes, dtype=float), 0.0)
    bb = np.maximum(np.asarray(bwd_bytes, dtype=float), 0.0)
    cols = [np.log1p(dur), np.log1p(fb), np.log1p(bb), fb / (fb + bb + 1e-9)]
    if view in ("extended", "sat_aug"):
        if fwd_pkts is None or bwd_pkts is None:
            raise ValueError("extended view needs packet counts (unavailable for this dataset)")
        fp = np.maximum(np.asarray(fwd_pkts, dtype=float), 0.0)
        bp = np.maximum(np.asarray(bwd_pkts, dtype=float), 0.0)
        tot = fp + bp
        cols += [np.log1p(fp), np.log1p(bp), (fb + bb) / np.maximum(tot, 1.0),
                 np.log1p(tot / np.maximum(dur, 1e-3))]
    return np.stack(cols, axis=-1)


# ------------------------------------------------------------------ terrestrial -------
def terrestrial_to_fc(df: pd.DataFrame, dataset: str, view: str = "core"):
    """Map a raw terrestrial dataframe to (Fc matrix, binary labels). 1 = attack."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if dataset == "unsw_nb15":
        dur, fb, bb = df["dur"], df["sbytes"], df["dbytes"]
        fp, bp = df["spkts"], df["dpkts"]
        y = df["label"].astype(int).to_numpy()
    elif dataset == "cicids2017":
        dur = df["Flow Duration"] / 1e6                       # microseconds -> seconds
        fb, bb = df["Total Length of Fwd Packets"], df["Total Length of Bwd Packets"]
        fp, bp = df["Total Fwd Packets"], df["Total Backward Packets"]
        y = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int).to_numpy()
    elif dataset == "nsl_kdd":
        if view != "core":
            raise ValueError("NSL-KDD has no packet counts: only view='core' is available")
        dur, fb, bb = df["duration"], df["src_bytes"], df["dst_bytes"]
        fp = bp = None
        y = (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int).to_numpy()
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    X = fc_from_natural(dur.to_numpy(float), fb.to_numpy(float), bb.to_numpy(float),
                        None if fp is None else fp.to_numpy(float), None if bp is None else bp.to_numpy(float), view)
    keep = np.isfinite(X).all(axis=1)           # drops inf/NaN rows (documented cleaning step)
    return X[keep], y[keep]


# ------------------------------------------------------------------ satellite ---------
def window_to_fc(w: Window, view: str = "core") -> np.ndarray:
    """Fc vector (raw, pre-scaling) of one CCSDS packet window."""
    wl = w.wire_lengths().astype(float)
    fwd = w.ptype == TC
    dur = float(w.ts[-1] - w.ts[0]) if len(w) > 1 else 0.0
    base = fc_from_natural(dur, wl[fwd].sum(), wl[~fwd].sum(), int(fwd.sum()), int((~fwd).sum()), view)
    if view != "sat_aug":
        return base
    _, counts = np.unique(w.apid, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log2(p)).sum())
    gaps = []
    for a in np.unique(w.apid):
        s = w.seqc[w.apid == a]
        if len(s) > 1:
            gaps.extend(((np.diff(s)) % (1 << 14)).tolist())
    gaps = np.asarray(gaps if gaps else [1.0], dtype=float)
    return np.concatenate([base, [ent, gaps.mean(), gaps.max(), float(len(counts))]])


def windows_to_fc(ws: Sequence[Window], view: str = "core") -> np.ndarray:
    return np.stack([window_to_fc(w, view) for w in ws]) if len(ws) else np.zeros((0, len(view_names(view))))


@dataclass
class MinMaxScaler:
    """Min-max scaler fitted on terrestrial TRAINING data (linear, so gradients chain trivially)."""

    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        return self

    @property
    def scale(self) -> np.ndarray:
        return (self.hi - self.lo) + 1e-9

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / self.scale

    def inverse(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.scale + self.lo


@dataclass
class FeaturePipeline:
    """View + scaler: maps windows (satellite) or raw Fc rows (terrestrial) to model input."""

    view: str
    scaler: MinMaxScaler

    def window(self, w: Window) -> np.ndarray:
        return self.scaler.transform(window_to_fc(w, self.view))

    def windows(self, ws: Sequence[Window]) -> np.ndarray:
        return self.scaler.transform(windows_to_fc(ws, self.view))
