"""Loading, cleaning, splitting and scaling of terrestrial and satellite data.

Every returned bundle carries a ``source`` field ('real' or 'synthetic'); experiment scripts copy it into
every results row so that tables built from synthetic data are always identifiable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # allow `python data/<script>.py`

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import feature_schema as fs
from data.ccsds_parser import Window, get_format, packets_to_window, read_stream, set_format
from data.synthetic_ccsds import make_windows
from data.synthetic_terrestrial import make_unsw_like


class DataNotFoundError(FileNotFoundError):
    pass


def _count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))


def _read_terrestrial(path: str, dataset: str, max_rows: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    """Read a terrestrial dataset. CICIDS2017 CSVs are not UTF-8 (a Windows-1252 dash appears in the Thursday
    labels), so they are read as latin-1. For a CICIDS2017 folder with ``max_rows`` set, every file is sampled
    with the same fraction while loading, so peak memory stays at roughly one daily file."""
    if not os.path.exists(path):
        raise DataNotFoundError(
            f"{dataset}: file not found at {path}. Run `python data/download_datasets.py --help`, "
            "obtain the dataset from its official source, and set the path in the config.")
    if dataset == "nsl_kdd":
        return pd.read_csv(path, header=None, names=fs.NSL_KDD_COLUMNS)
    enc = "latin-1" if dataset == "cicids2017" else "utf-8"
    if os.path.isdir(path):                                # CICIDS2017 ships as several daily CSVs
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(".csv")]
        if not files:
            raise DataNotFoundError(f"{dataset}: no .csv files in {path}")
        total = sum(max(_count_lines(f) - 1, 0) for f in files)
        frac = min(1.0, max_rows / total) if max_rows and total else 1.0
        parts = []
        for f in files:
            d = pd.read_csv(f, low_memory=False, encoding=enc)
            parts.append(d.sample(frac=frac, random_state=seed) if frac < 1.0 else d)
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path, low_memory=False, encoding=enc)


def load_terrestrial(cfg: Dict[str, Any], dataset: str, seed: int) -> Dict[str, Any]:
    """Return scaled train/val/test splits: 60/20/20 stratified from one file, or, if the dataset entry has `test_path`,
    `path` is the training partition (75/25 train/val) and `test_path` the official test partition."""
    dcfg = cfg["data"]["terrestrial"][dataset]
    view = dcfg.get("view", cfg["features"]["view"])
    source = dcfg.get("source", "real")
    if source == "synthetic":
        df = make_unsw_like(int(dcfg.get("n_rows", 6000)), seed=1000 + seed)
        ds_for_map = "unsw_nb15"
    else:
        df = _read_terrestrial(dcfg["path"], dataset, dcfg.get("max_rows"), seed)
        ds_for_map = dataset
    max_rows = dcfg.get("max_rows")
    if max_rows and len(df) > max_rows:
        df = df.sample(int(max_rows), random_state=seed)
    X, y = fs.terrestrial_to_fc(df, ds_for_map, view)
    test_path = dcfg.get("test_path")
    if test_path and source != "synthetic":
        # official partition: `path` = training file (split 75/25 into train/validation), `test_path` = official test file
        Xte, yte = fs.terrestrial_to_fc(_read_terrestrial(test_path, dataset, None, seed), ds_for_map, view)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    else:                                                  # one file: 60/20/20 stratified random split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    scaler = fs.MinMaxScaler().fit(Xtr)
    return dict(dataset=dataset, source=source, view=view, scaler=scaler,
                X_train=scaler.transform(Xtr), y_train=ytr, X_val=scaler.transform(Xva), y_val=yva,
                X_test=scaler.transform(Xte), y_test=yte, X_test_raw=Xte)


def windows_from_stream(bin_path: str, labels_csv: str, size: int, stride: int, label_rule: str = "majority") -> List[Window]:
    """Cut a captured/synthetic packet stream into windows.

    label_rule: "majority" -> window is an attack if >= 50% of its packets are labelled attack;
                "any"      -> attack if at least one packet is (use for sparse scripted attacks).
    If the labels file has an ``arrival_time`` column (written by tools/pcap_to_stream.py), it overrides the packet
    timestamps; this is REQUIRED for the cFS wire format, whose packets carry no usable timestamp.
    """
    if label_rule not in ("majority", "any"):
        raise ValueError("label_rule must be 'majority' or 'any'")
    if not os.path.exists(bin_path) or not os.path.exists(labels_csv):
        raise DataNotFoundError(f"satellite stream or labels missing: {bin_path} / {labels_csv}")
    with open(bin_path, "rb") as fh:
        pk = read_stream(fh.read())
    with open(labels_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(pk):
        raise ValueError(f"{len(pk)} packets but {len(rows)} label rows")
    if rows and "arrival_time" in rows[0]:
        for p_, r in zip(pk, rows):
            p_.timestamp = float(r["arrival_time"])
    elif not get_format().ts_in_header:
        raise ValueError("wire format 'cfs' needs an 'arrival_time' column in the labels file (tools/pcap_to_stream.py writes it)")
    out = []
    for s in range(0, len(pk) - size + 1, stride):
        lab = [int(r["label"]) for r in rows[s:s + size]]
        types = [r.get("attack_type", "") for r in rows[s:s + size] if int(r["label"]) == 1]
        l = int(np.mean(lab) >= 0.5) if label_rule == "majority" else int(max(lab) == 1)
        at = max(set(types), key=types.count) if (l and types) else "benign"
        out.append(packets_to_window(pk[s:s + size], l, at))
    return out


def satellite_window_splits(cfg: Dict[str, Any], seed: int, size: Optional[int] = None, label_rule: Optional[str] = None):
    """(train, val, test) window lists. Real streams: chronological 60/20/20 (no leakage across time). Synthetic: independent seeds.
    ``size`` / ``label_rule`` override the config (used by the in-domain reference and tools/diagnose_separability.py)."""
    scfg = cfg["data"]["satellite"]
    cfg_size = int(cfg["features"]["window_size"])
    size = int(size or cfg_size)
    set_format(scfg.get("wire_format", "project"), **(scfg.get("wire_overrides") or {}))
    if scfg.get("source", "real") == "synthetic":
        n = int(scfg.get("n_windows", 800))
        return (make_windows(n, 2000 + seed, 0.4, size), make_windows(max(n // 4, 50), 3000 + seed, 0.4, size),
                make_windows(max(n // 2, 100), 4000 + seed, 0.4, size))
    stride = int(scfg.get("stride", size)) if size == cfg_size else size
    wins = windows_from_stream(scfg["bin_path"], scfg["labels_path"], size, stride, label_rule or scfg.get("label_rule", "majority"))
    a, b = int(0.6 * len(wins)), int(0.8 * len(wins))
    return wins[:a], wins[a:b], wins[b:]


def load_satellite(cfg: Dict[str, Any], bundle: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Satellite windows, scaled with the TERRESTRIAL scaler (the satellite domain is only ever tested with terrestrial-trained
    models in the main benchmark)."""
    scfg = cfg["data"]["satellite"]
    wtr, wva, wte = satellite_window_splits(cfg, seed)
    pipe = fs.FeaturePipeline(bundle["view"], bundle["scaler"])
    f = lambda ws: (pipe.windows(ws), np.array([w.label for w in ws]))
    Xtr, ytr = f(wtr); Xva, yva = f(wva); Xte, yte = f(wte)
    return dict(source=scfg.get("source", "real"), pipeline=pipe, w_train=wtr, w_val=wva, w_test=wte,
                X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte)


if __name__ == "__main__":
    import argparse
    from config import load_config
    ap = argparse.ArgumentParser(description="Preprocess a terrestrial dataset and print split sizes")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()
    c = load_config(a.config)
    for ds in ([a.dataset] if a.dataset else c["datasets"]):
        b = load_terrestrial(c, ds, seed=c["seeds"][0])
        print(ds, b["source"], b["X_train"].shape, b["X_val"].shape, b["X_test"].shape)
