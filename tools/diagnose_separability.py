"""Are benign and attack windows separable in the shared features at all? (in-domain diagnostic on the satellite stream)

    python tools/diagnose_separability.py --config configs/my_nos3.yaml [--views core extended sat_aug] [--sizes 8 16 32] [--rules any majority]

Prints, for the configured satellite stream: (1) benign/attack counts per chronological split and per attack type, (2) label
purity (fraction of attack packets inside attack windows: sparse attacks are diluted in long windows), (3) per-feature
standardised mean differences (Cohen's d) of benign vs each attack type, and (4) an in-domain Random-Forest check with
TIME-BLOCKED cross-validation inside the training split plus a grid over feature view x window size x label rule.

DISCIPLINE: the grid is scored on the VALIDATION split only (the test split is never used here), because choosing a design
by test performance would invalidate the test. Choose the design from this table, write the choice into the amendment, then
run the experiments. The shared features (core/extended) contain no APID/opcode information; 'sat_aug' adds APID entropy,
sequence-gap and distinct-APID features, which exist only on the satellite side (in-domain detectors only).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config
from data import feature_schema as fs
from data.preprocess import satellite_window_splits


def cohens_d(a, b):
    sa, sb = a.std(ddof=1) if len(a) > 1 else 0.0, b.std(ddof=1) if len(b) > 1 else 0.0
    sp = np.sqrt((sa ** 2 + sb ** 2) / 2) + 1e-12
    return (b.mean() - a.mean()) / sp


def rf_scores(Xtr, ytr, Xev, yev, seed):
    if len(set(ytr)) < 2 or len(set(yev)) < 2:
        return float("nan"), float("nan")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=1, random_state=seed).fit(Xtr, ytr)
    p = rf.predict_proba(Xev)[:, 1]
    return balanced_accuracy_score(yev, (p >= 0.5).astype(int)), roc_auc_score(yev, p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--views", nargs="*", default=["core", "extended", "sat_aug"])
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--rules", nargs="*", default=["any", "majority"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    size0 = int(cfg["features"]["window_size"])
    sizes = a.sizes or [8, 16, size0]
    rule0 = cfg["data"]["satellite"].get("label_rule", "majority")
    wtr, wva, wte = satellite_window_splits(cfg, a.seed)
    print(f"== 1. composition (config: window {size0}, label rule {rule0})")
    for name, ws in (("train", wtr), ("val", wva), ("test", wte)):
        c = Counter(w.attack_type for w in ws)
        print(f"  {name:5s} {len(ws):6d} windows, attack share {100 * np.mean([w.label for w in ws]):5.1f}%   {dict(c)}")

    scfg = cfg["data"]["satellite"]
    if scfg.get("source", "real") != "synthetic":
        with open(scfg["labels_path"], newline="") as fh:
            lab = np.array([int(r["label"]) for r in csv.DictReader(fh)])
        stride = int(scfg.get("stride", size0))
        fr = np.array([lab[s:s + size0].mean() for s in range(0, len(lab) - size0 + 1, stride)])
        pos = fr[fr > 0]
        print(f"\n== 2. label purity: share of attack packets inside windows that contain any attack packet (window {size0})")
        if len(pos):
            print("  percentiles 10/50/90 of the attack-packet fraction:", np.round(np.percentile(pos, [10, 50, 90]), 3).tolist())
            print(f"  attack windows with < 25% attack packets: {100 * np.mean(pos < 0.25):.1f}%  (diluted: hard to see in aggregate features)")
        print(f"  packet-level attack share of the whole stream: {100 * lab.mean():.1f}%")

    view0 = cfg["features"]["view"]
    Xtr_raw = fs.windows_to_fc(wtr, "sat_aug"); ytr = np.array([w.label for w in wtr])
    names = fs.view_names("sat_aug")
    print("\n== 3. standardised mean difference (Cohen's d) vs benign, TRAIN split (|d| < 0.2 = barely separable)")
    types = sorted({w.attack_type for w in wtr} - {"benign"})
    ben = Xtr_raw[ytr == 0]
    print("  " + f"{'feature':16s}" + "".join(f"{t:>12s}" for t in types))
    for j, n in enumerate(names):
        print("  " + f"{n:16s}" + "".join(f"{cohens_d(ben[:, j], Xtr_raw[np.array([w.attack_type == t for w in wtr])][:, j]):12.2f}" if (np.array([w.attack_type == t for w in wtr])).any() else f"{'n/a':>12s}" for t in types))

    print("\n== 4a. in-domain RF, TIME-BLOCKED 5-fold CV inside the training split (config view)")
    X = fs.windows_to_fc(wtr, view0); k = 5; idx = np.array_split(np.arange(len(wtr)), k); bas, aucs = [], []
    for i in range(k):
        te = idx[i]; tr = np.concatenate([idx[j] for j in range(k) if j != i])
        ba, au = rf_scores(X[tr], ytr[tr], X[te], ytr[te], a.seed); bas.append(ba); aucs.append(au)
    print(f"  balanced accuracy {np.nanmean(bas):.3f}  AUC {np.nanmean(aucs):.3f}   (0.5 = chance)")

    print("\n== 4b. design grid scored on the VALIDATION split only: balanced accuracy / AUC of an in-domain RF")
    print(f"  {'view':10s}{'window':>8s}{'rule':>10s}{'train n':>9s}{'val n':>7s}{'val attack%':>12s}{'bal.acc':>9s}{'AUC':>7s}")
    for view in a.views:
        for size in sizes:
            for rule in a.rules:
                tr, va, _ = satellite_window_splits(cfg, a.seed, size, rule)
                Xa, ya = fs.windows_to_fc(tr, view), np.array([w.label for w in tr])
                Xb, yb = fs.windows_to_fc(va, view), np.array([w.label for w in va])
                ba, au = rf_scores(Xa, ya, Xb, yb, a.seed)
                print(f"  {view:10s}{size:8d}{rule:>10s}{len(tr):9d}{len(va):7d}{100 * yb.mean():12.1f}{ba:9.3f}{au:7.3f}")
    print("\nRead: balanced accuracy near 0.5 / AUC near 0.5 = benign and attack windows are not separable with that design.")
    print("Pick the design from THIS table (validation), record it in the amendment, then rerun; never tune on the test split.")


if __name__ == "__main__":
    main()
