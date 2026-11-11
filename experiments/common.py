"""Shared plumbing for the experiment scripts (argument parsing, data bundles, model cache, CSV output)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from attacks.ccsds_conformant import CcsdsAttackConfig
from attacks.validator import MissionProfile
from config import load_config, results_dir
from data.preprocess import load_satellite, load_terrestrial
from models import build_model
from models.federated import FedAvg, dirichlet_partition


def parse_args(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--seeds", type=int, nargs="*", help="override seeds (default: config.seeds = 1..5)")
    ap.add_argument("--no-cache", action="store_true", help="retrain models instead of using results/cache")
    ap.add_argument("--out-dir", help="override output.dir (use a different folder per seed/dataset, then tools/merge_results.py)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg.setdefault("output", {})["dir"] = args.out_dir
    return cfg, (args.seeds or cfg["seeds"]), args


def load_bundles(cfg: Dict[str, Any], dataset: str, seed: int):
    terr = load_terrestrial(cfg, dataset, seed)
    sat = load_satellite(cfg, terr, seed)
    return terr, sat


def common_cols(cfg, terr, sat, seed: int, dataset: str) -> Dict[str, Any]:
    return dict(seed=seed, dataset=dataset, terr_source=terr["source"], sat_source=sat["source"], view=terr["view"])


def profile_of(cfg) -> MissionProfile:
    sat = cfg.get("data", {}).get("satellite", {})
    if sat.get("source") == "real" and sat.get("wire_format", "project") != "project" and "tc_apid_list" not in (cfg.get("profile") or {}):
        raise SystemExit("config: wire_format is not 'project' but `profile:` has no tc_apid_list. Run "
                         "tools/derive_profile.py on your capture and paste its `profile:` block into the config (docs/NOS3_GUIDE.md).")
    return MissionProfile.from_dict(cfg.get("profile"))


def atk_cfg_of(cfg, seed: int) -> CcsdsAttackConfig:
    c = CcsdsAttackConfig.from_dict(cfg["attacks"].get("ccsds"))
    c.seed = seed
    return c


def subset_for_attack(windows: Sequence, max_attack: int) -> List:
    """Longest prefix of ``windows`` holding at most ``max_attack`` attack-class windows."""
    out, n = [], 0
    for w in windows:
        if w.label == 1:
            if n >= max_attack:
                break
            n += 1
        out.append(w)
    return out


# ---------------------------------------------------------------------------------- training
def _fed_params(cfg) -> Tuple[int, float]:
    f = cfg["federated"]
    return int(f["baseline_k"]), float(f["baseline_alpha"])


def train_model(cfg, name: str, terr, seed: int):
    """Train a centralised family on TERRESTRIAL data only. Returns (model, wall-clock seconds)."""
    d = terr["X_train"].shape[1]
    t0 = time.perf_counter()
    if name == "fedavg":
        f = cfg["federated"]
        k, alpha = _fed_params(cfg)
        parts_idx = dirichlet_partition(terr["y_train"], k, alpha, np.random.default_rng(seed))
        parts = [(terr["X_train"][i], terr["y_train"][i]) for i in parts_idx]
        model = FedAvg(lambda: build_model(f["backbone"], d, cfg, seed), int(f["rounds"]), int(f["local_epochs"]),
                       seed=seed).fit(parts, terr["X_val"], terr["y_val"])
    else:
        model = build_model(name, d, cfg, seed).fit(terr["X_train"], terr["y_train"], terr["X_val"], terr["y_val"])
    return model, time.perf_counter() - t0


def get_model(cfg, name: str, dataset: str, seed: int, terr, use_cache: bool = True):
    """Cached baseline model. Neural models are cached as weights (their nets are not picklable closures)."""
    cdir = results_dir(cfg, "cache")
    path = os.path.join(cdir, f"{dataset}_{name}_{seed}_{terr['source']}_{terr['view']}.pkl")
    d = terr["X_train"].shape[1]
    backbone = cfg["federated"]["backbone"] if name == "fedavg" else name
    if use_cache and os.path.exists(path):
        with open(path, "rb") as fh:
            kind, payload, secs = pickle.load(fh)
        if kind == "object":
            return payload, secs
        m = build_model(backbone, d, cfg, seed)
        m.set_weights(payload)
        return m, secs
    model, secs = train_model(cfg, name, terr, seed)
    with open(path, "wb") as fh:
        if hasattr(model, "get_weights"):
            pickle.dump(("weights", model.get_weights(), secs), fh)
        else:
            pickle.dump(("object", model, secs), fh)
    return model, secs


def write_rows(cfg, filename: str, rows: List[Dict[str, Any]]) -> str:
    out = os.path.join(results_dir(cfg, "tables"), filename)
    pd.DataFrame(rows).to_csv(out, index=False)
    flag = "SYNTHETIC-DATA" if any(str(r.get("terr_source")) == "synthetic" or str(r.get("sat_source")) == "synthetic" for r in rows) else "real-data"
    print(f"wrote {len(rows)} rows -> {out}  [{flag}]")
    return out
