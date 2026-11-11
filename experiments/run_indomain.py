"""EXPLORATORY in-domain reference: detectors trained on the SATELLITE training windows (scaler fitted on them).

Why it exists: when terrestrial-trained detectors are near-constant predictors on satellite data (clean FPR ~ 100%),
robustness numbers measured on them are uninformative. This script answers two questions the main benchmark cannot:
(1) are benign and attack windows separable in the shared feature space at all, and (2) when the detector actually
works on satellite data, do conformant attacks evade it, compared with unconstrained ones?
It is a design addition made AFTER inspecting earlier runs: report it as exploratory.
Output: results/tables/indomain.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.ccsds_conformant import attack_windows
from attacks.standard_attacks import run_standard_attack
from data import feature_schema as fs
from data.preprocess import satellite_window_splits
from evaluation.metrics import evaluate
from experiments.common import (atk_cfg_of, common_cols, load_bundles, parse_args, profile_of, subset_for_attack,
                                write_rows)
from models import build_model


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    prof, rows = profile_of(cfg), []
    sa, eps = cfg["attacks"]["standard"], float(cfg["defenses"]["eps"])
    names = cfg.get("indomain", {}).get("models") or cfg["defenses"]["models"]
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            icfg = cfg.get("indomain", {})
            view = icfg.get("view") or terr["view"]                       # e.g. sat_aug: adds APID entropy / sequence-gap features
            size, rule = icfg.get("window_size"), icfg.get("label_rule")
            wtr, wva, wte = satellite_window_splits(cfg, seed, size, rule) if (size or rule) else (sat["w_train"], sat["w_val"], sat["w_test"])
            pipe = fs.FeaturePipeline(view, fs.MinMaxScaler().fit(fs.windows_to_fc(wtr, view)))
            ytr, yva = np.array([w.label for w in wtr]), np.array([w.label for w in wva])
            Xtr, Xva = pipe.windows(wtr), pipe.windows(wva)
            wins = subset_for_attack(wte, int(cfg["attacks"]["max_windows"]))
            yte, Xte = np.array([w.label for w in wins]), pipe.windows(wins)
            d = Xtr.shape[1]
            sur = build_model(cfg["attacks"]["surrogate"], d, cfg, seed + 20000).fit(Xtr, ytr, Xva, yva)
            for name in names:
                model = build_model(name, d, cfg, seed).fit(Xtr, ytr, Xva, yva)
                clean = evaluate(model, Xte, yte)
                gm = model if model.differentiable else sur
                std = evaluate(model, run_standard_attack("pgd", gm, Xte, yte, eps, (0.0, 1.0), seed, sa["pgd_steps"]), yte)
                kind = "pgd_ccsds" if model.differentiable else "transfer_ccsds"
                adv, st = attack_windows(kind, wins, model, pipe, prof, atk_cfg_of(cfg, seed), surrogate=sur)
                ccs = evaluate(model, pipe.windows(adv), yte)
                advg, stg = attack_windows("greedy_ccsds", wins, model, pipe, prof, atk_cfg_of(cfg, seed))
                grd = evaluate(model, pipe.windows(advg), yte)
                det = lambda s: [x for x in s if x["p_before"] >= 0.5]
                rate = lambda s: float(np.mean([x["success"] for x in det(s)])) if det(s) else float("nan")
                rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), model=name, n_train=len(ytr), n_windows=len(wins),
                                 n_attacked=int(yte.sum()), base_rate_attack=float(yte.mean()),
                                 indomain_view=view, window_size=len(wins[0]), label_rule=rule or cfg["data"]["satellite"].get("label_rule", "majority"),
                                 train_attack_rate=float(ytr.mean()),
                                 acc_clean=clean["acc"], f1_clean=clean["f1"], fpr_clean=clean["fpr"], fnr_clean=clean["fnr"],
                                 acc_std=std["acc"], fnr_std=std["fnr"], ccsds_attack=kind,
                                 acc_ccsds=ccs["acc"], fnr_ccsds=ccs["fnr"], success_ccsds=rate(st),
                                 feasibility_ccsds=float(np.mean([x["feasible"] for x in st])) if st else float("nan"),
                                 acc_greedy=grd["acc"], fnr_greedy=grd["fnr"], success_greedy=rate(stg)))
                print(f"[indomain] seed={seed} {ds} {name}: clean acc {clean['acc']:.3f} (fpr {clean['fpr']:.3f}, fnr {clean['fnr']:.3f})")
    write_rows(cfg, "indomain.csv", rows)


if __name__ == "__main__":
    main()
