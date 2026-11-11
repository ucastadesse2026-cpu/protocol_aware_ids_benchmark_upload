"""Phase 3b/4: pgd-ccsds (white-box), transfer-ccsds (surrogate) and greedy-ccsds (black-box) on satellite windows.

Reports accuracy on the attacked window subset, attack success rate (among windows the model initially detects),
feasibility rate (fraction passing C1-C8), mean edit distance and mean queries.
Output: results/tables/ccsds_attacks.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.ccsds_conformant import attack_windows
from evaluation.metrics import evaluate
from experiments.common import (atk_cfg_of, common_cols, get_model, load_bundles, parse_args, profile_of,
                                subset_for_attack, write_rows)


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    prof, rows = profile_of(cfg), []
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            pipe = sat["pipeline"]
            surrogate, _ = get_model(cfg, cfg["attacks"]["surrogate"], ds, seed, terr, not args.no_cache)
            wins = subset_for_attack(sat["w_test"], int(cfg["attacks"]["max_windows"]))
            y = np.array([w.label for w in wins])
            X_clean = pipe.windows(wins)
            for name in cfg["model_families"]:
                target, _ = get_model(cfg, name, ds, seed, terr, not args.no_cache)
                clean_acc = evaluate(target, X_clean, y)["acc"]
                kinds = (["pgd_ccsds"] if target.differentiable else []) + \
                        (["transfer_ccsds"] if name != cfg["attacks"]["surrogate"] else []) + ["greedy_ccsds"]
                for kind in kinds:
                    adv, st = attack_windows(kind, wins, target, pipe, prof, atk_cfg_of(cfg, seed), surrogate=surrogate)
                    det = [s for s in st if s["p_before"] >= 0.5]
                    rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), model=name, attack=kind,
                                     n_windows=len(wins), n_attacked=len(st), acc_clean_subset=clean_acc,
                                     attack_success_rate=float(np.mean([s["success"] for s in det])) if det else float("nan"),
                                     feasibility_rate=float(np.mean([s["feasible"] for s in st])) if st else float("nan"),
                                     mean_edits=float(np.mean([s["edits"] for s in st])) if st else float("nan"),
                                     mean_queries=float(np.mean([s["queries"] for s in st])) if st else float("nan"),
                                     **evaluate(target, pipe.windows(adv), y)))
                print(f"[ccsds-attacks] seed={seed} {ds} {name}: {kinds}")
    write_rows(cfg, "ccsds_attacks.csv", rows)


if __name__ == "__main__":
    main()
