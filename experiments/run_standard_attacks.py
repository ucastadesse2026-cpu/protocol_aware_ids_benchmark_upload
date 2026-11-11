"""Phase 3a/4: FGSM / PGD / AutoAttack-style on terrestrial and satellite test features.

Only attack-class rows are perturbed. Non-differentiable models (Random Forest) are attacked by TRANSFER from the
differentiable surrogate. Output: results/tables/standard_attacks.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.standard_attacks import run_standard_attack
from attacks.validator import feature_space_realizable
from evaluation.metrics import evaluate
from experiments.common import common_cols, get_model, load_bundles, parse_args, profile_of, write_rows


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    sa, prof, rows = cfg["attacks"]["standard"], profile_of(cfg), []
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            surrogate, _ = get_model(cfg, cfg["attacks"]["surrogate"], ds, seed, terr, not args.no_cache)
            for name in cfg["model_families"]:
                target, _ = get_model(cfg, name, ds, seed, terr, not args.no_cache)
                grad_model, mode = (target, "whitebox") if target.differentiable else (surrogate, "transfer")
                for atk in sa["names"]:
                    if atk == "autoattack" and not (hasattr(grad_model, "net") or hasattr(grad_model, "clf")):
                        print(f"[std-attacks] skip autoattack for {name} (needs a torch model)"); continue
                    for eps in sa["eps"]:
                        for domain, X, y, clip in (("terrestrial", terr["X_test"], terr["y_test"], (0.0, 1.0)),
                                                   ("satellite", sat["X_test"], sat["y_test"], None)):
                            try:
                                Xa = run_standard_attack(atk, grad_model, X, y, eps, clip, seed, sa["pgd_steps"], grad_model)
                            except ImportError as exc:
                                print(f"[std-attacks] skip {atk}: {exc}"); break
                            pos = y == 1
                            realizable = feature_space_realizable(terr["scaler"].inverse(Xa[pos]), terr["view"], prof)
                            rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), model=name, attack=atk, eps=eps,
                                             domain=domain, mode=mode, validator_rejection=1.0 - float(realizable.mean()),
                                             **evaluate(target, Xa, y)))
                print(f"[std-attacks] seed={seed} {ds} {name} done ({mode})")
    write_rows(cfg, "standard_attacks.csv", rows)


if __name__ == "__main__":
    main()
