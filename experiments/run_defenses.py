"""Phase 5: baseline vs standard AT vs PAA vs AT+PAA (+ clean-satellite-augmentation CONTROL).

For every (model, defense) records accuracy under PGD (terrestrial), PGD (satellite), pgd/transfer-ccsds
(satellite), clean accuracies, total training time (incl. surrogate training and augmentation generation),
single-sample latency and model size. tau and training-time multiples are computed by make_tables.py.
Output: results/tables/defenses.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.ccsds_conformant import attack_windows
from attacks.standard_attacks import run_standard_attack
from defenses.protocol_aware_augmentation import build_paa_set
from defenses.standard_at import adversarial_training
from evaluation.metrics import evaluate
from evaluation.onboard_metrics import measure_latency_ms, model_footprint
from experiments.common import (atk_cfg_of, common_cols, get_model, load_bundles, parse_args, profile_of,
                                subset_for_attack, write_rows)
from models import build_model


def evaluate_defended(model, eval_surrogate, cfg, terr, sat, seed, prof, wins, ywin):
    """Evaluate one (defended) model.

    Non-differentiable models are attacked by transfer from ``eval_surrogate``, a surrogate trained with a DIFFERENT seed
    than the one that generated the augmentation data. Using the same surrogate for training-time augmentation and for
    evaluation lets the model memorise that surrogate's adversarial signature and inflates its 'robustness' (accuracy
    under attack can exceed clean accuracy). The black-box greedy-ccsds attack is run on EVERY model as the
    surrogate-free check (``acc_S_greedy``).
    """
    sa, eps = cfg["attacks"]["standard"], float(cfg["defenses"]["eps"])
    gm = model if model.differentiable else eval_surrogate
    out = {}
    out["acc_T_clean"] = evaluate(model, terr["X_test"], terr["y_test"])["acc"]
    out["acc_S_clean"] = evaluate(model, sat["X_test"], sat["y_test"])["acc"]
    Xa = run_standard_attack("pgd", gm, terr["X_test"], terr["y_test"], eps, (0.0, 1.0), seed, sa["pgd_steps"])
    out["acc_T_std"] = evaluate(model, Xa, terr["y_test"])["acc"]
    Xa = run_standard_attack("pgd", gm, sat["X_test"], sat["y_test"], eps, None, seed, sa["pgd_steps"])
    out["acc_S_std"] = evaluate(model, Xa, sat["y_test"])["acc"]
    kind = "pgd_ccsds" if model.differentiable else "transfer_ccsds"
    adv, _ = attack_windows(kind, wins, model, sat["pipeline"], prof, atk_cfg_of(cfg, seed), surrogate=eval_surrogate)
    out["acc_S_ccsds"] = evaluate(model, sat["pipeline"].windows(adv), ywin)["acc"]
    out["ccsds_attack"] = kind
    advg, _ = attack_windows("greedy_ccsds", wins, model, sat["pipeline"], prof, atk_cfg_of(cfg, seed))
    out["acc_S_greedy"] = evaluate(model, sat["pipeline"].windows(advg), ywin)["acc"]
    out["eval_surrogate"] = "independent" if not model.differentiable else "n/a (white-box)"
    out.update(measure_latency_ms(model, sat["X_test"][0], n=100, warmup=10))
    out.update(model_footprint(model))
    return out


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    prof, dcfg, rows = profile_of(cfg), cfg["defenses"], []
    eps, steps, mix = float(dcfg["eps"]), int(dcfg["at_steps"]), float(dcfg["at_mix"])
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            sname = cfg["attacks"]["surrogate"]
            surrogate, sur_time = get_model(cfg, sname, ds, seed, terr, not args.no_cache)
            wins = subset_for_attack(sat["w_test"], int(cfg["attacks"]["max_windows"]))
            ywin = np.array([w.label for w in wins])
            Xtr, ytr, Xv, yv = terr["X_train"], terr["y_train"], terr["X_val"], terr["y_val"]
            eval_sur = build_model(sname, Xtr.shape[1], cfg, seed + 10_000).fit(Xtr, ytr, Xv, yv)   # independent of the PAA/AT surrogate
            for name in dcfg["models"]:
                base, base_time = get_model(cfg, name, ds, seed, terr, not args.no_cache)
                d = Xtr.shape[1]
                epochs = int(getattr(base, "epochs", 1))
                variants = ["baseline", "std_at", "paa", "at_paa", "paa_clean_control"]
                for variant in variants:
                    gen_time, info, n_aug = 0.0, {}, 0
                    if variant == "baseline":
                        model, fit_time = base, base_time
                    else:
                        model = build_model(name, d, cfg, seed)
                        extra = None
                        if variant != "std_at":
                            X_aug, y_aug, info = build_paa_set(
                                sat["w_train"], surrogate, sat["pipeline"], prof, atk_cfg_of(cfg, seed), len(ytr),
                                float(dcfg["paa"]["lam"]), float(dcfg["paa"]["adv_fraction"]), seed,
                                clean_control=(variant == "paa_clean_control"),
                                max_attack_windows=int(dcfg["paa"].get("max_attack_windows", 500)))
                            gen_time = sur_time + info["gen_time_s"]; n_aug = len(y_aug)
                            extra = (X_aug, y_aug)
                        t0 = time.perf_counter()
                        if variant in ("std_at", "at_paa"):
                            adversarial_training(model, Xtr, ytr, eps, epochs, steps, mix, seed, surrogate, (0.0, 1.0),
                                                 extra=extra, X_val=Xv, y_val=yv)
                        else:                                           # paa / paa_clean_control
                            model.fit(np.vstack([Xtr, extra[0]]), np.concatenate([ytr, extra[1]]), Xv, yv)
                        fit_time = time.perf_counter() - t0
                        if variant in ("std_at", "at_paa") and not model.differentiable:
                            gen_time += 0.0   # surrogate time already counted via get_model cache when PAA is used
                    rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), model=name, defense=variant,
                                     train_time_s=fit_time + gen_time, fit_time_s=fit_time, gen_time_s=gen_time,
                                     n_aug=n_aug, n_adv=info.get("n_adv", 0), paa_feasibility=info.get("feasibility_rate", np.nan),
                                     **evaluate_defended(model, eval_sur, cfg, terr, sat, seed, prof, wins, ywin)))
                    print(f"[defenses] seed={seed} {ds} {name} {variant} done")
    write_rows(cfg, "defenses.csv", rows)


if __name__ == "__main__":
    main()
