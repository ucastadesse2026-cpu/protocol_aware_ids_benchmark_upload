"""Phase 6: FedAvg across K simulated satellite nodes with Dirichlet(alpha) label skew.

Variants: fedavg | fedavg_at (local PGD-AT) | federated_paa (each node augments with conformant adversarial windows
generated against a NODE-LOCAL surrogate) | federated_paa_clean_control (same windows, unperturbed).
Nodes whose shard contains a single class cannot train a meaningful surrogate and skip perturbation (documented).
Output: results/tables/federated.csv
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
from experiments.common import (atk_cfg_of, common_cols, load_bundles, parse_args, profile_of, subset_for_attack,
                                write_rows)
from models import build_model
from models.federated import FedAvg, dirichlet_partition


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    f, prof, rows = cfg["federated"], profile_of(cfg), []
    sa, dcfg = cfg["attacks"]["standard"], cfg["defenses"]
    eps, steps, mix = float(dcfg["eps"]), int(dcfg["at_steps"]), float(dcfg["at_mix"])
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            pipe, d = sat["pipeline"], terr["X_train"].shape[1]
            wins = subset_for_attack(sat["w_test"], int(cfg["attacks"]["max_windows"]))
            ywin = np.array([w.label for w in wins])
            for K in f["k"]:
                for alpha in f["alpha"]:
                    rng = np.random.default_rng(seed)
                    idx = dirichlet_partition(terr["y_train"], int(K), float(alpha), rng)
                    parts = [(terr["X_train"][i], terr["y_train"][i]) for i in idx]
                    shares = np.array_split(rng.permutation(len(sat["w_train"])), int(K))
                    for variant in f["variants"]:
                        t0 = time.perf_counter()
                        use_parts, local_fit, n_adv_nodes = parts, None, 0
                        if variant.startswith("federated_paa"):
                            use_parts = []
                            for k, (Xk, yk) in enumerate(parts):
                                nodew = [sat["w_train"][j] for j in shares[k]]
                                clean = variant.endswith("clean_control") or len(np.unique(yk)) < 2
                                sur = build_model("numpy_mlp", d, cfg, seed + k).fit(Xk, yk)
                                Xa, ya, info = build_paa_set(nodew, sur, pipe, prof, atk_cfg_of(cfg, seed + k), len(yk),
                                                             float(dcfg["paa"]["lam"]), float(dcfg["paa"]["adv_fraction"]),
                                                             seed + k, clean_control=clean)
                                n_adv_nodes += int(info["n_adv"] > 0)
                                use_parts.append((np.vstack([Xk, Xa]), np.concatenate([yk, ya])))
                        elif variant == "fedavg_at":
                            def local_fit(m, X, y, ep, node, _s=seed):
                                adversarial_training(m, X, y, eps, ep, steps, mix, _s + node, None, (0.0, 1.0))
                        fa = FedAvg(lambda: build_model(f["backbone"], d, cfg, seed), int(f["rounds"]),
                                    int(f["local_epochs"]), seed=seed)
                        model = fa.fit(use_parts, terr["X_val"], terr["y_val"], local_fit=local_fit)
                        secs = time.perf_counter() - t0
                        out = dict(acc_T_clean=evaluate(model, terr["X_test"], terr["y_test"])["acc"],
                                   acc_S_clean=evaluate(model, sat["X_test"], sat["y_test"])["acc"])
                        Xa = run_standard_attack("pgd", model, terr["X_test"], terr["y_test"], eps, (0.0, 1.0), seed, sa["pgd_steps"])
                        out["acc_T_std"] = evaluate(model, Xa, terr["y_test"])["acc"]
                        Xa = run_standard_attack("pgd", model, sat["X_test"], sat["y_test"], eps, None, seed, sa["pgd_steps"])
                        out["acc_S_std"] = evaluate(model, Xa, sat["y_test"])["acc"]
                        adv, _ = attack_windows("pgd_ccsds", wins, model, pipe, prof, atk_cfg_of(cfg, seed))
                        out["acc_S_ccsds"] = evaluate(model, pipe.windows(adv), ywin)["acc"]
                        rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), k=int(K), alpha=float(alpha),
                                         variant=variant, backbone=f["backbone"], rounds=int(f["rounds"]),
                                         comm_bytes_per_round=fa.bytes_per_round, train_time_s=secs,
                                         nodes_with_adv=n_adv_nodes, **out))
                        print(f"[federated] seed={seed} {ds} K={K} alpha={alpha} {variant} done")
    write_rows(cfg, "federated.csv", rows)


if __name__ == "__main__":
    main()
