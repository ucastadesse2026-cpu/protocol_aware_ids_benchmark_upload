"""Phase 2: train every family on TERRESTRIAL data only; evaluate clean terrestrial and clean satellite data.

    python experiments/run_baseline.py --config configs/default.yaml [--seeds 1 2 3 4 5]
Output: results/tables/baseline_clean.csv
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import evaluate
from experiments.common import common_cols, get_model, load_bundles, parse_args, write_rows


def main() -> None:
    cfg, seeds, args = parse_args(__doc__)
    rows = []
    for seed in seeds:
        for ds in cfg["datasets"]:
            terr, sat = load_bundles(cfg, ds, seed)
            for name in cfg["model_families"]:
                model, secs = get_model(cfg, name, ds, seed, terr, not args.no_cache)
                for domain, (X, y) in (("terrestrial", (terr["X_test"], terr["y_test"])),
                                       ("satellite", (sat["X_test"], sat["y_test"]))):
                    rows.append(dict(**common_cols(cfg, terr, sat, seed, ds), model=name, domain=domain,
                                     train_time_s=secs, **evaluate(model, X, y)))
                print(f"[baseline] seed={seed} {ds} {name} done")
    write_rows(cfg, "baseline_clean.csv", rows)


if __name__ == "__main__":
    main()
