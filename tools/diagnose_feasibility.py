"""Why are some pgd-ccsds attempts infeasible? Tally the violated constraints (C1-C8) over the attack windows.

    python tools/diagnose_feasibility.py --config configs/my_nos3.yaml --seed 1 --model cnn [--dataset unsw_nb15]

Trains/loads the baseline model (cache), runs pgd-ccsds on every attack-class test window and prints, for the windows
whose best candidate introduced NEW violations, how often each constraint code occurred and an example message.
Use it to decide whether the mission profile (limits, gaps, sequence tolerance) or the attack settings need adjusting.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.ccsds_conformant import ConformantAttacker
from experiments.common import atk_cfg_of, get_model, load_bundles, profile_of
from config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--dataset")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds = a.dataset or cfg["datasets"][0]
    terr, sat = load_bundles(cfg, ds, a.seed)
    model, _ = get_model(cfg, a.model, ds, a.seed, terr)
    if not model.differentiable:
        sys.exit("pick a differentiable model (cnn, lstm, transformer, numpy_mlp)")
    prof = profile_of(cfg)
    atk = ConformantAttacker(sat["pipeline"], prof, atk_cfg_of(cfg, a.seed))
    wins = [w for w in sat["w_test"] if w.label == 1]
    codes, examples, bad, edits0 = Counter(), {}, 0, 0
    for w in wins:
        r = atk.pgd(model, w)
        if not r.feasible:
            bad += 1
            for v in r.violations:
                c = v.split(":")[0]
                codes[c] += 1
                examples.setdefault(c, v)
        edits0 += r.n_edits > 0
    print(f"attack windows: {len(wins)}; infeasible: {bad} ({100 * bad / max(len(wins), 1):.1f}%); windows with >=1 edit kept: {edits0}")
    for c, n in codes.most_common():
        print(f"  {c}: {n} violations, e.g. {examples[c]}")
    if not codes:
        print("  no violations recorded (every attempt was feasible).")


if __name__ == "__main__":
    main()
