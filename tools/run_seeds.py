"""Run the whole experiment chain seed by seed (resumable), merge the results and build the tables.

    python tools/run_seeds.py --config configs/my_run.yaml --seeds 1 2 3 4 5 --final-dir results_final

For every seed the five experiment scripts run into their own folder results_s<seed>; a seed whose
results_s<seed>/tables/federated.csv already exists is skipped (use --force to redo it). Logs go to logs/.
After all seeds: tools/merge_results.py -> <final-dir>, then make_tables.py --out-dir <final-dir>.
If a script fails, the last lines of its log are printed and the runner stops; fix the problem and run the same
command again - finished seeds are not repeated. Elapsed time per script is printed (use it to plan the full run).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import load_config

SCRIPTS = ("run_baseline", "run_standard_attacks", "run_ccsds_attacks", "run_indomain", "run_defenses", "run_federated")


def run(cmd, log_path, env=None) -> int:
    with open(log_path, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def tail(path, n=15) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-n:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--final-dir", default="results_final")
    ap.add_argument("--force", action="store_true", help="redo seeds that are already complete")
    ap.add_argument("--jobs", type=int, default=1, help="run this many seeds in parallel (each seed is a separate process chain; CPU only)")
    ap.add_argument("--only", nargs="*", choices=SCRIPTS, help="rerun ONLY these scripts for the given seeds (their CSV is rewritten in results_s<seed>)")
    a = ap.parse_args()
    seeds = a.seeds or load_config(a.config)["seeds"]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    dirs = [f"results_s{s}" for s in seeds]
    env = dict(os.environ)
    if a.jobs > 1:                                   # avoid CPU oversubscription: split the cores among the jobs
        per = str(max(1, (os.cpu_count() or 1) // a.jobs))
        per = str(min(int(per), 2))                  # HARD CAP: max 2 BLAS threads per child
        env.update(OMP_NUM_THREADS=per, MKL_NUM_THREADS=per, OPENBLAS_NUM_THREADS=per)

    def one_seed(s):
        out = f"results_s{s}"
        if os.path.exists(os.path.join(ROOT, out, "tables", "federated.csv")) and not a.force and not a.only:
            print(f"seed {s}: already complete in {out}, skipping (use --force to redo)", flush=True)
            return None
        for sc in (a.only or SCRIPTS):
            log = os.path.join(ROOT, "logs", f"{sc}_s{s}.txt")
            t0 = time.time()
            print(f"seed {s}: {sc} ...", flush=True)
            rc = run([sys.executable, os.path.join(ROOT, "experiments", sc + ".py"), "--config", a.config,
                      "--seeds", str(s), "--out-dir", out], log, env)
            print(f"seed {s}: {sc} {'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
            if rc != 0:
                return (s, sc, log)
        return None

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        failures = [f for f in ex.map(one_seed, seeds) if f]
    for s, sc, log in failures:
        print(f"\n--- last lines of {log} ---\n{tail(log)}")
    if failures:
        sys.exit("failed: " + ", ".join(f"seed {s} {sc}" for s, sc, _ in failures) +
                 ". Fix the error and re-run the same command (finished seeds are kept).")
    rc = run([sys.executable, os.path.join(ROOT, "tools", "merge_results.py"), "--inputs", *dirs, "--out", a.final_dir],
             os.path.join(ROOT, "logs", "merge.txt"))
    if rc != 0:
        sys.exit("merge failed:\n" + tail(os.path.join(ROOT, "logs", "merge.txt")))
    rc = run([sys.executable, os.path.join(ROOT, "make_tables.py"), "--config", a.config, "--out-dir", a.final_dir],
             os.path.join(ROOT, "logs", "make_tables.txt"))
    if rc != 0:
        sys.exit("make_tables failed:\n" + tail(os.path.join(ROOT, "logs", "make_tables.txt")))
    print(f"\nDone. Tables: {a.final_dir}\\tables   Figures: {a.final_dir}\\figures   (Table 9: table9_rq_summary.md)")


if __name__ == "__main__":
    main()
