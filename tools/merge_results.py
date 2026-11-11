"""Merge result CSVs from several runs (different seeds / datasets / machines) into one results folder.

    python tools/merge_results.py --inputs results_s1 results_s2 results_s3 --out results_merged
    python make_tables.py --config configs/my_run.yaml --out-dir results_merged

Each input is an output.dir (it must contain a `tables` sub-folder). Rows are concatenated and exact duplicates are
dropped. Rows from different runs must not overlap in (seed, dataset): the tool stops if the same (seed, dataset,
model/variant ...) appears in two inputs with different numbers, which would mean two runs disagree.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

FILES = ("baseline_clean.csv", "standard_attacks.csv", "ccsds_attacks.csv", "indomain.csv", "defenses.csv", "federated.csv")


def merge(inputs, out) -> dict:
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    report = {}
    for name in FILES:
        frames = []
        for d in inputs:
            p = os.path.join(d, "tables", name)
            if os.path.exists(p):
                f = pd.read_csv(p)
                f["_src"] = d
                frames.append(f)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        key_cols = [c for c in df.columns if c not in ("_src",)]
        before = len(df)
        df = df.drop_duplicates(subset=key_cols)
        ident = [c for c in ("seed", "dataset", "model", "domain", "attack", "eps", "defense", "variant", "k", "alpha") if c in df.columns]
        dup = df[df.duplicated(subset=ident, keep=False)]
        if len(dup):
            raise SystemExit(f"{name}: {len(dup)} rows share an identity key but differ in values (overlapping runs that disagree). "
                             f"Example key: {dup[ident].iloc[0].to_dict()}")
        df.drop(columns=["_src"]).to_csv(os.path.join(out, "tables", name), index=False)
        report[name] = (before, len(df))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = merge(a.inputs, a.out)
    if not rep:
        sys.exit("no result CSVs found in the given input folders")
    for k, (b, n) in rep.items():
        print(f"{k}: {b} rows in, {n} rows out")


if __name__ == "__main__":
    main()
