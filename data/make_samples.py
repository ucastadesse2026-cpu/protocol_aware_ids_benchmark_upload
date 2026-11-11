"""Create the small sample files shipped with the repository.

    python data/make_samples.py                 # SYNTHETIC samples (always works; labelled synthetic)
    python data/make_samples.py --real --config configs/default.yaml
                                                # 10,000-row preprocessed Fc samples from the REAL datasets
                                                # you downloaded (data/raw/...); written to data/samples/
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from data import feature_schema as fs
from data.synthetic_ccsds import write_sample_files
from data.synthetic_terrestrial import make_unsw_like

OUT = os.path.join(ROOT, "data", "samples")


def synthetic() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = make_unsw_like(10_000, seed=7)
    df.to_csv(os.path.join(OUT, "SYNTHETIC_unsw_like_10k.csv"), index=False)
    print("wrote SYNTHETIC_unsw_like_10k.csv (synthetic flow records, UNSW-NB15-style columns)")
    print("wrote", write_sample_files(OUT, n_windows=24, seed=7))


def real(config: str) -> None:
    from config import load_config
    from data.preprocess import _read_terrestrial
    cfg = load_config(config)
    for ds in cfg["datasets"]:
        dcfg = cfg["data"]["terrestrial"][ds]
        df = _read_terrestrial(dcfg["path"], ds).sample(10_000, random_state=0)
        view = dcfg.get("view", cfg["features"]["view"])
        X, y = fs.terrestrial_to_fc(df, ds, view)
        out = pd.DataFrame(X, columns=fs.view_names(view)); out["label"] = y
        out.to_csv(os.path.join(OUT, f"{ds}_fc_10k.csv"), index=False)
        print(f"wrote {ds}_fc_10k.csv ({len(out)} rows, view={view})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true"); ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    real(a.config) if a.real else synthetic()
