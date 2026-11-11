"""Dataset acquisition helper.

It (i) prints the official dataset pages and the download routes checked on 2026-10-10 (see DATASETS below),
(ii) checks that the expected files exist under data/raw/, and (iii) can fetch a file from a URL you supply.
Only NSL-KDD has a direct file URL (a third-party mirror); UNSW-NB15 (SharePoint folder) and CICIDS2017
(registration form) must be downloaded in a browser. None of the files were downloaded by the authoring session.

    python data\\download_datasets.py --check
    python data\\download_datasets.py --fetch nsl_kdd --url <direct-file-url>
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

DATASETS = {
    "unsw_nb15": dict(
        landing="Official page: https://research.unsw.edu.au/projects/unsw-nb15-dataset  (its 'HERE' link opens a UNSW SharePoint folder: "
                "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS ). "
                "Browser download only; no stable direct file URL. Checked 2026-10-10.",
        expect=["unsw_nb15/UNSW_NB15_training-set.csv"],
        note="Use the provided training-set CSV (columns include dur, spkts, dpkts, sbytes, dbytes, label)."),
    "cicids2017": dict(
        landing="Official page: https://www.unb.ca/cic/datasets/ids-2017.html ; its download button leads to a registration form at "
                "http://cicresearch.ca/CICDataset/CIC-IDS-2017/ (name, e-mail, organisation, country) - no direct file URL. "
                "UNOFFICIAL mirror (not verified for integrity): https://huggingface.co/datasets/bencorn/CICIDS2017 . Checked 2026-10-10.",
        expect=["cicids2017/"],
        note="Use the 'MachineLearningCSV' daily CSV files; put all CSVs in data/raw/cicids2017/."),
    "nsl_kdd": dict(
        landing="Official page https://www.unb.ca/cic/datasets/nsl.html says the dataset is NO LONGER AVAILABLE there (licence allows mirrors). "
                "Third-party GitHub mirror: https://github.com/Jehuty4949/NSL_KDD (file KDDTrain+.txt; raw: "
                "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTrain%2B.txt - not downloaded or verified by the authoring session). Checked 2026-10-10.",
        expect=["nsl_kdd/KDDTrain+.txt"],
        note="Header-less CSV; column names are supplied by data/feature_schema.py."),
}


def check() -> int:
    missing = 0
    for name, d in DATASETS.items():
        for rel in d["expect"]:
            p = os.path.join(RAW, rel)
            ok = os.path.exists(p) and (os.path.isfile(p) or any(f.endswith('.csv') for f in os.listdir(p)))
            print(f"[{'ok' if ok else 'MISSING'}] {name}: {rel}")
            missing += 0 if ok else 1
    return missing


def fetch(name: str, url: str, retries: int = 5) -> None:
    """Download with retries into a .part file; the final file appears only if the download completed
    (size must match Content-Length when the server sends it)."""
    import time
    d = DATASETS[name]
    dest = os.path.join(RAW, d["expect"][0])
    if dest.endswith("/") or dest.endswith("\\"):
        os.makedirs(dest, exist_ok=True)
        dest = os.path.join(dest, os.path.basename(url.split("?")[0]) or "download.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading ({attempt}/{retries}) {url} -> {dest}")
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:
                total = r.headers.get("Content-Length")
                got = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
            if total is not None and got != int(total):
                raise IOError(f"incomplete download: got {got} of {total} bytes")
            os.replace(part, dest)
            print(f"OK: {got} bytes written to {dest}")
            return
        except Exception as exc:                     # network errors are retried; anything else is also reported
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(min(2 * attempt, 10))
    if os.path.exists(part):
        os.remove(part)
    raise SystemExit(f"Download failed after {retries} attempts. Download the file in a browser or with curl instead "
                     f"(see data/README.md) and save it to {dest}")


def main() -> int:
    epilog = "\n".join(f"- {n}: {d['landing']}\n    expected: data/raw/{d['expect'][0]}" for n, d in DATASETS.items())
    ap = argparse.ArgumentParser(description=__doc__, epilog="Dataset sources:\n" + epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fetch", choices=list(DATASETS))
    ap.add_argument("--url")
    a = ap.parse_args()
    for n, d in DATASETS.items():
        print(f"- {n}: {d['landing']}\n    {d['note']}")
    if a.fetch:
        if not a.url:
            print("--fetch requires --url (a direct link taken from the official page)", file=sys.stderr)
            return 2
        fetch(a.fetch, a.url)
    return 1 if (a.check and check()) else 0


if __name__ == "__main__":
    sys.exit(main())
