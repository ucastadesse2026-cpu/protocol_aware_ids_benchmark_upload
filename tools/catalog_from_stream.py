"""Build a command-catalog SKELETON from real captured cFS commands (output of tools/pcap_to_stream.py).

    python tools/catalog_from_stream.py --stream data/raw/nos3/nos3 --out tools/my_catalog.json [--max-payload 64]

Takes every distinct (APID, function code) seen in TC packets of the stream, keeps its most frequent payload, and writes
catalog entries for tools/nos3_traffic.py. Class suggestion (REVIEW IT): commands seen at least as often as the median are
proposed 'benign', rarer ones 'sensitive'. The tool cannot know what a command DOES: before using the catalog, read each
entry, delete anything that could reset, delete, power off or otherwise harm the simulation run, and make sure at least
one benign and one sensitive command remain. Commands with payloads longer than --max-payload are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ccsds_parser import TC, read_stream, set_format


def build(stream: str, max_payload: int = 64, fmt: str = "cfs", **overrides) -> list:
    set_format(fmt, **overrides)
    with open(stream + ".bin", "rb") as fh:
        pk = read_stream(fh.read())
    counts = Counter()
    payloads = defaultdict(Counter)
    for p in pk:
        if p.ptype == TC and p.opcode is not None:
            counts[(p.apid, p.opcode)] += 1
            payloads[(p.apid, p.opcode)][p.payload] += 1
    if not counts:
        raise SystemExit("no TC packets in the stream")
    med = float(np.median(list(counts.values())))
    cmds = []
    for (apid, fc), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        payload = payloads[(apid, fc)].most_common(1)[0][0]
        if len(payload) > max_payload:
            continue
        cmds.append(dict(name=f"apid{apid:#05x}_fc{fc}", apid=apid, fc=fc, payload_hex=payload.hex(),
                         klass="benign" if n >= med else "sensitive", observed_count=n,
                         _note="AUTO-SUGGESTED class: REVIEW what this command does before use"))
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-payload", type=int, default=64)
    ap.add_argument("--format", default="cfs", choices=["cfs", "project"])
    a = ap.parse_args()
    cmds = build(a.stream, a.max_payload, a.format)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"_comment": "AUTO-GENERATED SKELETON from captured commands. Review every entry before use.",
                   "commands": cmds}, fh, indent=2)
    nb = sum(c["klass"] == "benign" for c in cmds)
    print(f"wrote {len(cmds)} commands to {a.out}: {nb} suggested benign, {len(cmds) - nb} suggested sensitive.")
    print("NOW: open the file, check what each command does, delete risky ones, fix the classes.")
    if nb == 0 or nb == len(cmds):
        print("WARNING: the generator needs at least one benign AND one sensitive command; edit klass values.")


if __name__ == "__main__":
    main()
