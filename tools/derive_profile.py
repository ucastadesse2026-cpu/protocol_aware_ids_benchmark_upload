"""Derive a mission-profile fragment and sanity checks from a converted capture (benign + attack stream).

    python tools/derive_profile.py --stream data/raw/nos3/nos3 [--catalog tools/my_catalog.json]

Prints (1) which cFS checksum initial value (0xFF or 0x00) validates the captured TC packets, (2) a YAML
`profile:` fragment (APIDs, legal function codes, size bounds, sequence tolerance) to paste into your config,
(3) the fraction of windows that are already valid under that profile, and (4) window/attack counts per
chronological split. REVIEW every line: legal commands taken from a capture are only as complete as the capture;
a --catalog (the commands you declare legal, e.g. the generator's catalog) overrides the observed set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.validator import MissionProfile, validate_window
from data.ccsds_parser import SEQ_MOD, TC, TM, get_format, packet_crc, read_stream, set_format
from data.preprocess import windows_from_stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", required=True, help="prefix written by pcap_to_stream.py (<prefix>.bin / .labels.csv)")
    ap.add_argument("--format", default="cfs", choices=["cfs", "project"])
    ap.add_argument("--catalog", help="JSON catalog of legal commands (overrides observed legal function codes)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--label-rule", default="any", choices=["any", "majority"])
    ap.add_argument("--pad-headroom", type=int, default=64)
    a = ap.parse_args()
    bin_p, lab_p = a.stream + ".bin", a.stream + ".labels.csv"

    set_format(a.format)
    with open(bin_p, "rb") as fh:
        pk = read_stream(fh.read())
    with open(lab_p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"packets: {len(pk)}  (TC {sum(p.ptype == TC for p in pk)}, TM {sum(p.ptype == TM for p in pk)}); "
          f"attack-labelled: {sum(int(r['label']) for r in rows)}")

    xor_init = None
    if a.format == "cfs":
        tcs = [p for p in pk if p.ptype == TC and p.sec_hdr]
        best = {}
        for init in (0xFF, 0x00):
            set_format("cfs", xor_init=init)
            best[init] = sum(1 for p in tcs if p.pec == packet_crc(p)) / max(len(tcs), 1)
        print(f"cFS TC checksum validates: init 0xFF -> {best[0xFF]:.1%}, init 0x00 -> {best[0x00]:.1%}")
        xor_init = max(best, key=best.get)
        if best[xor_init] < 0.99:
            print("WARNING: neither initial value validates >99% of TC packets; the checksum rule of your cFE version "
                  "differs (check CFE_MSG checksum code) or the capture contains corrupted/foreign packets.")
        set_format("cfs", xor_init=xor_init)

    tm_apids = sorted({p.apid for p in pk if p.ptype == TM})
    tc_apids = sorted({p.apid for p in pk if p.ptype == TC})
    legal = defaultdict(set)
    if a.catalog:
        with open(a.catalog) as fh:
            for c in json.load(fh)["commands"]:
                legal[int(c["apid"])].add(int(c["fc"]))
        tc_apids = sorted(set(tc_apids) | set(legal))
    else:
        for p in pk:
            if p.ptype == TC and p.opcode is not None:
                legal[p.apid].add(p.opcode)
    mx = lambda t: max([len(p.payload) for p in pk if p.ptype == t] or [0])
    steps = []
    last = {}
    for p in pk:
        if p.apid in last:
            steps.append((p.seq_count - last[p.apid]) % SEQ_MOD)
        last[p.apid] = p.seq_count
    tol = int(max(0, np.percentile(steps, 99.9) - 1)) if steps else 0
    print("\n# ---- paste into your config (REVIEW each line) ----")
    print("profile:")
    print("  version: 0")
    print(f"  tm_apid_list: {tm_apids}")
    print(f"  tc_apid_list: {tc_apids}")
    print("  legal_opcodes:")
    for apid in sorted(legal):
        print(f"    {apid}: {sorted(legal[apid])}")
    print(f"  max_user_data_tm: {mx(TM) + a.pad_headroom}   # observed max {mx(TM)} + headroom: SET YOUR MISSION'S REAL LIMIT")
    print(f"  max_user_data_tc: {mx(TC) + a.pad_headroom}   # observed max {mx(TC)} + headroom: SET YOUR MISSION'S REAL LIMIT")
    print(f"  seq_tolerance: {tol}   # 99.9th percentile of observed per-APID sequence steps minus 1")
    print("  max_gap_s: 60.0")
    if xor_init is not None:
        print("# data.satellite:  wire_format: cfs   wire_overrides: {xor_init: %d}   label_rule: %s" % (xor_init, a.label_rule))

    # checks on windows
    import yaml
    prof = MissionProfile.from_dict(yaml.safe_load(
        "version: 0\ntm_apid_list: %s\ntc_apid_list: %s\nlegal_opcodes: {%s}\nmax_user_data_tm: %d\nmax_user_data_tc: %d\nseq_tolerance: %d\n"
        % (tm_apids, tc_apids, ", ".join(f"{k}: {sorted(v)}" for k, v in legal.items()), mx(TM) + a.pad_headroom,
           mx(TC) + a.pad_headroom, tol)))
    wins = windows_from_stream(bin_p, lab_p, a.window, a.window, a.label_rule)
    valid = [not validate_window(w, prof) for w in wins]
    print(f"\nwindows ({a.window} packets, rule={a.label_rule}): {len(wins)}; attack windows: {sum(w.label for w in wins)}; "
          f"already valid under this profile: {np.mean(valid):.1%}")
    n = len(wins)
    for name, lo, hi in (("train", 0, int(.6 * n)), ("val", int(.6 * n), int(.8 * n)), ("test", int(.8 * n), n)):
        part = wins[lo:hi]
        print(f"  {name:5s}: {len(part):6d} windows, {sum(w.label for w in part):6d} attack")
    if any(sum(w.label for w in wins[lo:hi]) == 0 for lo, hi in ((0, int(.6 * n)), (int(.8 * n), n))):
        print("WARNING: a chronological split has NO attack windows: interleave benign and attack sessions over the capture.")


if __name__ == "__main__":
    main()
