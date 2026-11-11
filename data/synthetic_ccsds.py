"""SYNTHETIC CCSDS-style packet windows for tests and the smoke configuration.

THIS IS NOT REAL SATELLITE TRAFFIC. Traffic statistics are invented to exercise the code paths
(parser, validator, attacks). Results obtained on it carry no scientific meaning.

Attack classes (all are *valid* streams under the default mission profile, so C1-C8 can hold):
    flood      high-rate TC bursts with small payloads
    injection  TC with rarely used (but legal) opcodes and large payloads
    recon      TC sweeping across allocated APIDs with tiny payloads
Replay is deliberately not generated: a replay violates C3/C7 by construction.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # allow `python data/<script>.py`

import csv
import os
from typing import List, Optional, Tuple

import numpy as np

from data.ccsds_parser import SEQ_MOD, TC, TM, CcsdsPacket, Window, packets_to_window, write_stream

ATTACKS = ("flood", "injection", "recon")


def _seq_assign(apids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seqc = np.zeros(len(apids), dtype=np.int64)
    start = {}
    cnt = {}
    for i, a in enumerate(apids):
        if a not in start:
            start[a] = int(rng.integers(0, SEQ_MOD))
            cnt[a] = 0
        seqc[i] = (start[a] + cnt[a]) % SEQ_MOD
        cnt[a] += 1
    return seqc


def make_window(rng: np.random.Generator, attack: Optional[str], size: int = 32, t0: Optional[float] = None) -> Window:
    t0 = float(rng.uniform(1e6, 2e6)) if t0 is None else t0
    if attack is None:                                    # benign: periodic housekeeping TM + occasional TC
        is_tc = rng.random(size) < 0.25
        gaps = rng.exponential(0.8, size)
        plen_tm = rng.integers(48, 81, size)
        plen_tc = rng.integers(4, 13, size)
        opc = rng.integers(1, 17, size)
        apid = np.where(is_tc, 0x201, 0x101)
    elif attack == "flood":
        is_tc = rng.random(size) < 0.9
        gaps = rng.exponential(0.04, size)
        plen_tm = rng.integers(48, 81, size)
        plen_tc = rng.integers(2, 9, size)
        opc = rng.integers(17, 21, size)
        apid = np.where(is_tc, 0x201, 0x101)
    elif attack == "injection":
        is_tc = rng.random(size) < 0.7
        gaps = rng.exponential(0.3, size)
        plen_tm = rng.integers(48, 81, size)
        plen_tc = rng.integers(100, 201, size)
        opc = rng.integers(21, 33, size)
        apid = np.where(is_tc, 0x202, 0x101)
    elif attack == "recon":
        is_tc = rng.random(size) < 0.85
        gaps = rng.exponential(0.12, size)
        plen_tm = rng.integers(48, 81, size)
        plen_tc = rng.integers(0, 3, size)
        opc = rng.integers(1, 33, size)
        apid = np.where(is_tc, rng.integers(0x200, 0x210, size), 0x101)
    else:
        raise ValueError(attack)
    gaps = np.round(np.maximum(gaps, 1e-4) * 65536) / 65536
    gaps[0] = 0.0
    ts = t0 + np.cumsum(gaps)
    ptype = np.where(is_tc, TC, TM).astype(np.int64)
    base = np.where(is_tc, plen_tc, plen_tm).astype(np.int64)
    opcode = np.where(is_tc, opc, -1).astype(np.int64)
    apid = apid.astype(np.int64)
    return Window(ptype=ptype, apid=apid, seqc=_seq_assign(apid, rng), ts=ts, base_plen=base,
                  pad=np.zeros(size, dtype=np.int64), opcode=opcode, sec=np.ones(size, dtype=bool),
                  label=0 if attack is None else 1, attack_type=attack or "benign")


def make_windows(n: int, seed: int, attack_fraction: float = 0.4, size: int = 32) -> List[Window]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if rng.random() < attack_fraction:
            out.append(make_window(rng, ATTACKS[int(rng.integers(0, len(ATTACKS)))], size))
        else:
            out.append(make_window(rng, None, size))
    return out


def write_sample_files(out_dir: str, n_windows: int = 24, seed: int = 7, size: int = 32) -> Tuple[str, str]:
    """Write ``synthetic_ccsds_sample.bin`` and a per-packet ``.labels.csv`` (label, attack_type)."""
    from data.ccsds_parser import window_to_packets
    os.makedirs(out_dir, exist_ok=True)
    wins = make_windows(n_windows, seed, 0.4, size)
    pk, labels = [], []
    for w in wins:
        pk += window_to_packets(w)
        labels += [(w.label, w.attack_type)] * len(w)
    bin_path = os.path.join(out_dir, "synthetic_ccsds_sample.bin")
    with open(bin_path, "wb") as fh:
        fh.write(write_stream(pk))
    lab_path = os.path.join(out_dir, "synthetic_ccsds_sample.labels.csv")
    with open(lab_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["packet_index", "label", "attack_type"])
        for i, (l, a) in enumerate(labels):
            wr.writerow([i, l, a])
    return bin_path, lab_path


if __name__ == "__main__":
    print(write_sample_files(os.path.join(os.path.dirname(__file__), "samples")))
