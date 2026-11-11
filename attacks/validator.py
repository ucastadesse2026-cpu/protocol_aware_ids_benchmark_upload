"""C1-C8 constraint validator for CCSDS-conformant adversarial examples.

C1 version / type / secondary-header consistency      (per packet)
C2 APID within the allocated set for the packet type   (per packet)
C3 sequence flags legal and sequence count continuous  (per stream, per APID, modulo 2**14)
C4 length consistency (legal user-data bounds; length field matches the bytes)
C5 error-control field (PEC) valid
C6 command legality (TC opcode legal for the APID; TM carries no opcode)
C7 timing consistency (timestamps present, monotone, bounded gaps, representable)
C8 functionality preservation (immutable fields and label unchanged between original and adversarial)

The profile is a *mission profile*: the defaults below are the synthetic project profile used by the
test-suite and the smoke configuration. Real missions must supply their own values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from data.ccsds_parser import (SEQ_MOD, TC, TM, CcsdsPacket, ParseError, Window, decode, get_format, packet_crc,
                               window_to_packets, wire_length)


@dataclass
class Violation:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class MissionProfile:
    version: int = 0
    tm_apids: Set[int] = field(default_factory=lambda: set(range(0x100, 0x110)))
    tc_apids: Set[int] = field(default_factory=lambda: set(range(0x200, 0x210)))
    sec_hdr_required: Dict[int, bool] = field(default_factory=lambda: {TM: True, TC: True})
    seq_flags_allowed: Set[int] = field(default_factory=lambda: {3})
    legal_opcodes: Dict[int, Set[int]] = field(default_factory=dict)   # per TC APID
    default_opcodes: Set[int] = field(default_factory=lambda: set(range(1, 33)))
    min_user_data: int = 0
    max_user_data: Dict[int, int] = field(default_factory=lambda: {TM: 1024, TC: 256})
    seq_tolerance: int = 0          # extra forward skips tolerated between consecutive packets of an APID
    max_gap_s: float = 60.0
    max_timestamp: float = float(2 ** 32 - 1)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "MissionProfile":
        d = dict(d or {})
        p = MissionProfile()
        for k in ("version", "seq_tolerance", "min_user_data", "max_gap_s"):
            if k in d:
                setattr(p, k, d[k])
        if "tm_apid_list" in d:
            p.tm_apids = set(int(a) for a in d["tm_apid_list"])
        if "tc_apid_list" in d:
            p.tc_apids = set(int(a) for a in d["tc_apid_list"])
        if "legal_opcodes" in d:
            p.legal_opcodes = {int(a): set(int(c) for c in cs) for a, cs in d["legal_opcodes"].items()}
        if "tm_apids" in d:
            p.tm_apids = set(range(d["tm_apids"][0], d["tm_apids"][1] + 1))
        if "tc_apids" in d:
            p.tc_apids = set(range(d["tc_apids"][0], d["tc_apids"][1] + 1))
        if "default_opcodes" in d:
            p.default_opcodes = set(range(d["default_opcodes"][0], d["default_opcodes"][1] + 1))
        if "default_opcode_list" in d:
            p.default_opcodes = set(int(c) for c in d["default_opcode_list"])
        if "max_user_data_tm" in d:
            p.max_user_data[TM] = d["max_user_data_tm"]
        if "max_user_data_tc" in d:
            p.max_user_data[TC] = d["max_user_data_tc"]
        return p

    def allowed_apids(self, ptype: int) -> Set[int]:
        return self.tc_apids if ptype == TC else self.tm_apids

    def opcodes_for(self, apid: int) -> Set[int]:
        return self.legal_opcodes.get(apid, self.default_opcodes)


# ---------------------------------------------------------------------------------------
def validate_packet(pkt: CcsdsPacket, prof: MissionProfile) -> List[Violation]:
    """Per-packet constraints C1, C2, C4 (bounds), C5, C6, C7 (timestamp presence/range)."""
    v: List[Violation] = []
    # C1
    if pkt.version != prof.version:
        v.append(Violation("C1", f"version {pkt.version} != {prof.version}"))
    if pkt.ptype not in (TM, TC):
        v.append(Violation("C1", f"invalid packet type {pkt.ptype}"))
        return v
    if bool(pkt.sec_hdr) != prof.sec_hdr_required.get(pkt.ptype, True):
        v.append(Violation("C1", "secondary-header flag inconsistent with profile"))
    # C2
    if pkt.apid not in prof.allowed_apids(pkt.ptype):
        v.append(Violation("C2", f"APID {pkt.apid:#x} not allocated for type {pkt.ptype}"))
    # C3 (per-packet part)
    if pkt.seq_flags not in prof.seq_flags_allowed:
        v.append(Violation("C3", f"sequence flags {pkt.seq_flags} not allowed"))
    if not 0 <= pkt.seq_count < SEQ_MOD:
        v.append(Violation("C3", "sequence count outside 14-bit range"))
    # C4
    n = len(pkt.payload)
    if not prof.min_user_data <= n <= prof.max_user_data[pkt.ptype]:
        v.append(Violation("C4", f"user data length {n} outside legal bounds"))
    # C5
    try:
        if pkt.pec is not None and pkt.pec != packet_crc(pkt):
            v.append(Violation("C5", "packet error control field invalid"))
    except Exception as exc:  # unrepresentable packet cannot have a valid PEC
        v.append(Violation("C5", f"cannot compute PEC: {exc}"))
    # C6
    if pkt.ptype == TC:
        if pkt.opcode is None or pkt.opcode not in prof.opcodes_for(pkt.apid):
            v.append(Violation("C6", f"opcode {pkt.opcode} not legal for APID {pkt.apid:#x}"))
    elif pkt.opcode is not None:
        v.append(Violation("C6", "TM packet carries an opcode"))
    # C7 (per-packet part)
    if pkt.sec_hdr:
        if pkt.timestamp is None:
            if get_format().ts_in_header:               # cFS packets carry no timestamp: arrival time is attached later
                v.append(Violation("C7", "timestamp missing"))
        elif not 0 <= pkt.timestamp <= prof.max_timestamp:
            v.append(Violation("C7", "timestamp not representable"))
    return v


def validate_bytes(raw: bytes, prof: MissionProfile) -> List[Violation]:
    """Validate a single packet from raw bytes, including length-field consistency (C4) and PEC (C5)."""
    try:
        pkt, end = decode(raw, 0)
    except ParseError as exc:
        return [Violation("C4", f"length/structure error: {exc}")]
    v = validate_packet(pkt, prof)
    if end != len(raw):
        v.append(Violation("C4", f"{len(raw) - end} trailing bytes after the packet"))
    return v


def validate_stream(packets: Sequence[CcsdsPacket], prof: MissionProfile) -> List[Violation]:
    """Per-packet constraints plus stream-level C3 (sequence continuity per APID) and C7 (monotone time)."""
    v: List[Violation] = []
    last_seq: Dict[int, int] = {}
    last_ts: Optional[float] = None
    for i, p in enumerate(packets):
        v += [Violation(x.code, f"[pkt {i}] {x.message}") for x in validate_packet(p, prof)]
        if p.apid in last_seq:
            diff = (p.seq_count - last_seq[p.apid]) % SEQ_MOD
            if not 1 <= diff <= 1 + prof.seq_tolerance:
                v.append(Violation("C3", f"[pkt {i}] APID {p.apid:#x} sequence step {diff} not in 1..{1 + prof.seq_tolerance}"))
        last_seq[p.apid] = p.seq_count
        if p.sec_hdr and p.timestamp is not None:
            if last_ts is not None:
                gap = p.timestamp - last_ts
                if gap < 0:
                    v.append(Violation("C7", f"[pkt {i}] timestamp decreases"))
                elif gap > prof.max_gap_s:
                    v.append(Violation("C7", f"[pkt {i}] gap {gap:.2f}s exceeds {prof.max_gap_s}s"))
            last_ts = p.timestamp
    return v


def check_functionality(orig: Window, adv: Window) -> List[Violation]:
    """C8: immutable fields (packet count, type, opcode, parameter length, first timestamp) and label preserved."""
    v: List[Violation] = []
    if len(orig) != len(adv):
        return [Violation("C8", "packet count changed")]
    if orig.label != adv.label:
        v.append(Violation("C8", "label changed"))
    if not np.array_equal(orig.ptype, adv.ptype):
        v.append(Violation("C8", "packet type changed"))
    if not np.array_equal(orig.opcode, adv.opcode):
        v.append(Violation("C8", "opcode changed"))
    if not np.array_equal(orig.base_plen, adv.base_plen):
        v.append(Violation("C8", "parameter bytes changed"))
    if abs(float(orig.ts[0]) - float(adv.ts[0])) > 1e-9:
        v.append(Violation("C8", "stream start time changed"))
    return v


def validate_window(w: Window, prof: MissionProfile, original: Optional[Window] = None) -> List[Violation]:
    """Full C1-C8 validation of a window (C8 only when ``original`` is given)."""
    v = validate_stream(window_to_packets(w), prof)
    if original is not None:
        v += check_functionality(original, w)
    return v


def is_valid(violations: Iterable[Violation]) -> bool:
    return not list(violations)


# ---------------------------------------------------------------------------------------
def feature_space_realizable(X_raw: np.ndarray, view: str, prof: MissionProfile, rel_tol: float = 0.05) -> np.ndarray:
    """Necessary-condition realizability proxy for *feature-space* adversarial examples.

    Feature-space examples have no packet, so true C1-C8 validation is impossible. This returns, per row,
    whether the raw (pre-scaling) feature vector could come from *some* packet window: non-negative
    quantities, byte totals within [n_pkts * min_len, n_pkts * max_len], and mean length / rate
    consistent with the redundant totals. The core view has no redundancy, so only range checks
    apply and the rejection rate is a LOWER BOUND on the true rejection rate.
    """
    from data import feature_schema as fs

    X = np.asarray(X_raw, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    idx = {n: i for i, n in enumerate(fs.view_names(view))}
    dur = np.expm1(X[:, idx["log_duration"]])
    fb = np.expm1(X[:, idx["log_fwd_bytes"]])
    bb = np.expm1(X[:, idx["log_bwd_bytes"]])
    ok &= (dur >= -1e-9) & (fb >= -1e-9) & (bb >= -1e-9)
    ok &= (X[:, idx["fwd_byte_ratio"]] >= -1e-9) & (X[:, idx["fwd_byte_ratio"]] <= 1 + 1e-9)
    if "log_fwd_pkts" in idx:
        fp = np.expm1(X[:, idx["log_fwd_pkts"]])
        bp = np.expm1(X[:, idx["log_bwd_pkts"]])
        min_tc, min_tm = wire_length(True, TC, prof.min_user_data), wire_length(True, TM, prof.min_user_data)
        max_tc, max_tm = wire_length(True, TC, prof.max_user_data[TC]), wire_length(True, TM, prof.max_user_data[TM])
        ok &= (fp >= -1e-9) & (bp >= -1e-9)
        ok &= (fb >= fp * min_tc * (1 - rel_tol)) & (fb <= fp * max_tc * (1 + rel_tol))
        ok &= (bb >= bp * min_tm * (1 - rel_tol)) & (bb <= bp * max_tm * (1 + rel_tol))
        tot_p = fp + bp
        mean_len = X[:, idx["mean_pkt_len"]]
        ok &= np.abs(mean_len - (fb + bb) / np.maximum(tot_p, 1.0)) <= rel_tol * np.maximum(mean_len, 1.0) + 1e-6
        rate = np.expm1(X[:, idx["log_pkt_rate"]])
        ok &= np.abs(rate - tot_p / np.maximum(dur, 1e-3)) <= rel_tol * np.maximum(rate, 1e-6) + 1e-6
    return ok
