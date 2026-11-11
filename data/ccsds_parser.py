"""CCSDS Space Packet (CCSDS 133.0-B-2) encoder/decoder plus array-based packet windows.

Primary header (6 octets, big-endian):
    word0: version(3) | type(1) | secondary-header-flag(1) | APID(11)
    word1: sequence-flags(2) | sequence-count(14)
    word2: packet data length = (octets in packet data field) - 1

Two WIRE FORMATS are supported (select with ``set_format``; the default is "project"):
  "project"  the synthetic profile described below (used by the unit tests and the synthetic smoke data);
  "cfs"      NASA core Flight System as used by NOS3: TC = primary header + 1-octet function code + 1-octet
             XOR checksum (no timestamp, no CRC); TM = primary header + 6-octet time secondary header (no CRC).
             Arrival time comes from the capture, not from the header. VERIFY these against the cFE version you run
             (``tools/derive_profile.py`` auto-detects the checksum initial value from your capture).

PROJECT MISSION PROFILE (a modelling choice of this benchmark, NOT part of 133.0-B-2):
    * secondary header = 6 octets: uint32 seconds + uint16 sub-seconds (1/65536 s)
    * telecommand (TC) user data begins with a 2-octet opcode
    * the last 2 octets of the data field are a packet error control (PEC) field,
      CRC-16-CCITT (poly 0x1021, init 0xFFFF) over header + data field (excluding the PEC)
Real missions define these differently; adapt ``MissionProfile`` and this module accordingly.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

PRIMARY_HEADER_LEN = 6
SEC_HDR_LEN = 6
OPCODE_LEN = 2
PEC_LEN = 2
SEQ_MOD = 1 << 14
TM, TC = 0, 1
SUBSEC_RES = 65536


@dataclass(frozen=True)
class WireFormat:
    """Layout of everything after the 6-octet primary header."""

    name: str
    tm_sec_len: int            # TM secondary-header octets
    tc_sec_len: int            # TC secondary-header octets
    opcode_len: int            # opcode octets at the start of TC user data (0 if the opcode lives in the secondary header)
    pec_len: int               # trailing error-control octets
    ts_in_header: bool         # True: timestamps are read from secondary headers; False: supplied by the capture
    tc_sec_is_cmd_hdr: bool    # True: TC secondary header = function code + checksum (cFS)
    xor_init: int = 0xFF       # initial value of the cFS XOR checksum (verify with tools/derive_profile.py)


PROJECT = WireFormat("project", 6, 6, 2, 2, True, False)
CFS = WireFormat("cfs", 6, 2, 0, 0, False, True)
FORMATS = {"project": PROJECT, "cfs": CFS}
_FMT = PROJECT


def set_format(fmt="project", **overrides) -> WireFormat:
    """Select the active wire format (process-wide). ``overrides`` replace WireFormat fields, e.g. xor_init=0."""
    global _FMT
    base = FORMATS[fmt] if isinstance(fmt, str) else fmt
    _FMT = replace(base, **overrides) if overrides else base
    return _FMT


def get_format() -> WireFormat:
    return _FMT


def xor_checksum(raw_with_zero_checksum: bytes, init: int = 0xFF) -> int:
    """cFS-style command checksum: XOR of all octets (checksum octet zeroed) starting from ``init``."""
    x = init
    for b in raw_with_zero_checksum:
        x ^= b
    return x & 0xFF


class ParseError(ValueError):
    """Raised when raw bytes cannot be decoded as a packet of the project profile."""


class EncodeError(ValueError):
    """Raised when a packet object holds values that do not fit the bit fields."""


def _make_crc_table() -> List[int]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


_CRC_TABLE = _make_crc_table()


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE. ``crc16_ccitt(b'123456789') == 0x29B1``."""
    crc = init
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


@dataclass
class CcsdsPacket:
    """One Space Packet. ``pec=None`` means "compute a correct PEC on encode"."""

    ptype: int
    apid: int
    seq_count: int
    version: int = 0
    sec_hdr: bool = True
    seq_flags: int = 3
    timestamp: Optional[float] = None
    opcode: Optional[int] = None
    payload: bytes = b""
    pec: Optional[int] = None


def wire_length(sec_hdr: bool, ptype: int, payload_len: int) -> int:
    """Total octets on the wire for a packet of the ACTIVE wire format."""
    f = _FMT
    sec = (f.tc_sec_len if ptype == TC else f.tm_sec_len) if sec_hdr else 0
    return PRIMARY_HEADER_LEN + sec + (f.opcode_len if ptype == TC else 0) + payload_len + f.pec_len


def _split_time(ts: float) -> Tuple[int, int]:
    sec = int(np.floor(ts))
    sub = int(round((ts - sec) * SUBSEC_RES))
    if sub >= SUBSEC_RES:
        sec, sub = sec + 1, 0
    return sec, sub


def _time_bytes(ts, n: int) -> bytes:
    if ts is None or ts < 0 or ts >= 2 ** 32:
        raise EncodeError("timestamp missing or outside uint32 seconds range")
    sec, sub = _split_time(ts)
    return (struct.pack(">IH", sec, sub) + bytes(max(n - 6, 0)))[:n]


def _body_bytes(pkt: CcsdsPacket) -> bytes:
    f = _FMT
    body = b""
    if pkt.ptype == TC and f.tc_sec_is_cmd_hdr:
        if pkt.sec_hdr:
            if pkt.opcode is None or not 0 <= pkt.opcode <= 0xFF:
                raise EncodeError("cFS TC packet requires an 8-bit function code")
            body += bytes([pkt.opcode, 0])                       # checksum octet patched in encode()
        elif pkt.opcode is not None:
            raise EncodeError("opcode needs the secondary header")
        return body + bytes(pkt.payload)
    if pkt.sec_hdr:
        body += _time_bytes(pkt.timestamp, f.tc_sec_len if pkt.ptype == TC else f.tm_sec_len)
    if pkt.ptype == TC:
        if pkt.opcode is None or not 0 <= pkt.opcode <= 0xFFFF:
            raise EncodeError("TC packet requires a 16-bit opcode")
        body += struct.pack(">H", pkt.opcode)
    elif pkt.opcode is not None:
        raise EncodeError("TM packet must not carry an opcode")
    return body + bytes(pkt.payload)


def _header_bytes(pkt: CcsdsPacket, data_field_len: int) -> bytes:
    if not (0 <= pkt.version <= 7 and pkt.ptype in (0, 1) and 0 <= pkt.apid <= 2047
            and 0 <= pkt.seq_flags <= 3 and 0 <= pkt.seq_count < SEQ_MOD):
        raise EncodeError("header field does not fit its bit width")
    if not 1 <= data_field_len <= 65536:
        raise EncodeError("data field length outside 1..65536")
    w0 = (pkt.version << 13) | (pkt.ptype << 12) | (int(pkt.sec_hdr) << 11) | pkt.apid
    w1 = (pkt.seq_flags << 14) | pkt.seq_count
    return struct.pack(">HHH", w0, w1, data_field_len - 1)


def packet_crc(pkt: CcsdsPacket) -> Optional[int]:
    """The correct error-control value for ``pkt`` (ignores ``pkt.pec``): CRC-16 (project), the XOR checksum of a
    cFS TC packet, or None when the format carries no check on that packet."""
    f = _FMT
    body = _body_bytes(pkt)
    header = _header_bytes(pkt, len(body) + f.pec_len)
    if f.tc_sec_is_cmd_hdr:
        if pkt.ptype == TC and pkt.sec_hdr:
            return xor_checksum(header + body, f.xor_init)
        return None
    return crc16_ccitt(header + body)


def encode(pkt: CcsdsPacket) -> bytes:
    f = _FMT
    body = _body_bytes(pkt)
    header = _header_bytes(pkt, len(body) + f.pec_len)
    if f.tc_sec_is_cmd_hdr:
        raw = bytearray(header + body)
        if pkt.ptype == TC and pkt.sec_hdr:
            raw[PRIMARY_HEADER_LEN + 1] = (xor_checksum(bytes(raw), f.xor_init) if pkt.pec is None else pkt.pec) & 0xFF
        return bytes(raw)
    pec = crc16_ccitt(header + body) if pkt.pec is None else pkt.pec
    return header + body + struct.pack(">H", pec & 0xFFFF)


def decode(raw: bytes, offset: int = 0) -> Tuple[CcsdsPacket, int]:
    """Decode one packet at ``offset``; returns (packet, next_offset). ``packet.pec`` holds the *received* check
    value (CRC-16, or the cFS TC checksum octet; None when the format has none for that packet)."""
    f = _FMT
    if len(raw) - offset < PRIMARY_HEADER_LEN + max(f.pec_len, 1):
        raise ParseError("truncated header")
    w0, w1, dlen = struct.unpack_from(">HHH", raw, offset)
    total = PRIMARY_HEADER_LEN + dlen + 1
    if dlen + 1 < f.pec_len or len(raw) - offset < total:
        raise ParseError("data length field inconsistent with available bytes")
    end = offset + total
    body = raw[offset + PRIMARY_HEADER_LEN:end - f.pec_len]
    pec = struct.unpack_from(">H", raw, end - f.pec_len)[0] if f.pec_len else None
    ptype, sec_flag = (w0 >> 12) & 1, bool((w0 >> 11) & 1)
    ts, pos, opcode = None, 0, None
    if ptype == TC and f.tc_sec_is_cmd_hdr:
        if sec_flag:
            if len(body) < 2:
                raise ParseError("cFS command header truncated")
            opcode, pec, pos = body[0], body[1], 2
    else:
        if sec_flag:
            n = f.tc_sec_len if ptype == TC else f.tm_sec_len
            if len(body) < max(n, 6):
                raise ParseError("secondary header truncated")
            s, sub = struct.unpack_from(">IH", body, 0)
            ts, pos = s + sub / SUBSEC_RES, n
        if ptype == TC:
            if len(body) - pos < f.opcode_len:
                raise ParseError("TC user data shorter than opcode")
            opcode = struct.unpack_from(">H", body, pos)[0]
            pos += f.opcode_len
    pkt = CcsdsPacket(ptype=ptype, apid=w0 & 0x7FF, seq_count=w1 & 0x3FFF, version=(w0 >> 13) & 7,
                      sec_hdr=sec_flag, seq_flags=(w1 >> 14) & 3, timestamp=ts, opcode=opcode,
                      payload=bytes(body[pos:]), pec=pec)
    return pkt, end


def read_stream(raw: bytes) -> List[CcsdsPacket]:
    out, off = [], 0
    while off < len(raw):
        pkt, off = decode(raw, off)
        out.append(pkt)
    return out


def write_stream(packets: List[CcsdsPacket]) -> bytes:
    return b"".join(encode(p) for p in packets)


# --------------------------------------------------------------------------------------
# Array representation used by feature extraction and attacks
# --------------------------------------------------------------------------------------
@dataclass
class Window:
    """A fixed-length run of packets stored as arrays (payload bytes are not retained).

    ``base_plen`` is the immutable parameter part of the user data, ``pad`` the mutable trailing
    padding. Payload content is irrelevant to the benchmark features, so reconstructing packets
    uses zero bytes.
    """

    ptype: np.ndarray
    apid: np.ndarray
    seqc: np.ndarray
    ts: np.ndarray
    base_plen: np.ndarray
    pad: np.ndarray
    opcode: np.ndarray            # -1 for TM
    sec: np.ndarray               # bool
    label: int = 0
    attack_type: str = "benign"

    def __len__(self) -> int:
        return int(len(self.ptype))

    @property
    def plen(self) -> np.ndarray:
        return self.base_plen + self.pad

    def copy(self) -> "Window":
        return Window(self.ptype.copy(), self.apid.copy(), self.seqc.copy(), self.ts.copy(),
                      self.base_plen.copy(), self.pad.copy(), self.opcode.copy(), self.sec.copy(),
                      self.label, self.attack_type)

    def wire_lengths(self) -> np.ndarray:
        f = _FMT
        tc = self.ptype == TC
        sec = np.where(tc, f.tc_sec_len, f.tm_sec_len) * self.sec.astype(np.int64)
        return (PRIMARY_HEADER_LEN + sec + f.opcode_len * tc + self.plen + f.pec_len).astype(np.int64)


def packets_to_window(packets: List[CcsdsPacket], label: int = 0, attack_type: str = "benign") -> Window:
    return Window(
        ptype=np.array([p.ptype for p in packets], dtype=np.int64),
        apid=np.array([p.apid for p in packets], dtype=np.int64),
        seqc=np.array([p.seq_count for p in packets], dtype=np.int64),
        ts=np.array([p.timestamp if p.timestamp is not None else 0.0 for p in packets], dtype=np.float64),
        base_plen=np.array([len(p.payload) for p in packets], dtype=np.int64),
        pad=np.zeros(len(packets), dtype=np.int64),
        opcode=np.array([p.opcode if p.opcode is not None else -1 for p in packets], dtype=np.int64),
        sec=np.array([p.sec_hdr for p in packets], dtype=bool),
        label=label, attack_type=attack_type)


def window_to_packets(w: Window) -> List[CcsdsPacket]:
    pk = []
    for i in range(len(w)):
        pk.append(CcsdsPacket(
            ptype=int(w.ptype[i]), apid=int(w.apid[i]), seq_count=int(w.seqc[i]), sec_hdr=bool(w.sec[i]),
            timestamp=float(w.ts[i]) if w.sec[i] else None,
            opcode=int(w.opcode[i]) if w.ptype[i] == TC else None,
            payload=bytes(int(w.plen[i]))))
    return pk
