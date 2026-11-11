"""Convert a classic-pcap capture of NOS3/cFS UDP traffic into the benchmark's packet stream + labels.

    python tools/pcap_to_stream.py --pcap nos3.pcap --cmd-port <CI port> --tlm-port <ground TLM port> \
        --intervals attacks.csv --out data/raw/nos3/nos3

Writes <out>.bin (concatenated CCSDS packets) and <out>.labels.csv with columns
    packet_index,label,attack_type,arrival_time,direction,src_port,dst_port
Direction is decided by the UDP DESTINATION port: dst == --cmd-port -> TC (ground->cFS command ingest),
dst == --tlm-port -> TM (cFS telemetry output -> ground). Find both ports yourself (docs/NOS3_GUIDE.md).

Labels: a TC packet whose arrival time falls inside an interval of --intervals (CSV: start,end,attack_type; epoch
seconds, written by tools/nos3_traffic.py) is an attack packet; TM packets and all other TC packets are benign.

Input must be classic pcap (tcpdump -w). For pcapng (Wireshark default) convert first:
    editcap -F pcap in.pcapng out.pcap
Supported link types: Ethernet (1), raw IP (101), BSD loopback (0), Linux cooked SLL (113). IPv4/UDP only.
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ccsds_parser import ParseError, TC, decode, set_format


def read_pcap(path: str) -> Iterator[Tuple[float, int, bytes]]:
    """Yield (epoch_seconds, linktype, frame) from a classic pcap file (micro- or nanosecond, either endianness)."""
    with open(path, "rb") as fh:
        gh = fh.read(24)
        if len(gh) < 24:
            raise ValueError("not a pcap file (too short)")
        magic = gh[:4]
        if magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            end, nano = ">", magic == b"\xa1\xb2\x3c\x4d"
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            end, nano = "<", magic == b"\x4d\x3c\xb2\xa1"
        elif magic == b"\x0a\x0d\x0d\x0a":
            raise ValueError("this is pcapng; convert with: editcap -F pcap in.pcapng out.pcap")
        else:
            raise ValueError("unrecognised capture file magic")
        linktype = struct.unpack(end + "I", gh[20:24])[0]
        while True:
            rh = fh.read(16)
            if len(rh) < 16:
                return
            sec, frac, incl, _orig = struct.unpack(end + "IIII", rh)
            data = fh.read(incl)
            if len(data) < incl:
                return
            yield sec + frac / (1e9 if nano else 1e6), linktype, data


def udp_payload(frame: bytes, linktype: int) -> Optional[Tuple[str, int, str, int, bytes]]:
    """Return (src_ip, src_port, dst_ip, dst_port, payload) for an unfragmented IPv4/UDP frame, else None."""
    if linktype == 1:                                  # Ethernet (+ optional single VLAN tag)
        if len(frame) < 14:
            return None
        et, off = struct.unpack(">H", frame[12:14])[0], 14
        if et == 0x8100 and len(frame) >= 18:
            et, off = struct.unpack(">H", frame[16:18])[0], 18
        if et != 0x0800:
            return None
    elif linktype == 113:                              # Linux cooked capture
        if len(frame) < 16 or struct.unpack(">H", frame[14:16])[0] != 0x0800:
            return None
        off = 16
    elif linktype == 0:                                # BSD loopback
        off = 4
    elif linktype == 101:                              # raw IP
        off = 0
    else:
        return None
    ip = frame[off:]
    if len(ip) < 20 or (ip[0] >> 4) != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    flags_frag = struct.unpack(">H", ip[6:8])[0]
    if ip[9] != 17 or (flags_frag & 0x3FFF) != 0 or len(ip) < ihl + 8:
        return None
    sp, dp, ulen = struct.unpack(">HHH", ip[ihl:ihl + 6])
    payload = ip[ihl + 8:ihl + ulen]
    ipstr = lambda b: ".".join(str(x) for x in b)
    return ipstr(ip[12:16]), sp, ipstr(ip[16:20]), dp, payload


def load_intervals(path: Optional[str]) -> List[Tuple[float, float, str]]:
    if not path:
        return []
    with open(path, newline="") as fh:
        return [(float(r["start"]), float(r["end"]), r["attack_type"]) for r in csv.DictReader(fh)]


def convert(pcap: str, cmd_port: int, tlm_port: int, out: str, intervals_csv: Optional[str] = None,
            dedupe_ms: float = 1.0, fmt: str = "cfs", **fmt_overrides) -> dict:
    set_format(fmt, **fmt_overrides)
    intervals = load_intervals(intervals_csv)
    stats = dict(frames=0, udp_selected=0, packets=0, parse_errors=0, direction_mismatch=0, duplicates=0, attack_packets=0)
    seen = {}
    rows = []
    for ts, lt, frame in read_pcap(pcap):
        stats["frames"] += 1
        u = udp_payload(frame, lt)
        if u is None:
            continue
        _sip, sp, _dip, dp, payload = u
        if dp == cmd_port:
            direction = "TC"
        elif dp == tlm_port:
            direction = "TM"
        else:
            continue
        stats["udp_selected"] += 1
        key = (payload, sp, dp)                              # the same datagram seen on several interfaces
        if key in seen and (ts - seen[key]) * 1e3 <= dedupe_ms:
            stats["duplicates"] += 1
            continue
        seen[key] = ts
        if len(seen) > 100000:
            seen.clear()
        off = 0
        while off < len(payload):
            start = off
            try:
                pkt, off = decode(payload, off)
            except ParseError:
                stats["parse_errors"] += 1
                break
            if (pkt.ptype == TC) != (direction == "TC"):
                stats["direction_mismatch"] += 1
            label, atype = 0, "benign"
            if direction == "TC":
                for a_, b_, t_ in intervals:
                    if a_ <= ts <= b_:
                        label, atype = 1, t_
                        break
            stats["attack_packets"] += label
            rows.append((label, atype, ts, direction, sp, dp, payload[start:off]))     # original captured bytes
    with open(out + ".bin", "wb") as fb, open(out + ".labels.csv", "w", newline="") as fl:
        wr = csv.writer(fl)
        wr.writerow(["packet_index", "label", "attack_type", "arrival_time", "direction", "src_port", "dst_port"])
        for i, (label, atype, ts, direction, sp, dp, raw) in enumerate(rows):
            fb.write(raw)
            wr.writerow([i, label, atype, f"{ts:.6f}", direction, sp, dp])
    stats["packets"] = len(rows)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--cmd-port", type=int, required=True, help="UDP destination port of commands (cFS CI)")
    ap.add_argument("--tlm-port", type=int, required=True, help="UDP destination port of telemetry (ground side)")
    ap.add_argument("--out", required=True, help="output prefix, e.g. data/raw/nos3/nos3")
    ap.add_argument("--intervals", help="attack intervals CSV (start,end,attack_type; epoch seconds)")
    ap.add_argument("--format", default="cfs", choices=["cfs", "project"])
    ap.add_argument("--xor-init", type=lambda x: int(x, 0), help="cFS checksum initial value (default 0xFF)")
    ap.add_argument("--dedupe-ms", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ov = {"xor_init": a.xor_init} if a.xor_init is not None else {}
    st = convert(a.pcap, a.cmd_port, a.tlm_port, a.out, a.intervals, a.dedupe_ms, a.format, **ov)
    print(st)
    if st["packets"] == 0:
        sys.exit("No packets extracted: check the ports (they are destination ports) and that the capture holds UDP/IPv4.")
    if st["parse_errors"] > 0.05 * max(st["udp_selected"], 1):
        print("WARNING: >5% of selected datagrams did not parse; the wire format may not match (see docs/NOS3_GUIDE.md).")


if __name__ == "__main__":
    main()
