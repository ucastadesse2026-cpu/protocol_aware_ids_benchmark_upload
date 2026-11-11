"""List the UDP flows in a classic pcap and guess which one is cFS command ingest (CI) and which is telemetry (TO).

    python tools/pcap_flows.py --pcap nos3.pcap [--top 15]

For every (destination IP:port) it tries to decode each datagram as a cFS packet (data/ccsds_parser.py, format 'cfs')
and reports: datagrams, parse success, TC/TM split, number of distinct APIDs, packets per second and mean length.
Heuristic suggestion: --cmd-port = the destination port whose parsed packets are (almost) all TC; --tlm-port = the
destination port whose parsed packets are (almost) all TM, with the most datagrams. CHECK the table yourself.
Pcapng must be converted first (editcap -F pcap in.pcapng out.pcap).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ccsds_parser import ParseError, TC, decode, set_format
from tools.pcap_to_stream import read_pcap, udp_payload


def analyse(pcap: str, fmt: str = "cfs", **overrides):
    set_format(fmt, **overrides)
    flows = defaultdict(lambda: dict(n=0, ok=0, tc=0, tm=0, apids=set(), bytes=0, t0=None, t1=None, srcs=set()))
    for ts, lt, frame in read_pcap(pcap):
        u = udp_payload(frame, lt)
        if u is None:
            continue
        sip, sp, dip, dp, payload = u
        f = flows[(dip, dp)]
        f["n"] += 1
        f["bytes"] += len(payload)
        f["srcs"].add(f"{sip}:{sp}")
        f["t0"] = ts if f["t0"] is None else f["t0"]
        f["t1"] = ts
        try:
            pkt, _ = decode(payload, 0)
        except ParseError:
            continue
        f["ok"] += 1
        f["tc" if pkt.ptype == TC else "tm"] += 1
        f["apids"].add(pkt.apid)
    return flows


def suggest(flows):
    cand_cmd = [(f["tc"], k) for k, f in flows.items() if f["ok"] >= 3 and f["tc"] >= 0.95 * f["ok"]]
    cand_tlm = [(f["tm"], k) for k, f in flows.items() if f["ok"] >= 3 and f["tm"] >= 0.95 * f["ok"]]
    cmd = max(cand_cmd)[1] if cand_cmd else None
    tlm = max(cand_tlm)[1] if cand_tlm else None
    return cmd, tlm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--format", default="cfs", choices=["cfs", "project"])
    a = ap.parse_args()
    flows = analyse(a.pcap, a.format)
    if not flows:
        sys.exit("No UDP/IPv4 datagrams found (wrong interface, pcapng instead of pcap, or no traffic).")
    print(f"{'dst ip:port':24s} {'dgrams':>8s} {'parsed':>8s} {'TC':>7s} {'TM':>7s} {'APIDs':>6s} {'pkt/s':>8s} {'mean len':>9s}  sources")
    for (dip, dp), f in sorted(flows.items(), key=lambda kv: -kv[1]["n"])[:a.top]:
        dur = max((f["t1"] or 0) - (f["t0"] or 0), 1e-6)
        print(f"{dip + ':' + str(dp):24s} {f['n']:8d} {f['ok']:8d} {f['tc']:7d} {f['tm']:7d} {len(f['apids']):6d} "
              f"{f['n'] / dur:8.1f} {f['bytes'] / f['n']:9.1f}  {', '.join(sorted(f['srcs'])[:2])}")
    cmd, tlm = suggest(flows)
    print()
    print(f"suggested --cmd-port: {cmd[1] if cmd else 'none found (no flow of parseable TC packets)'}"
          + (f"   (host {cmd[0]})" if cmd else ""))
    print(f"suggested --tlm-port: {tlm[1] if tlm else 'none found (no flow of parseable TM packets)'}"
          + (f"   (host {tlm[0]})" if tlm else ""))
    print("These are guesses from packet content. If nothing parses, the wire format differs from the cFS layout assumed here.")


if __name__ == "__main__":
    main()
