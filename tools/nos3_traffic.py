"""Scripted cFS command traffic for YOUR OWN local NOS3 simulator (benign operations + three scripted attack classes).

    python tools/nos3_traffic.py --host 127.0.0.1 --port <CI port> --catalog tools/my_catalog.json \
        --mode session --cycles 20 --benign-s 60 --attack-s 20 --intervals-out attacks.csv --i-own-this-simulator

Modes
  benign     low-rate random 'benign'-class commands from the catalog
  flood      high-rate 'benign'-class commands (volume attack)
  injection  'sensitive'-class commands (legal but unusual) with optional extra payload bytes
  recon      sweep through every catalog command / APID once per round
  session    cycles of: benign period, then each attack class in turn separated by benign periods

Every attack period is appended to --intervals-out (start,end,attack_type; epoch seconds) -> use it with
tools/pcap_to_stream.py for ground-truth labels. Capture the traffic separately (tcpdump) on the same machine.

SAFETY: refuses any destination that is not loopback / private / link-local, and refuses to run without
--i-own-this-simulator. The catalog decides what is sent: choose commands that are harmless in the simulator
(no resets/deletes/power-offs unless you are prepared to restart NOS3). Never point this at real hardware.
"""
from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import random
import socket
import sys
import time
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ccsds_parser import TC, CcsdsPacket, encode, set_format

ATTACKS = ("flood", "injection", "recon")


def load_catalog(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        cmds = json.load(fh)["commands"]
    for c in cmds:
        for k in ("apid", "fc"):
            if not isinstance(c.get(k), int):
                raise ValueError(f"catalog entry {c.get('name')!r}: '{k}' must be an integer (replace the TODO placeholders "
                                 "with real values from your NOS3 command definitions)")
        if c.get("klass") not in ("benign", "sensitive"):
            raise ValueError(f"catalog entry {c.get('name')!r}: klass must be 'benign' or 'sensitive'")
        bytes.fromhex(c.get("payload_hex", ""))
    if not any(c["klass"] == "benign" for c in cmds) or not any(c["klass"] == "sensitive" for c in cmds):
        raise ValueError("catalog needs at least one 'benign' and one 'sensitive' command")
    return cmds


def check_destination(host: str) -> None:
    ip = ipaddress.ip_address(socket.gethostbyname(host))
    if not (ip.is_loopback or ip.is_private or ip.is_link_local):
        raise SystemExit(f"refusing to send to {ip}: only loopback/private/link-local simulator hosts are allowed")


class Sender:
    def __init__(self, sock, dest, seed: int = 0, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.sock, self.dest, self.rng, self.clock, self.sleep = sock, dest, random.Random(seed), clock, sleep
        self.seq: Dict[int, int] = {}
        set_format("cfs")

    def send(self, cmd: Dict, extra: int = 0) -> None:
        apid = cmd["apid"]
        n = self.seq.get(apid, 0)
        self.seq[apid] = (n + 1) % (1 << 14)
        payload = bytes.fromhex(cmd.get("payload_hex", "")) + bytes(self.rng.randrange(256) for _ in range(extra))
        self.sock.sendto(encode(CcsdsPacket(TC, apid, n, sec_hdr=True, seq_flags=3, opcode=cmd["fc"], payload=payload)), self.dest)

    def run(self, mode: str, seconds: float, catalog: List[Dict], rate: float, extra: int = 0) -> int:
        ben = [c for c in catalog if c["klass"] == "benign"]
        sens = [c for c in catalog if c["klass"] == "sensitive"]
        end, sent, i = self.clock() + seconds, 0, 0
        while self.clock() < end:
            if mode == "benign":
                self.send(self.rng.choice(ben)); gap = self.rng.expovariate(rate)
            elif mode == "flood":
                self.send(self.rng.choice(ben)); gap = 1.0 / rate
            elif mode == "injection":
                self.send(self.rng.choice(sens), extra); gap = self.rng.expovariate(rate)
            elif mode == "recon":
                self.send(catalog[i % len(catalog)]); i += 1; gap = 1.0 / rate
            else:
                raise ValueError(mode)
            sent += 1
            self.sleep(gap)
        return sent


def log_interval(path: str, start: float, end: float, kind: str) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["start", "end", "attack_type"])
        wr.writerow([f"{start:.6f}", f"{end:.6f}", kind])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True, help="UDP port of the cFS command-ingest app (CI)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--mode", choices=("benign", "flood", "injection", "recon", "session"), default="session")
    ap.add_argument("--seconds", type=float, default=60, help="duration for a single mode")
    ap.add_argument("--rates", default="benign=1,flood=100,injection=2,recon=5", help="packets/s per mode")
    ap.add_argument("--extra-payload-bytes", type=int, default=0, help="random bytes appended to injection commands")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--benign-s", type=float, default=60)
    ap.add_argument("--attack-s", type=float, default=20)
    ap.add_argument("--intervals-out", default="attacks.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i-own-this-simulator", action="store_true")
    a = ap.parse_args()
    if not a.i_own_this_simulator:
        raise SystemExit("add --i-own-this-simulator to confirm the target is your own local simulator")
    check_destination(a.host)
    cat = load_catalog(a.catalog)
    rates = {k: float(v) for k, v in (x.split("=") for x in a.rates.split(","))}
    snd = Sender(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (a.host, a.port), a.seed)

    def segment(mode: str, secs: float) -> None:
        t0 = time.time()
        n = snd.run(mode, secs, cat, rates[mode], a.extra_payload_bytes if mode == "injection" else 0)
        t1 = time.time()
        if mode in ATTACKS:
            log_interval(a.intervals_out, t0, t1, mode)
        print(f"{mode:9s} {secs:6.1f}s  sent {n}")

    if a.mode == "session":
        for c in range(a.cycles):
            for atk in ATTACKS:
                segment("benign", a.benign_s)
                segment(atk, a.attack_s)
    else:
        segment(a.mode, a.seconds)


if __name__ == "__main__":
    main()
