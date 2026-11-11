"""Tests for the cFS wire format, pcap conversion, profile derivation, traffic generator and an end-to-end
conformant attack on converted cFS-format data. All data are SYNTHETIC cFS-format packets built by the tests."""
import csv
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from attacks.ccsds_conformant import CcsdsAttackConfig, attack_windows
from attacks.validator import MissionProfile, validate_bytes, validate_packet, validate_window
from data import feature_schema as fs
from data.ccsds_parser import (TC, TM, CcsdsPacket, ParseError, decode, encode, packet_crc, read_stream, set_format,
                               wire_length, xor_checksum)
from data.preprocess import load_satellite, load_terrestrial, windows_from_stream
from models.numpy_mlp import NumpyMLP
from tools import nos3_traffic as gen
from tools.pcap_to_stream import convert, read_pcap


def tc(apid=0x1A0, seq=0, fc=3, payload=b"", ts=None):
    return CcsdsPacket(TC, apid, seq, opcode=fc, payload=payload, timestamp=ts)


def tm(apid=0x0A0, seq=0, payload=b"x" * 20, ts=100.0):
    return CcsdsPacket(TM, apid, seq, timestamp=ts, payload=payload)


def codes(v):
    return {x.code for x in v}


def frame_eth(payload, dport, sport=40000):
    udp = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 1, 0, 64, 17, 0, bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2])) + udp
    return b"\x00" * 12 + b"\x08\x00" + ip, ip


def pcap_bytes(frames, linktype=1, end="<", nano=False):
    magic = {("<", False): b"\xd4\xc3\xb2\xa1", (">", False): b"\xa1\xb2\xc3\xd4", ("<", True): b"\x4d\x3c\xb2\xa1"}[(end, nano)]
    out = magic + struct.pack(end + "HHIIII", 2, 4, 0, 0, 65535, linktype)
    for ts, data in frames:
        sec = int(ts); frac = int(round((ts - sec) * (1e9 if nano else 1e6)))
        out += struct.pack(end + "IIII", sec, frac, len(data), len(data)) + data
    return out


class CfsCase(unittest.TestCase):
    def setUp(self):
        set_format("cfs")

    def tearDown(self):
        set_format("project")


class TestCfsFormat(CfsCase):
    def test_tc_layout_and_checksum(self):
        raw = encode(tc(0x1A0, 5, 3, b"\x01\x02"))
        self.assertEqual(len(raw), 6 + 2 + 2); self.assertEqual(wire_length(True, TC, 2), len(raw))
        self.assertEqual(struct.unpack(">H", raw[:2])[0], 0x1800 | 0x1A0)       # 0x18xx command message id
        self.assertEqual(raw[6], 3)
        self.assertEqual(xor_checksum(raw, 0xFF), 0)                              # whole packet XORs to 0 with init 0xFF
        p, end = decode(raw)
        self.assertEqual((p.opcode, p.payload, p.seq_count, p.pec), (3, b"\x01\x02", 5, raw[7]))
        self.assertEqual(packet_crc(p), p.pec)

    def test_tm_layout_no_check(self):
        raw = encode(tm(payload=b"y" * 10, ts=12.5))
        self.assertEqual(len(raw), 6 + 6 + 10); self.assertEqual(struct.unpack(">H", raw[:2])[0] & 0xFF00, 0x0800)
        p, _ = decode(raw)
        self.assertAlmostEqual(p.timestamp, 12.5, places=4); self.assertIsNone(p.pec); self.assertIsNone(packet_crc(p))

    def test_xor_init_override(self):
        set_format("cfs", xor_init=0x00)
        raw = encode(tc(payload=b"ab"))
        self.assertEqual(xor_checksum(raw, 0x00), 0)
        set_format("cfs")
        p, _ = decode(raw)
        self.assertNotEqual(p.pec, packet_crc(p))                                   # wrong init is detected

    def test_validator_on_cfs(self):
        prof = MissionProfile.from_dict({"tm_apid_list": [0xA0], "tc_apid_list": [0x1A0], "legal_opcodes": {0x1A0: [3, 4]},
                                         "max_user_data_tm": 64, "max_user_data_tc": 16})
        raw = encode(tc(0x1A0, 0, 3, b"abc"))
        self.assertEqual(validate_bytes(raw, prof), [])
        bad = bytearray(raw); bad[8] ^= 0x55                                        # corrupt payload -> checksum fails
        self.assertIn("C5", codes(validate_bytes(bytes(bad), prof)))
        self.assertIn("C6", codes(validate_packet(tc(0x1A0, 0, 9), prof)))          # illegal function code
        self.assertIn("C2", codes(validate_packet(tc(0x1A1, 0, 3), prof)))
        self.assertIn("C4", codes(validate_packet(tc(0x1A0, 0, 3, b"x" * 40), prof)))
        self.assertEqual(validate_bytes(encode(tm()), prof), [])

    def test_truncated_command_header(self):
        raw = bytearray(encode(tc())); raw[4:6] = struct.pack(">H", 0)             # data length 1 -> only one octet
        with self.assertRaises(ParseError):
            decode(bytes(raw[:7]))

    def test_window_lengths(self):
        from data.ccsds_parser import packets_to_window
        w = packets_to_window([tc(payload=b"ab", ts=1.0), tm(ts=2.0)])
        self.assertEqual(w.wire_lengths().tolist(), [10, 6 + 6 + 20])


def make_capture(tmp, linktype=1, end="<", nano=False):
    """Synthetic NOS3-like capture: TM at ~2 Hz, benign TC every ~3 s, a 5-s TC flood; plus noise and a duplicate."""
    CMD, TLM = 1234, 5678
    frames, t, seqs = [], 1_700_000_000.0, {}
    flood_start = flood_end = None
    def add(pkt, dport, ts, dup=False):
        eth, ip = frame_eth(encode(pkt), dport)
        data = eth if linktype == 1 else (b"\x00" * 14 + b"\x08\x00" + ip if linktype == 113 else ip)
        frames.append((ts, data))
        if dup:
            frames.append((ts + 0.0001, data))
    for i in range(400):
        t += 0.5
        seqs[0xA0] = seqs.get(0xA0, 0) + 1
        add(tm(0xA0, seqs[0xA0], b"z" * 20), TLM, t, dup=(i % 50 == 0))
        if i % 6 == 0:
            seqs[0x1A0] = seqs.get(0x1A0, 0) + 1
            add(tc(0x1A0, seqs[0x1A0], 3, b""), CMD, t + 0.01)
        if i == 200:
            flood_start = t + 0.02
            for k in range(100):
                seqs[0x1A0] += 1
                add(tc(0x1A0, seqs[0x1A0], 4, b""), CMD, t + 0.02 + k * 0.01)
            flood_end = t + 0.02 + 100 * 0.01
        if i == 10:
            frames.append((t + 0.001, frame_eth(b"not ccsds noise", 9999)[0] if linktype == 1 else b""))
    frames.sort(key=lambda x: x[0])
    pcap = os.path.join(tmp, "cap.pcap")
    with open(pcap, "wb") as fh:
        fh.write(pcap_bytes([(a, b) for a, b in frames if b], linktype, end, nano))
    iv = os.path.join(tmp, "iv.csv")
    with open(iv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["start", "end", "attack_type"]); w.writerow([flood_start - 0.001, flood_end, "flood"])
    return pcap, iv, CMD, TLM


class TestPcapConversion(CfsCase):
    def _run(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            pcap, iv, CMD, TLM = make_capture(d, **kw)
            out = os.path.join(d, "o")
            st = convert(pcap, CMD, TLM, out, iv)
            with open(out + ".labels.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            with open(out + ".bin", "rb") as fh:
                pk = read_stream(fh.read())
            return st, rows, pk

    def test_ethernet_le(self):
        st, rows, pk = self._run()
        self.assertEqual(len(rows), len(pk)); self.assertEqual(st["parse_errors"], 0)
        self.assertEqual(sum(p.ptype == TC for p in pk), 400 // 6 + 1 + 100)
        self.assertEqual(sum(p.ptype == TM for p in pk), 400)                       # duplicates removed
        self.assertEqual(st["duplicates"], 8)
        att = [r for r in rows if r["label"] == "1"]
        self.assertEqual(len(att), 100); self.assertTrue(all(r["direction"] == "TC" and r["attack_type"] == "flood" for r in att))
        self.assertTrue(all(r["label"] == "0" for r in rows if r["direction"] == "TM"))
        times = [float(r["arrival_time"]) for r in rows]
        self.assertEqual(times, sorted(times))

    def test_other_linktypes_and_formats(self):
        for kw in (dict(linktype=113), dict(linktype=101), dict(end=">"), dict(nano=True)):
            st, rows, pk = self._run(**kw)
            self.assertEqual(len(pk), 400 + 400 // 6 + 1 + 100, msg=str(kw))

    def test_pcapng_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.pcapng"); open(p, "wb").write(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40)
            with self.assertRaises(ValueError):
                list(read_pcap(p))


class TestCfsLoaderAndAttack(CfsCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        set_format("cfs")
        pcap, iv, CMD, TLM = make_capture(cls.tmp.name)
        cls.out = os.path.join(cls.tmp.name, "o")
        convert(pcap, CMD, TLM, cls.out, iv)
        set_format("project")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_arrival_time_required_and_label_rules(self):
        with self.assertRaises(ValueError):                                         # labels without arrival_time
            with tempfile.TemporaryDirectory() as d:
                import shutil
                shutil.copy(self.out + ".bin", d + "/a.bin")
                rows = list(csv.DictReader(open(self.out + ".labels.csv")))
                with open(d + "/a.csv", "w", newline="") as fh:
                    w = csv.writer(fh); w.writerow(["packet_index", "label", "attack_type"])
                    for r in rows: w.writerow([r["packet_index"], r["label"], r["attack_type"]])
                windows_from_stream(d + "/a.bin", d + "/a.csv", 32, 32)
        anyw = windows_from_stream(self.out + ".bin", self.out + ".labels.csv", 32, 32, "any")
        maj = windows_from_stream(self.out + ".bin", self.out + ".labels.csv", 32, 32, "majority")
        self.assertGreater(sum(w.label for w in anyw), 0); self.assertGreaterEqual(sum(w.label for w in anyw), sum(w.label for w in maj))
        self.assertTrue(all(w.sec.all() and w.ts[-1] >= w.ts[0] for w in anyw))

    def test_derive_profile_cli(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "derive_profile.py"), "--stream", self.out],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("init 0xFF -> 100.0%", r.stdout); self.assertIn("tc_apid_list: [416]", r.stdout)
        self.assertIn("legal_opcodes:", r.stdout); self.assertIn("already valid under this profile", r.stdout)

    def test_end_to_end_conformant_attack_on_cfs_windows(self):
        cfg = {"features": {"view": "extended", "window_size": 32},
               "data": {"terrestrial": {"unsw_nb15": {"source": "synthetic", "n_rows": 3000}},
                        "satellite": {"source": "real", "bin_path": self.out + ".bin", "labels_path": self.out + ".labels.csv",
                                      "wire_format": "cfs", "label_rule": "any", "stride": 32}}}
        terr = load_terrestrial(cfg, "unsw_nb15", 1)
        sat = load_satellite(cfg, terr, 1)
        prof = MissionProfile.from_dict({"tm_apid_list": [0xA0], "tc_apid_list": [0x1A0], "legal_opcodes": {0x1A0: [3, 4]},
                                         "max_user_data_tm": 200, "max_user_data_tc": 100})
        allw = sat["w_train"] + sat["w_val"] + sat["w_test"]
        self.assertTrue(all(not validate_window(w, prof) for w in allw))           # synthetic capture is valid
        mlp = NumpyMLP(8, hidden=(16,), epochs=10, seed=0).fit(terr["X_train"], terr["y_train"])
        att = [w for w in allw if w.label == 1][:5]
        self.assertTrue(att)
        adv, st = attack_windows("pgd_ccsds", att, mlp, sat["pipeline"], prof, CcsdsAttackConfig(steps=5))
        self.assertTrue(all(s["feasible"] for s in st))
        for w0, w1 in zip(att, adv):
            self.assertEqual(validate_window(w1, prof, original=w0), [])
        set_format("cfs")                                                           # wire lengths use the cFS layout
        self.assertEqual(int(att[0].wire_lengths()[att[0].ptype == TC][0]), 6 + 2 + int(att[0].plen[att[0].ptype == TC][0]))


class TestGenerator(CfsCase):
    def _catalog(self, d):
        p = os.path.join(d, "cat.json")
        json.dump({"commands": [
            {"name": "noop", "apid": 0x1A0, "fc": 0, "payload_hex": "", "klass": "benign"},
            {"name": "hk", "apid": 0x1A1, "fc": 1, "payload_hex": "00", "klass": "benign"},
            {"name": "sens", "apid": 0x1A2, "fc": 7, "payload_hex": "0102", "klass": "sensitive"}]}, open(p, "w"))
        return p

    def test_catalog_validation(self):
        with self.assertRaises(ValueError):
            gen.load_catalog(os.path.join(ROOT, "tools", "command_catalog.example.json"))

    def test_destination_guard(self):
        gen.check_destination("127.0.0.1"); gen.check_destination("192.168.1.5")
        with self.assertRaises(SystemExit):
            gen.check_destination("8.8.8.8")

    def test_cli_requires_acknowledgement(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "nos3_traffic.py"), "--port", "9", "--catalog",
                                self._catalog(d), "--mode", "benign"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0); self.assertIn("i-own-this-simulator", r.stderr)

    def test_session_sends_valid_cfs_packets_and_logs_intervals(self):
        with tempfile.TemporaryDirectory() as d:
            srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); srv.bind(("127.0.0.1", 0)); srv.settimeout(0.2)
            port = srv.getsockname()[1]
            iv = os.path.join(d, "iv.csv")
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "nos3_traffic.py"), "--port", str(port), "--catalog",
                                self._catalog(d), "--mode", "session", "--cycles", "1", "--benign-s", "0.15", "--attack-s", "0.2",
                                "--rates", "benign=50,flood=200,injection=50,recon=100", "--intervals-out", iv,
                                "--i-own-this-simulator"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            got = []
            try:
                while True:
                    got.append(srv.recv(2048))
            except socket.timeout:
                pass
            self.assertGreater(len(got), 30)
            prof = MissionProfile.from_dict({"tm_apid_list": [1], "tc_apid_list": [0x1A0, 0x1A1, 0x1A2],
                                             "legal_opcodes": {0x1A0: [0], 0x1A1: [1], 0x1A2: [7]}, "max_user_data_tc": 64})
            self.assertTrue(all(validate_bytes(g, prof) == [] for g in got))
            rows = list(csv.DictReader(open(iv)))
            self.assertEqual([x["attack_type"] for x in rows], ["flood", "injection", "recon"])
            self.assertTrue(all(float(x["end"]) > float(x["start"]) for x in rows))


if __name__ == "__main__":
    unittest.main()


class TestDiscoveryTools(CfsCase):
    def test_flows_and_catalog_from_capture(self):
        from tools import catalog_from_stream as cat
        from tools import pcap_flows as fl
        with tempfile.TemporaryDirectory() as d:
            pcap, iv, CMD, TLM = make_capture(d)
            flows = fl.analyse(pcap)
            c, t = fl.suggest(flows)
            self.assertEqual(c[1], CMD); self.assertEqual(t[1], TLM)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "pcap_flows.py"), "--pcap", pcap],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertIn(f"suggested --cmd-port: {CMD}", r.stdout)
            out = os.path.join(d, "o")
            convert(pcap, CMD, TLM, out, iv)
            cmds = cat.build(out)
            self.assertEqual({(x["apid"], x["fc"]) for x in cmds}, {(0x1A0, 3), (0x1A0, 4)})
            self.assertTrue(all(x["klass"] in ("benign", "sensitive") for x in cmds))
            p = os.path.join(d, "cat.json")
            subprocess.run([sys.executable, os.path.join(ROOT, "tools", "catalog_from_stream.py"), "--stream", out, "--out", p],
                           check=True, capture_output=True)
            self.assertEqual(len(json.load(open(p))["commands"]), 2)
