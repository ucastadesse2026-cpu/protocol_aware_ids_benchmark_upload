"""Unit tests for the CCSDS parser and the C1-C8 validator."""
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.validator import (MissionProfile, check_functionality, feature_space_realizable, is_valid, validate_bytes,
                               validate_packet, validate_stream, validate_window)
from data.ccsds_parser import (SEQ_MOD, TC, TM, CcsdsPacket, EncodeError, ParseError, crc16_ccitt, decode, encode,
                               packet_crc, read_stream, window_to_packets, write_stream)
from data.feature_schema import window_to_fc
from data.synthetic_ccsds import make_window

PROF = MissionProfile()


def tm(seq=0, ts=10.0, apid=0x101, payload=b"x" * 10):
    return CcsdsPacket(TM, apid, seq, timestamp=ts, payload=payload)


def tc(seq=0, ts=10.0, apid=0x201, opcode=3, payload=b"ab"):
    return CcsdsPacket(TC, apid, seq, timestamp=ts, opcode=opcode, payload=payload)


def codes(v):
    return {x.code for x in v}


class TestParser(unittest.TestCase):
    def test_crc_check_value(self):
        self.assertEqual(crc16_ccitt(b"123456789"), 0x29B1)

    def test_roundtrip(self):
        for p in (tm(5, 12.25), tc(7, 99.5, opcode=31, payload=b"\x01\x02\x03")):
            q, end = decode(encode(p))
            self.assertEqual(end, len(encode(p)))
            self.assertEqual((q.ptype, q.apid, q.seq_count, q.opcode, q.payload), (p.ptype, p.apid, p.seq_count, p.opcode, p.payload))
            self.assertAlmostEqual(q.timestamp, p.timestamp, places=4)
            self.assertEqual(q.pec, packet_crc(p))

    def test_stream_roundtrip(self):
        pk = [tm(0, 1.0), tc(0, 1.5), tm(1, 2.0)]
        out = read_stream(write_stream(pk))
        self.assertEqual([p.apid for p in out], [p.apid for p in pk])

    def test_truncated_raises(self):
        with self.assertRaises(ParseError):
            decode(encode(tm())[:-3])

    def test_unrepresentable_fields_raise(self):
        with self.assertRaises(EncodeError):
            encode(CcsdsPacket(TM, 5000, 0, timestamp=1.0))
        with self.assertRaises(EncodeError):
            encode(CcsdsPacket(TC, 0x201, 0, timestamp=1.0, opcode=None))


class TestConstraints(unittest.TestCase):
    def test_valid_packet(self):
        self.assertTrue(is_valid(validate_packet(tm(), PROF)))
        self.assertTrue(is_valid(validate_packet(tc(), PROF)))

    def test_c1_version_and_header_flags(self):
        p = tm(); p.version = 1
        self.assertIn("C1", codes(validate_packet(p, PROF)))
        p = tm(); p.sec_hdr = False
        self.assertIn("C1", codes(validate_packet(p, PROF)))

    def test_c2_apid(self):
        self.assertIn("C2", codes(validate_packet(tm(apid=0x300), PROF)))
        self.assertIn("C2", codes(validate_packet(tc(apid=0x101), PROF)))   # TM APID used for TC

    def test_c3_sequence(self):
        self.assertTrue(is_valid(validate_stream([tm(5, 1.0), tm(6, 2.0)], PROF)))
        self.assertTrue(is_valid(validate_stream([tm(SEQ_MOD - 1, 1.0), tm(0, 2.0)], PROF)))   # wrap-around
        self.assertIn("C3", codes(validate_stream([tm(5, 1.0), tm(5, 2.0)], PROF)))            # duplicate
        self.assertIn("C3", codes(validate_stream([tm(5, 1.0), tm(9, 2.0)], PROF)))            # gap
        p = tm(); p.seq_flags = 1
        self.assertIn("C3", codes(validate_packet(p, PROF)))
        self.assertIn("C3", codes(validate_packet(tm(seq=SEQ_MOD), PROF)))

    def test_c3_tolerance(self):
        prof = MissionProfile(seq_tolerance=3)
        self.assertTrue(is_valid(validate_stream([tm(5, 1.0), tm(8, 2.0)], prof)))

    def test_c4_length(self):
        self.assertIn("C4", codes(validate_packet(tc(payload=b"x" * 300), PROF)))
        raw = encode(tm())
        self.assertIn("C4", codes(validate_bytes(raw[:-3], PROF)))                 # truncated
        self.assertIn("C4", codes(validate_bytes(raw + b"\x00", PROF)))            # trailing byte
        bad = bytearray(raw); bad[5] += 4                                           # length field too large
        self.assertIn("C4", codes(validate_bytes(bytes(bad), PROF)))
        self.assertTrue(is_valid(validate_bytes(raw, PROF)))

    def test_c5_crc(self):
        raw = bytearray(encode(tc()))
        raw[-1] ^= 0xFF
        self.assertIn("C5", codes(validate_bytes(bytes(raw), PROF)))
        p = tc(); p.pec = (packet_crc(p) + 1) & 0xFFFF
        self.assertIn("C5", codes(validate_packet(p, PROF)))

    def test_c6_opcode(self):
        self.assertIn("C6", codes(validate_packet(tc(opcode=99), PROF)))
        p = tm(); p.opcode = 3
        self.assertIn("C6", codes(validate_packet(p, PROF)))
        prof = MissionProfile(legal_opcodes={0x201: {7}})
        self.assertIn("C6", codes(validate_packet(tc(opcode=3), prof)))
        self.assertTrue(is_valid(validate_packet(tc(opcode=7), prof)))

    def test_c7_timing(self):
        self.assertIn("C7", codes(validate_stream([tm(0, 5.0), tm(1, 4.0)], PROF)))          # decreasing
        self.assertIn("C7", codes(validate_stream([tm(0, 5.0), tm(1, 500.0)], PROF)))        # gap too large
        p = tm(); p.timestamp = None
        self.assertIn("C7", codes(validate_packet(p, PROF)))

    def test_c8_functionality(self):
        rng = np.random.default_rng(0)
        w = make_window(rng, "flood")
        self.assertEqual(check_functionality(w, w.copy()), [])
        w2 = w.copy(); w2.label = 0
        self.assertIn("C8", codes(check_functionality(w, w2)))
        w3 = w.copy(); i = int(np.where(w3.ptype == TC)[0][0]); w3.opcode[i] += 1
        self.assertIn("C8", codes(check_functionality(w, w3)))
        w4 = w.copy(); w4.base_plen[0] += 1
        self.assertIn("C8", codes(check_functionality(w, w4)))

    def test_generated_windows_are_valid(self):
        rng = np.random.default_rng(1)
        for a in (None, "flood", "injection", "recon"):
            w = make_window(rng, a)
            self.assertEqual(validate_window(w, PROF), [], msg=a)
            self.assertEqual(len(window_to_packets(w)), len(w))

    def test_feature_realizability(self):
        rng = np.random.default_rng(2)
        X = np.stack([window_to_fc(make_window(rng, a), "extended") for a in (None, "flood", "recon")])
        self.assertTrue(feature_space_realizable(X, "extended", PROF).all())
        Xbad = X.copy(); Xbad[:, 5] += 2.0                                       # bwd packet count no longer matches bytes
        self.assertFalse(feature_space_realizable(Xbad, "extended", PROF).any())
        self.assertFalse(feature_space_realizable(np.full((1, 8), np.nan), "extended", PROF).any())


if __name__ == "__main__":
    unittest.main()
