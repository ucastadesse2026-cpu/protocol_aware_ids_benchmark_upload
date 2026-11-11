"""Tests for the common feature schema Fc and the dataset mappings."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import feature_schema as fs
from data.ccsds_parser import TC, TM, Window
from data.synthetic_ccsds import make_window


def small_window():
    return Window(ptype=np.array([TC, TM, TC]), apid=np.array([0x201, 0x101, 0x201]), seqc=np.array([0, 0, 1]),
                  ts=np.array([100.0, 101.0, 102.0]), base_plen=np.array([10, 20, 10]), pad=np.zeros(3, dtype=np.int64),
                  opcode=np.array([1, -1, 2]), sec=np.ones(3, dtype=bool), label=0)


class TestFeatureSchema(unittest.TestCase):
    def test_views(self):
        self.assertEqual(len(fs.view_names("core")), 4)
        self.assertEqual(len(fs.view_names("extended")), 8)
        self.assertEqual(len(fs.view_names("sat_aug")), 12)
        with self.assertRaises(ValueError):
            fs.view_names("nope")

    def test_window_features_exact(self):
        w = small_window()
        # wire lengths: TC = 6+6+2+10+2 = 26 ; TM = 6+6+0+20+2 = 34
        x = fs.window_to_fc(w, "extended")
        self.assertAlmostEqual(x[0], np.log1p(2.0))            # duration
        self.assertAlmostEqual(x[1], np.log1p(52.0))           # fwd bytes (2 TC)
        self.assertAlmostEqual(x[2], np.log1p(34.0))           # bwd bytes (1 TM)
        self.assertAlmostEqual(x[3], 52.0 / (52.0 + 34.0), places=6)
        self.assertAlmostEqual(x[6], 86.0 / 3.0)               # mean packet length
        self.assertAlmostEqual(x[7], np.log1p(3.0 / 2.0))      # packet rate

    def test_padding_changes_bytes_feature(self):
        w = small_window(); w2 = w.copy(); w2.pad[0] = 16
        self.assertAlmostEqual(np.expm1(fs.window_to_fc(w2, "core")[1]), 52.0 + 16.0)

    def test_sat_aug_features(self):
        x = fs.window_to_fc(make_window(np.random.default_rng(0), None), "sat_aug")
        self.assertEqual(x.shape, (12,)); self.assertTrue(np.isfinite(x).all())

    def test_same_function_for_both_domains(self):
        direct = fs.fc_from_natural(2.0, 52.0, 34.0, 2, 1, "extended")
        np.testing.assert_allclose(direct, fs.window_to_fc(small_window(), "extended"))

    def test_unsw_mapping(self):
        df = pd.DataFrame({"dur": [2.0], "spkts": [2], "dpkts": [1], "sbytes": [52], "dbytes": [34], "label": [1]})
        X, y = fs.terrestrial_to_fc(df, "unsw_nb15", "extended")
        np.testing.assert_allclose(X[0], fs.window_to_fc(small_window(), "extended"))
        self.assertEqual(y.tolist(), [1])

    def test_cicids_mapping_strips_column_spaces(self):
        df = pd.DataFrame({" Flow Duration": [2_000_000], " Total Fwd Packets": [2], " Total Backward Packets": [1],
                           "Total Length of Fwd Packets": [52], " Total Length of Bwd Packets": [34], " Label": ["BENIGN"]})
        X, y = fs.terrestrial_to_fc(df, "cicids2017", "extended")
        self.assertAlmostEqual(X[0, 0], np.log1p(2.0)); self.assertEqual(y.tolist(), [0])

    def test_nsl_kdd_core_only(self):
        df = pd.DataFrame({"duration": [1], "src_bytes": [10], "dst_bytes": [20], "label": ["neptune"]})
        X, y = fs.terrestrial_to_fc(df, "nsl_kdd", "core")
        self.assertEqual(X.shape, (1, 4)); self.assertEqual(y.tolist(), [1])
        with self.assertRaises(ValueError):
            fs.terrestrial_to_fc(df, "nsl_kdd", "extended")

    def test_bad_rows_dropped(self):
        df = pd.DataFrame({"dur": [1.0, np.inf], "spkts": [1, 1], "dpkts": [1, 1], "sbytes": [1, 1], "dbytes": [1, 1], "label": [0, 1]})
        X, y = fs.terrestrial_to_fc(df, "unsw_nb15", "core")
        self.assertEqual(len(X), 1)

    def test_scaler_inverse_and_linearity(self):
        X = np.random.default_rng(0).normal(size=(50, 4))
        sc = fs.MinMaxScaler().fit(X)
        np.testing.assert_allclose(sc.inverse(sc.transform(X)), X, atol=1e-8)
        self.assertGreaterEqual(sc.transform(X).min(), -1e-6)


if __name__ == "__main__":
    unittest.main()
