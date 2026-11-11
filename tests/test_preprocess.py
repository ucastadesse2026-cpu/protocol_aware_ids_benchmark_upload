"""Loader tests with tiny fake CICIDS2017-style CSVs (latin-1 bytes, leading-space column names)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import feature_schema as fs
from data.preprocess import DataNotFoundError, _read_terrestrial, load_terrestrial


def write_fake_cicids(folder, n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = [" Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets",
            " Total Length of Bwd Packets", " Label"]
    for i, name in enumerate(["Monday.csv", "Thursday.csv"]):
        y = rng.random(n) < 0.3
        df = pd.DataFrame({cols[0]: rng.integers(1, 5_000_000, n), cols[1]: rng.integers(1, 50, n),
                           cols[2]: rng.integers(0, 50, n), cols[3]: rng.integers(0, 9000, n),
                           cols[4]: rng.integers(0, 9000, n),
                           cols[5]: np.where(y, "Web Attack \x96 Brute Force" if i else "DoS", "BENIGN")})
        df.to_csv(os.path.join(folder, name), index=False, encoding="latin-1")


class TestCicidsLoader(unittest.TestCase):
    def test_latin1_files_and_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            write_fake_cicids(d)
            full = _read_terrestrial(d, "cicids2017")
            self.assertEqual(len(full), 800)
            small = _read_terrestrial(d, "cicids2017", max_rows=200, seed=1)
            self.assertTrue(180 <= len(small) <= 220)
            X, y = fs.terrestrial_to_fc(small, "cicids2017", "extended")
            self.assertEqual(X.shape[1], 8); self.assertTrue(set(y) <= {0, 1})

    def test_load_terrestrial_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            write_fake_cicids(d)
            cfg = {"features": {"view": "extended"}, "data": {"terrestrial": {"cicids2017": {"path": d, "source": "real", "max_rows": 500}}}}
            b = load_terrestrial(cfg, "cicids2017", seed=1)
            self.assertEqual(b["X_train"].shape[1], 8)
            self.assertGreater(len(b["y_test"]), 50)

    def test_official_test_partition(self):
        from data.synthetic_terrestrial import make_unsw_like
        with tempfile.TemporaryDirectory() as d:
            make_unsw_like(800, 1).to_csv(os.path.join(d, "train.csv"), index=False)
            make_unsw_like(300, 2).to_csv(os.path.join(d, "test.csv"), index=False)
            cfg = {"features": {"view": "extended"}, "data": {"terrestrial": {"unsw_nb15": {
                "path": os.path.join(d, "train.csv"), "test_path": os.path.join(d, "test.csv"), "source": "real"}}}}
            b = load_terrestrial(cfg, "unsw_nb15", seed=1)
            self.assertEqual(len(b["y_test"]), 300)                       # the whole official test file
            self.assertEqual(len(b["y_train"]) + len(b["y_val"]), 800)    # training file split 75/25
            self.assertEqual(len(b["y_val"]), 200)

    def test_missing_path_message(self):
        with self.assertRaises(DataNotFoundError):
            _read_terrestrial("does/not/exist", "unsw_nb15")

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DataNotFoundError):
                _read_terrestrial(d, "cicids2017")


if __name__ == "__main__":
    unittest.main()


class TestSeparabilityTool(unittest.TestCase):
    def test_runs_on_synthetic_and_overrides_work(self):
        import subprocess
        from config import load_config
        from data.preprocess import satellite_window_splits
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "tools", "diagnose_separability.py"), "--config",
                            os.path.join(root, "configs", "smoke.yaml"), "--sizes", "8", "--rules", "any"],
                           capture_output=True, text=True, cwd=root)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("design grid scored on the VALIDATION split only", r.stdout)
        cfg = load_config(os.path.join(root, "configs", "smoke.yaml"))
        a = satellite_window_splits(cfg, 1); b = satellite_window_splits(cfg, 1, size=8)
        self.assertEqual(len(a[0][0]), 32); self.assertEqual(len(b[0][0]), 8)
