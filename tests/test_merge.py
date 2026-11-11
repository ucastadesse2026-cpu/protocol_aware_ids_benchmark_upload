import os, sys, tempfile, unittest
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.merge_results import merge


def write(d, name, rows):
    os.makedirs(os.path.join(d, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "tables", name), index=False)


class TestMerge(unittest.TestCase):
    def test_merge_concat_dedupe_and_conflict(self):
        with tempfile.TemporaryDirectory() as t:
            a, b, c, out = (os.path.join(t, x) for x in "abco")
            write(a, "baseline_clean.csv", [dict(seed=1, dataset="u", model="m", domain="terrestrial", acc=0.9)])
            write(b, "baseline_clean.csv", [dict(seed=2, dataset="u", model="m", domain="terrestrial", acc=0.8)])
            rep = merge([a, b], out)
            self.assertEqual(rep["baseline_clean.csv"], (2, 2))
            self.assertEqual(len(pd.read_csv(os.path.join(out, "tables", "baseline_clean.csv"))), 2)
            self.assertEqual(merge([a, a], os.path.join(t, "o2"))["baseline_clean.csv"], (2, 1))   # identical rows collapse
            write(c, "baseline_clean.csv", [dict(seed=1, dataset="u", model="m", domain="terrestrial", acc=0.5)])
            with self.assertRaises(SystemExit):                                                     # same key, different value
                merge([a, c], os.path.join(t, "o3"))


if __name__ == "__main__":
    unittest.main()
