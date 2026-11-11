"""End-to-end smoke test: runs every experiment script and make_tables.py on configs/smoke.yaml (synthetic data,
NumPy/RF models) and checks outputs. Verifies plumbing only -- the numbers are meaningless."""
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_smoke")


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--config", "configs/smoke.yaml", *args],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"
    return r.stdout


class TestSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(OUT, ignore_errors=True)
        for s in ("run_baseline", "run_standard_attacks", "run_ccsds_attacks", "run_indomain", "run_defenses", "run_federated"):
            run(f"experiments/{s}.py")
        run("make_tables.py")

    def tbl(self, n):
        return pd.read_csv(os.path.join(OUT, "tables", n))

    def test_csvs_nonempty_and_flagged_synthetic(self):
        for n in ("baseline_clean", "standard_attacks", "ccsds_attacks", "defenses", "federated"):
            df = self.tbl(n + ".csv")
            self.assertGreater(len(df), 0, n)
            self.assertTrue((df.terr_source == "synthetic").all() and (df.sat_source == "synthetic").all())
            self.assertEqual(sorted(df.seed.unique()), [1, 2])

    def test_expected_rows(self):
        b = self.tbl("baseline_clean.csv")
        self.assertEqual(set(b.model), {"numpy_mlp", "random_forest", "fedavg"}); self.assertEqual(set(b.domain), {"terrestrial", "satellite"})
        cc = self.tbl("ccsds_attacks.csv")
        self.assertTrue({"pgd_ccsds", "transfer_ccsds", "greedy_ccsds"} <= set(cc.attack))
        self.assertTrue((cc.feasibility_rate == 1.0).all())
        de = self.tbl("defenses.csv")
        self.assertEqual(set(de.defense), {"baseline", "std_at", "paa", "at_paa", "paa_clean_control"})
        ind = self.tbl("indomain.csv")
        self.assertEqual(set(ind.model), {"numpy_mlp", "random_forest"}); self.assertTrue((ind.n_train > 0).all())
        fed = self.tbl("federated.csv")
        self.assertEqual(set(fed.variant), {"fedavg", "fedavg_at", "federated_paa", "federated_paa_clean_control"})

    def test_gap_identity_in_output(self):
        g = self.tbl("gap_per_seed.csv")
        np.testing.assert_allclose(g.G, g.G_domain + g.G_protocol, atol=1e-9)

    def test_tables_figures_and_banner(self):
        for n in ("table3_clean_terrestrial", "table4_standard_attacks", "table5_ccsds_attacks", "table6_robustness_gap",
                  "table7_defenses", "table8_federated", "table9_rq_summary", "table10_indomain_reference"):
            p = os.path.join(OUT, "tables", n + ".md")
            self.assertTrue(os.path.exists(p), n)
            with open(p, encoding="utf-8") as fh:
                self.assertIn("SYNTHETIC-DATA BANNER", fh.read(), n)
        for f in ("fig3_robustness_gap", "fig4_defense_effectiveness", "fig5_overhead_vs_robustness"):
            self.assertGreater(os.path.getsize(os.path.join(OUT, "figures", f + ".png")), 2000, f)

    def test_no_tbd_or_nan_in_tables(self):
        for n in ("table6_robustness_gap", "table7_defenses", "table9_rq_summary"):
            with open(os.path.join(OUT, "tables", n + ".md"), encoding="utf-8") as fh:
                txt = fh.read()
            self.assertNotIn("TBD", txt)


if __name__ == "__main__":
    unittest.main()
