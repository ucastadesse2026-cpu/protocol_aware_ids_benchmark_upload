"""Tests for standard attacks, CCSDS-conformant attacks, defenses, FedAvg, metrics."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.ccsds_conformant import CcsdsAttackConfig, ConformantAttacker, attack_windows
from attacks.standard_attacks import fgsm, pgd, perturb_positives, run_standard_attack
from attacks.validator import MissionProfile, validate_window
from data import feature_schema as fs
from data.ccsds_parser import TC, TM
from data.synthetic_ccsds import make_windows
from data.synthetic_terrestrial import make_unsw_like
from defenses.protocol_aware_augmentation import build_paa_set
from defenses.standard_at import adversarial_training
from evaluation.metrics import classification_metrics
from evaluation.robustness_gap import decompose_gap
from evaluation.transferability import transferability_score
from models.federated import FedAvg, average_weights, dirichlet_partition
from models.numpy_mlp import NumpyMLP
from models.random_forest import RandomForestIDS

PROF = MissionProfile()


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = make_unsw_like(3000, 1)
        X, y = fs.terrestrial_to_fc(df, "unsw_nb15", "extended")
        cls.sc = fs.MinMaxScaler().fit(X)
        cls.Xs, cls.y = cls.sc.transform(X), y
        cls.mlp = NumpyMLP(8, hidden=(32, 16), epochs=15, seed=0).fit(cls.Xs, cls.y)
        cls.rf = RandomForestIDS(8, n_estimators=30, seed=0).fit(cls.Xs, cls.y)
        cls.pipe = fs.FeaturePipeline("extended", cls.sc)
        cls.wins = make_windows(60, 3)
        cls.ywin = np.array([w.label for w in cls.wins])


class TestStandardAttacks(Fixture):
    def test_fgsm_bounds_and_clip(self):
        Xa = fgsm(self.mlp, self.Xs[:200], self.y[:200], 0.05, (0.0, 1.0))
        self.assertLessEqual(np.abs(Xa - self.Xs[:200]).max(), 0.05 + 1e-9)
        self.assertGreaterEqual(Xa.min(), 0.0); self.assertLessEqual(Xa.max(), 1.0)

    def test_pgd_bounds(self):
        Xa = pgd(self.mlp, self.Xs[:200], self.y[:200], 0.1, steps=5, clip=None)
        self.assertLessEqual(np.abs(Xa - self.Xs[:200]).max(), 0.1 + 1e-9)

    def test_only_positives_perturbed(self):
        Xa = run_standard_attack("pgd", self.mlp, self.Xs[:300], self.y[:300], 0.1, None, 0, 3)
        neg = self.y[:300] == 0
        np.testing.assert_array_equal(Xa[neg], self.Xs[:300][neg])
        self.assertTrue((Xa[~neg] != self.Xs[:300][~neg]).any())

    def test_pgd_reduces_detection(self):
        m = self.y == 1
        base = (self.mlp.predict(self.Xs[m]) == 1).mean()
        Xa = pgd(self.mlp, self.Xs[m], self.y[m], 0.15, steps=10, clip=None)
        self.assertLess((self.mlp.predict(Xa) == 1).mean(), base - 0.2)

    def test_transfer_to_random_forest_runs(self):
        Xa = run_standard_attack("pgd", self.mlp, self.Xs[:300], self.y[:300], 0.1, None, 0, 3)
        self.assertEqual(self.rf.predict_proba(Xa).shape, (300, 2))

    def test_unknown_attack(self):
        with self.assertRaises(ValueError):
            run_standard_attack("nope", self.mlp, self.Xs[:5], self.y[:5], 0.1, None)


class TestConformantAttacks(Fixture):
    def setUp(self):
        self.cfg = CcsdsAttackConfig(steps=8, greedy_iters=12, greedy_candidates=10, seed=0)

    def _check_adv(self, orig, adv, stats, kind):
        for w0, w1 in zip(orig, adv):
            self.assertEqual(validate_window(w1, PROF, original=w0), [], msg=kind)      # C1-C8 hold
            if w0.label == 0:
                self.assertIs(w0, w1)                                                    # benign untouched
            self.assertTrue(np.array_equal(w0.opcode, w1.opcode))
            tm = w0.ptype == TM
            self.assertTrue(np.array_equal(w0.pad[tm], w1.pad[tm]))                      # TM packets immutable
            self.assertTrue(np.array_equal(w0.apid[tm], w1.apid[tm]))
            self.assertLessEqual(int((w0.pad != w1.pad).sum()), self.cfg.budget_packets) if kind != "greedy_ccsds" else None
            self.assertLessEqual(int(w1.pad.max()), self.cfg.max_pad)
        self.assertTrue(all(s["feasible"] for s in stats), msg=kind)

    def test_pgd_ccsds_valid_and_effective(self):
        adv, st = attack_windows("pgd_ccsds", self.wins, self.mlp, self.pipe, PROF, self.cfg)
        self._check_adv(self.wins, adv, st, "pgd_ccsds")
        before = (self.mlp.predict(self.pipe.windows(self.wins)) == self.ywin).mean()
        after = (self.mlp.predict(self.pipe.windows(adv)) == self.ywin).mean()
        self.assertLessEqual(after, before)
        self.assertGreater(sum(s["edits"] for s in st), 0)

    def test_budget_limits_edited_packets(self):
        cfg = CcsdsAttackConfig(steps=8, budget_packets=3)
        adv, st = attack_windows("pgd_ccsds", self.wins, self.mlp, self.pipe, PROF, cfg)
        for w0, w1 in zip(self.wins, adv):
            edited = (w0.pad != w1.pad) | np.concatenate([[False], np.abs(np.diff(w0.ts) - np.diff(w1.ts)) > 1e-9])
            self.assertLessEqual(int(edited.sum()), 3)

    def test_greedy_black_box_valid(self):
        adv, st = attack_windows("greedy_ccsds", self.wins, self.rf, self.pipe, PROF, self.cfg)
        self._check_adv(self.wins, adv, st, "greedy_ccsds")
        self.assertTrue(all(s["queries"] >= 1 for s in st))

    def test_transfer_valid(self):
        adv, st = attack_windows("transfer_ccsds", self.wins, self.rf, self.pipe, PROF, self.cfg, surrogate=self.mlp)
        self._check_adv(self.wins, adv, st, "transfer_ccsds")
        with self.assertRaises(ValueError):
            attack_windows("transfer_ccsds", self.wins, self.rf, self.pipe, PROF, self.cfg)

    def test_infeasible_candidates_are_rejected(self):
        atk = ConformantAttacker(self.pipe, PROF, self.cfg)
        w0 = next(w for w in self.wins if w.label == 1)
        bad = w0.copy(); bad.ts[3] = bad.ts[2] - 1.0                          # time goes backwards -> C7
        ok, viol = atk._feasible(w0, bad)
        self.assertFalse(ok); self.assertTrue(any(v.startswith("C7") for v in viol))
        bad2 = w0.copy(); i = int(np.where(bad2.ptype == TC)[0][0]); bad2.opcode[i] += 1
        ok, viol = atk._feasible(w0, bad2)
        self.assertFalse(ok)                                                   # C8 immutable field changed

    def test_no_editable_packets(self):
        cfg = CcsdsAttackConfig(editable_types=())
        adv, st = attack_windows("pgd_ccsds", self.wins[:10], self.mlp, self.pipe, PROF, cfg)
        for w0, w1 in zip(self.wins[:10], adv):
            self.assertTrue(np.array_equal(w0.pad, w1.pad))

    def test_edits_change_features_as_expected(self):
        atk = ConformantAttacker(self.pipe, PROF, self.cfg)
        w0 = next(w for w in self.wins if w.label == 1)
        caps = atk._caps(w0)
        w1 = atk.realize(w0, w0.pad.astype(float) + 10, np.zeros(len(w0)), caps)
        self.assertTrue((w1.pad >= 10).all() or (w1.pad == caps).any())
        self.assertGreater(self.pipe.window(w1)[1], self.pipe.window(w0)[1] - 1e-12)   # fwd bytes feature grew


class TestDefensesAndFL(Fixture):
    def test_standard_at_runs_on_mlp_and_rf(self):
        m = NumpyMLP(8, hidden=(16,), epochs=4, seed=1)
        adversarial_training(m, self.Xs, self.y, 0.05, 4, 2, 0.5, 0, None, (0, 1))
        self.assertEqual(m.predict(self.Xs[:10]).shape, (10,))
        rf = RandomForestIDS(8, n_estimators=10)
        adversarial_training(rf, self.Xs, self.y, 0.05, 1, 2, 0.5, 0, surrogate=self.mlp)
        with self.assertRaises(ValueError):
            adversarial_training(RandomForestIDS(8, n_estimators=5), self.Xs, self.y, 0.05, 1)

    def test_paa_set(self):
        cfg = CcsdsAttackConfig(steps=5, seed=0)
        X, y, info = build_paa_set(self.wins, self.mlp, self.pipe, PROF, cfg, n_terrestrial=100, lam=0.5, adv_fraction=1.0, seed=0)
        self.assertEqual(len(X), len(y)); self.assertGreater(info["n_adv"], 0); self.assertEqual(info["feasibility_rate"], 1.0)
        Xc, yc, infoc = build_paa_set(self.wins, self.mlp, self.pipe, PROF, cfg, 100, 0.5, 1.0, 0, clean_control=True)
        self.assertEqual(infoc["n_adv"], 0)
        np.testing.assert_array_equal(y, yc)                                 # same windows, same labels
        self.assertTrue((X != Xc).any())                                     # but perturbed features differ

    def test_dirichlet_partition_covers_all(self):
        parts = dirichlet_partition(self.y, 5, 0.5, np.random.default_rng(0))
        allidx = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(allidx, np.arange(len(self.y)))

    def test_average_weights(self):
        out = average_weights([[np.ones(2)], [np.zeros(2)]], [3, 1])
        np.testing.assert_allclose(out[0], [0.75, 0.75])

    def test_fedavg_learns(self):
        parts = [(self.Xs[i], self.y[i]) for i in dirichlet_partition(self.y, 4, 1.0, np.random.default_rng(0))]
        g = FedAvg(lambda: NumpyMLP(8, hidden=(16,), epochs=1, seed=2), rounds=8, local_epochs=2, seed=0).fit(parts)
        self.assertGreater((g.predict(self.Xs) == self.y).mean(), 0.9)


class TestMetrics(unittest.TestCase):
    def test_classification_metrics(self):
        m = classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
        self.assertAlmostEqual(m["acc"], 0.5); self.assertAlmostEqual(m["fnr"], 0.5); self.assertAlmostEqual(m["fpr"], 0.5)

    def test_gap_decomposition_identity(self):
        d = decompose_gap(0.9, 0.7, 0.5)
        self.assertAlmostEqual(d["G"], d["G_domain"] + d["G_protocol"]); self.assertAlmostEqual(d["G"], 40.0)

    def test_transferability(self):
        t = transferability_score([0.6, 0.62, 0.61], [0.5, 0.5, 0.5], [0.9, 0.92, 0.91], [0.5, 0.5, 0.5])
        self.assertTrue(t["applicable"]); self.assertAlmostEqual(t["tau"], 0.11 / 0.41, places=2)
        na = transferability_score([0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
        self.assertFalse(na["applicable"])
        tiny = transferability_score([0.6, 0.6, 0.6], [0.5, 0.5, 0.5], [0.505, 0.506, 0.504], [0.5, 0.5, 0.5])
        self.assertFalse(tiny["applicable"])        # terrestrial gain < 1 pp -> tau not reported


if __name__ == "__main__":
    unittest.main()
