"""Tests for the PyTorch models, GAN-augmented model and ART AutoAttack wrapper. SKIPPED when torch is absent.
These tests were NOT executed in the authoring sandbox (no torch): run them on your machine first."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.torch_base import HAS_TORCH

try:
    import art  # noqa: F401
    HAS_ART = True
except ImportError:
    HAS_ART = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchModels(unittest.TestCase):
    cfg = {"models": {"cnn": {"epochs": 3}, "lstm": {"epochs": 2, "hidden": 16, "layers": 1},
                      "transformer": {"epochs": 2, "d_model": 16, "heads": 2, "ff": 32, "layers": 1},
                      "gan_augmented": {"gan_iters": 20, "z_dim": 8}}}

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.random((200, 8)); self.y = (self.X[:, 0] + self.X[:, 1] > 1).astype(int)

    def test_each_model_interface(self):
        from models import build_model
        for name in ("cnn", "lstm", "transformer", "gan_augmented"):
            m = build_model(name, 8, self.cfg, seed=0).fit(self.X, self.y)
            p = m.predict_proba(self.X)
            self.assertEqual(p.shape, (200, 2)); np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
            self.assertEqual(m.loss_gradient(self.X[:5], self.y[:5]).shape, (5, 8))
            w = m.get_weights(); m.set_weights(w)
            np.testing.assert_allclose(m.predict_proba(self.X[:5]), p[:5], atol=1e-5)
            self.assertGreater(m.n_params, 0); self.assertGreater(m.size_bytes(), 0)

    def test_gradient_matches_finite_difference(self):
        from models import build_model
        m = build_model("cnn", 8, self.cfg, seed=0).fit(self.X, self.y)
        g = m.loss_gradient(self.X[:1], self.y[:1])[0]; eps = 1e-3
        for j in range(3):
            d = np.zeros(8); d[j] = eps
            lp = -np.log(m.predict_proba(self.X[:1] + d)[0, self.y[0]]); lm = -np.log(m.predict_proba(self.X[:1] - d)[0, self.y[0]])
            self.assertAlmostEqual(g[j], (lp - lm) / (2 * eps), delta=5e-2)

    def test_fedavg_with_torch_backbone(self):
        from models import build_model
        from models.federated import FedAvg
        parts = [(self.X[i::3], self.y[i::3]) for i in range(3)]
        g = FedAvg(lambda: build_model("cnn", 8, self.cfg, 0), rounds=2, local_epochs=1).fit(parts)
        self.assertEqual(g.predict_proba(self.X).shape, (200, 2))

    @unittest.skipUnless(HAS_ART, "ART not installed")
    def test_autoattack_wrapper(self):
        from attacks.standard_attacks import autoattack_art
        from models import build_model
        m = build_model("cnn", 8, self.cfg, seed=0).fit(self.X, self.y)
        Xa = autoattack_art(m, self.X[:20], self.y[:20], 0.1, (0.0, 1.0), max_iter=5)
        self.assertEqual(Xa.shape, (20, 8))


if __name__ == "__main__":
    unittest.main()
