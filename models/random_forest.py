"""Random Forest IDS (scikit-learn). Non-differentiable: attacked via transfer or query-based attacks."""
from __future__ import annotations

import pickle
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestIDS:
    name = "random_forest"
    differentiable = False

    def __init__(self, d: int, n_estimators: int = 200, max_depth=None, seed: int = 0, **_):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced",
                                          n_jobs=1, random_state=seed)
        self.fitted = False

    def fit(self, X, y, X_val=None, y_val=None) -> "RandomForestIDS":
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def partial_fit(self, X, y, epochs: int = 1, reset_optimizer: bool = False):
        raise NotImplementedError("Random Forests are refit from scratch; use fit()")

    def predict_proba(self, X) -> np.ndarray:
        p = self.clf.predict_proba(X)
        if p.shape[1] == 1:                                  # only one class seen
            p = np.hstack([1 - p, p]) if self.clf.classes_[0] == 1 else np.hstack([p, 1 - p])
        return p

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_params(self) -> int:
        return int(sum(t.tree_.node_count for t in self.clf.estimators_))

    def size_bytes(self) -> int:
        return len(pickle.dumps(self.clf))
