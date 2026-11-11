"""FedAvg implemented from scratch (no FL library). Works with any model exposing
get_weights/set_weights/partial_fit (NumpyMLP and every TorchIDS)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def dirichlet_partition(y: np.ndarray, k: int, alpha: float, rng: np.random.Generator, min_size: int = 10) -> List[np.ndarray]:
    """Label-skewed non-IID split: per class, proportions ~ Dirichlet(alpha) over k nodes.
    Small alpha (0.1) = strong skew; large alpha = near-IID. Resamples until every node has >= min_size rows."""
    y = np.asarray(y)
    for _ in range(200):
        parts: List[List[int]] = [[] for _ in range(k)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for node, chunk in enumerate(np.split(idx, cuts)):
                parts[node] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("could not build a Dirichlet partition with the requested minimum node size")


def average_weights(ws: Sequence[List[np.ndarray]], sizes: Sequence[int]) -> List[np.ndarray]:
    tot = float(sum(sizes))
    return [sum(w[i] * (n / tot) for w, n in zip(ws, sizes)) for i in range(len(ws[0]))]


class FedAvg:
    """Synchronous FedAvg. ``factory()`` returns a fresh model of the common architecture."""

    def __init__(self, factory: Callable[[], object], rounds: int = 50, local_epochs: int = 2,
                 fraction: float = 1.0, seed: int = 0):
        self.factory, self.rounds, self.local_epochs, self.fraction = factory, rounds, local_epochs, fraction
        self.rng = np.random.default_rng(seed)
        self.history: List[Dict[str, float]] = []
        self.bytes_per_round = 0

    def fit(self, parts: Sequence[Tuple[np.ndarray, np.ndarray]], X_val=None, y_val=None,
            local_fit: Optional[Callable[[object, np.ndarray, np.ndarray, int, int], None]] = None):
        glob = self.factory()
        k = len(parts)
        n_sel = max(1, int(round(self.fraction * k)))
        clients = [self.factory() for _ in range(k)]
        wbytes = sum(w.size * 4 for w in glob.get_weights())        # float32 on the wire
        self.bytes_per_round = int(2 * wbytes * n_sel)               # down + up per selected node
        for r in range(self.rounds):
            sel = self.rng.choice(k, size=n_sel, replace=False)
            gw = glob.get_weights()
            locals_, sizes = [], []
            for i in sel:
                clients[i].set_weights(gw)
                if local_fit is None:
                    clients[i].partial_fit(parts[i][0], parts[i][1], epochs=self.local_epochs, reset_optimizer=True)
                else:                                   # e.g. local adversarial training; hook gets (model, X, y, epochs, node)
                    local_fit(clients[i], parts[i][0], parts[i][1], self.local_epochs, int(i))
                locals_.append(clients[i].get_weights())
                sizes.append(len(parts[i][1]))
            glob.set_weights(average_weights(locals_, sizes))
            if X_val is not None:
                acc = float((glob.predict_proba(X_val).argmax(1) == y_val).mean())
                self.history.append({"round": r + 1, "val_acc": acc})
        return glob
