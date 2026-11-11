"""Onboard-feasibility metrics: single-sample latency, serialised model size, peak Python memory.

Latency is measured on the host CPU restricted to one thread; it is a *proxy*, not a flight-hardware
measurement. Report the host CPU in the paper.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Dict

import numpy as np


def measure_latency_ms(model, x: np.ndarray, n: int = 200, warmup: int = 20) -> Dict[str, float]:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    x1 = np.asarray(x, dtype=float).reshape(1, -1)
    for _ in range(warmup):
        model.predict_proba(x1)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict_proba(x1)
        ts.append((time.perf_counter() - t0) * 1e3)
    return dict(latency_ms_median=float(np.median(ts)), latency_ms_p95=float(np.percentile(ts, 95)))


def peak_memory_mb(fn: Callable[[], object]) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def model_footprint(model) -> Dict[str, float]:
    return dict(n_params=int(model.n_params), size_bytes=int(model.size_bytes()))
