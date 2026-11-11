"""YAML configuration loading with ``_base`` inheritance and deep merge."""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config. A ``_base`` key names a parent config (relative to this file's directory)."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("_base", None)
    if base:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


def results_dir(cfg: Dict[str, Any], sub: str = "") -> str:
    d = os.path.join(ROOT, cfg.get("output", {}).get("dir", "results"), sub)
    os.makedirs(d, exist_ok=True)
    return d
