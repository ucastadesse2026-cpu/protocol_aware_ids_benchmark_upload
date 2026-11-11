"""Verify a frozen configuration: compare the EFFECTIVE config (after `_base` inheritance) at a freeze commit with the working tree.

    python tools/check_freeze.py --freeze 9f621fe --config configs/nos3.yaml

Prints the freeze commit's date, the effective `preregistration:` values at freeze vs now, every other effective setting that
differs (data, features, attacks, defenses, federated, gap, profile, models, ...), and the commits after the freeze that touched
`configs/`. Use its output as the evidence behind "thresholds unchanged"; `grep preregistration <your config>` proves nothing when the
block lives in a base config. Needs git and the repository checked out.
"""
from __future__ import annotations

import argparse
import copy
import os
import posixpath
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def effective_at(commit: str, path: str) -> dict:
    """Effective config at ``commit`` (None = working tree), resolving `_base` from the same snapshot."""
    if commit is None:
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = git("show", f"{commit}:{path.replace(os.sep, '/')}")
    cfg = yaml.safe_load(text) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = deep_merge(effective_at(commit, posixpath.normpath(posixpath.join(posixpath.dirname(path.replace(os.sep, "/")), base))), cfg)
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", required=True, help="freeze commit hash")
    ap.add_argument("--config", required=True, help="config path relative to the repo root, e.g. configs/nos3.yaml")
    a = ap.parse_args()
    cfgp = a.config.replace("\\", "/")
    print("freeze commit:", git("log", "-1", "--format=%H  author-date %ad  committer-date %cd  %s", "--date=iso", a.freeze).strip())
    try:
        then = flatten(effective_at(a.freeze, cfgp))
    except SystemExit as exc:
        sys.exit(f"{exc}\nThe config (or its base) did not exist at the freeze commit: the freeze did not contain it.")
    now = flatten(effective_at(None, cfgp))
    keys = sorted(set(then) | set(now))
    pre = [k for k in keys if k.startswith("preregistration.")]
    print("\nEffective preregistration block (freeze -> now):")
    for k in pre:
        flag = "" if then.get(k) == now.get(k) else "   <-- CHANGED"
        print(f"  {k:44s} {then.get(k)!s:>8} -> {now.get(k)!s:<8}{flag}")
    if not pre:
        print("  (none found: the effective config has no preregistration block at all)")
    diff = [k for k in keys if not k.startswith("preregistration.") and then.get(k) != now.get(k)]
    print(f"\nOther effective settings that differ ({len(diff)}):")
    for k in diff:
        print(f"  {k}: {then.get(k)!r} -> {now.get(k)!r}")
    print("\nCommits after the freeze that touched configs/:")
    log = git("log", f"{a.freeze}..HEAD", "--format=  %h %ad %s", "--date=iso", "--", "configs").strip()
    print(log or "  (none)")
    changed = [k for k in pre if then.get(k) != now.get(k)]
    print("\nRESULT:", "thresholds UNCHANGED" if pre and not changed else "thresholds CHANGED or missing: disclose in the amendment")


if __name__ == "__main__":
    main()
