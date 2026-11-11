#!/usr/bin/env bash
# Assemble protocol_aware_ids_benchmark.zip (GitHub/Zenodo-ready). Run from the repository root:  bash build_zip.sh
set -euo pipefail
NAME="protocol_aware_ids_benchmark"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../${NAME}.zip}"
cd "$HERE"

# 1. regenerate the (synthetic) sample files so the archive is self-consistent
python3 data/make_samples.py >/dev/null

# 2. build the archive, excluding caches, smoke results and raw datasets
python3 - "$HERE" "$OUT" "$NAME" <<'PY'
import os, sys, zipfile
root, out, name = sys.argv[1:4]
skip_dirs = {"__pycache__", ".git", ".venv", "results_smoke", "cache"}
skip_prefix = ("data/raw/",)
keep_raw = {"data/raw/.gitkeep"}
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in skip_dirs]
        for f in files:
            full = os.path.join(d, f); rel = os.path.relpath(full, root).replace(os.sep, "/")
            if f.endswith((".pyc", ".zip")) or (rel.startswith(skip_prefix) and rel not in keep_raw):
                continue
            if rel.startswith("results/") and not rel.endswith(".gitkeep"):
                continue
            z.write(full, f"{name}/{rel}"); n += 1
print(f"wrote {out} ({n} files)")
PY
touch results/tables/.gitkeep results/figures/.gitkeep 2>/dev/null || true
