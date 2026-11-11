#!/usr/bin/env python3
"""Verify preregistration thresholds unchanged since the freeze commit."""
import subprocess, re, sys

FREEZE_COMMIT = "9f621fe"
FREEZE_FILE   = "configs/default.yaml"
CURRENT_FILE  = "configs/nos3.yaml"


def extract_prereg(raw: str) -> dict:
    block, in_block = [], False
    for line in raw.split("\n"):
        if line.startswith("preregistration:"):
            in_block = True
            continue
        if in_block:
            if line.strip() == "" or re.match(r"^[a-z]", line):
                break
            block.append(line)
    out = {}
    for line in block:
        line = line.split("#")[0].rstrip()
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    freeze = extract_prereg(subprocess.check_output(
        ["git", "show", f"{FREEZE_COMMIT}:{FREEZE_FILE}"], text=True))
    current = extract_prereg(open(CURRENT_FILE).read())

    print(f"Freeze: {FREEZE_COMMIT}:{FREEZE_FILE}")
    print(f"Current: {CURRENT_FILE}")
    print(f"\n{'Key':30s} {'Freeze':>10s} {'Current':>10s} {'Match':>8s}")
    print("-" * 62)
    all_match = True
    for k in sorted(set(freeze) | set(current)):
        f = freeze.get(k, "<missing>")
        c = current.get(k, "<missing>")
        match = "OK" if f == c else "DIFF"
        if f != c:
            all_match = False
        print(f"{k:30s} {f:>10s} {c:>10s} {match:>8s}")
    print()
    print("PASS: thresholds unchanged" if all_match else "FAIL: thresholds differ")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
