#!/usr/bin/env python3
"""
Recompute the training-set medians used as the last-resort imputation fallback.

Run this whenever ml/data/synthetic_builds.csv is regenerated. An imputed value
drawn from a different distribution than the model was trained on is a quiet bug:
the score looks confident and is built on a number the model has never seen.

    python tools/recompute_medians.py            # print the block
    python tools/recompute_medians.py --check    # exit 1 if app/imputation.py drifted

The --check form is suitable for CI.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "app"))

from features import FEATURES  # noqa: E402

CSV = REPO / "ml" / "data" / "synthetic_builds.csv"
TARGET = REPO / "app" / "imputation.py"
TOLERANCE = 0.02  # relative; medians shift slightly with regeneration seeds


def measured() -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(CSV)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"dataset is missing feature columns: {missing}")
    return {f: round(float(df[f].median()), 3) for f in FEATURES}


def current() -> dict[str, float]:
    src = TARGET.read_text()
    block = re.search(r"TRAINING_MEDIANS:\s*dict\[str,\s*float\]\s*=\s*\{(.*?)\}", src, re.S)
    if not block:
        raise SystemExit(f"could not find TRAINING_MEDIANS in {TARGET}")
    return {
        k: float(v)
        for k, v in re.findall(r'"([a-z_]+)":\s*(-?[0-9.]+)', block.group(1))
    }


def render(vals: dict[str, float]) -> str:
    rows = "\n".join(f'    "{f}": {vals[f]},' for f in FEATURES)
    return "TRAINING_MEDIANS: dict[str, float] = {\n" + rows + "\n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if app/imputation.py has drifted")
    args = ap.parse_args()

    if not CSV.exists():
        print(f"dataset not found: {CSV}")
        return 0 if not args.check else 1

    new, old = measured(), current()

    drifted = []
    for f in FEATURES:
        a, b = old.get(f), new[f]
        if a is None:
            drifted.append((f, a, b)); continue
        scale = max(abs(a), abs(b), 1e-6)
        if abs(a - b) / scale > TOLERANCE:
            drifted.append((f, a, b))

    if not drifted:
        print(f"✓ TRAINING_MEDIANS matches {CSV.name} (within {TOLERANCE:.0%})")
        return 0

    print(f"✗ {len(drifted)} median(s) drifted from {CSV.name}:\n")
    for f, a, b in drifted:
        print(f"    {f:22} imputation.py={a!s:>8}   dataset={b:>8}")
    print("\nPaste this into app/imputation.py:\n")
    print(render(new))
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
