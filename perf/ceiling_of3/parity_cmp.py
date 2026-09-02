#!/usr/bin/env python3
"""Compare two OpenFold3 output directories bit-exactly.

Coordinates are compared as parsed numbers rather than by hashing the file, because an mmCIF
carries a write timestamp and a run id that differ between two identical folds and would make
every comparison fail for a reason that has nothing to do with the model. Confidence scores are
compared the same way, since a footprint change that moved a number would show there first.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def coords(d: str) -> list[tuple]:
    out = []
    for path in sorted(glob.glob(os.path.join(d, "**", "*.cif"), recursive=True)):
        for line in open(path):
            if line.startswith(("ATOM", "HETATM")):
                f = line.split()
                out.append((os.path.basename(path), f[1], tuple(f[10:13])))
    return out


def scores(d: str) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(d, "**", "results.json"), recursive=True)):
        for rec in json.load(open(path)):
            out.append({k: v for k, v in sorted(rec.items()) if k not in ("path", "out_dir")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    ca, cb = coords(args.a), coords(args.b)
    sa, sb = scores(args.a), scores(args.b)
    if not ca:
        print(f"{args.label}: NO COORDINATES in {args.a} -- nothing was compared")
        return 2
    same_c, same_s = ca == cb, sa == sb
    n = sum(1 for x, y in zip(ca, cb) if x != y)
    print(f"{args.label}: atoms={len(ca)}/{len(cb)} coords_bit_exact={same_c} "
          f"differing_atoms={n} scores_bit_exact={same_s}")
    if not same_s:
        print(f"  A scores: {sa}")
        print(f"  B scores: {sb}")
    return 0 if (same_c and same_s) else 1


if __name__ == "__main__":
    sys.exit(main())
