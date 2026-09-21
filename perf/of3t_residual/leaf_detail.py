#!/usr/bin/env python3
"""Per-tensor detail for one leaf class, across every arm sidecar handed to it."""
import argparse
import json
import re
import sys


def load(path):
    d = json.load(open(path))
    rows = d["per_tensor"] if isinstance(d, dict) else d
    return {r["param"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pat", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()
    arms = []
    for spec in a.arms:
        lab, _, p = spec.partition("=")
        arms.append((lab, load(p)))
    rx = re.compile(a.pat)
    names = [p for p in arms[0][1] if rx.search(p)]
    names.sort(key=lambda p: -arms[0][1][p]["diff_norm"] ** 2)
    print(f"{'param':<86}{'mass%':>9}{'ref_norm':>11}", end="")
    for lab, _ in arms:
        print(f"{lab + ' rel':>16}{lab + ' r':>11}{lab + ' cos':>11}", end="")
    print()
    for p in names[:a.limit]:
        r0 = arms[0][1][p]
        print(f"{p[-85:]:<86}{r0['pct_of_model_mass']:>9.4f}{r0['ref_norm']:>11.4e}", end="")
        for lab, m in arms:
            r = m.get(p)
            if r is None:
                print(f"{'--':>16}{'--':>11}{'--':>11}", end="")
            else:
                print(f"{r['rel_l2']:>16.4e}{r['r']:>11.4f}{r['cos']:>11.4f}", end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
