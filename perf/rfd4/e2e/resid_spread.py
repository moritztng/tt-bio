"""Seed floor against same-seed deviation for a resid_abab.sh run made with `--rng per_design`.

    python perf/rfd4/e2e/resid_spread.py <abab dir> [--base elide] [--arm pairbf16] [--first 1] [--out <json>]

Each leg holds designs r0..rK, one seed each (design.py --reseed). r0 is the compile design and is
skipped by default: at 256 it does not repeat the warm design of the same seed. The floor is every pair of different seeds in the base
leg (what a re-roll costs you anyway); the deviation is the arms rk against the bases rk, same seed.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from grade import compare, designed_and_target, load


def stats(xs):
    return {"n": len(xs), "median": float(np.median(xs)), "mean": float(np.mean(xs)),
            "min": float(min(xs)), "max": float(max(xs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("abab")
    ap.add_argument("--base", default="elide")
    ap.add_argument("--arm", default="pairbf16")
    ap.add_argument("--first", type=int, default=1, help="first design index graded")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    for base in sorted(Path(a.abab).glob(f"*-{a.base}1")):
        size = base.name.split("-")[0]
        arm = base.with_name(f"{size}-{a.arm}1")
        cifs = lambda leg: {i: p for p in leg.glob("*_r*_generated.cif")
                            if (i := int(p.name.split("_r")[1].split("_")[0])) >= a.first}
        b, m = cifs(base), (cifs(arm) if arm.exists() else {})
        designed, targets = designed_and_target(load(str(b[min(b)])))
        rmsd = lambda x, y: compare(load(str(x)), load(str(y)), designed, targets)["designed_ca_rmsd_a"]
        floor = [rmsd(b[i], b[j]) for i, j in itertools.combinations(sorted(b), 2)]
        dev = {k: rmsd(m[k], b[k]) for k in sorted(set(b) & set(m))}
        out[size] = {"seed_floor_a": stats(floor), "same_seed_dev_a": dev,
                     "same_seed_dev": stats(list(dev.values())) if dev else None}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
