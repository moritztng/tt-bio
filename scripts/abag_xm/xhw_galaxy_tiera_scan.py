#!/usr/bin/env python3
"""Galaxy-vs-tier_a oracle delta scan (the od 9sbb artifact-class prevalence tool).

For every target with BOTH a galaxy deepn N=64 labels.json and a tier_a (qb1, 50-sample)
label file: oracle delta = max DockQ(galaxy n64) - max DockQ(tier_a 50). Reports n,
median signed delta, and the targets whose |delta| exceeds 0.2 in either direction.
The od galaxy artifact class (pipeline mis-fold condemned by the model's own ptm, e.g.
9sbb: tier_a 0.951 vs galaxy 0.313) shows as galaxy-worse-by->0.2; per-model exclusion
sets for the galaxy rungs are taken from this scan (state doc passes 59/61/71).

Runs on qb1. usage: xhw_galaxy_tiera_scan.py [model ...]   (default: opendde)
"""
import json
import statistics as st
import sys
from pathlib import Path

DEEPN = Path.home() / "abag_xm" / "deepn" / "galaxy"
TIERA = Path.home() / "abag_xm" / "tier_a" / "labels"
MDIR = {"opendde": "opendde", "boltz2": "boltz2",
        "protenix": "protenix", "esmfold2": "esmfold2"}
TPREFIX = {"opendde": "opendde_abag", "boltz2": "boltz2",
           "protenix": "protenix_v2", "esmfold2": "esmfold2"}


def getdq(s):
    d = s.get("dockq")
    return d.get("dockq") if isinstance(d, dict) else d


def oracle(path):
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    s = d.get("samples", d)
    vals = [getdq(x) for x in s]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def main():
    models = sys.argv[1:] or ["opendde"]
    for m in models:
        rows = []
        for d in sorted((DEEPN / MDIR[m]).glob("*_n64")):
            t = d.name[:-4]
            g = oracle(d / "labels.json")
            ta = oracle(TIERA / f"{TPREFIX[m]}_{t}.json")
            if g is None or ta is None:
                continue
            rows.append((t, g, ta, g - ta))
        if not rows:
            print(f"{m}: no paired targets")
            continue
        deltas = [r[3] for r in rows]
        worse = [r for r in rows if r[3] < -0.2]
        better = [r for r in rows if r[3] > 0.2]
        print(f"{m}: n={len(rows)} med_delta={st.median(deltas):+.4f} "
              f"galaxy-worse>0.2: {len(worse)}  galaxy-better>0.2: {len(better)}")
        for t, g, ta, d in sorted(worse, key=lambda r: r[3]):
            print(f"   WORSE {t}: galaxy {g:.3f} vs tier_a {ta:.3f} (d={d:+.3f})")
        for t, g, ta, d in sorted(better, key=lambda r: -r[3]):
            print(f"   better {t}: galaxy {g:.3f} vs tier_a {ta:.3f} (d={d:+.3f})")


if __name__ == "__main__":
    main()
