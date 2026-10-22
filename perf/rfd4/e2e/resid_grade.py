"""Summarise a resid_abab.sh run: warm trunk timings per leg and CA RMSD between legs.

    python perf/rfd4/e2e/resid_grade.py <abab dir> [--base base] [--out <json>]

Every leg writes two designs of one seed; r1 is the warm one. For each size and arm the warm r1 is
compared to the base leg of the same round (and the base rounds to each other), superposed on the
target CA as grade.py does. The floor is base r0 vs base r1: two honest designs of the same run.
`identical` says whether the two CIFs carry the same atom records, the exactness check for a
lever that must not move a bit.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from grade import compare, designed_and_target, load


def warm(leg):
    rows = [json.loads(l) for l in (leg / "runs.jsonl").read_text().splitlines() if l.strip()]
    return next((r for r in rows if r.get("kind") == "warm"), None)


def atoms(cif):
    return [l for l in Path(cif).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("abab")
    ap.add_argument("--base", default="base")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.abab)
    legs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "runs.jsonl").exists() and warm(p)):
        m = re.fullmatch(r"(\d+)-(\w+?)(\d)", d.name)
        size, arm, rnd = int(m[1]), m[2], int(m[3])
        legs[(size, arm, rnd)] = d
    out = {"timing": {}, "rmsd": {}}
    for (size, arm, rnd), d in sorted(legs.items()):
        w = warm(d)
        out["timing"].setdefault(str(size), {}).setdefault(arm, []).append(
            {"round": rnd, "trunk_s": w["trunk_s"], "design_s": w["design_s"], "host_s": w["host_s"],
             "trunk_call_min_s": w["trunk_call_s"]["min"], "aiclk_median": w["aiclk_mhz"]})
    for (size, arm, rnd), d in sorted(legs.items()):
        base = legs.get((size, a.base, rnd))
        if base is None:
            continue
        cif = lambda leg, r: str(next(leg.glob(f"*_r{r}_generated.cif")))
        ref = load(cif(base, 1))
        designed, targets = designed_and_target(ref)
        row = {"vs": f"{a.base}{rnd}", "identical": atoms(cif(d, 1)) == atoms(cif(base, 1))}
        row["rmsd_a"] = compare(load(cif(d, 1)), ref, designed, targets)["designed_ca_rmsd_a"]
        row["floor_a"] = compare(load(cif(base, 0)), ref, designed, targets)["designed_ca_rmsd_a"]
        if arm == a.base and (size, a.base, 1) in legs and rnd > 1:
            b1 = legs[(size, a.base, 1)]
            row["vs"] = f"{a.base}1"
            row["identical"] = atoms(cif(d, 1)) == atoms(cif(b1, 1))
            row["rmsd_a"] = compare(load(cif(d, 1)), load(cif(b1, 1)), designed, targets)["designed_ca_rmsd_a"]
        elif arm == a.base:
            continue
        out["rmsd"].setdefault(str(size), {})[f"{arm}{rnd}"] = row
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
