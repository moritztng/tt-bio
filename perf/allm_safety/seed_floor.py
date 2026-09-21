#!/usr/bin/env python3
"""M18's structural effect against a MULTI-SEED floor, not a single pair.

Pass 6 said M18 moves OpenFold3's 298 aa structure 2.253x further than re-seeding does, on ONE
seed pair. A single pair is a point estimate with no spread of its own, so it cannot say whether
18.6 A is outside seed variation or just a large draw from it. This scores every off-arm seed
against every other -- the seed floor as a DISTRIBUTION -- and puts the effect beside it.

    seed_floor.py <out-dir> --size 298
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
import cif_rmsd as CR  # noqa: E402


def load(d: Path):
    p = sorted(d.glob("*.cif")) or sorted(d.glob("*.pdb"))
    if not p:
        raise SystemExit(f"no structure in {d}")
    cols, rows = CR.atom_site_table(p[0])
    xi, yi, zi = (cols["_atom_site.Cartn_x"], cols["_atom_site.Cartn_y"],
                  cols["_atom_site.Cartn_z"])
    si, ai = cols["_atom_site.label_seq_id"], cols["_atom_site.label_atom_id"]
    xyz = np.array([(float(r[xi]), float(r[yi]), float(r[zi])) for r in rows])
    res = np.array([int(r[si]) for r in rows])
    nm = np.array([r[ai].strip().strip('"') for r in rows])
    return xyz, res, nm


def kab(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.sqrt((((B @ R.T) - A) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--size", type=int, default=298)
    a = ap.parse_args()

    pat = re.compile(rf"^cif_m18_{a.model}_{a.size}_s(\d+)_leg\d+_(off|on)$")
    legs = {}
    for d in sorted(a.out.iterdir()):
        m = pat.match(d.name)
        if m:
            legs.setdefault((int(m.group(1)), m.group(2)), d)

    ref = None
    store = {}
    for k, d in legs.items():
        xyz, res, nm = load(d)
        if ref is None:
            ref = (res, nm)
        else:
            assert (res == ref[0]).all() and (nm == ref[1]).all(), \
                f"atom order differs in {d}; correspondence unsafe"
        store[k] = xyz
    ca = ref[1] == "CA"

    print(f"{a.model} {a.size} aa -- M18 effect against a multi-seed floor (CA-only, Angstrom)")
    print(f"  legs: {', '.join(f's{s}/{arm}' for s, arm in sorted(store))}\n")

    offs = sorted(s for s, arm in store if arm == "off")
    floor = []
    print("  SEED FLOOR -- off arm, every seed pair (unmodified main against itself):")
    for x, y in itertools.combinations(offs, 2):
        v = kab(store[(x, "off")][ca], store[(y, "off")][ca])
        floor.append(v)
        print(f"    seed {x} vs seed {y}          {v:9.4f}")
    if floor:
        print(f"    -> n={len(floor)}  min {min(floor):.4f}  median {np.median(floor):.4f}  "
              f"max {max(floor):.4f}")

    print("\n  M18 EFFECT -- off vs on at the SAME seed:")
    eff = []
    for s in sorted({s for s, arm in store}):
        if (s, "off") in store and (s, "on") in store:
            v = kab(store[(s, "off")][ca], store[(s, "on")][ca])
            eff.append(v)
            print(f"    seed {s} off vs on         {v:9.4f}")
    if eff:
        print(f"    -> n={len(eff)}  min {min(eff):.4f}  max {max(eff):.4f}")

    ons = sorted(s for s, arm in store if arm == "on")
    if len(ons) > 1:
        print("\n  ON-ARM seed spread (does the lever's own output vary like main's?):")
        for x, y in itertools.combinations(ons, 2):
            print(f"    seed {x} vs seed {y} (on)     "
                  f"{kab(store[(x, 'on')][ca], store[(y, 'on')][ca]):9.4f}")

    if floor and eff:
        print()
        lo, hi, med = min(floor), max(floor), float(np.median(floor))
        e = float(np.median(eff))
        print(f"  effect median {e:.4f} A vs seed-floor range [{lo:.4f}, {hi:.4f}] A "
              f"(median {med:.4f})")
        if e > hi:
            print(f"  => the effect EXCEEDS every seed pair measured ({e/hi:.2f}x the largest). "
                  f"M18 is not a re-draw.")
        elif e < lo:
            print(f"  => the effect is BELOW every seed pair measured ({lo/e:.2f}x smaller than "
                  f"the smallest). Inside accepted variation.")
        else:
            print(f"  => the effect sits INSIDE the seed-floor range, so on this evidence it is "
                  f"indistinguishable from a different draw.")


if __name__ == "__main__":
    main()
