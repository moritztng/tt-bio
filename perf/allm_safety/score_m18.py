#!/usr/bin/env python3
"""Score M18's structural move against the bar, with this model's OWN seed floor beside it.

Three numbers, all Kabsch on the same parser (`perf/other512/cif_rmsd.py`), all-atom and CA-only:

  A/A   two legs of the SAME arm at the SAME seed -- the instrument's own negative control.
  EFFECT off vs on at the same seed -- what M18 does to the structure.
  SEED   off at seed 0 vs off at seed 1 -- variation the campaign already accepts.

The bar is 0.60 A. Quoting the effect without the seed floor beside it is unreadable, and
inheriting a seed floor measured on another model at another size is
`precision-change-298aa-control-blind-to-512aa-failure`.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
import cif_rmsd as CR  # noqa: E402

BAR = 0.60


def coords(p: Path, ca_only: bool):
    cols, rows = CR.atom_site_table(p)
    xi, yi, zi = (cols["_atom_site.Cartn_x"], cols["_atom_site.Cartn_y"],
                  cols["_atom_site.Cartn_z"])
    ai = cols.get("_atom_site.label_atom_id", cols.get("_atom_site.auth_atom_id"))
    out = []
    for r in rows:
        if ca_only and ai is not None and r[ai].strip().strip('"') != "CA":
            continue
        out.append((float(r[xi]), float(r[yi]), float(r[zi])))
    return np.asarray(out, dtype=np.float64)


def kabsch(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt((((B @ R.T) - A) ** 2).sum(1).mean()))


def one(d: Path):
    c = sorted(d.glob("*.cif")) or sorted(d.glob("*.pdb"))
    if not c:
        raise SystemExit(f"no structure in {d}")
    return c[0]


def score(label, d1: Path, d2: Path):
    p, q = one(d1), one(d2)
    a = kabsch(coords(p, False), coords(q, False))
    c = kabsch(coords(p, True), coords(q, True))
    n = len(coords(p, False))
    print(f"  {label:34s} {a:9.6f} {c:9.6f}   {n:6d} atoms")
    return a


def main():
    out = Path(sys.argv[1])
    pre = "cif_m18_openfold3_512"
    legs = {d.name: d for d in sorted(out.iterdir()) if d.name.startswith(pre)}
    if not legs:
        raise SystemExit(f"no {pre}_* dirs under {out}")
    print("OpenFold3 512 aa, M18 (fused HiFi4) -- Kabsch RMSD, Angstrom")
    print(f"  bar {BAR:.2f} A. Seed floor measured on THIS model at THIS size on THIS card, below.")
    print(f"  legs: {', '.join(sorted(legs))}\n")
    print(f"  {'comparison':34s} {'all-atom':>9s} {'CA-only':>9s}")

    def pick(seed, arm):
        return [d for n, d in legs.items() if f"_s{seed}_" in n and n.endswith(f"_{arm}")]

    aa = []
    for seed in (0, 1):
        for arm in ("off", "on"):
            ds = pick(seed, arm)
            for x, y in itertools.combinations(ds, 2):
                aa.append(score(f"A/A  s{seed} {arm} x{len(ds)}", x, y))
    eff = [score("EFFECT  s0 off vs on", a, b)
           for a in pick(0, "off")[:1] for b in pick(0, "on")[:1]]
    sd = [score("SEED    s0 off vs s1 off", a, b)
          for a in pick(0, "off")[:1] for b in pick(1, "off")[:1]]

    print()
    if aa:
        print(f"  A/A floor (max)      {max(aa):.6f} A   <- the instrument's negative control")
    if eff:
        e = max(eff)
        print(f"  M18 EFFECT           {e:.6f} A")
        print(f"  {'INSIDE' if e < BAR else 'OUTSIDE'} the {BAR:.2f} A bar", end="")
        if sd:
            s = max(sd)
            print(f"; seed floor {s:.6f} A -> the effect is {e/s:.3f}x the seed floor")
        else:
            print(" (seed floor not measured this run)")
    if sd and not eff:
        print(f"  SEED floor           {max(sd):.6f} A")


if __name__ == "__main__":
    main()
