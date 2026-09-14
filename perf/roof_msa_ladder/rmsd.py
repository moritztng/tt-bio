#!/usr/bin/env python3
"""All-atom RMSD, superposed, between every pair of folds this leg kept.

Three readings out of one set of files, which is the point of keeping every fold rather than one
per arm:

  A/A   two folds of the SAME arm in the same session. On a card with a known matmul
        nondeterminism this is not a formality; it is the floor any arm number has to clear.
  A/B   the paired, same-seed, same-stack reading the campaign's 0.60 A bar is defined against.
  S/S   the same arm across two processes, which carries the process-to-process part of the floor.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import gemmi
import numpy as np


def coords(p: Path):
    st = gemmi.read_structure(str(p))
    st.setup_entities()
    out, names = [], []
    for ch in st[0]:
        for res in ch:
            for at in res:
                out.append([at.pos.x, at.pos.y, at.pos.z])
                names.append((ch.name, res.seqid.num, at.name))
    return np.asarray(out, dtype=np.float64), names


def rmsd(a, b):
    """Kabsch-superposed RMSD in fp64. The structures share an atom order by construction."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ r) - b) ** 2).sum(1).mean()))


def main() -> int:
    out = {}
    for size in ("512", "298"):
        files = {}
        for d in (Path("perf/roof_msa_ladder/cif"), Path("perf/roof_msa_ladder/cif_s2")):
            s = "s1" if d.name == "cif" else "s2"
            for f in sorted(d.glob(f"{size}_warm*")):
                tag, arm = f.name.split("_")[1], f.name.split("_")[2]
                files[(s, tag, arm)] = f
        if not files:
            continue
        xyz = {k: coords(v) for k, v in files.items()}
        n = {k: len(v[1]) for k, v in xyz.items()}
        assert len(set(n.values())) == 1, f"atom counts differ: {n}"
        rows = {"AA_same_arm_same_session": [], "AB_paired_same_tag": [],
                "AB_all_cross_arm": [], "SS_same_arm_cross_session": []}
        for k1, k2 in itertools.combinations(sorted(files), 2):
            r = round(rmsd(xyz[k1][0], xyz[k2][0]), 5)
            lab = f"{k1[0]}/{k1[1]}/{k1[2]} vs {k2[0]}/{k2[1]}/{k2[2]}"
            if k1[2] == k2[2]:
                rows["AA_same_arm_same_session" if k1[0] == k2[0]
                     else "SS_same_arm_cross_session"].append((lab, r))
            else:
                rows["AB_all_cross_arm"].append((lab, r))
                if k1[0] == k2[0] and k1[1] == k2[1]:
                    rows["AB_paired_same_tag"].append((lab, r))
        out[size] = {"atoms": next(iter(n.values())),
                     "pairs": {k: v for k, v in rows.items()},
                     "median": {k: (round(float(np.median([x[1] for x in v])), 5) if v else None)
                                for k, v in rows.items()},
                     "max": {k: (round(max(x[1] for x in v), 5) if v else None)
                             for k, v in rows.items()}}
        print(f"\n=== {size} aa, {out[size]['atoms']} atoms ===")
        for k, v in rows.items():
            if not v:
                continue
            print(f"  {k}: median {out[size]['median'][k]:.5f} A, max {out[size]['max'][k]:.5f} A, "
                  f"n={len(v)}")
            for lab, r in v:
                print(f"      {lab:42s} {r:.5f}")
    Path("perf/roof_msa_ladder/rmsd.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
