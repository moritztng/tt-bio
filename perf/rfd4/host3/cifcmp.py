"""Pair the designs of two fair-harness arms seed by seed and compare their atoms.

    python perf/rfd4/host3/cifcmp.py <arm A dir> <arm B dir> [--design 0] [--out grade.json]

An arm dir is what `perf/rfd4/fair/run.py --out` wrote for one width (e.g. `256-A/b1`, `256-B/b2l1`). For every
`measure-s<seed>` present on both sides, design `--design` of each is read and the ATOM/HETATM records compared:
`identical` when every record matches byte for byte, else all-atom RMSD and max atom deviation in Angstrom (atom
identity, sequence and annotations must still match; a mismatch there is reported as `different_atoms`).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def atoms(path):
    """(identity fields, xyz) of each atom record; the coordinates are columns -5..-3 of the atom_site loop."""
    ids, xyz = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            ids.append(" ".join(f[:-5]))
            xyz.append([float(v) for v in f[-5:-2]])
    return ids, np.array(xyz)


def compare(a, b):
    ia, xa = atoms(a)
    ib, xb = atoms(b)
    if ia != ib:
        return {"different_atoms": True, "n": [len(ia), len(ib)]}
    if np.array_equal(xa, xb):
        return {"identical": True, "n": len(ia)}
    d = np.linalg.norm(xa - xb, axis=1)
    return {"identical": False, "n": len(ia), "rmsd_a": round(float(np.sqrt((d ** 2).mean())), 5),
            "max_a": round(float(d.max()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--design", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    def by_seed(root):
        out = {}
        for d in root.glob("measure-s*"):
            cif = sorted(d.glob(f"*_d{a.design}.cif"))
            if cif:
                out[int(re.search(r"s(\d+)$", d.name).group(1))] = cif[0]
        return out

    A, B = by_seed(a.a), by_seed(a.b)
    seeds = sorted(A.keys() & B.keys())
    rows = {s: compare(A[s], B[s]) for s in seeds}
    for s, r in rows.items():
        print(s, json.dumps(r))
    diff = [r for r in rows.values() if not r.get("identical")]
    summary = {"a": str(a.a), "b": str(a.b), "design": a.design, "seeds": len(seeds),
               "identical": len(seeds) - len(diff),
               "different_atoms": sum(1 for r in diff if r.get("different_atoms")),
               "rmsd_a_max": max((r["rmsd_a"] for r in diff if "rmsd_a" in r), default=0.0),
               "max_atom_a": max((r["max_a"] for r in diff if "max_a" in r), default=0.0)}
    if diff and not summary["different_atoms"]:
        summary["rmsd_a_median"] = float(np.median([r["rmsd_a"] for r in diff]))
    print(json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=1))
    return 0 if seeds else 1


if __name__ == "__main__":
    sys.exit(main())
