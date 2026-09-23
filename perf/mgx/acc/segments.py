#!/usr/bin/env python3
"""CA-RMSD and CA-lDDT of two single-chain structures, window by window.

    python3 perf/mgx/acc/segments.py a.cif b.cif.gz [--window 298] [--trim 8]

cdk2x2_* tiles one 298-residue CDK2 chain, so a 298 window scores each copy on its own
superposition. It separates "each copy folds the same" from "the copies are packed differently",
which the whole-chain RMSD mixes. --trim drops the last residues of every window: the CDK2
C-terminus sits at the junction between copies, where upstream opendde's own seeds are 23 A apart.
"""
import argparse
import gzip
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "perf/mgx/ref")]
import ca_rmsd  # noqa: E402
import score  # noqa: E402


def ca(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        tmp.write(gzip.open(path).read())
        tmp.close()
        path = tmp.name
    _, cas = next(iter(ca_rmsd.ca_chains(path).values()))
    return np.array([[cas[i].x, cas[i].y, cas[i].z] for i in sorted(cas)])


def kabsch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    s[-1] *= np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(max(0.0, ((a * a).sum() + (b * b).sum() - 2 * s.sum()) / len(a))))


def lddt(a: np.ndarray, b: np.ndarray) -> float:
    pts = lambda x: [types.SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in x]
    return score.lddt_ca(pts(a), pts(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--window", type=int, default=298)
    ap.add_argument("--trim", type=int, default=0, help="residues dropped from each window's end")
    args = ap.parse_args()
    a, b = ca(args.a), ca(args.b)
    n, w = min(len(a), len(b)), args.window
    for i in range(0, n, w):
        j = min(i + w, n) - (args.trim if i + w <= n else 0)
        print(f"[{i}:{j}] {kabsch(a[i:j], b[i:j]):.2f} A  lDDT {lddt(a[i:j], b[i:j]):.3f}")


if __name__ == "__main__":
    main()
