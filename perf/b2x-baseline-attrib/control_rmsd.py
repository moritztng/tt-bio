#!/usr/bin/env python3
"""cdk2x2_298 control: how far a non-bit-exact arm's structure moved from the shipped default.

Host only. All-atom and CA Kabsch RMSD, atoms matched on (chain, seq id, atom name) so a
different atom count between arms is an error rather than a silent reindex. Parser and
superposition come from ``perf/other512/cif_rmsd.py`` unmodified, so these numbers are
comparable with the rest of this lineage.

The bar is pre-fixed in ``state/answered/4649-decision.md``: <= 0.35 A pass, 0.35-0.60 A hold
for a decision, > 0.60 A reject.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms          # noqa: E402

PASS_A, HOLD_A = 0.35, 0.60


def matched(pa: Path, pb: Path):
    ka, xa = read_atoms(pa)
    kb, xb = read_atoms(pb)
    ia = {k: i for i, k in enumerate(ka)}
    common = [k for k in kb if k in ia]
    if not common:
        raise SystemExit(f"no atoms in common between {pa} and {pb}")
    ib = {k: i for i, k in enumerate(kb)}
    A = np.asarray([xa[ia[k]] for k in common])
    B = np.asarray([xb[ib[k]] for k in common])
    ca = [i for i, k in enumerate(common) if k[2] == "CA"]
    return A, B, ca, len(ka), len(kb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True, help="shipped-default arm CIF")
    ap.add_argument("--arm", type=Path, required=True, help="the arm under test")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    A, B, ca, na, nb = matched(a.ref, a.arm)
    allatom = kabsch_rmsd(A, B)
    caonly = kabsch_rmsd(A[ca], B[ca]) if ca else None
    verdict = ("pass" if allatom <= PASS_A else
               "hold" if allatom <= HOLD_A else "reject")
    res = {"label": a.label, "ref": str(a.ref), "arm": str(a.arm),
           "n_atoms_ref": na, "n_atoms_arm": nb, "n_matched": len(A), "n_CA": len(ca),
           "allatom_rmsd_A": round(float(allatom), 4),
           "ca_rmsd_A": round(float(caonly), 4) if caonly is not None else None,
           "bar": {"pass_A": PASS_A, "hold_A": HOLD_A}, "verdict": verdict,
           "identical_bytes": a.ref.read_bytes() == a.arm.read_bytes()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
