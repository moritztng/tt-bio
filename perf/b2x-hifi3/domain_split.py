#!/usr/bin/env python3
"""Per-domain RMSD on the cdk2x2_512 chimera, which is one chain and two domains.

`perf/b2x-baseline-attrib/domain_rmsd.py` splits by chain, and this fixture has only chain A:
CDK2 fused to a truncated copy of itself, hinge unconstrained (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). A whole-structure RMSD on it reads
the hinge angle. Superposing each domain on itself separates the hinge from the fold, and the
residue ranges below are the ones the baseline task used, so the numbers are comparable.

Parser and superposition are imported from that script rather than re-implemented.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_src = ROOT / "perf" / "b2x-baseline-attrib" / "domain_rmsd.py"
_spec = importlib.util.spec_from_file_location("b2x_domain_rmsd", _src)
_dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--ranges", default="1-290,301-512")
    ap.add_argument("--label", default="pair")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    ka, xa, _ba = _dr.read(a.ref)
    kb, xb, _bb = _dr.read(a.arm)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    common = [k for k in kb if k in ia]
    A = np.asarray([xa[ia[k]] for k in common])
    B = np.asarray([xb[ib[k]] for k in common])

    res = {"label": a.label, "ref": str(a.ref), "arm": str(a.arm),
           "n_matched": len(common),
           "whole_allatom_rmsd_A": round(_dr.kabsch(A, B), 4), "domains": {}}
    for rng in a.ranges.split(","):
        lo, hi = (int(v) for v in rng.split("-"))
        sel = [i for i, k in enumerate(common) if lo <= int(k[1]) <= hi]
        ca = [i for i in sel if common[i][2] == "CA"]
        res["domains"][rng] = {
            "n_atoms": len(sel),
            "allatom_rmsd_A": round(_dr.kabsch(A[sel], B[sel]), 4),
            "ca_rmsd_A": round(_dr.kabsch(A[ca], B[ca]), 4) if ca else None,
        }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
