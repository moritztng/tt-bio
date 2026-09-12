"""Per-DOMAIN RMSD on cdk2x2_512, which is one chain containing two.

`perf/b2x-baseline-attrib/domain_rmsd.py` splits by chain, and this fixture is CDK2 fused to a
truncated copy of itself inside a single chain A. So a per-chain RMSD on it is the whole-structure
RMSD, and the whole-structure RMSD reads the unconstrained hinge angle between the two halves
rather than the arithmetic (`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). This
splits on the residue index instead and superposes each half on itself.

Parser and superposition come from `perf/other512/cif_rmsd.py` unmodified -- the same two functions
`control_rmsd.py` uses -- so these numbers sit in the same lineage and no second Kabsch exists to
get the rotation inverted in (`kabsch-inverse-rotation-swap-phantom-rmsd`).
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


def score(pa: Path, pb: Path, lo: int, hi: int):
    ka, xa = read_atoms(pa)
    kb, xb = read_atoms(pb)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    common = [k for k in kb if k in ia and lo <= int(k[1]) <= hi]
    if not common:
        raise SystemExit(f"no atoms in residues {lo}-{hi}")
    A = np.asarray([xa[ia[k]] for k in common])
    B = np.asarray([xb[ib[k]] for k in common])
    ca = [i for i, k in enumerate(common) if k[2] == "CA"]
    return {"lo": lo, "hi": hi, "n_atoms": len(common), "n_CA": len(ca),
            "allatom_rmsd_A": round(float(kabsch_rmsd(A, B)), 4),
            "ca_rmsd_A": round(float(kabsch_rmsd(A[ca], B[ca])), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--split", type=int, default=298,
                    help="last residue of domain 1; cdk2x2_512 is CDK2(298) + CDK2(1..214)")
    ap.add_argument("--label", default="pair")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ka, _ = read_atoms(a.ref)
    last = max(int(k[1]) for k in ka)
    doms = [score(a.ref, a.arm, 1, a.split), score(a.ref, a.arm, a.split + 1, last)]
    worst = max(d["allatom_rmsd_A"] for d in doms)
    out = {"label": a.label, "ref": str(a.ref), "arm": str(a.arm),
           "split_residue": a.split, "last_residue": last, "domains": doms,
           "worst_domain_allatom_A": worst,
           "bar": {"pass_A": PASS_A, "hold_A": HOLD_A},
           "verdict": "pass" if worst <= PASS_A else ("hold" if worst <= HOLD_A else "reject")}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
