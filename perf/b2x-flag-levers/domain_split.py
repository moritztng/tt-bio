#!/usr/bin/env python3
"""cdk2x2_512: superpose the two pseudo-domains separately, and report the hinge rotation.

`cdk2x2_512` is CDK2 fused to a truncated copy of itself in one chain, with no real inter-domain
interface. The hinge between the two pseudo-domains is the softest degree of freedom in the
structure, so it saturates whole-molecule RMSD for ANY non-bit-exact change and four unrelated
boltz-2 levers once all landed in the same 7.7-9.0 A band (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). The fixture-artifact signature that
same investigation established is: superposed SEPARATELY, each domain stays inside 0.5-1.5 A and
the entire whole-molecule difference is a rigid hinge rotation.

That is the only reading `cdk2x2_512` supports, and it is what this file computes. It does not
replace the `cdk2x2_298` control; it is what is available when the control comes back bit-exact
and therefore cannot exercise the difference under test.

    domain_split.py <dir-of-arm-dirs> --split 298 [--ref base]

Superposition and parser are `perf/other512/cif_rmsd.py`'s, so the numbers stay on the same basis
as the rest of this lineage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms                                  # noqa: E402


def kabsch(P, Q):
    """Rotation that takes centred P onto centred Q, and the RMSD after applying it.

    Same SVD convention as `cif_rmsd.kabsch_rmsd`, and `main` asserts the RMSD returned here
    reproduces that function's to 1e-9 -- an inverted rotation reads as a phantom RMSD and the
    check is cheaper than the confusion (memory `kabsch-inverse-rotation-swap-phantom-rmsd`).
    """
    P, Q = P - P.mean(0), Q - Q.mean(0)
    V, _S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return R, float(np.sqrt((((P @ R) - Q) ** 2).sum() / len(P)))


def angle_between(R1, R2) -> float:
    """Degrees of the relative rotation R1 -> R2."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", type=int, required=True, help="last label_seq_id of domain 1")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    data = {}
    for d in sorted(a.root.iterdir()):
        if not d.is_dir():
            continue
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        keys, xyz = read_atoms(cifs[0])
        assert len(keys[0]) == 4, f"expected 4 key columns, got {keys[0]}"
        seq = np.array([int(k[1]) for k in keys])
        arm = d.name.split("_")[1]
        data[arm] = {"keys": keys, "xyz": xyz,
                     "d1": np.where(seq <= a.split)[0], "d2": np.where(seq > a.split)[0],
                     "ca": np.array([i for i, k in enumerate(keys) if k[2] == "CA"])}
        print(f"  {arm:5s} {len(xyz):6d} atoms  domain1 {len(data[arm]['d1'])}  "
              f"domain2 {len(data[arm]['d2'])}  CA {len(data[arm]['ca'])}")
    if a.ref not in data:
        raise SystemExit(f"no {a.ref} arm in {a.root}")
    ref_keys = data[a.ref]["keys"]
    for arm, v in data.items():
        if v["keys"] != ref_keys:
            raise SystemExit(f"atom identity differs in {arm}")

    out = {"root": str(a.root), "split_seq_id": a.split, "ref": a.ref,
           "n_atoms": len(ref_keys), "arms": {}}
    R = data[a.ref]
    print(f"\n  each arm against `{a.ref}` -- whole molecule, then each pseudo-domain alone:")
    for arm, v in sorted(data.items()):
        if arm == a.ref:
            continue
        whole = kabsch_rmsd(v["xyz"], R["xyz"])
        _r, chk = kabsch(v["xyz"], R["xyz"])
        assert abs(whole - chk) < 1e-9, f"rotation convention mismatch: {whole} vs {chk}"
        wca = kabsch_rmsd(v["xyz"][v["ca"]], R["xyz"][R["ca"]])
        R1, r1 = kabsch(v["xyz"][v["d1"]], R["xyz"][R["d1"]])
        R2, r2 = kabsch(v["xyz"][v["d2"]], R["xyz"][R["d2"]])
        hinge = angle_between(R1, R2)
        artifact = max(r1, r2) < 1.5 and whole > 2.0 * max(r1, r2)
        out["arms"][arm] = {
            "whole_all_atom_A": round(whole, 5), "whole_ca_A": round(wca, 5),
            "domain1_all_atom_A": round(r1, 5), "domain2_all_atom_A": round(r2, 5),
            "hinge_deg": round(hinge, 3), "hinge_artifact_signature": bool(artifact)}
        print(f"    {arm:5s} whole {whole:8.4f} A (CA {wca:7.4f})   "
              f"domain1 {r1:7.4f}   domain2 {r2:7.4f}   hinge {hinge:7.2f} deg   "
              f"{'ARTIFACT' if artifact else 'not the artifact signature'}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print("\n  wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
