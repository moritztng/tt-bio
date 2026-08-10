#!/usr/bin/env python3
"""p3-sdpa deliverable 1: chunk 64 vs chunk 320, structure and end-of-trunk tensors.

Host-only, no device. Reuses the fixed-1:1-alignment kabsch / tm_score / lddt primitives from
`scripts/boltz2_fast_parity.py` -- the same metric code the parity work already uses, so only
WHAT is compared changes.

    python3 perf/p3_sdpa/compare.py --a perf/p3_sdpa/fold_c64 --b perf/p3_sdpa/fold_c320 \
        --out perf/p3_sdpa/parity.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "b2fp", REPO_ROOT / "scripts" / "boltz2_fast_parity.py")
B2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B2)


def read_atoms(cif: Path):
    """Every atom, keyed so the two structures pair 1:1 and in a deterministic order."""
    import gemmi
    st = gemmi.read_structure(str(cif))
    out = {}
    for ch in st[0]:
        for res in ch:
            for at in res:
                out[(ch.name, res.seqid.num, res.name, at.name, at.altloc)] = \
                    np.array([at.pos.x, at.pos.y, at.pos.z])
    return out


def paired(a: dict, b: dict, ca_only=False):
    keys = [k for k in a if k in b and (not ca_only or k[3] == "CA")]
    keys.sort()
    return (np.array([a[k] for k in keys]), np.array([b[k] for k in keys]), len(keys))


def tensor_stats(x: "np.ndarray", y: "np.ndarray"):
    """Deviation of y (arm B) against x (arm A, the reference)."""
    import torch
    x = x.flatten().double()
    y = y.flatten().double()
    d = y - x
    rms_ref = float(torch.sqrt((x ** 2).mean()))
    rmsd = float(torch.sqrt((d ** 2).mean()))
    xc, yc = x - x.mean(), y - y.mean()
    pcc = float((xc * yc).sum() / (xc.norm() * yc.norm()))
    return dict(rms_of_reference=rms_ref, rmsd=rmsd,
                relative_rmsd=rmsd / rms_ref if rms_ref else float("nan"),
                max_abs_deviation=float(d.abs().max()),
                max_abs_of_reference=float(x.abs().max()),
                pcc=pcc, torch_equal=bool(torch.equal(x, y)), n=int(x.numel()))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference arm dir (chunk 64)")
    ap.add_argument("--b", type=Path, required=True, help="test arm dir (chunk 320)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cif_a = sorted(args.a.glob("*.cif"))[0]
    cif_b = sorted(args.b.glob("*.cif"))[0]
    A, B = read_atoms(cif_a), read_atoms(cif_b)

    res = {"cif_a": str(cif_a), "cif_b": str(cif_b),
           "n_atoms_a": len(A), "n_atoms_b": len(B)}

    Pa, Pb, n_all = paired(A, B)
    rmsd_all, al_all, ref_all = B2.kabsch(Pb, Pa)
    res["all_atom"] = dict(n=n_all, kabsch_rmsd_A=rmsd_all,
                           unaligned_rmsd_A=float(np.sqrt(((Pb - Pa) ** 2).sum(1).mean())))

    Ca, Cb, n_ca = paired(A, B, ca_only=True)
    rmsd_ca, al_ca, ref_ca = B2.kabsch(Cb, Ca)
    res["ca"] = dict(n=n_ca, kabsch_rmsd_A=rmsd_ca,
                     unaligned_rmsd_A=float(np.sqrt(((Cb - Ca) ** 2).sum(1).mean())),
                     tm_score=B2.tm_score(al_ca, ref_ca),
                     lddt=B2.lddt(al_ca, ref_ca))

    ta = torch.load(args.a / "trunk_out.pt")
    tb = torch.load(args.b / "trunk_out.pt")
    res["trunk"] = {k: tensor_stats(ta[k], tb[k]) for k in ("s", "z")}
    res["trunk_shapes"] = {k: list(ta[k].shape) for k in ("s", "z")}

    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
