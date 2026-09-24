#!/usr/bin/env python3
"""Is the ceiling step's trunk output a structure? Host only, no card.

`ladder.py --save-out` keeps the device trunk's (msa, pair) after the gradient step. This runs
them through the model's OWN structure module and pLDDT head (float32 reference weights), and
does the same for a float32 reference trunk fed the identical embedding, so the device
structure is graded against a reference structure and not only against geometry.

The sequence is the step's own: argmax of the seeded logits, one recycle, no MSA, no template.
That is a random design at step 0, so pLDDT is low by construction on BOTH arms and is not the
claim. The claims are geometric (CA-CA bond lengths, no CA clashes) and comparative (device vs
reference CA RMSD after superposition, and the pLDDT gap between them).

  python3 perf/bcx_large/structure_check.py --n 352
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402

OUT = ROOT / "perf" / "bcx_large"
RESTYPES = "ARNDCQEGHILKMFPSTWYV"


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def geometry(ca):
    bond = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    dist = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    iu = np.triu_indices(len(ca), k=3)
    return {"ca_ca_median": float(np.median(bond)), "ca_ca_min": float(bond.min()),
            "ca_ca_max": float(bond.max()),
            "ca_ca_frac_3p6_4p0": float(((bond > 3.6) & (bond < 4.0)).mean()),
            "ca_clashes_lt_3A": int((dist[iu] < 3.0).sum()),
            "radius_of_gyration": float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))}


def fold(model, m, z, feats):
    from tt_bio.af2_confidence import plddt_per_residue
    with torch.no_grad():
        single = model.single_activations(m[0].to(model.trunk_dtype)).float()
        st = model.structure(single, z.float(), feats)
        pl = plddt_per_residue(model.heads["plddt"](st["representations/structure_module"]))
    ca = st["final_atom_positions"][:, 1].double().numpy()
    return ca, pl.double().numpy()


def write_ca_pdb(path, ca, seq, plddt):
    three = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
             "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
             "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
    with open(path, "w") as f:
        for i, (x, y, zz) in enumerate(ca):
            f.write(f"ATOM  {i + 1:5d}  CA  {three[seq[i]]} A{i + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{zz:8.3f}  1.00{100 * plddt[i]:6.2f}           C\n")
        f.write("END\n")


def ref_trunk(model, logits):
    t0 = time.time()
    with torch.no_grad():
        m, z = A.embed(model, logits.float(), torch.arange(logits.shape[0]))
        m, z = A.ref_stack(model, m, z, 4, 48)
    return m, z, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ref-only", action="store_true",
                    help="compute and cache the float32 reference trunk from the seed alone")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.ref_only:                      # the same draws ladder.py makes, in its order
        torch.manual_seed(args.seed)
        logits = torch.randn(args.n, 20) * 2.0
        _, ref = A.load_models(args.params, device_arm=False)
        m, z, sec = ref_trunk(ref["f32"], logits)
        torch.save({"logits": logits, "m": m, "z": z, "seconds": sec},
                   OUT / f"ref_trunk_n{args.n}.pt")
        print(f"reference trunk n={args.n}: {sec:.1f} s")
        return
    saved = torch.load(OUT / f"out_n{args.n}.pt")
    logits, m_dev, z_dev = saved["logits"], saved["m"], saved["z"]
    n = logits.shape[0]
    from tt_bio.af2_data import monomer_features
    seq = "".join(RESTYPES[i] for i in logits.argmax(-1).tolist())
    feats = {k: torch.as_tensor(v) for k, v in monomer_features(seq).items()}
    _, ref = A.load_models(args.params, device_arm=False)
    model = ref["f32"]
    cache = OUT / f"ref_trunk_n{n}.pt"
    if cache.exists():                     # `--ref-only` wrote it, possibly while the card ran
        c = torch.load(cache)
        assert torch.equal(c["logits"], logits), "reference trunk cached for other logits"
        m_ref, z_ref, ref_trunk_s = c["m"], c["z"], c["seconds"]
    else:
        m_ref, z_ref, ref_trunk_s = ref_trunk(model, logits)
    ca_dev, pl_dev = fold(model, m_dev, z_dev, feats)
    ca_ref, pl_ref = fold(model, m_ref, z_ref, feats)
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())  # noqa: E731
    blob = {"n": n, "sequence": seq, "ref_trunk_f32_s": ref_trunk_s,
            "trunk_rel_l2_vs_f32": {"msa": rel(m_dev, m_ref), "pair": rel(z_dev, z_ref)},
            "device": {**geometry(ca_dev), "plddt_mean": float(pl_dev.mean())},
            "f32_reference": {**geometry(ca_ref), "plddt_mean": float(pl_ref.mean())},
            "ca_rmsd_device_vs_f32_A": kabsch_rmsd(ca_dev, ca_ref),
            "plddt_mean_abs_diff": float(np.abs(pl_dev - pl_ref).mean())}
    write_ca_pdb(OUT / f"ca_n{n}_device.pdb", ca_dev, seq, pl_dev)
    write_ca_pdb(OUT / f"ca_n{n}_f32ref.pdb", ca_ref, seq, pl_ref)
    (OUT / f"structure_n{n}.json").write_text(json.dumps(blob, indent=1))
    print(json.dumps({k: v for k, v in blob.items() if k != "sequence"}, indent=1))


if __name__ == "__main__":
    main()
