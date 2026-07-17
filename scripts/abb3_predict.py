"""End-to-end parity + predict for the hybrid ABodyBuilder3 ttnn port.

Runs the on-device/host hybrid StructureModuleTT on the 6yio H0-L0 Fv and
compares its atom37 Cα to the pure-PyTorch reference (Cα-RMSD after Kabsch
alignment). Also writes a PDB. This is the end-to-end parity gate for the port.

On device (bf16, fp32 dest acc): input embeddings, IPA projections + linear_out,
post-IPA LayerNorm, Transition, BackboneUpdate, AngleResnet linears, pLDDT head.
On host fp32 (the documented ceiling): IPA rigid-apply + scalar/point attention +
value aggregation, quaternion backbone compose, torsion_angles_to_frames + atom14.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio.abodybuilder3 import abb3_compute_kernel_config, StructureModuleTT
from tt_bio._vendor.abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from tt_bio._vendor.abodybuilder3.openfold.utils.feats import atom14_to_atom37
from tt_bio._vendor.abodybuilder3.openfold.np.protein import Protein, to_pdb
from abodybuilder3_reference import string_to_input, compute_plddt, EXAMPLE_HEAVY, EXAMPLE_LIGHT


def kabsch_rmsd(a, b):
    a = a.double(); b = b.double()
    ca, cb = a.mean(0), b.mean(0)
    a, b = a - ca, b - cb
    h = a.T @ b
    u, s, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, d], dtype=a.dtype, device=a.device))
    r = u @ corr @ vt
    a = (r @ a.T).T
    return float(torch.sqrt(((a - b) ** 2).sum(-1).mean()))


def main():
    cache = os.environ.get("TT_BIO_CACHE", "/tmp/abb3_cache")
    heavy = os.environ.get("ABB3_HEAVY", EXAMPLE_HEAVY)
    light = os.environ.get("ABB3_LIGHT", EXAMPLE_LIGHT)
    inp = string_to_input(heavy, light, "cpu")
    single, pair, aatype = inp["single"], inp["pair"], inp["aatype"]
    mask = torch.ones(single.shape[:-1], dtype=single.dtype)

    # Reference (host fp32)
    from abodybuilder3_reference import load_reference_model
    ref = load_reference_model(cache)
    with torch.no_grad():
        rout = ref({"single": single, "pair": pair}, aatype)
    ref_atom14 = rout["positions"][-1, 0]
    batch = make_atom14_masks({"aatype": aatype.squeeze(0)})
    ref_atom37 = atom14_to_atom37(ref_atom14, batch)
    ref_ca = ref_atom37[:, 1]

    # Hybrid (on-device projections + standard components; host attention/atom14)
    sd = torch.load(ensure_abb3_weights(cache), map_location="cpu", weights_only=True)
    ck = abb3_compute_kernel_config()
    model = StructureModuleTT(sd, ck, ABB3_CONFIG)
    t0 = time.time()
    with torch.no_grad():
        out = model(single, pair, aatype, mask)
    dt = time.time() - t0
    atom14 = out["positions"][-1, 0]
    atom37 = atom14_to_atom37(atom14, batch)
    ca = atom37[:, 1]
    rmsd = kabsch_rmsd(ca, ref_ca)
    plddt = compute_plddt(out["plddt"][0])
    plddt_ref = compute_plddt(rout["plddt"][0])
    plddt_pcc = float(((plddt - plddt.mean()) * (plddt_ref - plddt_ref.mean())).sum() /
                      ((plddt - plddt.mean()).norm() * (plddt_ref - plddt_ref.mean()).norm()))

    print(f"hybrid ABodyBuilder3: {ca.shape[0]} residues")
    print(f"  Cα-RMSD vs reference (Kabsch): {rmsd:.4f} Å")
    print(f"  pLDDT PCC vs reference: {plddt_pcc:.5f}  (mean {plddt.mean():.2f} vs ref {plddt_ref.mean():.2f})")
    print(f"  wall time (incl compile): {dt:.2f} s")

    aatype_np = aatype.squeeze(0).cpu().numpy().astype(int)
    chain_index = 1 - inp["is_heavy"].cpu().numpy().astype(int)
    atom_mask = batch["atom37_atom_exists"].cpu().numpy().astype(int)
    b_factors = (plddt.cpu().numpy()[:, None] * atom_mask).astype(atom_mask.dtype)
    protein = Protein(aatype=aatype_np, atom_positions=atom37.cpu().numpy(),
                      atom_mask=atom_mask, residue_index=np.arange(len(aatype_np)),
                      b_factors=b_factors, chain_index=chain_index)
    pdb = to_pdb(protein)
    out_pdb = os.environ.get("ABB3_PDB", "/tmp/abb3_cache/abb3_hybrid_pred.pdb")
    Path(out_pdb).write_text(pdb)
    print(f"  wrote PDB -> {out_pdb}")


if __name__ == "__main__":
    main()
