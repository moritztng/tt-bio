"""ABodyBuilder3 reference harness — the PyTorch ground truth for the tt-bio port.

Loads the trained ABodyBuilder3 ``plddt-loss`` checkpoint (one-hot, MSA-free
antibody Fv structure module) into the vendored reference StructureModule and runs
inference on a paired heavy+light sequence, producing atom37 coordinates + a
per-residue pLDDT. This is the parity oracle for the ttnn port: the port must match
these outputs (per-module PCC > 0.98, end-to-end Cα-RMSD vs this reference).

Usage::

    python tests/abodybuilder3_reference.py [--pdb out.pdb]

The checkpoint is resolved via tt_bio.abodybuilder3_weights (Zenodo download +
Lightning -> plain state_dict conversion on first use; set $ABB3_CKPT to point at a
local ``plddt-loss/best_second_stage.ckpt`` to skip the download).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make `tt_bio` importable when run directly from the repo root / tests dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio._vendor.abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from tt_bio._vendor.abodybuilder3.openfold.model.structure_module import StructureModule
from tt_bio._vendor.abodybuilder3.openfold.np.protein import Protein, to_pdb
from tt_bio._vendor.abodybuilder3.openfold.np.residue_constants import restype_order_with_x
from tt_bio._vendor.abodybuilder3.openfold.utils.feats import atom14_to_atom37

REL_POS_DIM = 64
DEVICE = "cpu"


def string_to_input(heavy: str, light: str, device: str = DEVICE) -> dict:
    """Build the one-hot ABodyBuilder3 input from heavy + light chain strings.

    Mirrors abodybuilder3.utils.string_to_input + ABDataset.single_and_double_from_datapoint:
      single: (1, N, 23)  aatype one-hot(21) + is_heavy one-hot(2)
      pair:   (1, N, N, 132)  edge-chain one-hot(3) + relative-position one-hot(2*64+1)
      aatype: (N,)  residue indices (20 = X)
      residue_index: (N,)  heavy 0..H-1, light 500..500+L-1 (the AF2 chain gap)
      is_heavy: (N,)  1 heavy / 0 light
    """
    aatype, is_heavy = [], []
    for c in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x[c])
    for c in light:
        is_heavy.append(0)
        aatype.append(restype_order_with_x[c])
    is_heavy = torch.tensor(is_heavy, dtype=torch.long)
    aatype = torch.tensor(aatype, dtype=torch.long)
    residue_index = torch.cat(
        (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
    )

    single_aa = F.one_hot(aatype, 21)
    single_chain = F.one_hot(is_heavy, 2)
    single = torch.cat((single_aa, single_chain), dim=-1).float()  # (N, 23)

    pair = residue_index[None] - residue_index[:, None]
    pair = pair.clamp(-REL_POS_DIM, REL_POS_DIM) + REL_POS_DIM
    pair = F.one_hot(pair, 2 * REL_POS_DIM + 1)
    edge_chain = 2 * is_heavy.outer(is_heavy) + (1 - is_heavy).outer(1 - is_heavy)
    edge_chain = F.one_hot(edge_chain.long())
    pair = torch.cat((edge_chain, pair), dim=-1).float()  # (N, N, 132)

    return {
        "single": single.unsqueeze(0).to(device),
        "pair": pair.unsqueeze(0).to(device),
        "aatype": aatype.to(device),
        "residue_index": residue_index.to(device),
        "is_heavy": is_heavy.to(device),
    }


def load_reference_model(cache: Path | None = None, device: str = DEVICE) -> StructureModule:
    """Build the vendored StructureModule and load the trained pLDDT weights."""
    cache = Path(cache or os.environ.get("TT_BIO_CACHE", Path.home() / ".ttbio"))
    weights = ensure_abb3_weights(cache)
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model = StructureModule(**ABB3_CONFIG)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(device)
    return model


def compute_plddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """(B, N, 50) logits -> (B, N) pLDDT scores (the upstream inference helper)."""
    pdf = F.softmax(plddt_logits, dim=-1)
    vbins = torch.arange(1, 101, 2, dtype=plddt_logits.dtype, device=plddt_logits.device)
    return pdf @ vbins


def predict(model: StructureModule, heavy: str, light: str,
            device: str = DEVICE) -> dict:
    """Run the reference forward; return atom37 coords, pLDDT, and a PDB string."""
    inp = string_to_input(heavy, light, device)
    with torch.no_grad():
        out = model({"single": inp["single"], "pair": inp["pair"]}, inp["aatype"])
    atom14 = out["positions"][-1, 0]  # (N, 14, 3) — final block
    batch = make_atom14_masks({"aatype": inp["aatype"].squeeze(0)})
    atom37 = atom14_to_atom37(atom14, batch)  # (N, 37, 3)
    plddt = compute_plddt(out["plddt"][0])  # (N,)
    aatype_np = inp["aatype"].squeeze(0).cpu().numpy().astype(int)
    chain_index = 1 - inp["is_heavy"].cpu().numpy().astype(int)
    atom_mask = batch["atom37_atom_exists"].cpu().numpy().astype(int)
    plddt_np = plddt.cpu().numpy()
    b_factors = (plddt_np[:, None] * atom_mask).astype(atom_mask.dtype)
    protein = Protein(
        aatype=aatype_np,
        atom_positions=atom37.cpu().numpy(),
        atom_mask=atom_mask,
        residue_index=np.arange(len(aatype_np)),
        b_factors=b_factors,
        chain_index=chain_index,
    )
    pdb = to_pdb(protein)
    return {
        "atom37": atom37.cpu(),
        "plddt": plddt.cpu(),
        "pdb": pdb,
        "aatype": inp["aatype"].squeeze(0).cpu(),
        "is_heavy": inp["is_heavy"].cpu(),
    }


# The paired Fv from ABodyBuilder3's example notebook (PDB 6yio H0-L0).
EXAMPLE_HEAVY = (
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQG"
    "RVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS"
)
EXAMPLE_LIGHT = (
    "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGS"
    "GSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", default=EXAMPLE_HEAVY)
    ap.add_argument("--light", default=EXAMPLE_LIGHT)
    ap.add_argument("--pdb", default=None, help="write the predicted structure here")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    model = load_reference_model(args.cache)
    res = predict(model, args.heavy, args.light)
    ca = res["atom37"][:, 1]  # CA is atom index 1
    n = ca.shape[0]
    print(f"ABodyBuilder3 reference: {n} residues "
          f"(H={int(res['is_heavy'].sum())}, L={n - int(res['is_heavy'].sum())})")
    print(f"  CA coords range: {ca.min(0).values.tolist()} .. {ca.max(0).values.tolist()}")
    print(f"  mean pLDDT: {res['plddt'].mean().item():.2f}")
    print(f"  param count: {sum(p.numel() for p in model.parameters())}")
    if args.pdb:
        Path(args.pdb).write_text(res["pdb"])
        print(f"  wrote PDB -> {args.pdb}")


if __name__ == "__main__":
    main()
