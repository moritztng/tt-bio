#!/usr/bin/env python3
"""A/A: the shipped default path with token_mask unset, before and after the mask patch.

`token_mask` defaults to None and the claim is that nothing moves when it is unset. A claim
is not a measurement, so this dumps the unmasked head outputs and the two runs are compared
byte for byte. Run once with the pre-patch file in place and once with the patched one.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

BOUND = "/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt"
CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
TREE = "/home/ttuser/of3t_rebase/of3pkg043"


def sq(t, rank):
    while t.dim() > rank:
        assert t.shape[0] == 1, t.shape
        t = t[0]
    return t


def main() -> int:
    out = Path(sys.argv[1])
    sys.path.insert(0, TREE)
    from openfold3.core.utils.atomize_utils import (
        broadcast_token_feat_to_atoms, get_token_representative_atoms)
    B = torch.load(BOUND, map_location="cpu", weights_only=False)
    kw = dict(B["inputs"]["kwargs"]); del B
    batch, outd = kw["batch"], kw["output"]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    del sd
    xpred = outd["atom_positions_predicted"].to(dtype=outd["si_trunk"].dtype)
    repr_x, _ = get_token_representative_atoms(batch=batch, x=xpred,
                                               atom_mask=batch["atom_mask"])
    mam = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"], num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"], max_num_atoms_per_token=23).reshape(-1).float()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    port = OF3ConfidenceHead(aux, dev, ckc)
    with torch.no_grad():
        o = port.forward(si_input=sq(kw["si_input"], 2).float(),
                         si_trunk=sq(outd["si_trunk"], 2).float(),
                         zij_trunk=sq(outd["zij_trunk"], 3).float(),
                         repr_x_pred=sq(repr_x, 2).float(),
                         max_atom_per_token_mask=mam,
                         use_zij_trunk_embedding=bool(kw.get("use_zij_trunk_embedding", True)))
    torch.save({k: v.detach().float().clone() for k, v in o.items() if torch.is_tensor(v)}, out)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
