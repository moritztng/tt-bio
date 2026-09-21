#!/usr/bin/env python3
"""A/A on `forward_device` with both masks unset: the shipped default must not have moved.

of3t-auxfind's control 6/11 ran this on the HOST `forward` only (`default_aa.py` calls
`port.forward`). The gradient this row re-takes is taken on `forward_device`, the training
entry point, which got its `pair_mask_d`/`attn_mask_d` in the same pass and was never A/A'd.
So the control that protects the default on the path this row actually uses did not exist.

Two arms on one card, one boundary: `origin/wk/of3t`'s own `openfold3_confidence.py`, and this
branch's. Both run `forward_device` with no masks. All seven outputs must be `torch.equal`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "of3t_confidence"))

KEYS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits", "pde_logits",
        "distogram_logits", "si_conf", "zij_conf"]


def main():
    out = Path(sys.argv[1])
    boundary = Path(sys.argv[2])
    ckpt = Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from openfold3.core.utils.atomize_utils import get_token_representative_atoms

    B = torch.load(boundary, map_location="cpu", weights_only=False)
    kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
    batch = kw["batch"] if "batch" in kw else pos[0]
    si_input = kw["si_input"] if "si_input" in kw else pos[1]
    outd = kw["output"] if "output" in kw else pos[2]
    use_ztrunk = bool(kw.get("use_zij_trunk_embedding", True))
    si_trunk, zij_trunk = outd["si_trunk"], outd["zij_trunk"]
    xpred = outd["atom_positions_predicted"].to(dtype=si_trunk.dtype)
    repr_x, _ = get_token_representative_atoms(
        batch=batch, x=xpred, atom_mask=batch["atom_mask"])

    def sq(t, r):
        while t.dim() > r:
            assert t.shape[0] == 1
            t = t[0]
        return t

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)

    N = int(sq(si_trunk, 2).shape[0])
    ft = lambda t, d: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)
    si_d = ft(sq(si_input, 2).float().unsqueeze(0), ttnn.bfloat16)
    st_d = ft(sq(si_trunk, 2).float().unsqueeze(0), ttnn.float32)
    zt_d = ft(sq(zij_trunk, 3).float().unsqueeze(0), ttnn.bfloat16)
    oh_d = head.distance_onehot(sq(repr_x, 2).float())

    with torch.no_grad():
        o = head.forward_device(si_d, st_d, zt_d, oh_d, use_zij_trunk_embedding=use_ztrunk)
    dump = {k: torch.Tensor(ttnn.to_torch(o[k])).clone() for k in KEYS}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, out)
    print(f"wrote {out}: {N} tokens, {len(dump)} tensors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
