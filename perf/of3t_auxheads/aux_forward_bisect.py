#!/usr/bin/env python3
"""Localise the aux_heads forward gap, which A18s first clause makes mandatory.

`aux_instrument.py` read four of five heads outside the 5.0e-02 bar at the 0.4.3 boundary,
worst plddt_logits 3.6515e-01, with only distogram_logits (the one head that does not go
through the confidence Pairformer) inside it. A18 then forbids reading the gradient there
and requires the forward to be localised instead. This is that bisection, and it is the
same shape as of3t-rebases one-block DiT bisection: upstream 0.4.3s own module in float64
against ours on the card, stage by stage, at the boundary the capture recorded.

Two arms, and the pair is the point:

  padded   the full 384-token crop exactly as the boundary holds it. Upstream applies
           single_mask and pair_mask; our forward_device applies none.
  real     the 56 real tokens sliced out, where every mask upstream would apply is
           identically one. Our unmasked port computes upstreams function here by
           construction, so anything left in this arm is arithmetic, not masking.

A gap that vanishes in real is a masking gap. One that survives is ours.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "of3t_confidence"))


def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def squeeze_leading(t, rank):
    while t.dim() > rank:
        assert t.shape[0] == 1, t.shape
        t = t[0]
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--arm", default="real", choices=["real", "padded"])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.perf_counter()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from openfold3.core.model.heads.prediction_heads import PairformerEmbedding
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.core.utils.atomize_utils import get_token_representative_atoms

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
    batch = kw["batch"] if "batch" in kw else pos[0]
    si_input = squeeze_leading(kw["si_input"] if "si_input" in kw else pos[1], 2)
    outd = kw["output"] if "output" in kw else pos[2]
    use_ztrunk = bool(kw.get("use_zij_trunk_embedding", True))
    si_trunk = squeeze_leading(outd["si_trunk"], 2)
    zij_trunk = squeeze_leading(outd["zij_trunk"], 3)
    xpred = outd["atom_positions_predicted"].to(dtype=outd["si_trunk"].dtype)
    token_mask = batch["token_mask"].reshape(-1)
    repr_x, repr_mask = get_token_representative_atoms(
        batch=batch, x=xpred, atom_mask=batch["atom_mask"])
    repr_x = squeeze_leading(repr_x, 2)
    repr_mask = repr_mask.reshape(-1)

    if a.arm == "real":
        sel = token_mask.bool()
        si_input, si_trunk = si_input[sel], si_trunk[sel]
        zij_trunk = zij_trunk[sel][:, sel]
        repr_x, repr_mask, token_mask = repr_x[sel], repr_mask[sel], token_mask[sel]
    n = int(si_trunk.shape[0])
    print("arm %s: N=%d, single_mask ones %d, use_zij_trunk_embedding=%s"
          % (a.arm, n, int(repr_mask.sum()), use_ztrunk), flush=True)

    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    pe_cfg = cfg.architecture.heads["pairformer_embedding"]
    pe = PairformerEmbedding(**pe_cfg).to(torch.float64)
    pe_sd = {k[len("pairformer_embedding."):]: v.to(torch.float64)
             for k, v in aux.items() if k.startswith("pairformer_embedding.")}
    pe.load_state_dict(pe_sd, strict=True)
    pe.eval()
    for m in pe.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
            m.p = 0.0
        if hasattr(m, "r"):
            try:
                m.eval()
                m.r = 0.0
            except Exception:
                pass

    their = {}
    hooks = []
    for i, blk in enumerate(pe.pairformer_stack.blocks):
        def mk(i):
            def hook(mod, args, output):
                s_, z_ = output
                their["block%d_s" % i] = s_.detach().double().reshape(-1, s_.shape[-1]).clone()
                their["block%d_z" % i] = z_.detach().double().reshape(
                    z_.shape[-3], z_.shape[-2], z_.shape[-1]).clone()
            return hook
        hooks.append(blk.register_forward_hook(mk(i)))

    si_in64 = si_input.double().unsqueeze(0)
    si64 = si_trunk.double().unsqueeze(0)
    z64 = zij_trunk.double().unsqueeze(0)
    x64 = repr_x.double().reshape(1, 1, n, 3)
    sm = repr_mask.double().reshape(1, n)
    pm = token_mask.double().reshape(1, n, 1) * token_mask.double().reshape(1, 1, n)
    z_in = z64 if use_ztrunk else z64 * 0
    with torch.no_grad():
        z_emb = pe.embed_zij(si_input=si_in64, zij=z_in, x_pred=x64)
        # pairformer_emb, not forward: forward carries a batch-dimension validator that
        # rejects a sample axis on x_pred alone, and with apply_per_sample False (which is
        # what grad-enabled training always takes) forward dispatches straight to this.
        s_out, z_out = pe.pairformer_emb(
            si_input=si_in64, si=si64, zij=z_in, x_pred=x64,
            single_mask=sm, pair_mask=pm, chunk_size=None, _mask_trans=True)
    for h in hooks:
        h.remove()
    their["embed_z"] = z_emb.detach().double().reshape(n, n, -1).clone()
    their["si_conf"] = s_out.detach().double().reshape(-1, s_out.shape[-1]).clone()
    their["zij_conf"] = z_out.detach().double().reshape(n, n, -1).clone()
    print("[%.0fs] their float64 side done, %d intermediates"
          % (time.perf_counter() - t0, len(their)), flush=True)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    dn = lambda t: torch.Tensor(ttnn.to_torch(getattr(t, "value", t))).double()
    si_d = ttnn.from_torch(si_input.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    st_d = ttnn.from_torch(si_trunk.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.float32)
    zt_d = ttnn.from_torch(zij_trunk.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    oh_d = head.distance_onehot(repr_x)

    pe_p = "pairformer_embedding."
    C_Z = int(zij_trunk.shape[-1])
    z = zt_d if use_ztrunk else ttnn.multiply(zt_d, 0.0)
    z = ttnn.add(z, ttnn.reshape(head._lin(si_d, pe_p + "linear_i.weight"), (1, n, 1, C_Z)))
    z = ttnn.add(z, ttnn.reshape(head._lin(si_d, pe_p + "linear_j.weight"), (1, 1, n, C_Z)))
    z = ttnn.add(z, head._lin(oh_d, pe_p + "linear_distance.weight"))
    rows = [{"stage": "embed_z", "rel_l2": rel_l2(dn(z).reshape(n, n, -1), their["embed_z"]),
             "ref_norm": float(their["embed_z"].norm())}]
    print("  %-12s rel_l2 %.4e" % ("embed_z", rows[-1]["rel_l2"]), flush=True)

    s = st_d
    for i, blk in enumerate(head.pf.blocks):
        s, z = blk(s, z)
        pairs = (("s", dn(s).reshape(-1, their["block%d_s" % i].shape[-1]),
                  their["block%d_s" % i]),
                 ("z", dn(z).reshape(n, n, -1), their["block%d_z" % i]))
        for tag, ours, ref in pairs:
            rows.append({"stage": "block%d_%s" % (i, tag), "rel_l2": rel_l2(ours, ref),
                         "ref_norm": float(ref.norm()), "our_norm": float(ours.norm())})
            print("  %-12s rel_l2 %.4e   |ref| %.6g  |ours| %.6g"
                  % (rows[-1]["stage"], rows[-1]["rel_l2"], rows[-1]["ref_norm"],
                     rows[-1]["our_norm"]), flush=True)

    report = {
        "instrument": "aux_heads forward bisection against upstream 0.4.3 float64",
        "arm": a.arm, "n_tokens": n, "use_zij_trunk_embedding": use_ztrunk,
        "boundary": str(a.boundary), "bar": 5.0e-2, "rows": rows,
        "first_stage_over_bar": next((r["stage"] for r in rows if r["rel_l2"] > 5e-2), None),
        "worst": max(rows, key=lambda r: r["rel_l2"]),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("first stage over the bar: %s" % report["first_stage_over_bar"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
