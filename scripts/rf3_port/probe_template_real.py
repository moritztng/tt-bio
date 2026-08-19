#!/usr/bin/env python3
"""Score the template embedder's pairformer sub-blocks on REAL input.

The sub-blocks pass in isolation on synthetic N(0,1) (>= 0.99980, better than the
trunk) and the chain fails on the real template-conditioned input (17-20% relative
RMS). The input is the only remaining variable, so this feeds the isolated
sub-blocks exactly what the reference feeds them, captured by hooking the INPUT to
each pairformer block rather than its output.

Reports the input's own statistics too: if a sub-block only misbehaves at a
particular scale or sparsity, that shows up here.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/probe_template_real.py \\
        --ckpt ... --golden /path/to/tmpl_io_on.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PREFIX = "shadow.recycler.template_embedder."
C = 64


def pcc(a, b) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()

    import ttnn

    from tt_bio._vendor.rf3.model.layers.attention import (
        TriangleAttention as RefTriAtt,
        TriangleMultiplication as RefTriMul,
    )
    from tt_bio._vendor.rf3.model.layers.layer_utils import Transition as RefTransition
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import RF3TemplateEmbedder
    from tt_bio.rf3.remap import remap_pairformer_block
    from tt_bio.tenstorrent import (
        TriangleAttention, TriangleMultiplication, Transition, get_device,
    )

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    block_sd = {k[len(PREFIX):]: v.float() for k, v in sd.items()
                if k.startswith(PREFIX)}
    gold = torch.load(args.golden, weights_only=False)
    f_in, z_in = gold["in"][0], gold["in"][1]

    raw_dim = int(block_sd["emb_templ.weight"].shape[1])
    ref = RF3TemplateEmbedder(n_block=2, raw_template_dim=raw_dim, c_z=128, c=C,
                              p_drop=0.0)
    ref.load_state_dict(block_sd, strict=False)
    ref.eval()

    captured = {}

    def tap_in(name):
        def hook(_m, inputs, _o):
            captured[name] = inputs[1].detach().float()   # (S_I, Z_II) -> Z_II
        return hook

    handles = [ref.pairformer[i].register_forward_pre_hook(
        lambda m, i_, n=f"in.{i}": captured.__setitem__(n, i_[1].detach().float()))
        for i in range(2)]
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            ref(dict(f_in), z_in.clone())
    finally:
        for h in handles:
            h.remove()

    real = captured[f"in.{args.block}"]
    # The reference sub-blocks want a leading batch dim; the hook captures the
    # unbatched [I, I, C] the template embedder passes around internally.
    if real.dim() == 3:
        real = real.unsqueeze(0)
    prefix = f"pairformer.{args.block}."
    sub_sd = {k[len(prefix):]: v for k, v in block_sd.items()
              if k.startswith(prefix)}
    mapped = remap_pairformer_block(sub_sd)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(
        x.float().unsqueeze(0) if x.dim() == 3 else x.float(),
        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def scope(p):
        return {k[len(p) + 1:]: v for k, v in mapped.items() if k.startswith(p + ".")}

    def ref_sub(mod, name, x, autocast):
        mod.load_state_dict({k[len(name) + 1:]: v for k, v in sub_sd.items()
                             if k.startswith(name + ".")}, strict=True)
        mod.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16,
                                             enabled=autocast):
            return mod(x).float()

    cases = [
        ("tri_mul_out", "tri_mul_outgoing",
         lambda: RefTriMul(d_pair=C, d_hidden=C, direction="outgoing", bias=True),
         lambda w: TriangleMultiplication(False, w, cfg)),
        ("tri_mul_in", "tri_mul_incoming",
         lambda: RefTriMul(d_pair=C, d_hidden=C, direction="incoming", bias=True),
         lambda w: TriangleMultiplication(True, w, cfg)),
        ("tri_att_start", "tri_attn_start",
         lambda: RefTriAtt(C, n_head=4, d_hidden=C, start_node=True),
         lambda w: TriangleAttention(C, 4, False, w, cfg, scale_pair_bias=False,
                                     fp32_softmax=True)),
        ("tri_att_end", "tri_attn_end",
         lambda: RefTriAtt(C, n_head=4, d_hidden=C, start_node=False),
         lambda w: TriangleAttention(C, 4, True, w, cfg, scale_pair_bias=False,
                                     fp32_softmax=True, transpose_bias=False)),
        ("transition_z", "z_transition",
         lambda: RefTransition(c=C, n=4),
         lambda w: Transition(w, cfg)),
    ]

    out = {}
    for tt_name, ref_name, make_ref, make_tt in cases:
        try:
            rb = ref_sub(make_ref(), ref_name, real.clone(), autocast=True)
            rf = ref_sub(make_ref(), ref_name, real.clone(), autocast=False)
            got = torch.Tensor(ttnn.to_torch(
                make_tt(scope(tt_name))(to_tt(real)))).float().reshape(rb.shape)
            diff = (got - rb).abs()
            out[tt_name] = {
                "device": round(pcc(got, rb), 6),
                "ceiling": round(pcc(rb, rf), 6),
                "maxabs": round(float(diff.max()), 6),
                "rel_rms": round(float(diff.pow(2).mean().sqrt() / rb.std()), 6),
            }
        except Exception as exc:
            out[tt_name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(json.dumps({tt_name: out[tt_name]}), flush=True)

    print(json.dumps({
        "block": args.block,
        "input_stats": {
            "shape": list(real.shape),
            "std": round(float(real.std()), 6),
            "absmax": round(float(real.abs().max()), 6),
            "frac_exact_zero": round(float((real == 0).float().mean()), 6),
        },
        "sub_blocks": out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
