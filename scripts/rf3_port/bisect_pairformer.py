#!/usr/bin/env python3
"""Localise the RF3 -> tt-bio Pairformer PCC loss to a sub-block.

The whole block scores s_pcc 0.977 / z_pcc 0.824. That is either a remap error in
one sub-block or the bf16 conditioning ceiling; scoring each sub-block on the same
input separates them, and also reports a CPU bf16-vs-fp32 control so the ceiling is
measured rather than assumed.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/bisect_pairformer.py --ckpt ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

STACK = "shadow.recycler.pairformer_stack."
C_Z = 128


def pcc(a, b) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import ttnn

    from tt_bio._vendor.rf3.model.layers.attention import (
        TriangleAttention as RefTriAtt,
        TriangleMultiplication as RefTriMul,
    )
    from tt_bio._vendor.rf3.model.layers.layer_utils import Transition as RefTransition
    from tt_bio.rf3.remap import remap_pairformer_block
    from tt_bio.tenstorrent import (
        TriangleAttention, TriangleMultiplication, Transition, get_device,
    )

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    pre = f"{STACK}0."
    block = {k[len(pre):]: v.float() for k, v in sd.items() if k.startswith(pre)}
    mapped = remap_pairformer_block(block)

    torch.manual_seed(args.seed)
    z = torch.randn(1, args.n, args.n, C_Z)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.bfloat16)

    def scope(prefix):
        return {k[len(prefix) + 1:]: v for k, v in mapped.items()
                if k.startswith(prefix + ".")}

    def ref_sub(mod, sub, x, autocast):
        mod.load_state_dict({k[len(sub) + 1:]: v for k, v in block.items()
                             if k.startswith(sub + ".")}, strict=True)
        mod.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16,
                                             enabled=autocast):
            return mod(x).float()

    out = {}
    cases = [
        ("tri_mul_out", "tri_mul_outgoing",
         lambda: RefTriMul(d_pair=C_Z, d_hidden=128, direction="outgoing", bias=True),
         lambda w: TriangleMultiplication(False, w, cfg)),
        ("tri_mul_in", "tri_mul_incoming",
         lambda: RefTriMul(d_pair=C_Z, d_hidden=128, direction="incoming", bias=True),
         lambda w: TriangleMultiplication(True, w, cfg)),
        ("tri_att_start", "tri_attn_start",
         lambda: RefTriAtt(C_Z, n_head=4, d_hidden=32, start_node=True),
         lambda w: TriangleAttention(32, 4, False, w, cfg, scale_pair_bias=False,
                                     fp32_softmax=True)),
        ("tri_att_end", "tri_attn_end",
         lambda: RefTriAtt(C_Z, n_head=4, d_hidden=32, start_node=False),
         lambda w: TriangleAttention(32, 4, True, w, cfg, scale_pair_bias=False,
                                     fp32_softmax=True)),
        ("transition_z", "z_transition",
         lambda: RefTransition(c=C_Z, n=4),
         lambda w: Transition(w, cfg)),
    ]

    for tt_name, ref_name, make_ref, make_tt in cases:
        try:
            ref_bf16 = ref_sub(make_ref(), ref_name, z.clone(), autocast=True)
            ref_fp32 = ref_sub(make_ref(), ref_name, z.clone(), autocast=False)
            mod = make_tt(scope(tt_name))
            got = torch.Tensor(ttnn.to_torch(mod(to_tt(z)))).float().reshape(ref_bf16.shape)
            out[tt_name] = {
                "device_vs_bf16": round(pcc(got, ref_bf16), 6),
                "device_vs_fp32": round(pcc(got, ref_fp32), 6),
                "cpu_bf16_vs_fp32": round(pcc(ref_bf16, ref_fp32), 6),
                "ref_std": round(float(ref_fp32.std()), 4),
            }
        except Exception as exc:
            out[tt_name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(json.dumps({tt_name: out[tt_name]}), flush=True)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
