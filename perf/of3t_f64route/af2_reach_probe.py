#!/usr/bin/env python3
"""of3t-f64route: does the host float64 gate reach AF2's two sites AT RUNTIME?

AF2 is not on `PREDICT_MODELS`. It is reached through `scripts/af2_port`, and its parameters are
the 4.67 GB `af2-params` archive, which is not on this host and cannot safely be fetched onto a
root filesystem at 100 % with 19 GiB free. So the fold-level A/B the other three models get is
not available for AF2, and this is the honest substitute: build the two modules with SYNTHETIC
weights of the right shapes, run one attention call through each, and count what arrives at the
gate.

It is a MODULE-LEVEL probe, not a fold, and the claim it supports is narrower: the two AF2 call
sites reach `host_softmax_or_none` at runtime with the site's own answer. It says nothing about
an AF2 fold's output, and it is labelled that way wherever it is quoted.

`TT_BIO_HOST_F64_SOFTMAX_AB=all` selects both sites, no tape is open, so every arrival is
`refused` and the device path runs -- which is also the inference guarantee at these two sites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tt_bio.tenstorrent as T                                          # noqa: E402
from tt_bio.af2 import AF2Attention                       # noqa: E402


def synth(names, shapes):
    return {n: torch.randn(s, dtype=torch.float32) * 0.05
            for n, s in zip(names, shapes)}


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    dev = T.get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    before = dict(T.HOST_F64_SOFTMAX_STATS)
    report = {"what": "a MODULE-LEVEL reach probe, not a fold: AF2 has no tt-bio predict entry "
                      "point and its 4.67 GB parameter archive is not on this host",
              "env": "TT_BIO_HOST_F64_SOFTMAX_AB=all",
              "before": before, "sites": {}}

    # ---- af2.msa: AF2Attention with a pair bias takes _fp32_softmax_attention -------------
    H, D, C, S, Z = 8, 32, 256, 64, 128
    w = synth(["layer_norm.weight", "layer_norm.bias",
               "linear_q.weight", "linear_k.weight", "linear_v.weight",
               "linear_g.weight", "linear_g.bias", "linear_o.weight", "linear_o.bias",
               "pair_norm.weight", "pair_norm.bias", "linear.weight"],
              [(C,), (C,), (H * D, C), (H * D, C), (H * D, C), (H * D, C), (H * D,),
               (C, H * D), (C,), (Z,), (Z,), (H, Z)])
    att = AF2Attention(w, ckc, n_heads=H, head_dim=D, pair_bias=True)
    q = ttnn.from_torch(torch.randn(1, H, S, D) * 0.1, layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    k = ttnn.from_torch(torch.randn(1, H, S, D) * 0.1, layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    v = ttnn.from_torch(torch.randn(1, H, S, D) * 0.1, layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(torch.randn(1, H, S, S) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    mid = dict(T.HOST_F64_SOFTMAX_STATS)
    o = att._attend(q, k, v, bias)
    after_msa = dict(T.HOST_F64_SOFTMAX_STATS)
    report["sites"]["af2.msa"] = {
        "selector": att._softmax_f64,
        "arrivals": (after_msa["refused"] + after_msa["declined"] + after_msa["served"]
                     - mid["refused"] - mid["declined"] - mid["served"]),
        "refused": after_msa["refused"] - mid["refused"],
        "tail": after_msa["tail"] - mid["tail"],
        "out_shape": [int(d) for d in o.shape],
    }
    for t in (q, k, v, bias, o):
        ttnn.deallocate(t)

    # ---- af2.tri_att: AF2PairBlock's two TriangleAttentions -------------------------------
    report["sites"]["af2.tri_att"] = {
        "selector_start": None, "selector_end": None,
        "note": "resolved at construction; the block needs the full AF2 pair-block weight set, "
                "so only the selector is read here",
    }
    try:
        tri = T.TriangleAttention(
            D, H, False, synth(
                ["layer_norm.weight", "layer_norm.bias", "linear_q.weight", "linear_k.weight",
                 "linear_v.weight", "linear_g.weight", "linear_g.bias", "linear_o.weight",
                 "linear_o.bias", "linear.weight"],
                [(Z,), (Z,), (H * D, Z), (H * D, Z), (H * D, Z), (H * D, Z), (H * D,),
                 (Z, H * D), (Z,), (H, Z)]),
            ckc, scale_pair_bias=False, fp32_softmax=True, bias_in_matmul="o",
            l1_padded_plan=True, softmax_site="af2.tri_att")
        report["sites"]["af2.tri_att"]["selector_start"] = tri._softmax_f64
    except Exception as exc:                                             # noqa: BLE001
        report["sites"]["af2.tri_att"]["construction_error"] = repr(exc)[:300]

    report["after"] = dict(T.HOST_F64_SOFTMAX_STATS)
    report["selected_per_site"] = dict(T.HOST_F64_SOFTMAX_SITES)
    report["reach"] = T.host_f64_softmax_reach()
    print(json.dumps(report, indent=1))
    if out:
        json.dump(report, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
