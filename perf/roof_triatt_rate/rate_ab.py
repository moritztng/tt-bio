#!/usr/bin/env python3
"""Why the catalogue prices TriangleAttention 1.61x above what the fold spends on it.

`perf/roof_quiet/QUIET_REFOLD.md` measured the class at 24.85 TFLOP/s inside a quiet 512 aa fold
and 15.45 TFLOP/s in `perf/roof_shape/`'s standalone arms, and read the gap as an isolated-arm
effect: a rate taken on one op alone being a lower bound on what the fold reaches when the op is
embedded. That reading has a cheaper rival. The catalogue's TriangleAttention arms are
`ttnn.linear` and `ttnn.transformer.scaled_dot_product_attention`, and the fold's
TriangleAttention issues neither -- its capture holds three `ttnn.generic_op` calls and no matmul
at all. The arms may simply be a different implementation of the same arithmetic.

The two readings predict opposite things about the SAME standalone measurement, so one session
separates them:

  triatt      the shipped TriangleAttention unit, alone, at the fold's 512 aa shape. The fold
              spends 4.676 ms on this call (`roof_quiet/out_quiet_trimmed/ROOF_BUDGET.md`, the
              32-call unmasked row; the 528-call masked row is 4.574 ms).
              isolated-arm reading -> materially slower than 4.676 ms
              wrong-kernel reading -> lands on it
  cat_sdpa    the catalogue's winning SDPA arm, `triatt_sdpa_q256`, verbatim
  ship_sdpa   the same arithmetic through the shipped dispatcher `_tri_att_sdpa`
  cat_in      the catalogue's winning in-projection arm, `triatt_in_flat`, verbatim
  ship_in     the same arithmetic through the shipped `triatt_qkv.qkvgb_heads`, the one pass over
              the normed pair tensor that writes q, k, v, the gate and the pair bias
  cat_out     the catalogue's winning output-projection arm, `pair_out128_flat`, verbatim
  ship_out    the same arithmetic through the shipped `triatt_qkv.out_proj`
  triatt_fq   the same unit with `TT_BIO_TRIATT_FUSE_QKV` on, which folds the qkv projection
              into the SDPA kernel so the projection program does not run at all. Built and
              merged by `roof-qkv-sdpa-build`, default OFF, measured there at 1.4982x / 1.5502x
              on pc. This is the one pre-built lever aimed at this class.
  cube        the dense 4096^3, so every rate here has this session's own denominator

Arms interleave per rep (`op-ab-must-interleave-arms-compile-warmup-bias`), each timed region is
bracketed by `ttnn.synchronize_device`, the first reps are warmup, and the statistic is the
minimum over reps. Harness shape follows `perf/roof_gate_epilogue/screen.py`, which already builds
a real `TriangleAttention` from random weights.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio import tenstorrent as T                                           # noqa: E402
from tt_bio import triatt_sdpa as TS                                          # noqa: E402
from tt_bio import triatt_qkv as TQ                                           # noqa: E402


def weights(c_z, n_heads, head_dim):
    torch.manual_seed(0)
    d = n_heads * head_dim
    return {
        "layer_norm.weight": torch.ones(c_z),
        "layer_norm.bias": torch.zeros(c_z),
        "linear_q.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_k.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_v.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_g.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_o.weight": torch.randn(c_z, d) * (d ** -0.5),
        "linear.weight": torch.randn(n_heads, c_z) * (c_z ** -0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", default="perf/roof_triatt_rate/rate_ab_512_qb2c3.json")
    a = ap.parse_args()
    S, C, H, D = a.n, a.c_z, a.heads, a.head_dim

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)

    meta = {"n": S, "c_z": C, "heads": H, "head_dim": D, "arch": str(dev.arch()),
            "grid": list(T.COMPUTE_GRID_MAIN), "card": os.environ.get("TT_VISIBLE_DEVICES"),
            "host": os.uname().nodename, "loadavg": os.getloadavg(),
            "reps": a.reps, "warm": a.warm}
    print(json.dumps(meta), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    att = T.TriangleAttention(D, H, False, weights(C, H, D), ckc)
    z = dram(torch.randn(1, S, S, C).to(torch.bfloat16) * 0.1)
    zflat = dram(torch.randn(S * S, C).to(torch.bfloat16) * 0.1)
    w544 = dram(torch.randn(C, 3 * H * D + C + 32).to(torch.bfloat16))
    w128 = dram(torch.randn(C, C).to(torch.bfloat16))
    q = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    k = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    v = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16) * 0.1)
    zn = dram(torch.randn(S, S, C).to(torch.bfloat16) * 0.1)   # the normed pair tensor
    cube_a = dram(torch.randn(4096, 4096).to(torch.bfloat16))
    cube_b = dram(torch.randn(4096, 4096).to(torch.bfloat16))

    cc = dev.compute_with_storage_grid_size()
    stock_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                      q_chunk_size=256, k_chunk_size=512)
    scale = D ** -0.5

    def f_triatt():
        TS._FUSE_QKV = False
        ttnn.deallocate(att(z))

    def f_triatt_fq():
        # The flag is read inside `sdpa_fused_qkv` at call time, so assigning it here is what
        # lets both arms interleave in one process -- the same pattern `TriangleAttention.
        # fused_hifi` documents for `_TRIATT_FUSED_HIFI`.
        TS._FUSE_QKV = True
        try:
            ttnn.deallocate(att(z))
        finally:
            TS._FUSE_QKV = False

    def f_cat_sdpa():
        ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale, program_config=stock_pc))

    def f_ship_sdpa():
        ttnn.deallocate(T._tri_att_sdpa(q, k, v, bias, scale))

    def f_cat_in():
        ttnn.deallocate(ttnn.linear(zflat, w544, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))

    def f_ship_in():
        r = TQ.qkvgb_heads(zn, att.qkvgb_weight, att.o_weight, ckc, H, D, ttnn.bfloat16,
                           T._qkv_mm_config(zn, att.qkvgb_weight),
                           int(att.bias_weight.shape[-1]))
        assert r is not None, "shipped in-projection declined"
        (qq, kk, vv), gg, bb = r
        for t_ in (qq, kk, vv, gg, bb):
            ttnn.deallocate(t_)

    def f_ship_out():
        ttnn.deallocate(TQ.out_proj(q, att.o_weight, ckc, ttnn.bfloat16))

    def f_cat_out():
        ttnn.deallocate(ttnn.linear(zflat, w128, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))

    def f_cube():
        ttnn.deallocate(ttnn.matmul(cube_a, cube_b, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))

    # FLOP each arm executes, so every row carries its own rate. The unit's three terms are the
    # census's own: in-projection [q|k|v|g] 128->544 minus the 32 bias columns the census counts
    # with it, the fused SDPA (QK^T and AV), and the 128->128 output projection.
    SDPA_F = 2 * 2 * (S * H) * S * S * D
    IN_F = 2 * S * S * C * (3 * H * D + C + 32)
    OUT_F = 2 * S * S * C * C
    UNIT_F = IN_F + SDPA_F + OUT_F
    arms = {
        "triatt": (f_triatt, UNIT_F),
        "triatt2": (f_triatt, UNIT_F),
        "triatt_fq": (f_triatt_fq, UNIT_F),
        "triatt_fq2": (f_triatt_fq, UNIT_F),
        "cat_sdpa": (f_cat_sdpa, SDPA_F),
        "ship_sdpa": (f_ship_sdpa, SDPA_F),
        "cat_in": (f_cat_in, IN_F),
        "ship_in": (f_ship_in, IN_F),
        "cat_out": (f_cat_out, OUT_F),
        "ship_out": (f_ship_out, OUT_F),
        "cube": (f_cube, 2 * 4096 ** 3),
    }
    ts = {kk: [] for kk in arms}
    for i in range(a.warm + a.reps):
        for name, (fn, _f) in arms.items():
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            if i >= a.warm:
                ts[name].append(dt)
        print("rep %d  %s" % (i, "  ".join("%s %.3fms" % (nm, 1e3 * ts[nm][-1])
                                           for nm in arms if ts[nm])), flush=True)

    res = {"meta": meta, "arms": {}}
    for name, (_fn, f) in arms.items():
        v_ = ts[name]
        res["arms"][name] = {"min_ms": 1e3 * min(v_), "median_ms": 1e3 * st.median(v_),
                             "GFLOP": f / 1e9, "TFLOPs_at_min": f / min(v_) / 1e12,
                             "reps": len(v_)}
    res["meta"]["loadavg_after"] = os.getloadavg()
    res["meta"]["fuse_rejects"] = dict(TS.FUSE_REJECTS)
    res["meta"]["sdpa_chunk_picks"] = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
    res["meta"]["sdpa_route_counts"] = dict(T.SDPA_ROUTE_COUNTS)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["arms"], indent=1))
    for one, two in (("triatt", "triatt2"), ("triatt_fq", "triatt_fq2")):
        aa, bb = res["arms"][one]["min_ms"], res["arms"][two]["min_ms"]
        print("A/A floor %-10s %.2f %%" % (one, 100 * abs(aa - bb) / min(aa, bb)))
    print("fuse_qkv speedup on the unit  %.4fx"
          % (res["arms"]["triatt"]["min_ms"] / res["arms"]["triatt_fq"]["min_ms"]))
    print("fuse rejects: %s" % json.dumps(res["meta"]["fuse_rejects"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
