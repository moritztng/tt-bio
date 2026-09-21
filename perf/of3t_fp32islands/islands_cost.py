#!/usr/bin/env python3
"""Per-island precision cost for OpenFold3's bf16-mixed training step.

Protocol. Every arm sees the SAME bf16 input values; the reference is the op evaluated in
float64 on those identical values, so what is scored is the OP, not the input rounding.
    rel = ||arm - ref||_2 / ||ref||_2     (relative L2 == relative RMS at equal count)

`floor_bf16_out` is the reference rounded to bf16 and scored the same way: the price of
storing the island's OUTPUT in bf16, which upstream pays at every island it upcasts and then
casts back. An arm at that floor is as good as the island can be inside a bf16-mixed step.
"""
import json
import os
import sys

import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tt_bio.autograd import precise_config  # noqa: E402
import tt_bio.autograd as ag  # noqa: E402

torch.manual_seed(20260921)
ROWS = []
N = 384          # the training crop
C_Z, C_S = 128, 384
H_APB, H_TRI, D_HEAD = 16, 4, 32


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return float(torch.linalg.vector_norm(a - ref) / torch.linalg.vector_norm(ref))


def bf(x):
    return x.to(torch.bfloat16)


def tt(x, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def row(island, site, shape, upstream, arms):
    ROWS.append(dict(island=island, site=site, shape=[int(s) for s in shape],
                     upstream=upstream, arms=arms))
    print(f"\n== {island}   {site}   shape={tuple(shape)}   upstream={upstream}", flush=True)
    for k, v in arms.items():
        print(f"     {k:26s} {v:.6e}", flush=True)


def main():
    dev = ttnn.open_device(device_id=0)
    cfg = precise_config()
    try:
        # ---------------- island: softmax (attention weights) --------------------------
        for site, shp, h in (("trunk AttentionPairBias", (1, H_APB, N, N), H_APB),
                             ("triangle attention (16 of 384 i-slices)", (16, H_TRI, N, N), H_TRI)):
            x64 = (torch.randn(*shp, dtype=torch.float64) * 3.0)
            xb = bf(x64)
            ref = torch.softmax(xb.double(), dim=-1)
            arms = {
                "floor_bf16_out": rel(bf(ref), ref),
                "up_bf16 (0.4.3 softmax_no_cast)": rel(torch.softmax(xb, dim=-1), ref),
                "up_fp32 (0.5.0 high-prec APB)": rel(torch.softmax(xb.float(), dim=-1), ref),
            }
            t = tt(xb, dev)
            y = ttnn.softmax(t, dim=-1)
            arms["tt_default (no ckc)"] = rel(ttnn.to_torch(y), ref)
            ttnn.deallocate(y)
            y = ttnn.softmax(t, dim=-1, compute_kernel_config=cfg)
            arms["tt_precise_config"] = rel(ttnn.to_torch(y), ref)
            ttnn.deallocate(y)
            ttnn.deallocate(t)
            row("softmax", site, shp, "bf16 in 0.4.3 / fp32 in 0.5.0", arms)

        # ---------------- island: layer_norm ------------------------------------------
        for site, shp, c in (("single track s [1,384,384]", (1, N, C_S), C_S),
                             ("pair track z [1,384,384,128]", (1, N, N, C_Z), C_Z)):
            x64 = torch.randn(*shp, dtype=torch.float64)
            g64 = torch.randn(c, dtype=torch.float64) * 0.5 + 1.0
            b64 = torch.randn(c, dtype=torch.float64) * 0.1
            xb, gb, bb = bf(x64), bf(g64), bf(b64)
            ref = torch.nn.functional.layer_norm(xb.double(), (c,), gb.double(), bb.double(), 1e-5)
            arms = {
                "floor_bf16_out": rel(bf(ref), ref),
                "up_fp32 (their .float() LN)": rel(
                    torch.nn.functional.layer_norm(xb.float(), (c,), gb.float(), bb.float(), 1e-5), ref),
                "up_fp32_then_bf16_out": rel(bf(
                    torch.nn.functional.layer_norm(xb.float(), (c,), gb.float(), bb.float(), 1e-5)), ref),
                "bf16_compute (not theirs)": rel(
                    torch.nn.functional.layer_norm(xb, (c,), gb, bb, 1e-5), ref),
            }
            t, tg, tb = tt(xb, dev), tt(gb.reshape(1, -1), dev), tt(bb.reshape(1, -1), dev)
            y = ttnn.layer_norm(t, weight=tg, bias=tb, epsilon=1e-5)
            arms["tt_default (no ckc)"] = rel(ttnn.to_torch(y).reshape(shp), ref)
            ttnn.deallocate(y)
            y = ttnn.layer_norm(t, weight=tg, bias=tb, epsilon=1e-5, compute_kernel_config=cfg)
            arms["tt_precise_config"] = rel(ttnn.to_torch(y).reshape(shp), ref)
            ttnn.deallocate(y)
            for h in (t, tg, tb):
                ttnn.deallocate(h)
            row("layer_norm", site, shp, "fp32 (explicit .float(), and autocast fp32 list)", arms)

        # ---------------- island: sum / mean reductions --------------------------------
        shp = (1, N, N, C_Z)
        x64 = torch.randn(*shp, dtype=torch.float64)
        xb = bf(x64)
        ref = xb.double().sum(dim=-1, keepdim=True)
        arms = {
            "floor_bf16_out": rel(bf(ref), ref),
            "up_fp32 (autocast fp32_set_opt_dtype)": rel(xb.float().sum(dim=-1, keepdim=True), ref),
            "bf16_compute (not theirs)": rel(xb.sum(dim=-1, keepdim=True), ref),
        }
        t = tt(xb, dev)
        y = ttnn.sum(t, dim=-1, keepdim=True)
        arms["tt_default (no ckc)"] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
        ttnn.deallocate(y)
        y = ttnn.sum(t, dim=-1, keepdim=True, compute_kernel_config=cfg)
        arms["tt_precise_config"] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
        ttnn.deallocate(y)
        ttnn.deallocate(t)
        row("sum (reduction)", "pair channel reduction, 128 terms", shp,
            "fp32 (autocast sum/prod/cumsum/linalg_vector_norm)", arms)

        # ---------------- island: unary fp32 list --------------------------------------
        shp = (1, N, C_S)
        x64 = torch.rand(*shp, dtype=torch.float64) * 4.0 + 0.25
        xb = bf(x64)
        for name, fn, ttfn in (("exp", torch.exp, ttnn.exp),
                               ("rsqrt", torch.rsqrt, ttnn.rsqrt),
                               ("reciprocal", torch.reciprocal, ttnn.reciprocal)):
            ref = fn(xb.double())
            arms = {
                "floor_bf16_out": rel(bf(ref), ref),
                "up_fp32 (autocast fp32 list)": rel(fn(xb.float()), ref),
                "bf16_compute (not theirs)": rel(fn(xb), ref),
            }
            t = tt(xb, dev)
            y = ttfn(t)
            arms["tt_default (no ckc)"] = rel(ttnn.to_torch(y).reshape(shp), ref)
            ttnn.deallocate(y)
            ttnn.deallocate(t)
            row(name, "elementwise, single track", shp, "fp32 (autocast fp32 list)", arms)

        # ---------------- CONTROL: matmul, which is NOT an island -----------------------
        a64 = torch.randn(1, N, C_S, dtype=torch.float64)
        b64 = torch.randn(1, C_S, C_Z, dtype=torch.float64)
        ab, bb2 = bf(a64), bf(b64)
        ref = ab.double() @ bb2.double()
        arms = {
            "floor_bf16_out": rel(bf(ref), ref),
            "up_bf16 (autocast LOWER-precision list)": rel((ab @ bb2), ref),
            "fp32_compute (not theirs)": rel(ab.float() @ bb2.float(), ref),
        }
        ta, tb2 = tt(ab, dev), tt(bb2, dev)
        y = ttnn.matmul(ta, tb2)
        arms["tt_default (no ckc)"] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
        ttnn.deallocate(y)
        y = ttnn.matmul(ta, tb2, compute_kernel_config=cfg)
        arms["tt_precise_config"] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
        ttnn.deallocate(y)
        ttnn.deallocate(ta); ttnn.deallocate(tb2)
        row("matmul (CONTROL, not an island)", "single -> pair projection", (1, N, C_S),
            "bf16 (autocast lower-precision list)", arms)

        # ---------------- backward: the two islands D8's worst tensors sit in ------------
        # layer_norm affine backward -- `attn_pair_bias.layer_norm_a.weight` is D8's worst
        # tensor at blocks 23 and 47, and LN is upstream's most-executed fp32 island.
        shp = (1, N, C_S)
        x64 = torch.randn(*shp, dtype=torch.float64)
        g64 = torch.randn(C_S, dtype=torch.float64) * 0.5 + 1.0
        b64 = torch.zeros(C_S, dtype=torch.float64)
        c64 = torch.randn(*shp, dtype=torch.float64) * 0.1          # cotangent
        xb, gb, bb, cb = bf(x64), bf(g64), bf(b64), bf(c64)

        def ln_bwd(dtype):
            x = xb.to(dtype).requires_grad_(True)
            g = gb.to(dtype).requires_grad_(True)
            b = bb.to(dtype).requires_grad_(True)
            y = torch.nn.functional.layer_norm(x, (C_S,), g, b, 1e-5)
            y.backward(cb.to(dtype))
            return g.grad, x.grad

        ref_dg, ref_dx = ln_bwd(torch.float64)
        up_dg, up_dx = ln_bwd(torch.float32)
        bf_dg, bf_dx = ln_bwd(torch.bfloat16)

        tx = ag.Tensor(tt(xb, dev), requires_grad=True)
        tg = ag.Tensor(tt(gb.reshape(1, -1), dev), requires_grad=True)
        tbb = ag.Tensor(tt(bb.reshape(1, -1), dev), requires_grad=True)
        y = ag.layer_norm(tx, tg, tbb, eps=1e-5)
        y.backward(tt(cb, dev))
        tt_dg = ttnn.to_torch(tg.grad).reshape(-1)
        tt_dx = ttnn.to_torch(tx.grad).reshape(shp)
        row("layer_norm BACKWARD d(gamma)", "single track, one block's LN affine", shp,
            "fp32", {"floor_bf16_out": rel(bf(ref_dg), ref_dg),
                     "up_fp32": rel(up_dg, ref_dg),
                     "bf16_compute (not theirs)": rel(bf_dg, ref_dg),
                     "tt_autograd (as shipped)": rel(tt_dg, ref_dg)})
        row("layer_norm BACKWARD dx", "single track", shp, "fp32",
            {"floor_bf16_out": rel(bf(ref_dx), ref_dx),
             "up_fp32": rel(up_dx, ref_dx),
             "bf16_compute (not theirs)": rel(bf_dx, ref_dx),
             "tt_autograd (as shipped)": rel(tt_dx, ref_dx)})

        # softmax backward at the APB shape
        shp = (1, H_APB, N, N)
        x64 = torch.randn(*shp, dtype=torch.float64) * 3.0
        c64 = torch.randn(*shp, dtype=torch.float64) * 0.1
        xb, cb = bf(x64), bf(c64)

        def sm_bwd(dtype):
            x = xb.detach().clone().to(dtype).requires_grad_(True)
            torch.softmax(x, dim=-1).backward(cb.detach().clone().to(dtype))
            return x.grad

        ref_dx = sm_bwd(torch.float64)
        arms = {"floor_bf16_out": rel(bf(ref_dx), ref_dx),
                "up_bf16 (0.4.3)": rel(sm_bwd(torch.bfloat16), ref_dx),
                "up_fp32 (0.5.0)": rel(sm_bwd(torch.float32), ref_dx)}
        tx = ag.Tensor(tt(xb, dev), requires_grad=True)
        y = ag.softmax(tx, dim=-1)
        y.backward(tt(cb, dev))
        arms["tt_autograd (as shipped)"] = rel(ttnn.to_torch(tx.grad).reshape(shp), ref_dx)
        row("softmax BACKWARD dx", "trunk AttentionPairBias", shp,
            "bf16 in 0.4.3 / fp32 in 0.5.0", arms)

    finally:
        ttnn.close_device(dev)

    out = os.path.join(HERE, "islands_cost.json")
    with open(out, "w") as fh:
        json.dump(ROWS, fh, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
