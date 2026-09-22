#!/usr/bin/env python3
"""The island map differs between OpenFold3 0.4.3 (tt-bio's pin) and 0.5.0, in exactly the
two places the campaign's defects live. One process, one draw per shape, every arm.

  0.4.3  normalization.py:64-75   LayerNorm DISABLES autocast and runs bf16 (weight.to(bf16))
  0.5.0  normalization.py:60-74   LayerNorm upcasts: x.float(), weight.float(), out.to(bf16)
  0.4.3  attention.py:151-163     fp32 region covers scores+bias+softmax only, value product out
  0.5.0  attention.py:158-167     region covers the value product too; pairformer.py:199 turns it on
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
N, H_APB, H_TRI, D_HEAD, C_S = 384, 16, 4, 32, 384
ROWS = []


def rel(a, ref):
    return float(torch.linalg.vector_norm(a.double() - ref.double())
                 / torch.linalg.vector_norm(ref.double()))


def bf(x):
    return x.to(torch.bfloat16)


def tt(x, dev, dt=ttnn.bfloat16):
    return ttnn.from_torch(x.detach().clone().float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def row(name, arms):
    ROWS.append(dict(row=name, arms=arms))
    print(f"\n== {name}", flush=True)
    for k, v in arms.items():
        print(f"     {k:38s} {v:.6e}", flush=True)


def main():
    dev = ttnn.open_device(device_id=0)
    cfg = precise_config()
    try:
        # ---- LayerNorm forward, single track ----------------------------------------
        x = bf(torch.randn(1, N, C_S, dtype=torch.float64))
        g = bf(torch.randn(C_S, dtype=torch.float64) * 0.5 + 1.0)
        b = bf(torch.randn(C_S, dtype=torch.float64) * 0.1)
        ln = torch.nn.functional.layer_norm
        ref = ln(x.double(), (C_S,), g.double(), b.double(), 1e-5)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "0.4.3 (bf16, autocast disabled)": rel(ln(x, (C_S,), g, b, 1e-5), ref),
                "0.5.0 (fp32 upcast, bf16 out)": rel(bf(ln(x.float(), (C_S,), g.float(), b.float(), 1e-5)), ref),
                "0.5.0 fp32 value, unrounded": rel(ln(x.float(), (C_S,), g.float(), b.float(), 1e-5), ref)}
        for lbl, dt in (("ours bf16 + precise_config", ttnn.bfloat16),
                        ("ours fp32 storage + precise", ttnn.float32)):
            t, tg, tb = tt(x, dev, dt), tt(g.reshape(1, -1), dev, dt), tt(b.reshape(1, -1), dev, dt)
            y = ttnn.layer_norm(t, weight=tg, bias=tb, epsilon=1e-5, compute_kernel_config=cfg)
            arms[lbl] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
            for h in (y, t, tg, tb):
                ttnn.deallocate(h)
        row("layer_norm FORWARD  [1,384,384]", arms)

        # ---- LayerNorm backward ------------------------------------------------------
        c = bf(torch.randn(1, N, C_S, dtype=torch.float64) * 0.1)
        b0 = bf(torch.zeros(C_S, dtype=torch.float64))

        def bwd(dtype):
            xx = x.detach().clone().to(dtype).requires_grad_(True)
            gg = g.detach().clone().to(dtype).requires_grad_(True)
            bb = b0.detach().clone().to(dtype).requires_grad_(True)
            ln(xx, (C_S,), gg, bb, 1e-5).backward(c.detach().clone().to(dtype))
            return gg.grad, xx.grad

        ref_dg, ref_dx = bwd(torch.float64)
        dg43, dx43 = bwd(torch.bfloat16)
        dg50, dx50 = bwd(torch.float32)
        ag_dg, ag_dx = {}, {}
        for lbl, dt in (("ours bf16 (as shipped)", ttnn.bfloat16),
                        ("ours fp32 storage", ttnn.float32)):
            tx = ag.Tensor(tt(x, dev, dt), requires_grad=True)
            tg = ag.Tensor(tt(g.reshape(1, -1), dev, dt), requires_grad=True)
            tb = ag.Tensor(tt(b0.reshape(1, -1), dev, dt), requires_grad=True)
            ag.layer_norm(tx, tg, tb, eps=1e-5).backward(tt(c, dev, dt))
            ag_dg[lbl] = rel(ttnn.to_torch(tg.grad).reshape(-1), ref_dg)
            ag_dx[lbl] = rel(ttnn.to_torch(tx.grad).reshape(ref_dx.shape), ref_dx)
        row("layer_norm BACKWARD d(gamma)  [384]",
            {"floor_bf16_out": rel(bf(ref_dg), ref_dg), "0.4.3 (bf16)": rel(dg43, ref_dg),
             "0.5.0 (fp32)": rel(dg50, ref_dg), **ag_dg})
        row("layer_norm BACKWARD dx  [1,384,384]",
            {"floor_bf16_out": rel(bf(ref_dx), ref_dx), "0.4.3 (bf16)": rel(dx43, ref_dx),
             "0.5.0 (fp32)": rel(dx50, ref_dx), **ag_dx})

        # ---- attention region, both upstream shapes ----------------------------------
        q = bf(torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64))
        k = bf(torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64))
        v = bf(torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64))
        z = bf(torch.randn(16, H_TRI, N, N, dtype=torch.float64) * 0.5)
        sc = 1.0 / (D_HEAD ** 0.5)

        def region(dtype, value_inside):
            qq, kk, vv, zz = (t.to(dtype) for t in (q, k, v, z))
            s = (qq @ kk.transpose(-1, -2)) * sc + zz
            w = torch.softmax(s, dim=-1)
            if not value_inside:                      # 0.4.3: leaves the fp32 region here
                w = w.to(torch.bfloat16).to(dtype)
            return w @ vv

        ref = region(torch.float64, True)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "0.4.3 (bf16 throughout)": rel(region(torch.bfloat16, True), ref),
                "0.5.0 (fp32 region, value inside)": rel(region(torch.float32, True), ref),
                "0.4.3 shape w/ fp32 scores only": rel(region(torch.float32, False), ref)}
        for lbl, dt in (("ours bf16 + precise_config", ttnn.bfloat16),
                        ("ours fp32 storage + precise", ttnn.float32)):
            tq, tk, tv, tz = (tt(t, dev, dt) for t in (q, k, v, z))
            s = ttnn.add(ttnn.multiply(ttnn.matmul(tq, ttnn.permute(tk, (0, 1, 3, 2)),
                                                   compute_kernel_config=cfg), sc), tz)
            w = ttnn.softmax(s, dim=-1, compute_kernel_config=cfg)
            o = ttnn.matmul(w, tv, compute_kernel_config=cfg)
            arms[lbl] = rel(ttnn.to_torch(o).reshape(ref.shape), ref)
            for h in (s, w, o, tq, tk, tv, tz):
                ttnn.deallocate(h)
        row("attention REGION  tri-att [16,4,384,32]", arms)

        # ---- softmax alone, forward and backward -------------------------------------
        s0 = bf(torch.randn(1, H_APB, N, N, dtype=torch.float64) * 3.0)
        cs = bf(torch.randn(1, H_APB, N, N, dtype=torch.float64) * 0.1)
        ref = torch.softmax(s0.double(), dim=-1)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "0.4.3 (softmax_no_cast, bf16)": rel(torch.softmax(s0, dim=-1), ref),
                "0.5.0 (fp32 region)": rel(torch.softmax(s0.float(), dim=-1), ref)}
        for lbl, dt, kw in (("ours no compute_kernel_config", ttnn.bfloat16, {}),
                            ("ours bf16 + precise_config", ttnn.bfloat16, {"compute_kernel_config": cfg}),
                            ("ours fp32 storage + precise", ttnn.float32, {"compute_kernel_config": cfg})):
            t = tt(s0, dev, dt)
            y = ttnn.softmax(t, dim=-1, **kw)
            arms[lbl] = rel(ttnn.to_torch(y).reshape(ref.shape), ref)
            ttnn.deallocate(y); ttnn.deallocate(t)
        row("softmax FORWARD  APB [1,16,384,384]", arms)

        def sbwd(dtype):
            xx = s0.detach().clone().to(dtype).requires_grad_(True)
            torch.softmax(xx, dim=-1).backward(cs.detach().clone().to(dtype))
            return xx.grad

        ref_dx = sbwd(torch.float64)
        arms = {"floor_bf16_out": rel(bf(ref_dx), ref_dx),
                "0.4.3 (bf16)": rel(sbwd(torch.bfloat16), ref_dx),
                "0.5.0 (fp32)": rel(sbwd(torch.float32), ref_dx)}
        for lbl, dt in (("ours bf16 (as shipped)", ttnn.bfloat16),
                        ("ours fp32 storage", ttnn.float32)):
            tx = ag.Tensor(tt(s0, dev, dt), requires_grad=True)
            ag.softmax(tx, dim=-1).backward(tt(cs, dev, dt))
            arms[lbl] = rel(ttnn.to_torch(tx.grad).reshape(ref_dx.shape), ref_dx)
        row("softmax BACKWARD dx  APB [1,16,384,384]", arms)

    finally:
        ttnn.close_device(dev)
    with open(os.path.join(HERE, "version_split.json"), "w") as fh:
        json.dump(ROWS, fh, indent=1)


if __name__ == "__main__":
    main()
