#!/usr/bin/env python3
"""Pass 2: the arms pass 1 was missing.

  * device fp32 STORAGE -- can this silicon supply an fp32 island at all?
  * a multi-op fp32 REGION -- upstream's `autocast(fp32)` blocks do not round back to
    bf16 between ops, so error compounds and the bf16 output floor is not the bar.
  * the softmax backward, which pass 1 lost to a leaf-tensor bug.
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
N, H_APB, H_TRI, D_HEAD, C_S = 384, 16, 4, 32, 384


def rel(a, ref):
    return float(torch.linalg.vector_norm(a.double() - ref.double())
                 / torch.linalg.vector_norm(ref.double()))


def bf(x):
    return x.to(torch.bfloat16)


def tt(x, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x.detach().clone(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def row(island, site, shape, upstream, arms):
    ROWS.append(dict(island=island, site=site, shape=[int(s) for s in shape],
                     upstream=upstream, arms=arms))
    print(f"\n== {island}   {site}   shape={tuple(shape)}   upstream={upstream}", flush=True)
    for k, v in arms.items():
        print(f"     {k:34s} {v:.6e}", flush=True)


def main():
    dev = ttnn.open_device(device_id=0)
    cfg = precise_config()
    try:
        # --- can the device supply an fp32 island? softmax in ttnn.float32 storage ------
        shp = (1, H_APB, N, N)
        x64 = torch.randn(*shp, dtype=torch.float64) * 3.0
        xb = bf(x64)
        ref = torch.softmax(xb.double(), dim=-1)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "host_fp32 (what upstream gets)": rel(torch.softmax(xb.float(), dim=-1), ref)}
        for label, dt, kw in (("tt_bf16_default", ttnn.bfloat16, {}),
                              ("tt_bf16_precise", ttnn.bfloat16, {"compute_kernel_config": cfg}),
                              ("tt_fp32_default", ttnn.float32, {}),
                              ("tt_fp32_precise", ttnn.float32, {"compute_kernel_config": cfg})):
            t = tt(xb.float(), dev, dtype=dt)
            y = ttnn.softmax(t, dim=-1, **kw)
            arms[label] = rel(ttnn.to_torch(y).reshape(shp), ref)
            ttnn.deallocate(y); ttnn.deallocate(t)
        row("softmax, fp32 STORAGE", "trunk AttentionPairBias", shp, "fp32 region in 0.5.0", arms)

        # --- and for layer_norm, the most-executed island -------------------------------
        shp = (1, N, C_S)
        x64 = torch.randn(*shp, dtype=torch.float64)
        g64 = torch.randn(C_S, dtype=torch.float64) * 0.5 + 1.0
        b64 = torch.randn(C_S, dtype=torch.float64) * 0.1
        xb, gb, bb = bf(x64), bf(g64), bf(b64)
        ref = torch.nn.functional.layer_norm(xb.double(), (C_S,), gb.double(), bb.double(), 1e-5)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "host_fp32 (what upstream gets)": rel(
                    torch.nn.functional.layer_norm(xb.float(), (C_S,), gb.float(), bb.float(), 1e-5), ref)}
        for label, dt in (("tt_bf16_precise", ttnn.bfloat16), ("tt_fp32_precise", ttnn.float32)):
            t, tg, tb = (tt(xb.float(), dev, dt), tt(gb.reshape(1, -1).float(), dev, dt),
                         tt(bb.reshape(1, -1).float(), dev, dt))
            y = ttnn.layer_norm(t, weight=tg, bias=tb, epsilon=1e-5, compute_kernel_config=cfg)
            arms[label] = rel(ttnn.to_torch(y).reshape(shp), ref)
            for h in (y, t, tg, tb):
                ttnn.deallocate(h)
        row("layer_norm, fp32 STORAGE", "single track", shp, "fp32", arms)

        # --- a multi-op fp32 REGION: triangle attention, no bf16 rounding inside --------
        q64 = torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64)
        k64 = torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64)
        v64 = torch.randn(16, H_TRI, N, D_HEAD, dtype=torch.float64)
        z64 = torch.randn(16, H_TRI, N, N, dtype=torch.float64) * 0.5
        qb, kb, vb, zb = bf(q64), bf(k64), bf(v64), bf(z64)
        scale = 1.0 / (D_HEAD ** 0.5)

        def attn(dtype, round_between=False):
            q, k, v, z = (t.to(dtype) for t in (qb, kb, vb, zb))
            s = (q @ k.transpose(-1, -2)) * scale
            if round_between:
                s = bf(s).to(dtype)
            s = s + z
            w = torch.softmax(s, dim=-1)
            if round_between:
                w = bf(w).to(dtype)
            return w @ v

        ref = attn(torch.float64)
        arms = {"floor_bf16_out": rel(bf(ref), ref),
                "up_bf16 region (0.4.3)": rel(attn(torch.bfloat16), ref),
                "up_fp32 region (0.5.0, no rounding)": rel(attn(torch.float32), ref),
                "fp32 ops, bf16 between (island-by-island)": rel(attn(torch.float32, True), ref)}
        for label, dt in (("tt_bf16_precise", ttnn.bfloat16), ("tt_fp32_precise", ttnn.float32)):
            tq, tk, tv, tz = (tt(t.float(), dev, dt) for t in (qb, kb, vb, zb))
            s = ttnn.matmul(tq, ttnn.permute(tk, (0, 1, 3, 2)), compute_kernel_config=cfg)
            s = ttnn.multiply(s, scale)
            s = ttnn.add(s, tz)
            w = ttnn.softmax(s, dim=-1, compute_kernel_config=cfg)
            o = ttnn.matmul(w, tv, compute_kernel_config=cfg)
            arms[label] = rel(ttnn.to_torch(o).reshape(ref.shape), ref)
            for h in (s, w, o, tq, tk, tv, tz):
                ttnn.deallocate(h)
        row("fp32 REGION (4 ops)", "triangle attention q@kT -> +bias -> softmax -> @v",
            (16, H_TRI, N, D_HEAD), "bf16 in 0.4.3 / fp32 region in 0.5.0", arms)

        # --- softmax backward, fixed ----------------------------------------------------
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
                "up_fp32 (0.5.0, fp32 grads)": rel(sm_bwd(torch.float32), ref_dx)}
        for label, dt in (("tt_autograd bf16", ttnn.bfloat16), ("tt_autograd fp32", ttnn.float32)):
            tx = ag.Tensor(tt(xb.float(), dev, dt), requires_grad=True)
            ag.softmax(tx, dim=-1).backward(tt(cb.float(), dev, dt))
            arms[label] = rel(ttnn.to_torch(tx.grad).reshape(shp), ref_dx)
        row("softmax BACKWARD dx", "trunk AttentionPairBias", shp,
            "bf16 acts / fp32 GRADS (master weights are fp32)", arms)

        # --- layer_norm backward in fp32 storage ----------------------------------------
        shp = (1, N, C_S)
        x64 = torch.randn(*shp, dtype=torch.float64)
        g64 = torch.randn(C_S, dtype=torch.float64) * 0.5 + 1.0
        b64 = torch.zeros(C_S, dtype=torch.float64)
        c64 = torch.randn(*shp, dtype=torch.float64) * 0.1
        xb, gb, bb, cb = bf(x64), bf(g64), bf(b64), bf(c64)

        def ln_bwd(dtype):
            x = xb.detach().clone().to(dtype).requires_grad_(True)
            g = gb.detach().clone().to(dtype).requires_grad_(True)
            b = bb.detach().clone().to(dtype).requires_grad_(True)
            torch.nn.functional.layer_norm(x, (C_S,), g, b, 1e-5).backward(
                cb.detach().clone().to(dtype))
            return g.grad, x.grad

        ref_dg, ref_dx = ln_bwd(torch.float64)
        up_dg, up_dx = ln_bwd(torch.float32)
        arms_g = {"floor_bf16_out": rel(bf(ref_dg), ref_dg), "up_fp32": rel(up_dg, ref_dg)}
        arms_x = {"floor_bf16_out": rel(bf(ref_dx), ref_dx), "up_fp32": rel(up_dx, ref_dx)}
        for label, dt in (("tt_autograd bf16", ttnn.bfloat16), ("tt_autograd fp32", ttnn.float32)):
            tx = ag.Tensor(tt(xb.float(), dev, dt), requires_grad=True)
            tg = ag.Tensor(tt(gb.reshape(1, -1).float(), dev, dt), requires_grad=True)
            tb = ag.Tensor(tt(bb.reshape(1, -1).float(), dev, dt), requires_grad=True)
            ag.layer_norm(tx, tg, tb, eps=1e-5).backward(tt(cb.float(), dev, dt))
            arms_g[label] = rel(ttnn.to_torch(tg.grad).reshape(-1), ref_dg)
            arms_x[label] = rel(ttnn.to_torch(tx.grad).reshape(shp), ref_dx)
        row("layer_norm BACKWARD d(gamma)", "D8's worst tensor class", shp, "fp32 grads", arms_g)
        row("layer_norm BACKWARD dx", "single track", shp, "fp32 grads", arms_x)

    finally:
        ttnn.close_device(dev)

    out = os.path.join(HERE, "islands_cost2.json")
    with open(out, "w") as fh:
        json.dump(ROWS, fh, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
