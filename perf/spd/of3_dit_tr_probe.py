#!/usr/bin/env python3
"""OpenFold3's Wormhole --fast DiT transition and attention gate, as the fused block runs them, against cheaper op shapes.

dmmw (2026-10-21) put minimal_matmul out (slower than ttnn.linear at every DiT site) but read the fused epilogues as
real cost: in the pcen2 fold census the silu linear is 33.5 us against 24.9 plain (4,800 calls) and each sigmoid
linear 20.0 against 11.7 (19,200). This prices moving them, and merging the transition's two input linears:

  tr_site   b1 = linear(x, W_a, silu); b2 = linear(x, W_b); b1 * b2           (`_call_fused` today)
  tr_mulact b1 = linear(x, W_a); b2 = linear(x, W_b); multiply(b1, b2, silu on a)
  tr_cat    y = linear(x, [W_a | W_b]); multiply(y[..., :H], y[..., H:], silu on a)
  g_site    multiply(o, linear(x, W_g), sigmoid on b)                           (attention query gate today)
  g_epi     multiply(o, linear(x, W_g, sigmoid))

Traced us a call, rel. L2 against float64 of the same rounded operands. One chip, random operands.

    TT_VISIBLE_DEVICES=N python perf/spd/of3_dit_tr_probe.py --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import ops

with T.levers("dit_sigf32"):
    T._silu_f32_overlay()
    dev = T.get_device(trace="diffusion")
    torch.manual_seed(0)
    gx, gy = T.COMPUTE_GRID_MAIN
    res = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
           "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3], "cells": []}
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)

    def timed(fn, n):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        tid, out = T.capture_trace(dev, lambda: [fn() for _ in range(10)])
        for t in out:
            ttnn.deallocate(t)
        ts = []
        for _ in range(3):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(n // 10):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / (n // 10 * 10) * 1e6)
        ttnn.release_trace(dev, tid)
        return round(statistics.median(ts), 2)

    up = lambda t, d, mc=None: ttnn.from_torch(t, dtype=getattr(ttnn, d), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                               memory_config=mc)
    L1 = ttnn.L1_MEMORY_CONFIG
    M, K, H = a.rows, 768, 1536
    SILU, SIG = ttnn.UnaryOpType.SILU, ttnn.UnaryOpType.SIGMOID
    lin = lambda x, w, **k: ops.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,  # noqa: E731
                                       memory_config=L1, **k)
    d64 = lambda t: ttnn.to_torch(t).double()  # noqa: E731
    x = up(torch.randn(1, M, K), "bfloat16", L1)
    wa, wb = (up(torch.randn(K, H) / K ** 0.5, "bfloat8_b") for _ in range(2))
    wab = up(torch.cat([ttnn.to_torch(wa), ttnn.to_torch(wb)], -1).float(), "bfloat8_b")
    x64 = d64(x).reshape(M, K)
    ref_tr = torch.nn.functional.silu(x64 @ d64(wa)) * (x64 @ d64(wb))

    def tr_site():
        b1 = lin(x, wa, activation="silu"); b2 = lin(x, wb)
        o = ttnn.multiply(b1, b2, memory_config=L1); ttnn.deallocate(b1); ttnn.deallocate(b2); return o

    def tr_mulact():
        b1 = lin(x, wa); b2 = lin(x, wb)
        o = ttnn.multiply(b1, b2, input_tensor_a_activations=[SILU], memory_config=L1)
        ttnn.deallocate(b1); ttnn.deallocate(b2); return o

    def tr_cat():
        y = lin(x, wab)
        y1, y2 = y[:, :, :H], y[:, :, H:]
        o = ttnn.multiply(y1, y2, input_tensor_a_activations=[SILU], memory_config=L1)
        ttnn.deallocate(y); return o

    Kg, Ng = 768, 1024
    wg = up(torch.randn(Kg, Ng) / Kg ** 0.5, "bfloat8_b")
    og = up(torch.randn(1, M, Ng), "bfloat16", L1)
    ref_g = d64(og).reshape(M, Ng) * torch.sigmoid(x64 @ d64(wg))
    sig = T.mm_sigmoid(ckc)

    def g_site():
        g = lin(x, wg)
        o = ttnn.multiply(og, g, input_tensor_b_activations=[SIG], memory_config=L1); ttnn.deallocate(g); return o

    def g_epi():
        g = lin(x, wg, activation=sig)
        o = ttnn.multiply(og, g, memory_config=L1); ttnn.deallocate(g); return o

    for arm, fn, ref, calls in (("tr_site", tr_site, ref_tr, 4800), ("tr_mulact", tr_mulact, ref_tr, 4800),
                                ("tr_cat", tr_cat, ref_tr, 4800), ("g_site", g_site, ref_g, 4800),
                                ("g_epi", g_epi, ref_g, 4800)):
        c = {"arm": arm, "calls": calls}
        try:
            o = fn()
            got = d64(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)).reshape(M, -1)
            ttnn.deallocate(o)
            c["rel_l2"] = float((got - ref).norm() / ref.norm())
            c["us"] = timed(fn, a.reps)
        except Exception as e:  # noqa: BLE001
            c["error"] = f"{type(e).__name__}: {str(e)[:240]}"
        res["cells"].append(c)
        print(json.dumps(c), flush=True)
        a.out.write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    res["done"] = time.strftime("%FT%TZ", time.gmtime())
    a.out.write_text(json.dumps(res, indent=1))
    T.cleanup()
