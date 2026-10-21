#!/usr/bin/env python3
"""OpenFold3's Wormhole --fast DiT linears, as the block calls them, against minimal_matmul blocks.

The pcen2 census (WH fast, 1000 MHz) reads the nine `_lin` sites of the fused DiT block at 1.32 s a fold, 0.37-0.61
of roof, 4,800-9,600 calls each. Protenix-v2 moved its step linears to minimal_matmul for 1.04-1.35x (dit_step_mm),
but OpenFold3's block reaches ttnn.linear through `_lin`, not `DitLowp.linear`, so that lever never runs here.
This prices it per site before a lever is built. Traced us a call per arm:

  site       ops.linear exactly as `_lin` calls it (bf16 in L1, weight DRAM, HiFi2, fp32 dest, L1 out, bias,
             sigmoid via mm_sigmoid under dit_sigf32 or silu)
  site_nb    the same without bias and activation (what the epilogue costs)
  mm_Mb_Kb_Nb  minimal_matmul, same operands, ckc, bias and fused activation, L1 out. fp32 dest keeps K block 1
             (Wormhole's multi-K fp32 dest fault, tenstorrent.k1_linear)

with rel. L2 error against float64 of the same rounded operands (activation applied in float64). One chip,
random operands, no model weights.

    TT_VISIBLE_DEVICES=N python perf/spd/of3_dit_mm_probe.py --out r.json
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

# (name, K, N, weight dtype, bias, activation, calls a fold) from pcen2's linear rows of the fused block.
SITES = [("qkv", 768, 3072, "bfloat8_b", True, None, 4800),
         ("s_sig", 384, 768, "bfloat16", True, "sigmoid", 19200),
         ("s_plain", 384, 768, "bfloat16", False, None, 9600),
         ("la_silu", 768, 1536, "bfloat8_b", False, "silu", 4800),
         ("lb", 768, 1536, "bfloat8_b", False, None, 4800),
         ("lout", 1536, 768, "bfloat8_b", False, None, 4800),
         ("g", 768, 1024, "bfloat8_b", False, None, 4800),
         ("o", 1024, 768, "bfloat8_b", False, None, 4800)]
L1 = ttnn.L1_MEMORY_CONFIG

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
    ACT = {"sigmoid": ttnn.UnaryOpType.SIGMOID, "silu": ttnn.UnaryOpType.SILU}
    F64 = {None: lambda y: y, "sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu}
    M = a.rows
    for name, K, N, wd, has_b, act, calls in SITES:
        x = up(torch.randn(1, M, K), "bfloat16", L1)
        w = up(torch.randn(K, N) / K ** 0.5, wd)
        b = up(torch.randn(1, N) * 0.1, "bfloat16") if has_b else None
        y64 = ttnn.to_torch(x).double().reshape(M, K) @ ttnn.to_torch(w).double()
        if b is not None:
            y64 = y64 + ttnn.to_torch(b).double().reshape(1, N)[:, :N]
        ref = F64[act](y64)
        site_act = T.mm_sigmoid(ckc) if act == "sigmoid" else act
        arms = {"site": lambda: ops.linear(x, w, bias=b, activation=site_act, compute_kernel_config=ckc,
                                           core_grid=T.CORE_GRID_MAIN, memory_config=L1),
                "site_nb": lambda: ops.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                              memory_config=L1)}
        for Mb in (1, 2, 4, 8, 16):
            for Nb in (1, 2, 4, 8, 16):
                if Nb > N // 32:
                    continue
                sh, sw = max(((h, v) for h in (1, 2, 4) for v in (1, 2, 4) if Mb % h == 0 and Nb % v == 0 and h * v <= 4),
                             key=lambda p: (p[0] * p[1], p[1]))
                cfg = ttnn.MinimalMatmulConfig(M_block_size=Mb, K_block_size=1, N_block_size=Nb, subblock_h=sh,
                                               subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
                kw = {"fused_activation": (ACT[act], True)} if act else {}
                arms[f"mm_{Mb}_1_{Nb}"] = (lambda cfg=cfg, kw=kw: ttnn.experimental.minimal_matmul(
                    x, w, bias_tensor=b, compute_kernel_config=ckc, dtype=ttnn.bfloat16, memory_config=L1,
                    config=cfg, **kw))
        for arm, fn in arms.items():
            c = {"site": name, "K": K, "N": N, "calls": calls, "arm": arm}
            try:
                o = fn()
                got = ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)).double().reshape(M, -1)[:, :N]
                ttnn.deallocate(o)
                r = ref if arm != "site_nb" else y64 - (ttnn.to_torch(b).double().reshape(1, N) if b is not None else 0)
                c["rel_l2"] = float((got - r).norm() / r.norm())
                c["us"] = timed(fn, a.reps)
            except Exception as e:  # noqa: BLE001
                c["error"] = f"{type(e).__name__}: {str(e)[:240]}"
            res["cells"].append(c)
            print(json.dumps(c), flush=True)
            a.out.write_text(json.dumps(res, indent=1))
        for t in (x, w, b):
            if t is not None:
                ttnn.deallocate(t)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    res["done"] = time.strftime("%FT%TZ", time.gmtime())
    a.out.write_text(json.dumps(res, indent=1))
    T.cleanup()
