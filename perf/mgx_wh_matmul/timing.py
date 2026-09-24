#!/usr/bin/env python3
"""Price in0_block_w on the triangle product: the band's block against a candidate, same process.

[1, C, S, S] x [1, C, S, S]^T, trimul's program config family, HiFi4 + fp32 dest + packer L1 acc,
bf16 out, arms interleaved rep by rep after one warm call each, AICLK sampled DURING.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from perf.clocksample import during  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kt", type=int, nargs="+", default=[16, 32, 48])
ap.add_argument("--channels", type=int, default=128)
ap.add_argument("--w", type=int, nargs="+", default=[1], help="candidate blocks beside the band's")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)
gx, gy = T.COMPUTE_GRID_MAIN
rows = []
with during() as clk:
    for kt in a.kt:
        s = kt * 32
        A = ttnn.from_torch(torch.randn(1, a.channels, s, s).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
        B = ttnn.from_torch(torch.randn(1, a.channels, s, s).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
        M, N = -(-kt // gy), -(-kt // gx)
        w0 = T._triangle_mul_program_config(kt).in0_block_w
        ws = [w0] + [w for w in a.w if w != w0 and kt % w == 0]
        pcs = {w: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
            out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N, transpose_mcast=False,
            fused_activation=None, fuse_batch=False) for w in ws}
        ms = {w: [] for w in ws}
        for rep in range(a.reps + 1):
            for w in ws:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=pcs[w], dtype=ttnn.bfloat16,
                                transpose_b=True)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) * 1e3
                ttnn.deallocate(o)
                if rep:
                    ms[w].append(dt)
        for w in ws:
            x = sorted(ms[w])
            r = {"kt": kt, "channels": a.channels, "w": w, "band": w == w0, "median_ms": round(x[len(x) // 2], 3),
                 "min_ms": round(x[0], 3), "tflops": round(2 * a.channels * s ** 3 / (x[len(x) // 2] * 1e-3) / 1e12, 2)}
            rows.append(r)
            print(json.dumps(r), flush=True)
        ttnn.deallocate(A)
        ttnn.deallocate(B)
print(clk.line(), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "grid": [gx, gy], "aiclk": clk.summary(),
                                       "rows": rows}, indent=1) + "\n")
