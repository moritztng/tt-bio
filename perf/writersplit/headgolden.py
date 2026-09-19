#!/usr/bin/env python3
"""Golden check that survives stripping the A/B lever out of the build.

`--save` runs the shipped single-RISC program (TTNN_MM_WRITER_ON_IN0=0) and writes its outputs
to disk. `--check` runs whatever the build does by default and compares against that file with
torch.equal. The point is the final, unconditional build: once the measurement lever is gone
there is no shipped arm left in the binary to compare against in-process, so the reference has to
be carried across builds.

Same cases as headshapes.py, same pinned 1D mcast_in1 program config.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG

CASES = [
    (256, 4, 16, 1, 11, 10, True),
    (110, 4, 16, 1, 11, 10, True),
    (100, 2, 16, 1, 10, 1, True),
    (13, 3, 3, 1, 7, 1, True),
    (4, 2, 8, 1, 1, 1, True),
    (8, 2, 8, 3, 4, 1, False),
    (9, 5, 12, 2, 5, 1, False),
]


def subblocks(per_core_M, per_core_N, max_tiles=4):
    best = (1, 1)
    for h in range(1, per_core_M + 1):
        if per_core_M % h:
            continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w or h * w > max_tiles:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--path", default="/home/ttuser/wsh/golden_ship.pt")
    a = ap.parse_args()

    if a.save:
        os.environ["TTNN_MM_WRITER_ON_IN0"] = "0"
    device = ttnn.open_device(device_id=0)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {}
    ref = torch.load(a.path) if not a.save else None
    bad = 0
    for idx, (Mt, Kt, Nt, batch, gx, gy, fuse_batch) in enumerate(CASES):
        torch.manual_seed(1000 + idx)
        M, K, N = Mt * 32, Kt * 32, Nt * 32
        per_core_M = -(-Mt // (gx * gy))
        sh, sw = subblocks(per_core_M, Nt)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=Kt,
            out_subblock_h=sh, out_subblock_w=sw, per_core_M=per_core_M, per_core_N=Nt,
            fuse_batch=fuse_batch, fused_activation=None, mcast_in0=False)
        xt, wt = torch.randn(1, batch, M, K) * 0.05, torch.randn(K, N) * 0.05

        def dev(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM)

        x_d, w_d = dev(xt), dev(wt)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        outs = []
        for _ in range(2):
            r = ttnn.linear(x_d, w_d, compute_kernel_config=KC, memory_config=DRAM,
                            dtype=ttnn.bfloat16, program_config=pc)
            ttnn.synchronize_device(device)
            outs.append(ttnn.to_torch(r))
            ttnn.deallocate(r)
        tag = "Mt=%d Kt=%d Nt=%d b=%d %dx%d" % (Mt, Kt, Nt, batch, gx, gy)
        if a.save:
            out[idx] = outs[0]
            print("saved %s" % tag, flush=True)
        else:
            eq = torch.equal(outs[0], ref[idx]) and torch.equal(outs[1], ref[idx])
            d = (outs[0].float() - ref[idx].float()).abs().max().item()
            print("%-28s equal-to-shipped %-5s  maxdiff %.3e" % (tag, eq, d), flush=True)
            bad += 0 if eq else 1
        ttnn.deallocate(x_d)
        ttnn.deallocate(w_d)
    ttnn.close_device(device)
    if a.save:
        torch.save(out, a.path)
        print("wrote %s" % a.path, flush=True)
        return 0
    print("VERDICT: %d/%d cases bit-exact against the shipped program" % (len(CASES) - bad, len(CASES)), flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
