#!/usr/bin/env python3
"""Bit-exactness across the shapes the relocated writer's padding maths can break.

Every case is pinned to the 1D mcast_in1 program config, so the patched builder is the one that
runs; auto-selection on HEAD picks other factories for most shapes and would make the comparison
vacuous. The harness prints one MM1D-DESC marker line per program build, and the script refuses
to grade a shape whose two arms did not report writer_on_in0 = 0 then 1.

Cases: M and N off the per-core block multiple (H and W tails), a 1x1 grid where the mcast sender
core is the whole output, batch > 1 with fuse_batch off (the MtNt advance), and a subblock width
that does not divide per_core_N.
"""
from __future__ import annotations

import os
import sys

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG


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


# (Mt, Kt, Nt, batch, grid_x, grid_y, fuse_batch)
CASES = [
    (256, 4, 16, 1, 11, 10, True),    # the trimul key-A shape, H tail on the last cores
    (110, 4, 16, 1, 11, 10, True),    # no tail
    (100, 2, 16, 1, 10, 1, True),     # per_core_M exact, one row of cores
    (13, 3, 3, 1, 7, 1, True),        # N not a multiple of 4, ragged M
    (4, 2, 8, 1, 1, 1, True),         # single core, sender core is the whole output
    (8, 2, 8, 3, 4, 1, False),        # batch 3 unfused: exercises the MtNt advance
    (9, 5, 12, 2, 5, 1, False),       # batch 2 unfused with an H tail
]


def main() -> int:
    device = ttnn.open_device(device_id=0)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    bad = 0
    for (Mt, Kt, Nt, batch, gx, gy, fuse_batch) in CASES:
        M, K, N = Mt * 32, Kt * 32, Nt * 32
        ncores = gx * gy
        per_core_M = -(-Mt // ncores) if fuse_batch else -(-(Mt * batch) // ncores)
        if not fuse_batch:
            per_core_M = -(-Mt // ncores)
        per_core_N = Nt
        sh, sw = subblocks(per_core_M, per_core_N)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=Kt,
            out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=per_core_M, per_core_N=per_core_N,
            fuse_batch=fuse_batch, fused_activation=None, mcast_in0=False)
        xt = torch.randn(1, batch, M, K) * 0.05
        wt = torch.randn(K, N) * 0.05

        def dev(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM)

        x_d, w_d = dev(xt), dev(wt)

        def arm(flag):
            os.environ["TTNN_MM_WRITER_ON_IN0"] = flag
            device.disable_and_clear_program_cache()
            device.enable_program_cache()
            outs = []
            for _ in range(2):
                r = ttnn.linear(x_d, w_d, compute_kernel_config=KC, memory_config=DRAM,
                                dtype=ttnn.bfloat16, program_config=pc)
                ttnn.synchronize_device(device)
                outs.append(ttnn.to_torch(r))
                ttnn.deallocate(r)
            return outs

        tag = ("Mt=%-4d Kt=%-3d Nt=%-3d b=%d %dx%-2d pcM=%-3d sb=%dx%d fuse=%d"
               % (Mt, Kt, Nt, batch, gx, gy, per_core_M, sh, sw, int(fuse_batch)))
        print("SHAPE %s" % tag, flush=True)
        sys.stderr.flush()
        try:
            ship, split = arm("0"), arm("1")
        except Exception as e:  # noqa: BLE001
            print("  RAISED %s" % str(e).splitlines()[0][:160], flush=True)
            bad += 1
            continue
        eq = all(torch.equal(a, b) for a, b in zip(ship, split))
        selfeq = torch.equal(split[0], split[1])
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(ship, split))
        ref = (xt.to(torch.float32) @ wt.to(torch.float32))
        pcc = torch.corrcoef(torch.stack([
            split[0].float().flatten(), ref.reshape(split[0].shape).flatten()]))[0, 1].item()
        print("  equal %-5s  split self %-5s  maxdiff %.3e  split PCC %.6f"
              % (eq, selfeq, d, pcc), flush=True)
        bad += 0 if (eq and selfeq) else 1
        ttnn.deallocate(x_d)
        ttnn.deallocate(w_d)
    ttnn.close_device(device)
    print("VERDICT: %d/%d cases bit-exact" % (len(CASES) - bad, len(CASES)), flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
