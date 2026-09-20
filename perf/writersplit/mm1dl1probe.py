#!/usr/bin/env python3
"""Does occupying L1 change which block config ttnn picks?  That is the whole wide question.

`mm1dcalib.py` found the transcribed chooser bit-exact with ttnn's auto path on all 23 routed
shapes at every candidate budget from 700 kB to 1.6 MB -- and it built every operand in DRAM.
`get_max_l1_space` reads `device->lowest_occupied_compute_l1_address()`, so with L1 empty the
budget is not binding and the calibration was measuring a regime the wide arm never runs in.

This puts the same matmul under an L1 occupancy that rises tensor by tensor and asks, at each
level, whether `ttnn.linear(core_grid=...)` and `ttnn.linear(program_config=mine)` still agree
bit-exactly.  If they stop agreeing as L1 fills, the auto path is choosing a different block
config than the transcription can know about, and the wide extension cannot be made safe from
Python.  If they keep agreeing, the wide digest shift is something else and this rules the
chooser out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

#: Shapes the fold routes, spanning small and large per_core_M.
CASES = [([1, 140, 32, 128], [128, 256]),
         ([1, 140, 128, 128], [128, 256]),
         ([1, 512, 512, 128], [128, 128])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "mm1dl1probe.json"))
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T
    from tt_bio.mm1d_generic import systolic_1d_config

    device = T.get_device()
    G = T.CORE_GRID_MAIN
    grid = (G.x, G.y)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    torch.manual_seed(0)

    def probe(ash, bsh, mem, ballast):
        x = ttnn.from_torch(torch.randn(*ash) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        w = ttnn.from_torch(torch.randn(*bsh) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        try:
            ref = ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DR,
                              dtype=ttnn.bfloat16, core_grid=G)
            ttnn.synchronize_device(device)
            tr = ttnn.to_torch(ref)
            ttnn.deallocate(ref)
        except Exception as e:
            return {"auto": str(e)[:80]}
        pc, mc0 = systolic_1d_config(x, w, grid, True, ttnn.bfloat16)
        if pc is None or mc0:
            return {"cfg": None}
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid), in0_block_w=pc[1],
            out_subblock_h=pc[2], out_subblock_w=pc[3], out_block_h=pc[4], out_block_w=pc[5],
            per_core_M=pc[6], per_core_N=pc[7], fuse_batch=True, mcast_in0=False)
        try:
            got = ttnn.linear(x, w, program_config=cfg, compute_kernel_config=KC,
                              memory_config=DR, dtype=ttnn.bfloat16)
            ttnn.synchronize_device(device)
            eq = bool(torch.equal(tr, ttnn.to_torch(got)))
            ttnn.deallocate(got)
        except Exception as e:
            return {"mine": str(e)[:80], "out_block": [pc[4], pc[5]]}
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return {"equal": eq, "out_block": [pc[4], pc[5]], "per_core": [pc[6], pc[7]],
                "ballast_tiles": ballast}

    R = {"grid": grid, "runs": []}
    ballast = []
    # 64 KB of L1 per step, per core, so the occupancy climbs the way a fold's does.
    for step in range(0, 9):
        if step:
            try:
                ballast.append(ttnn.from_torch(
                    torch.zeros(1, 1, 32, 32 * 110 * 2), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1))
            except Exception as e:
                print("ballast step %d failed: %s" % (step, str(e)[:70]), flush=True)
                break
        for mem, tag in ((DR, "dram"), (L1, "l1")):
            for ash, bsh in CASES:
                r = probe(ash, bsh, mem, step)
                r.update({"a": ash, "b": bsh, "operands": tag, "ballast_step": step})
                R["runs"].append(r)
                print("step %d %-4s a=%-22s equal=%s out_block=%s %s"
                      % (step, tag, ash, r.get("equal"), r.get("out_block"),
                         r.get("auto") or r.get("mine") or ""), flush=True)

    bad = [r for r in R["runs"] if r.get("equal") is False]
    print("\n%d of %d probes disagree with the auto path" % (len(bad), len(R["runs"])))
    Path(a.out).write_text(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
