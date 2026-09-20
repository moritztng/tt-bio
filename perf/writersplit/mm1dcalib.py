#!/usr/bin/env python3
"""Calibrate the one number the chooser transcription cannot read: ttnn's live L1 budget.

`get_max_l1_space` reads `device->lowest_occupied_compute_l1_address()`, which Python cannot see,
and it is the only input to `create_matmul_1d_systolic_array_program_config` this module has to
guess.  Guessing it low made the fold's digest move: `mm1dfold.py` measured the transcription arm
and the split arm writing the SAME CIF digest as each other and a DIFFERENT one from the shipped
arm, which localises the difference to the block config and not to the split.

So calibrate it.  For each shape the fold actually routes, run `ttnn.linear(core_grid=...)` -- the
auto path, which resolves the budget in C++ -- and `ttnn.linear(program_config=mine)` at a
candidate budget, and compare bit-exactly.  A different out_block groups the accumulation
differently and lands on different bf16 roundings, so equality across every shape at one budget
identifies it.  Equality is evidence rather than proof, which is why it is taken over all
eighteen shapes at once and not one.
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

BUDGETS = [700_000, 900_000, 1_000_000, 1_100_000, 1_200_000, 1_250_000, 1_300_000,
           1_350_000, 1_400_000, 1_420_000, 1_440_000, 1_460_000, 1_500_000, 1_600_000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(HERE / "mm1dfold.jsonl"))
    ap.add_argument("--tag", default=None,
                    help="which mm1dfold record to take the shape list from; default is the last "
                         "tr arm. Pass a --wide run's tag to calibrate the wider set.")
    ap.add_argument("--out", default=str(HERE / "mm1dcalib.json"))
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T
    from tt_bio.mm1d_generic import systolic_1d_config

    recs = [json.loads(l) for l in open(a.jsonl)]
    picked = [x for x in recs if (x["tag"] == a.tag if a.tag else x["arm"] == "tr")]
    if not picked:
        print("no record in %s matching %r" % (a.jsonl, a.tag or "arm=tr"))
        return 2
    shapes = [(s["a"], s["b"], s["n"]) for s in picked[-1]["route_shapes"]
              if s["a"][-1] == s["b"][-2]]   # a transposed pair is not this path's to answer
    print("%d shapes from %s (wide=%s)" % (len(shapes), picked[-1]["tag"],
                                           picked[-1].get("wide")))

    device = T.get_device()
    G = T.CORE_GRID_MAIN
    grid = (G.x, G.y)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    DR = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)

    ops = []
    for ash, bsh, n in shapes:
        x = ttnn.from_torch(torch.randn(*ash) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DR)
        w = ttnn.from_torch(torch.randn(*bsh) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DR)
        ref = ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DR,
                          dtype=ttnn.bfloat16, core_grid=G)
        ttnn.synchronize_device(device)
        ops.append((ash, bsh, n, x, w, ttnn.to_torch(ref)))
        ttnn.deallocate(ref)

    R = {"grid": grid, "budgets": {}}
    for budget in BUDGETS:
        rows, ok = [], 0
        for ash, bsh, n, x, w, ref in ops:
            pc, mcast_in0 = systolic_1d_config(x, w, grid, True, ttnn.bfloat16, budget=budget)
            if pc is None or mcast_in0:
                rows.append({"a": ash, "b": bsh, "n": n, "cfg": None, "equal": False})
                continue
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid), in0_block_w=pc[1],
                out_subblock_h=pc[2], out_subblock_w=pc[3], out_block_h=pc[4], out_block_w=pc[5],
                per_core_M=pc[6], per_core_N=pc[7], fuse_batch=True, mcast_in0=False)
            try:
                got = ttnn.linear(x, w, program_config=cfg, compute_kernel_config=KC,
                                  memory_config=DR, dtype=ttnn.bfloat16)
                ttnn.synchronize_device(device)
                eq = bool(torch.equal(ref, ttnn.to_torch(got)))
                ttnn.deallocate(got)
            except Exception as e:
                eq, cfg = False, str(e)[:70]
            ok += eq
            rows.append({"a": ash, "b": bsh, "n": n,
                         "out_block": [pc[4], pc[5]], "subblock": [pc[2], pc[3]],
                         "per_core": [pc[6], pc[7]], "equal": eq})
        calls = sum(r["n"] for r in rows if r["equal"])
        tot = sum(r["n"] for r in rows)
        R["budgets"][budget] = {"shapes_equal": ok, "shapes": len(rows),
                                "calls_equal": calls, "calls": tot, "rows": rows}
        print("budget %9d: %2d/%2d shapes bit-exact with the auto path, %d/%d calls"
              % (budget, ok, len(rows), calls, tot), flush=True)

    best = max(R["budgets"], key=lambda b: (R["budgets"][b]["shapes_equal"],
                                            R["budgets"][b]["calls_equal"]))
    R["best_budget"] = best
    print("\nbest %d: %d/%d shapes, %d/%d calls"
          % (best, R["budgets"][best]["shapes_equal"], R["budgets"][best]["shapes"],
             R["budgets"][best]["calls_equal"], R["budgets"][best]["calls"]))
    for r in R["budgets"][best]["rows"]:
        if not r["equal"]:
            print("  still differs: n=%-6d a=%s b=%s %s" % (r["n"], r["a"], r["b"],
                                                            r.get("out_block")))
    Path(a.out).write_text(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
