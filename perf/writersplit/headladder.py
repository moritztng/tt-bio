#!/usr/bin/env python3
"""The n-ladder A/B for the writer split, re-run on the tt-metal HEAD build.

Both arms run in ONE process, interleaved ship split ship split, each arm's program rebuilt by
clearing the program cache so the factory re-reads TTNN_MM_WRITER_ON_IN0. Each arm is warmed
after every cache clear so no JIT lands inside a ladder point. An A/A twin of the shipped arm is
measured in the same session: the ratio is only readable against this session's own floor. kt=4
and kt=32 both, since a variant that only helps at one K would be a shape artefact.

The n-ladder fits wall = L + c*n over n calls per synchronize, so c is the per-call cost with the
host sync floor pushed into the intercept L.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("CLK_DIR", "."))

import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
B, M, N = 16, 512, 512
GRID = ttnn.CoreGrid(y=10, x=11)


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b, max(abs(y - (my - b * mx + b * x)) for x, y in zip(xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="headladder.json")
    a = ap.parse_args()

    device = ttnn.open_device(device_id=0)
    held = clk.force(a.clock, clk.nodes_open_by_this_process())
    t0 = time.time()
    while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()):
        print("REFUSING to measure: %r" % (reached,))
        return 2
    print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)
    sampler = clk.Sampler(held[0])

    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)

    # The 1D mcast_in1 program config is pinned, not auto-selected: on tt-metal HEAD the auto
    # path picks a different factory for this shape and the A/B would compare two copies of the
    # same program. Both arms use the identical config, so the ratio is the relocation alone.
    shapes, configs = {}, {}
    for kt, K in ((4, 128), (32, 1024)):
        shapes[kt] = (dev(torch.randn(1, B, M, K) * 0.05), dev(torch.randn(K, N) * 0.05))
        configs[kt] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(GRID.x, GRID.y), in0_block_w=4,
            out_subblock_h=1, out_subblock_w=4,
            per_core_M=-(-(B * M // 32) // (GRID.x * GRID.y)), per_core_N=N // 32,
            fuse_batch=True, fused_activation=None, mcast_in0=False)

    def call(kt):
        x, w = shapes[kt]
        return ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DRAM,
                           dtype=ttnn.bfloat16, program_config=configs[kt])

    def ladder(kt, ns=(1, 2, 4, 8, 16)):
        pts = []
        for n in ns:
            ts = []
            for i in range(a.reps + a.warm):
                ttnn.synchronize_device(device)
                s = time.perf_counter()
                rs = [call(kt) for _ in range(n)]
                ttnn.synchronize_device(device)
                e = time.perf_counter()
                for r in rs:
                    ttnn.deallocate(r)
                if i >= a.warm:
                    ts.append((e - s) * 1e3)
            pts.append((n, st.median(ts)))
        L, c, res = fit([p[0] for p in pts], [p[1] for p in pts])
        return {"pts": pts, "L": L, "c": c, "resid": res}

    def build(flag):
        os.environ["TTNN_MM_WRITER_ON_IN0"] = flag
        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for kt in shapes:
            for _ in range(3):
                r = call(kt)
                ttnn.synchronize_device(device)
                ttnn.deallocate(r)

    R = {"clock_target": a.clock, "arms": {}}
    plan = []
    for _ in range(a.rounds):
        plan += [("ship", "0"), ("split", "1")]
    plan.append(("ship_aa", "0"))

    for name, flag in plan:
        build(flag)
        for kt in (4, 32):
            key = "%s_kt%d" % (name, kt)
            res = ladder(kt)
            R["arms"].setdefault(key, []).append(res)
            print("  %-16s c = %.5f ms  L = %.5f  resid %.5f"
                  % (key, res["c"], res["L"], res["resid"]), flush=True)

    R["clock"] = sampler.stop()
    print("AICLK during: %r" % (R["clock"],), flush=True)

    def best(key):
        return st.median([r["c"] for r in R["arms"][key]])

    for kt in (4, 32):
        s, p = best("ship_kt%d" % kt), best("split_kt%d" % kt)
        R["ratio_kt%d" % kt] = s / p
        print("kt=%-3d ship %.5f ms  split %.5f ms  ratio %.4fx" % (kt, s, p, s / p), flush=True)
    aa = R["arms"]["ship_aa_kt4"][0]["c"]
    R["aa_floor_pct"] = 100 * abs(best("ship_kt4") - aa) / aa
    print("A/A floor kt=4: ship %.5f vs ship_aa %.5f = %.3f %%"
          % (best("ship_kt4"), aa, R["aa_floor_pct"]), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
