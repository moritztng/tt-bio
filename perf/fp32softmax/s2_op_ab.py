#!/usr/bin/env python3
"""Per-shape bit-exactness and speed of the three fp32-softmax block plans, on the helper itself.

Arm A is today's default: the tuned 8x8 rectangle wherever it can divide an affordable block, the
free core count only where it cannot (S1, shipped). Arm B is S2 as pass 3 built it -- the tallest
block the byte budget affords, at any core count. Arm C is S2 rebuilt against the objective pass 3's
own measurement implied: the tallest block that still admits a batched-matmul program config for
both of the matmuls it feeds.

`rows` defaults to the padded token count, so the block count per call is the one a real trunk
triangle-attention call pays and not a fraction of it.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 python3 perf/fp32softmax/s2_op_ab.py \
        --out perf/fp32softmax/results/s2_op_ab.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (heads, S). The heads=4 row is the OF3/OpenBind/RF3 trunk; 2 and 8 are the other pair-track head
# counts the shared helper serves, taken at the sizes where arms B and C disagree.
CASES = [(4, 256), (4, 512), (4, 576), (4, 640), (4, 768),
         (2, 512), (2, 576), (2, 1024), (8, 512), (8, 576)]
HEAD_DIM = 32
ARMS = (("A", False, False), ("B", True, False), ("C", True, True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=0, help="0 = the padded token count")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    cases = CASES
    if args.cases:
        cases = [tuple(int(x) for x in c.split("x")) for c in args.cases.split(",")]

    import torch
    import ttnn
    from tt_bio import tenstorrent as tt

    torch.set_grad_enabled(False)
    dev = tt.get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    real_served = tt._fp32_softmax_bmm_served
    rows_out = []
    for heads, S in cases:
        nrows = args.rows or S
        hpr = heads * S
        per_row = hpr * S * 4
        tuned = tt._fp32_softmax_l1_rows(per_row, hpr)
        g = torch.Generator().manual_seed(1234 + heads * 4096 + S)
        shp = (nrows, heads, S, HEAD_DIM)
        q = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
        k = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
        v = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
        b = (torch.randn((1, heads, S, S), generator=g, dtype=torch.float32) * 0.5
             ).to(torch.bfloat16)
        qd, kd, vd, bd = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=dev) for t in (q, k, v, b))
        row = {"heads": heads, "S": S, "rows": nrows, "tuned_rows": tuned}
        outs, times = {}, {}
        for arm, float_cores, bmm_aware in ARMS:
            tt._FP32_SOFTMAX_L1_FLOAT_CORES = float_cores
            # Arm B is arm C with the matmul-config constraint satisfied by everything, which is
            # exactly "the tallest block the byte budget affords" -- pass 3's objective.
            tt._fp32_softmax_bmm_served = (real_served if bmm_aware
                                           else (lambda rows, bmm: True))
            tt._fp32_softmax_l1_plan.cache_clear()
            tt._FP32_SOFTMAX_L1_ROW_CAP.clear()
            tt._FP32_SOFTMAX_L1_REFUSALS.clear()
            for key in tt.FP32_SOFTMAX_STATS:
                tt.FP32_SOFTMAX_STATS[key] = 0
            for key in tt.LATCH_STATS["bmm_cfg"]:
                if key != "why":
                    tt.LATCH_STATS["bmm_cfg"][key] = 0
            tt.LATCH_STATS["bmm_cfg"]["why"] = []
            tt._BMM_CFG_RUNG.clear()
            tt._BMM_CFG_REFUSED.clear()
            ts = []
            for r in range(args.reps + 1):          # first rep warms the program cache
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = tt._fp32_softmax_attention(qd, kd, vd, bd, scale_inv=HEAD_DIM ** 0.5,
                                               compute_kernel_config=ckc, bias_scale_inv=1.0)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                if r:
                    ts.append(dt)
                if r < args.reps:
                    ttnn.deallocate(o)
            outs[arm] = ttnn.to_torch(o)
            ttnn.deallocate(o)
            times[arm] = statistics.median(ts)
            st = dict(tt.FP32_SOFTMAX_STATS)
            row["plan_" + arm] = [st["l1_cores"], st["l1"]]
            row["stats_" + arm] = st
            row["bmm_" + arm] = {k2: v2 for k2, v2 in tt.LATCH_STATS["bmm_cfg"].items()
                                 if k2 != "why"}
            row["ms_" + arm] = round(1000 * times[arm], 3)
            row["reps_" + arm] = [round(1000 * t, 3) for t in ts]
        for t in (qd, kd, vd, bd):
            ttnn.deallocate(t)
        row["bit_exact_AB"] = bool(torch.equal(outs["A"], outs["B"]))
        row["bit_exact_AC"] = bool(torch.equal(outs["A"], outs["C"]))
        row["speedup_B"] = round(times["A"] / times["B"], 4)
        row["speedup_C"] = round(times["A"] / times["C"], 4)
        rows_out.append(row)
        print("heads=%d S=%4d rows=%4d | A %8.2f ms %4d blk cfg%5d/%-5d | B %8.2f ms %6.4fx "
              "%4d blk cfg%5d | C %8.2f ms %6.4fx %4d blk cfg%5d | exact AB=%s AC=%s"
              % (heads, S, nrows, row["ms_A"], row["stats_A"]["blocks"] or 1,
                 row["bmm_A"]["served"], row["bmm_A"]["declined"],
                 row["ms_B"], row["speedup_B"], row["stats_B"]["blocks"] or 1,
                 row["bmm_B"]["served"],
                 row["ms_C"], row["speedup_C"], row["stats_C"]["blocks"] or 1,
                 row["bmm_C"]["served"],
                 row["bit_exact_AB"], row["bit_exact_AC"]), flush=True)
        Path(args.out).write_text(json.dumps(
            {"grid": list(tt.COMPUTE_GRID_MAIN), "rows": rows_out}, indent=2) + "\n")

    print("all bit-exact A vs C: %s | A vs B: %s"
          % (all(r["bit_exact_AC"] for r in rows_out),
             all(r["bit_exact_AB"] for r in rows_out)))


if __name__ == "__main__":
    main()
