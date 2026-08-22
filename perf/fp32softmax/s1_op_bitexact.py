#!/usr/bin/env python3
"""Per-shape bit-exactness and speed of the S1 height-shard plan, straight on the helper.

The fold-level A/B answers "does this model still emit the same structure". This answers the
narrower question the bit-exactness argument actually makes: for one call of
`_fp32_softmax_attention`, does letting the shard pick its core count change a single bit of the
output. It runs both arms on the same inputs in one process and compares with `torch.equal`, over
the head counts and padded lengths the pair-track models use, including the sizes where the tuned
8x8 rectangle already serves (those must be unchanged, not merely equal).

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 perf/fp32softmax/s1_op_bitexact.py \
        --out perf/fp32softmax/results/s1_op_bitexact.json
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

HEADS = (2, 4, 8)
SIZES = (512, 515, 544, 704, 832, 960)
ROWS = 64
HEAD_DIM = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=ROWS)
    args = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import tenstorrent as tt

    torch.set_grad_enabled(False)
    dev = tt.get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    rows_out = []
    for heads in HEADS:
        for S in SIZES:
            hpr = heads * S
            per_row = hpr * S * 4
            tuned = tt._fp32_softmax_l1_rows(per_row, hpr)
            tt._fp32_softmax_l1_plan.cache_clear()
            plan = tt._fp32_softmax_l1_plan(per_row, hpr, S)
            g = torch.Generator().manual_seed(1234 + heads * 4096 + S)
            shp = (args.rows, heads, S, HEAD_DIM)
            q = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
            k = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
            v = torch.randn(shp, generator=g, dtype=torch.float32).to(torch.bfloat16)
            b = (torch.randn((1, heads, S, S), generator=g, dtype=torch.float32) * 0.5
                 ).to(torch.bfloat16)
            qd, kd, vd, bd = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                              device=dev) for t in (q, k, v, b))
            row = {"heads": heads, "S": S, "rows": args.rows, "tuned_rows": tuned,
                   "plan": list(plan)}
            outs, times = {}, {}
            for arm in ("A", "B"):
                tt._FP32_SOFTMAX_L1_ANY_CORES = (arm == "B")
                tt._fp32_softmax_l1_plan.cache_clear()
                tt._FP32_SOFTMAX_L1_ROW_CAP.clear()
                for key in tt.FP32_SOFTMAX_STATS:
                    tt.FP32_SOFTMAX_STATS[key] = 0
                ts = []
                for r in range(args.reps + 1):          # first rep warms the program cache
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    o = tt._fp32_softmax_attention(qd, kd, vd, bd, scale_inv=HEAD_DIM ** 0.5,
                                                   compute_kernel_config=ckc,
                                                   bias_scale_inv=1.0)
                    ttnn.synchronize_device(dev)
                    dt = time.perf_counter() - t0
                    if r:
                        ts.append(dt)
                    if r < args.reps:
                        ttnn.deallocate(o)
                outs[arm] = ttnn.to_torch(o)
                ttnn.deallocate(o)
                times[arm] = statistics.median(ts)
                row["stats_" + arm] = dict(tt.FP32_SOFTMAX_STATS)
                row["ms_" + arm] = round(1000 * times[arm], 3)
            for t in (qd, kd, vd, bd):
                ttnn.deallocate(t)
            row["bit_exact"] = bool(torch.equal(outs["A"], outs["B"]))
            row["max_abs_diff"] = float((outs["A"].float() - outs["B"].float()).abs().max())
            row["speedup"] = round(times["A"] / times["B"], 4)
            row["moved"] = row["stats_A"]["l1_cores"] != row["stats_B"]["l1_cores"]
            rows_out.append(row)
            print("heads=%d S=%4d tuned=%3d plan=%s  A %8.2f ms  B %8.2f ms  %6.3fx  "
                  "%s  moved=%s" % (heads, S, tuned, tuple(plan), row["ms_A"], row["ms_B"],
                                    row["speedup"],
                                    "BIT-EXACT" if row["bit_exact"] else "DIFFER %g"
                                    % row["max_abs_diff"], row["moved"]), flush=True)

    rep = {"grid": list(tt.COMPUTE_GRID_MAIN), "rows": rows_out,
           "all_bit_exact": all(r["bit_exact"] for r in rows_out),
           "moved": [[r["heads"], r["S"], r["speedup"]] for r in rows_out if r["moved"]],
           "unmoved": [[r["heads"], r["S"], r["speedup"]] for r in rows_out if not r["moved"]]}
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print("all bit-exact: %s   moved %d shapes, unmoved %d"
          % (rep["all_bit_exact"], len(rep["moved"]), len(rep["unmoved"])))


if __name__ == "__main__":
    main()
