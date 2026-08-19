#!/usr/bin/env python3
"""p67 -- which mechanism holds the DiT token linears at 7.4 TFLOP/s, and what a width fusion buys.

The DiT spends 47.058 ms/step (synced) in 432 `_tuned_linear` calls (p49 v2 `linear@265`), i.e.
0.109 ms/call on a [1,685,768] @ [768,768] whose MAC work is 8 us at the measured 102.02 TFLOP/s
roof. Two mechanisms fit that 13x deficit and they imply opposite levers:

  M1  per-output-tile pack/unpack, the mechanism arm E named for the pair matmuls (K = 4 tiles).
      Rate would then be invariant in M, growing the row count buys nothing, and BATCHING CANNOT
      HELP -- consistent with P3.15 closing batching NO-GO at 1.039x.
  M2  per-op fixed cost on a grid-starved shape. M = 685 rows = 22 row-tiles over ~130 cores is
      4 output tiles per core; if ~100 us of the 109 is fixed, time is FLAT in M until the grid
      fills, batching amortises it, and fusing four linears across N amortises it four times over.

PRE-REGISTERED PREDICTION, written before this ran:
  M2 holds. Arm A's ms/call is flat within 25 % from R=32 to R=685 and only then grows roughly
  linearly. If instead ms/call scales with R from R=32 -- doubling R doubles the time -- M1 holds,
  Lever E (width fusion) is worth little, and batching stays closed on a second independent
  measurement.

Arm B prices the lever either way: q/k/v/g are four [768,768] linears on the SAME input, so one
[768,3072] linear computes all four. `torch.equal` on the concatenated result against the four
separate calls is the bit-exactness gate -- each output column is the same dot product, so the only
question is whether the wider N changes the K-blocking.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD RFD3_TUNE_MATMUL=1 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p67_dit_linear_mechanism.py \
          perf/p67/dit_linear_mechanism.json
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                            # noqa: E402
from tt_bio.tenstorrent import get_device                              # noqa: E402
from tt_bio.rfd3 import model as M                                     # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p67/dit_linear_mechanism.json")
N = 6
C_TOKEN, C_S = 768, 384


def mk(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(torch.randn(*shape, generator=g), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=get_device())


def timeit(fn):
    dev = get_device()
    o = fn()
    ttnn.synchronize_device(dev)
    if isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)
    ts = []
    for _ in range(N):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return statistics.median(ts), min(ts), max(ts)


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    M.set_tune_matmul_for_atoms(6051)
    grid = M.BATCH_INVARIANT_GRID
    print("[p67] tune_matmul=%s grid=%s device_grid=%s"
          % (M._TUNE_MATMUL, grid, dev.compute_with_storage_grid_size()), flush=True)
    res = {"n": N, "host": "qb2", "card": int(os.environ.get("TT_VISIBLE_DEVICES", -1)),
           "torch": torch.__version__, "tune_matmul": bool(M._TUNE_MATMUL),
           "prediction": "M2: ms/call flat within 25pct from R=32 to R=685",
           "arm_a": [], "arm_b": {}, "arm_c": {}}

    w = mk((C_TOKEN, C_TOKEN), 1)
    for R in (32, 64, 128, 342, 685, 1370, 2740, 5480):
        x = mk((1, R, C_TOKEN), 100 + R)
        med, lo, hi = timeit(lambda: M._tuned_linear(x, w, ckc=ckc, dtype=ttnn.bfloat16,
                                                    core_grid=grid))
        gf = 2.0 * R * C_TOKEN * C_TOKEN / 1e9
        row = {"rows": R, "ms": round(med, 4), "min": round(lo, 4), "max": round(hi, 4),
               "gflop": round(gf, 3), "tflops": round(gf / med, 2),
               "pct_of_102_roof": round(100.0 * (gf / med) / 102.02, 1)}
        res["arm_a"].append(row)
        print("[A]", row, flush=True)
        ttnn.deallocate(x)

    # Arm B: q/k/v/g as four [768,768] linears vs one [768,3072].
    x = mk((1, 685, C_TOKEN), 7)
    ws = [mk((C_TOKEN, C_TOKEN), 200 + i) for i in range(4)]
    cat = torch.cat([ttnn.to_torch(v).float() for v in ws], dim=-1)
    wcat = ttnn.from_torch(cat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def four():
        return [M._tuned_linear(x, v, ckc=ckc, dtype=ttnn.bfloat16, core_grid=grid) for v in ws]

    def one():
        return M._tuned_linear(x, wcat, ckc=ckc, dtype=ttnn.bfloat16, core_grid=grid)

    def four_t():
        for t in four():
            ttnn.deallocate(t)
        return None

    med4, lo4, hi4 = timeit(four_t)
    med1, lo1, hi1 = timeit(one)
    fused = one()
    rank = len(fused.shape)
    lead = [int(fused.shape[i]) for i in range(rank - 1)]

    def slices():
        for i in range(4):
            ttnn.deallocate(ttnn.slice(fused, [0] * (rank - 1) + [C_TOKEN * i],
                                       lead + [C_TOKEN * (i + 1)]))
        return None

    meds, los, his = timeit(slices)
    sep = four()
    ref = torch.cat([ttnn.to_torch(t).float() for t in sep], dim=-1)
    got = ttnn.to_torch(fused).float()
    res["arm_b"] = {"four_ms": round(med4, 4), "one_ms": round(med1, 4),
                    "slice4_ms": round(meds, 4),
                    "one_plus_slice_ms": round(med1 + meds, 4),
                    "saved_ms_per_call": round(med4 - med1 - meds, 4),
                    "bit_exact": bool(torch.equal(ref, got)),
                    "maxabs": float((ref - got).abs().max())}
    print("[B]", res["arm_b"], flush=True)
    for t in sep:
        ttnn.deallocate(t)
    ttnn.deallocate(fused)

    # Arm C: the adaLN pair -- gain and bias share the input `s` [1,685,384] and the width 768.
    xs = mk((1, 685, C_S), 9)
    g1 = mk((C_S, C_TOKEN), 300)
    g2 = mk((C_S, C_TOKEN), 301)
    gcat = ttnn.from_torch(
        torch.cat([ttnn.to_torch(g1).float(), ttnn.to_torch(g2).float()], -1),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def two_t():
        for v in (g1, g2):
            ttnn.deallocate(M._tuned_linear(xs, v, ckc=ckc, dtype=ttnn.bfloat16,
                                            core_grid=grid))
        return None

    m2, l2, h2 = timeit(two_t)
    m1, l1, h1 = timeit(lambda: M._tuned_linear(xs, gcat, ckc=ckc, dtype=ttnn.bfloat16,
                                                core_grid=grid))
    gref = torch.cat([ttnn.to_torch(M._tuned_linear(xs, v, ckc=ckc, dtype=ttnn.bfloat16,
                                                    core_grid=grid)).float()
                      for v in (g1, g2)], dim=-1)
    ggot = ttnn.to_torch(M._tuned_linear(xs, gcat, ckc=ckc, dtype=ttnn.bfloat16,
                                         core_grid=grid)).float()
    res["arm_c"] = {"two_ms": round(m2, 4), "one_ms": round(m1, 4),
                    "saved_ms_per_call": round(m2 - m1, 4),
                    "bit_exact": bool(torch.equal(gref, ggot)),
                    "maxabs": float((gref - ggot).abs().max())}
    print("[C]", res["arm_c"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
