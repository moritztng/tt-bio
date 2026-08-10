#!/usr/bin/env python3
"""Q9 rescore, probe 3 — price the 62 ops `pf_block_ops.py` drops, so the gap can be split.

The five dropped classes all have L1 outputs, which is why the harness cannot re-run them: holding
`reps` extra outputs live throws. Time them one call at a time, freeing each output immediately, so
at most one extra output is live. `ttnn.deallocate` is 0.50-0.66 us/call (T4), under 1 % of every
row here, and it is counted in the number rather than subtracted.

Standalone, in a fresh allocator state -- a BOUND on the dropped bucket, not an in-block time.

    PYTHONPATH=<wt> python3 perf/rescore/r_probe3.py --out perf/rescore/r_probe3_qb2c0.json
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed_free(dev, fn, warm=3, iters=20, reps=3):
    """Seconds per call with each output freed immediately: only one extra output is ever live."""
    for _ in range(warm):
        o = fn()
        for t in (o if isinstance(o, (list, tuple)) else [o]):
            if isinstance(t, ttnn.Tensor):
                ttnn.deallocate(t)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            o = fn()
            for t in (o if isinstance(o, (list, tuple)) else [o]):
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / iters)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    R = {}

    def leg(name, fn, calls):
        try:
            s = timed_free(dev, fn)
            R[name] = {"us": round(s * 1e6, 2), "calls_per_block": calls,
                       "block_ms": round(s * calls * 1e3, 3)}
        except Exception as e:                                              # noqa: BLE001
            R[name] = {"err": str(e)[:120], "calls_per_block": calls}
        print(f"  {name}: {R[name]}", flush=True)

    z = ttnn.from_torch(torch.randn(1, 298, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    w128 = ttnn.from_torch(torch.randn(256, 128), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=DRAM)
    leg("minimal_matmul_1329", lambda: ttnn.experimental.minimal_matmul(
        z, w128, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=ckc), 16)
    g = None
    try:
        g = ttnn.experimental.minimal_matmul(z, w128, memory_config=L1, dtype=ttnn.bfloat16,
                                             compute_kernel_config=ckc)
        leg("chunk_1336", lambda: ttnn.chunk(g, chunks=4, dim=-1), 16)
    except Exception as e:                                                  # noqa: BLE001
        R["chunk_1336"] = {"err": str(e)[:120], "calls_per_block": 16}
        print(f"  chunk_1336 setup failed: {str(e)[:120]}", flush=True)
    if g is not None:
        ttnn.deallocate(g)
    ttnn.deallocate(z)
    ttnn.deallocate(w128)

    t = ttnn.from_torch(torch.randn(1, 30, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    gm = ttnn.from_torch(torch.randn(1, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    bm = ttnn.from_torch(torch.randn(1, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    leg("layer_norm_2063", lambda: ttnn.layer_norm(t, weight=gm, bias=bm, epsilon=1e-5,
                                                   memory_config=L1,
                                                   compute_kernel_config=ckc), 10)
    tl1 = ttnn.clone(t, memory_config=L1)
    w1024 = ttnn.from_torch(torch.randn(256, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
    leg("linear_2071_2080", lambda: ttnn.linear(tl1, w1024, memory_config=L1, dtype=ttnn.bfloat16,
                                                compute_kernel_config=ckc,
                                                core_grid=CORE_GRID_MAIN), 20)
    for x in (tl1, t, gm, bm, w1024):
        ttnn.deallocate(x)

    R["dropped_block_ms_bound"] = round(sum(v["block_ms"] for v in R.values()
                                            if isinstance(v, dict) and "block_ms" in v), 3)
    print(f"  dropped bucket bound = {R['dropped_block_ms_bound']} ms/block", flush=True)
    json.dump(R, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
