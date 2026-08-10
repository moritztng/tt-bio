#!/usr/bin/env python3
"""Deliverable 1: the descriptor cache, measured against X4's 155 us blocker.

Arms, all on one card in one process, synced on both sides of every timed region, median of 15:

  * the host cost of the cached path per call (everything except ``ttnn.generic_op``),
  * the same cost from cold, which is X4's 154.78 us figure reproduced on this build,
  * the full synced wall of the wired op against ``ttnn.permute`` for the same move,
  * the stale-descriptor tests: two different N alternating through one cache, and two different
    source/destination buffers at one N alternating through one cached descriptor,
  * the decomposition arm: reblock + transpose(-2,-1) against a single permute(0,3,2,1).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn

from tt_bio import reblock_permute as RP


def timeit(device, fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def host_only_us(x, device, mc, reps=200):
    """Everything reblock_permute does except ttnn.generic_op: allocate, key, cache hit, addresses."""
    N = int(x.shape[1])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        entry = RP._prepare(x, out, device)
        src, dst = x.buffer_address(), out.buffer_address()
        if RP.ADDR_WRITE_MODE == "in_place":
            pd = entry["pd"]
            pd.kernels[0].common_runtime_args = [src]
            pd.kernels[1].common_runtime_args = [dst]
        else:
            r, w, c = entry["kernels"]
            r.common_runtime_args = [src]
            w.common_runtime_args = [dst]
            entry["pd"] = ttnn.ProgramDescriptor(kernels=[r, w, c], semaphores=[], cbs=entry["cbs"])
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def cold_build_us(x, device, mc, reps=10):
    """X4's blocker, reproduced: the same construction with the cache defeated."""
    N = int(x.shape[1])
    saved = dict(RP._CACHE)
    ts = []
    for _ in range(reps):
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        RP._CACHE.clear()
        t0 = time.perf_counter()
        RP._prepare(x, out, device)
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    RP._CACHE.clear(); RP._CACHE.update(saved)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/p3_permute_op/wire_probe.json")
    a = ap.parse_args()
    RP.set_enabled(True)
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    R = {"grid": [g.x, g.y], "rows": [], "checks": {}}

    # ---- stale-descriptor test 1: two different N alternating through one cache ------------------
    refs, xs, goldens = {}, {}, {}
    for N in (320, 256):
        ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        refs[N] = ref
        goldens[N] = ref.permute(0, 3, 1, 2).contiguous()
        xs[N] = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    seq = []
    for _ in range(3):
        for N in (320, 256, 256, 320):
            r = RP.reblock_permute(xs[N], ttnn.L1_MEMORY_CONFIG, device)
            ok = torch.equal(ttnn.to_torch(r), goldens[N])
            seq.append({"N": N, "torch_equal": bool(ok)})
            ttnn.deallocate(r)
    R["checks"]["alternating_N_through_one_cache"] = seq
    R["checks"]["alternating_N_all_equal"] = all(s["torch_equal"] for s in seq)
    R["checks"]["cache_entries"] = len(RP._CACHE)
    R["checks"]["addr_write_mode"] = RP.ADDR_WRITE_MODE

    # ---- stale-descriptor test 2: two different buffers at one N, one cached descriptor ----------
    N = 320
    ra = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
    rb = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
    xa = ttnn.from_torch(ra, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    xb = ttnn.from_torch(rb, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    addr_seq = []
    for src_t, ref_t, tag in ((xa, ra, "A"), (xb, rb, "B"), (xa, ra, "A"), (xb, rb, "B")):
        r = RP.reblock_permute(src_t, ttnn.L1_MEMORY_CONFIG, device)
        addr_seq.append({"buf": tag, "src_addr": src_t.buffer_address(),
                         "torch_equal": bool(torch.equal(ttnn.to_torch(r),
                                                         ref_t.permute(0, 3, 1, 2).contiguous()))})
        ttnn.deallocate(r)
    R["checks"]["alternating_buffers_one_descriptor"] = addr_seq
    R["checks"]["alternating_buffers_all_equal"] = all(s["torch_equal"] for s in addr_seq)

    # ---- host cost, cached vs cold, and the full wall against the baseline ----------------------
    for N in (320, 256):
        for where in ("l1", "dram"):
            mc = ttnn.L1_MEMORY_CONFIG if where == "l1" else ttnn.DRAM_MEMORY_CONFIG
            ref = refs[N]
            x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                device=device, memory_config=mc)
            golden = goldens[N]
            r = RP.reblock_permute(x, mc, device)
            eq = torch.equal(ttnn.to_torch(r), golden)
            ttnn.deallocate(r)

            def wired(x=x, mc=mc):
                o = RP.reblock_permute(x, mc, device)
                ttnn.deallocate(o)

            def base(x=x, mc=mc):
                o = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                ttnn.deallocate(o)

            us_wired = timeit(device, wired)
            us_base = timeit(device, base)
            us_host = host_only_us(x, device, mc)
            us_cold = cold_build_us(x, device, mc)
            row = {"N": N, "buf": where, "wired_wall_us": round(us_wired, 2),
                   "ttnn_permute_us": round(us_base, 2),
                   "ratio_wall": round(us_base / us_wired, 3),
                   "host_per_call_cached_us": round(us_host, 2),
                   "host_per_call_cold_us": round(us_cold, 2),
                   "torch_equal": bool(eq)}
            R["rows"].append(row); print(row, flush=True)
            ttnn.deallocate(x)

    # ---- W6: the (0,3,2,1) half, decomposed --------------------------------------------------
    N = 320
    mc = ttnn.L1_MEMORY_CONFIG
    ref = refs[N]
    x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                        memory_config=mc)
    g2 = ref.permute(0, 3, 2, 1).contiguous()

    def single(x=x):
        o = ttnn.permute(x, (0, 3, 2, 1), memory_config=mc)
        ttnn.deallocate(o)

    def decomposed(x=x):
        o = RP.reblock_permute(x, mc, device)
        o2 = ttnn.transpose(o, -2, -1, memory_config=mc)
        ttnn.deallocate(o); ttnn.deallocate(o2)

    o = RP.reblock_permute(x, mc, device)
    o2 = ttnn.transpose(o, -2, -1, memory_config=mc)
    eq_dec = torch.equal(ttnn.to_torch(o2), g2)
    ttnn.deallocate(o); ttnn.deallocate(o2)
    us_single = timeit(device, single)
    us_dec = timeit(device, decomposed)
    R["decompose_0321"] = {"N": N, "buf": "l1", "single_permute_us": round(us_single, 2),
                           "reblock_plus_transpose_us": round(us_dec, 2),
                           "ratio": round(us_single / us_dec, 3), "torch_equal": bool(eq_dec)}
    print(R["decompose_0321"], flush=True)
    ttnn.deallocate(x)

    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print(json.dumps(R["checks"], indent=2))


if __name__ == "__main__":
    main()
