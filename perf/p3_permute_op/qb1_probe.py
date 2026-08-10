#!/usr/bin/env python3
"""p3-permute-qb1 deliverable 1: does the kernel run at all on ttnn 0.67.4, and is it still bit-exact.

Everything here is measured on qb1 card 0 at ttnn 0.67.4, in one process, with
`ttnn.synchronize_device` on both sides of every timed region. No figure is inherited: X6's qb2 /
0.68.0 numbers are printed alongside for comparison only, and the comparison is the deliverable.

The device is opened through `tt_bio.tenstorrent.get_device()` rather than `ttnn.open_device` so the
production grid rebind (11x10 -> 13x10) and the card lease both apply, and so
`_transpose_memory_config` is read on the grid the fold actually runs.

    TT_VISIBLE_DEVICES=0 python3 perf/p3_permute_op/qb1_probe.py
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


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


def host_cached_us(RP, x, device, mc, reps=200):
    """The per-call host cost on the cache-hit path: allocate, key, hit, write two addresses."""
    N = int(x.shape[1])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        entry = RP._prepare(x, out, device)
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = [x.buffer_address()]
        pd.kernels[1].common_runtime_args = [out.buffer_address()]
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def host_rebuild_us(RP, x, device, mc, reps=25):
    """X6's blocker, reproduced on this wheel: the descriptor rebuilt in Python on every call."""
    N = int(x.shape[1])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        writer_ct = [2, RP.OUT_CB, RP.TILE_H, RP.TILE_W, RP.FACE_H, RP.FACE_W, RP.STAGE_CB]
        writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        entry = RP._build(x, out, device, reader_ct, writer_ct)
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = [x.buffer_address()]
        pd.kernels[1].common_runtime_args = [out.buffer_address()]
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def cores_in(crs):
    n = 0
    for cr in crs.ranges():
        n += (cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "qb1_probe.json"))
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    from tt_bio import reblock_permute as RP
    RP.set_enabled(True)

    R = {"wheel": "0.67.4", "host": "qb1", "card": 0,
         "compute_grid_main_at_import": list(T.COMPUTE_GRID_MAIN)}

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    R["compute_grid_main_after_open"] = list(T.COMPUTE_GRID_MAIN)
    R["device_grid"] = [int(g.x), int(g.y)]
    R["per_core_l1_unreserved_B"] = int(ttnn.get_max_worker_l1_unreserved_size())

    # ---- Q1: does KernelDescriptor(kernel_source=...) JIT-compile against THIS wheel's headers ----
    ref298 = torch.randn(1, 298, 298, 32, dtype=torch.bfloat16)
    x298 = ttnn.from_torch(ref298, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
    try:
        first = RP.reblock_permute(x298, ttnn.L1_MEMORY_CONFIG, device)
        ttnn.synchronize_device(device)
        R["jit_compile"] = {"ok": True, "addr_write_mode": RP.ADDR_WRITE_MODE}
        ttnn.deallocate(first)
    except Exception:
        R["jit_compile"] = {"ok": False, "traceback": traceback.format_exc()}
        Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
        print(R["jit_compile"]["traceback"])
        print("JIT COMPILE FAILED at 0.67.4 -- that is the deliverable, stopping here.")
        return 1
    print("jit_compile:", R["jit_compile"], flush=True)

    # ---- Q4: core utilisation, read out of the work split rather than assumed -------------------
    entry = next(iter(RP._CACHE.values()))
    R["cores_engaged_N298"] = cores_in(entry["core_grid"])
    R["cores_on_grid"] = int(g.x) * int(g.y)

    # ---- Q2: parity ------------------------------------------------------------------------------
    refs, gold = {}, {}
    for N in (298, 320):
        refs[N] = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        gold[N] = refs[N].permute(0, 3, 1, 2).contiguous()

    parity = []
    for N in (298, 320):
        for where in ("l1", "dram"):
            xx = ttnn.from_torch(refs[N], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                 device=device, memory_config=MC[where])
            r = RP.reblock_permute(xx, MC[where], device)
            eq = bool(torch.equal(ttnn.to_torch(r), gold[N]))
            # Against the stock op INCLUDING the output tile-padding rows: those rows sit on the
            # contracted axis of the triangle matmul, so they are the check that matters.
            st = ttnn.permute(xx, (0, 3, 1, 2), memory_config=MC[where])
            eq_pad = bool(torch.equal(ttnn.to_torch(r), ttnn.to_torch(st)))
            parity.append({"N": N, "buf": where, "torch_equal_vs_torch": eq,
                           "torch_equal_vs_ttnn_permute_incl_padding": eq_pad})
            ttnn.deallocate(r); ttnn.deallocate(st); ttnn.deallocate(xx)
    R["parity_shapes"] = parity
    print("parity_shapes:", parity, flush=True)

    # Two different N alternating through one cache, and two buffers through one descriptor.
    xl = {N: ttnn.from_torch(refs[N], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                             memory_config=ttnn.L1_MEMORY_CONFIG) for N in (298, 320)}
    seq = []
    for _ in range(3):
        for N in (298, 320, 320, 298):
            r = RP.reblock_permute(xl[N], ttnn.L1_MEMORY_CONFIG, device)
            seq.append({"N": N, "eq": bool(torch.equal(ttnn.to_torch(r), gold[N]))})
            ttnn.deallocate(r)
    R["alternating_N"] = {"n": len(seq), "all_true": all(s["eq"] for s in seq),
                          "cache_entries": len(RP._CACHE), "seq": seq}
    print("alternating_N:", R["alternating_N"]["n"], R["alternating_N"]["all_true"], flush=True)

    refB = torch.randn(1, 298, 298, 32, dtype=torch.bfloat16)
    xB = ttnn.from_torch(refB, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    goldB = refB.permute(0, 3, 1, 2).contiguous()
    two = []
    for _ in range(2):
        for src, gd in ((xl[298], gold[298]), (xB, goldB)):
            r = RP.reblock_permute(src, ttnn.L1_MEMORY_CONFIG, device)
            two.append(bool(torch.equal(ttnn.to_torch(r), gd)))
            ttnn.deallocate(r)
    R["two_buffers_one_descriptor"] = {"n": len(two), "all_true": all(two),
                                       "addrs_differ": xl[298].buffer_address() != xB.buffer_address()}
    print("two_buffers:", R["two_buffers_one_descriptor"], flush=True)

    # ---- Q3: host cost, cached vs rebuilt, on this wheel ----------------------------------------
    R["host_us"] = {
        "cached_per_call": round(host_cached_us(RP, xl[298], device, ttnn.L1_MEMORY_CONFIG), 2),
        "rebuilt_per_call": round(host_rebuild_us(RP, xl[298], device, ttnn.L1_MEMORY_CONFIG), 2),
        "x6_qb2_0_68_0_cached": 4.54, "x6_qb2_0_68_0_rebuilt": 151.29,
    }
    print("host_us:", R["host_us"], flush=True)
    # host_rebuild_us poisoned the cache with fresh entries; drop them so the cache count above
    # stays meaningful for the wall measurements below.
    RP._CACHE.clear()

    # ---- roofs on THIS card, at THIS op's own tensor size ---------------------------------------
    roofs = []
    for N in (298, 320):
        nbytes = 1 * N * N * 32 * 2  # bf16, logical; the padded buffer is what moves
        for si in ("l1", "dram"):
            for so in ("l1", "dram"):
                xx = ttnn.from_torch(refs[N], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                     device=device, memory_config=MC[si])
                us = timeit(device, lambda xx=xx, so=so: ttnn.deallocate(
                    ttnn.clone(xx, memory_config=MC[so])))
                roofs.append({"N": N, "src": si, "dst": so, "clone_us": round(us, 2),
                              "one_way_GBs": round(nbytes / us / 1e3, 1)})
                ttnn.deallocate(xx)
    R["clone_roofs"] = roofs
    for r in roofs:
        print("roof:", r, flush=True)

    # ---- the gate ladder, wall-clock, host cost included ----------------------------------------
    rows = []
    for N in (256, 298, 320, 384):
        ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        gd = ref.permute(0, 3, 1, 2).contiguous()
        for where in ("l1", "dram"):
            mc = MC[where]
            xx = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                                 memory_config=mc)
            r = RP.reblock_permute(xx, mc, device)
            eq = bool(torch.equal(ttnn.to_torch(r), gd))
            ttnn.deallocate(r)
            uw = timeit(device, lambda xx=xx, mc=mc: ttnn.deallocate(
                RP.reblock_permute(xx, mc, device)))
            ub = timeit(device, lambda xx=xx, mc=mc: ttnn.deallocate(
                ttnn.permute(xx, (0, 3, 1, 2), memory_config=mc)))
            ent = RP._prepare(xx, ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc), device)
            rows.append({"N": N, "buf": where, "wired_us": round(uw, 2),
                         "ttnn_permute_us": round(ub, 2), "ratio": round(ub / uw, 3),
                         "torch_equal": eq, "cores": cores_in(ent["core_grid"]),
                         "groups": ((N + 31) // 32) ** 2})
            print("ladder:", rows[-1], flush=True)
            ttnn.deallocate(xx)
    R["ladder"] = rows

    # ---- Q9: which stock permute is cheaper on 0.67.4, and does the decomposition still lose ----
    N = 298
    ref = refs[N]
    xx = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    mc = ttnn.L1_MEMORY_CONFIG

    def decomposed(xx=xx, mc=mc):
        a1 = RP.reblock_permute(xx, mc, device)
        a2 = ttnn.transpose(a1, -2, -1, memory_config=mc)
        ttnn.deallocate(a1); ttnn.deallocate(a2)

    d_stock_3132 = timeit(device, lambda: ttnn.deallocate(
        ttnn.permute(xx, (0, 3, 1, 2), memory_config=mc)))
    d_stock_3231 = timeit(device, lambda: ttnn.deallocate(
        ttnn.permute(xx, (0, 3, 2, 1), memory_config=mc)))
    d_decomp = timeit(device, decomposed)
    # bit-exactness of the decomposition against the single stock call
    got = ttnn.permute(RP.reblock_permute(xx, mc, device), (0, 1, 3, 2), memory_config=mc)
    want = ttnn.permute(xx, (0, 3, 2, 1), memory_config=mc)
    R["q9"] = {
        "stock_0312_us": round(d_stock_3132, 2), "stock_0321_us": round(d_stock_3231, 2),
        "stock_0321_cheaper": bool(d_stock_3231 < d_stock_3132),
        "pct_cheaper": round((d_stock_3132 - d_stock_3231) / d_stock_3132 * 100, 1),
        "decomposed_us": round(d_decomp, 2),
        "decomposition_ratio_vs_stock_0321": round(d_stock_3231 / d_decomp, 3),
        "decomposition_torch_equal": bool(torch.equal(ttnn.to_torch(got), ttnn.to_torch(want))),
    }
    ttnn.deallocate(got); ttnn.deallocate(want)
    print("q9:", R["q9"], flush=True)

    # ---- Q8: W6's landed lever, at the fold's own pair shape, on the rebound grid ---------------
    t = ttnn.from_torch(torch.zeros(298, 320, 256, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    R["transpose_memory_config"] = {
        "branch": str(T._transpose_memory_config(t).buffer_type),
        "grid": list(T.COMPUTE_GRID_MAIN),
        "demand_MB": round(2.5 * 298 * 320 * 256 * 2 / 1e6, 2),
        "capacity_MB": round(R["per_core_l1_unreserved_B"] * T.COMPUTE_GRID_MAIN[0]
                             * T.COMPUTE_GRID_MAIN[1] / 1e6, 2),
    }
    ttnn.deallocate(t)
    print("transpose_memory_config:", R["transpose_memory_config"], flush=True)

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
