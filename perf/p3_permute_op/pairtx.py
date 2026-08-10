#!/usr/bin/env python3
"""P3 / p3-permute-op, deliverable 1 -- the TriangleAttention pair-tensor transpose.

The two sites T5 numbers `tenstorrent.py:1570/:1715` and the ledger numbers `1295/1448 TriAtt`
are `ttnn.permute(x, (1,0,2))` on the 3-D pair tensor. This measures, on THIS card:
  * the buffer-type ladder for that exact move at the 298 aa shape,
  * the staged arm (clone DRAM->L1, permute L1->L1, clone back),
  * what `_transpose_memory_config` actually returns here, i.e. what production does today,
  * this card's clone floors at the same byte count.
"""
import argparse, json, time
import torch
import ttnn

BF16 = 2


def timeit(device, fn, reps=9, warmup=2):
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def mem(k):
    return ttnn.L1_MEMORY_CONFIG if k == "l1" else ttnn.DRAM_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=int, default=298)
    ap.add_argument("--s1", type=int, default=320)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out", default="perf/p3_permute_op/pairtx.json")
    a = ap.parse_args()
    S0, S1, C = a.s0, a.s1, a.c
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y
    one_way = S0 * S1 * C * BF16
    R = {"shape": [S0, S1, C], "cores": ncores, "one_way_MB": one_way / 1e6, "arms": {}, "parity": {}}

    def rec(name, us, note=""):
        gbs = one_way / (us * 1e-6) / 1e9
        R["arms"][name] = {"us": round(us, 2), "GB_s_one_way": round(gbs, 1), "note": note}
        print(f"{name:52s} {us:9.2f} us   {gbs:7.1f} GB/s  {note}", flush=True)

    # what production does today
    import sys
    sys.path.insert(0, ".")
    try:
        from tt_bio.tenstorrent import _transpose_memory_config, COMPUTE_GRID_MAIN
        per_core = int(ttnn.get_max_worker_l1_unreserved_size())
        R["per_core_l1_unreserved"] = per_core
        R["compute_grid_main"] = list(COMPUTE_GRID_MAIN)
    except Exception as e:
        _transpose_memory_config = None
        R["import_error"] = repr(e)
        print("tt_bio import failed:", e, flush=True)

    ref = torch.randn(S0, S1, C, dtype=torch.bfloat16)
    golden = ref.permute(1, 0, 2).contiguous()

    def load(where):
        return ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               device=device, memory_config=mem(where))

    if _transpose_memory_config is not None:
        t = load("dram")
        mc = _transpose_memory_config(t)
        R["production_dest"] = str(mc.buffer_type)
        print("production _transpose_memory_config ->", mc.buffer_type,
              " per_core_l1_unreserved =", R.get("per_core_l1_unreserved"),
              " grid =", R.get("compute_grid_main"), flush=True)
        ttnn.deallocate(t)

    # ---- clone floors at the same byte count -----------------------------------------
    for src in ("dram", "l1"):
        for dst in ("dram", "l1"):
            try:
                t = load(src)
                us = timeit(device, lambda t=t, dst=dst: ttnn.clone(t, memory_config=mem(dst)), a.reps)
                rec(f"clone {src}->{dst}", us, "floor")
                ttnn.deallocate(t)
            except Exception as e:
                print(f"clone {src}->{dst} FAILED: {e}", flush=True)

    # ---- the production op, buffer-type ladder ---------------------------------------
    for src in ("dram", "l1"):
        for dst in ("dram", "l1"):
            try:
                t = load(src)
                us = timeit(device, lambda t=t, dst=dst: ttnn.permute(t, (1, 0, 2), memory_config=mem(dst)), a.reps)
                rec(f"permute(1,0,2) {src}->{dst}", us)
                out = ttnn.permute(t, (1, 0, 2), memory_config=mem(dst))
                ok = torch.equal(ttnn.to_torch(out), golden)
                R["parity"][f"permute {src}->{dst}"] = bool(ok)
                print("   torch.equal vs torch golden:", ok, flush=True)
                ttnn.deallocate(out)
                ttnn.deallocate(t)
            except Exception as e:
                print(f"permute {src}->{dst} FAILED: {e}", flush=True)

    # ---- the staged arms --------------------------------------------------------------
    def staged(t, final):
        s = ttnn.clone(t, memory_config=ttnn.L1_MEMORY_CONFIG)
        p = ttnn.permute(s, (1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(s)
        if final == "l1":
            return p
        o = ttnn.clone(p, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(p)
        return o

    for final in ("l1", "dram"):
        try:
            t = load("dram")
            us = timeit(device, lambda t=t, final=final: staged(t, final), a.reps)
            rec(f"staged dram->L1->permute->{final}", us, "3 stock calls" if final == "dram" else "2 stock calls")
            out = staged(t, final)
            ok = torch.equal(ttnn.to_torch(out), golden)
            R["parity"][f"staged dram->{final}"] = bool(ok)
            print("   torch.equal vs torch golden:", ok, flush=True)
            ttnn.deallocate(out)
            ttnn.deallocate(t)
        except Exception as e:
            print(f"staged -> {final} FAILED: {e}", flush=True)

    ttnn.close_device(device)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
