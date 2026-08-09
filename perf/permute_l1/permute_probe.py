#!/usr/bin/env python3
"""W12 -- why ttnn.permute(1,0,2) sits at ~19% of the DRAM copy roof, and what moves it.

The permuted tensor is [B, M, C] bf16 in TILE layout, so the tiled dim pair is (M, C) and
dim0=B is the untiled batch. permute(1,0,2) exchanges the batch dim with the tile-ROW dim:
element (b,m,c) moves from intra-tile row m%32 to intra-tile row b%32. That is a sub-tile
move, and in TILE layout the largest contiguous run that survives it is one face row --
16 bf16 elements = 32 B -- regardless of C, because faces are 16 wide.

HYPOTHESIS: the op is NOC-transaction-issue-bound at 32 B, not bandwidth-bound.
PREDICTION: tiled permute GB/s is FLAT in C (piece is always 32 B), while the same
permutation done in ROW_MAJOR layout -- where the contiguous run is 2*C bytes -- scales
its GB/s with C and approaches the copy roof once 2*C >= 256 B.
FALSIFIED IF: the row-major permute is also flat in C, or the tiled permute rises with C.

Every timed region amortizes N issues between two synchronize_device calls (W4: one op per
sync region costs 0.02-0.05 ms of region overhead, 100%+ of a small op).
One arm per process: an L1 overflow fragments the allocator and poisons later arms (W7).
"""
import argparse, json, os, sys, time

import torch
import ttnn

BF16 = ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def timeit(dev, fn, n, warmup=2):
    """Amortized device time per issue, median of 5 regions of n issues."""
    for _ in range(warmup):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    samples = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
            ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        samples.append((time.perf_counter() - t0) / n)
    samples.sort()
    return samples[len(samples) // 2]


def gbps(mb_moved, secs):
    return mb_moved * 2 ** 20 / secs / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--b", type=int, default=320)
    ap.add_argument("--m", type=int, default=320)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--rows", type=int, default=0, help="row block for blocked arms")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--perm", default="1,0,2")
    ap.add_argument("--shape", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    grid = dev.compute_with_storage_grid_size()
    B, M, C = a.b, a.m, a.c
    n_el = B * M * C
    mb = n_el * 2 / 2 ** 20          # one copy of the tensor
    rw = 2 * mb                       # read + write bytes for a copy/permute
    res = {"arm": a.arm, "B": B, "M": M, "C": C, "MB": round(mb, 2),
           "grid": [grid.x, grid.y], "ttnn": "0.68.0", "rows": a.rows, "iters": a.iters}

    torch.manual_seed(0)
    host = torch.randn(B, M, C, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=BF16, memory_config=DRAM)
    hg = (torch.randn([int(d) for d in a.shape.split(",")], dtype=torch.float32).to(torch.bfloat16)
          if a.shape else host)

    try:
        if a.arm == "roof_dram":
            t = timeit(dev, lambda: ttnn.clone(x, memory_config=DRAM), a.iters)
            res.update(ms=t * 1e3, gbps=gbps(rw, t))

        elif a.arm == "roof_l1":
            t = timeit(dev, lambda: ttnn.clone(x, memory_config=L1), max(2, a.iters // 4))
            res.update(ms=t * 1e3, gbps=gbps(rw, t))

        elif a.arm == "perm_dram":
            t = timeit(dev, lambda: ttnn.permute(x, (1, 0, 2), memory_config=DRAM), a.iters)
            res.update(ms=t * 1e3, gbps=gbps(rw, t))
            # parity: the permute is a pure index reorder, so it must be bit-exact
            got = ttnn.to_torch(ttnn.permute(x, (1, 0, 2), memory_config=DRAM))
            res["bit_exact"] = bool(torch.equal(got, host.permute(1, 0, 2)))

        elif a.arm == "perm_l1":
            t = timeit(dev, lambda: ttnn.permute(x, (1, 0, 2), memory_config=L1),
                       max(2, a.iters // 4))
            res.update(ms=t * 1e3, gbps=gbps(rw, t))
            got = ttnn.to_torch(ttnn.permute(x, (1, 0, 2), memory_config=L1))
            res["bit_exact"] = bool(torch.equal(got, host.permute(1, 0, 2)))

        elif a.arm == "rm_chain":
            # untilize -> row-major permute -> tilize, each leg timed separately.
            xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            t_un = timeit(dev, lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), max(2, a.iters // 2))
            t_pm = timeit(dev, lambda: ttnn.permute(xr, (1, 0, 2), memory_config=DRAM),
                          max(2, a.iters // 2))
            pr = ttnn.permute(xr, (1, 0, 2), memory_config=DRAM)
            t_ti = timeit(dev, lambda: ttnn.to_layout(pr, ttnn.TILE_LAYOUT), max(2, a.iters // 2))
            res.update(untilize_ms=t_un * 1e3, untilize_gbps=gbps(rw, t_un),
                       rm_permute_ms=t_pm * 1e3, rm_permute_gbps=gbps(rw, t_pm),
                       tilize_ms=t_ti * 1e3, tilize_gbps=gbps(rw, t_ti),
                       chain_ms=(t_un + t_pm + t_ti) * 1e3,
                       piece_bytes=2 * C)
            got = ttnn.to_torch(ttnn.to_layout(pr, ttnn.TILE_LAYOUT))
            res["bit_exact"] = bool(torch.equal(got, host.permute(1, 0, 2)))

        elif a.arm == "rm_chain_l1":
            # same chain, but the two intermediates live in L1. 50 MB each; only one is
            # live at a time, and the op running over it is the only kernel on the grid,
            # so W7's 18.6 MB block-wide budget (which is what the block's OWN kernels
            # leave underneath their CBs) does not apply to a transient like this.
            xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=L1)
            t_un = timeit(dev, lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=L1), 2)
            t_pm = timeit(dev, lambda: ttnn.permute(xr, (1, 0, 2), memory_config=L1), 2)
            pr = ttnn.permute(xr, (1, 0, 2), memory_config=L1)
            t_ti = timeit(dev, lambda: ttnn.to_layout(pr, ttnn.TILE_LAYOUT, memory_config=DRAM), 2)
            res.update(untilize_ms=t_un * 1e3, untilize_gbps=gbps(rw, t_un),
                       rm_permute_ms=t_pm * 1e3, rm_permute_gbps=gbps(rw, t_pm),
                       tilize_ms=t_ti * 1e3, tilize_gbps=gbps(rw, t_ti),
                       chain_ms=(t_un + t_pm + t_ti) * 1e3, piece_bytes=2 * C)
            got = ttnn.to_torch(ttnn.to_layout(pr, ttnn.TILE_LAYOUT, memory_config=DRAM))
            res["bit_exact"] = bool(torch.equal(got, host.permute(1, 0, 2)))

        elif a.arm == "rm_permute":
            # row-major permute only -- the piece-size sweep arm. piece = 2*C bytes.
            xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x)
            t = timeit(dev, lambda: ttnn.permute(xr, (1, 0, 2), memory_config=DRAM), a.iters)
            res.update(ms=t * 1e3, gbps=gbps(rw, t), piece_bytes=2 * C)

        elif a.arm == "tile_permute":
            # tiled permute only -- the flat-in-C control arm. piece is 32 B for every C.
            t = timeit(dev, lambda: ttnn.permute(x, (1, 0, 2), memory_config=DRAM), a.iters)
            res.update(ms=t * 1e3, gbps=gbps(rw, t), piece_bytes=32)

        elif a.arm in ("blocked_dram", "blocked_l1"):
            R = a.rows
            assert R > 0
            mc = L1 if a.arm == "blocked_l1" else DRAM

            def run():
                parts = []
                for s in range(0, M, R):
                    e = min(s + R, M)
                    blk = ttnn.slice(x, [0, s, 0], [B, e, C])
                    p = ttnn.permute(blk, (1, 0, 2), memory_config=mc)
                    ttnn.deallocate(blk)
                    parts.append(p)
                out = ttnn.concat(parts, dim=0, memory_config=DRAM)
                for p in parts:
                    ttnn.deallocate(p)
                return out

            t = timeit(dev, run, max(2, a.iters // 4), warmup=1)
            res.update(ms=t * 1e3, gbps=gbps(rw, t),
                       live_block_MB=round(R * B * C * 2 / 2 ** 20, 2))
            got = ttnn.to_torch(run())
            res["bit_exact"] = bool(torch.equal(got, host.permute(1, 0, 2)))

        elif a.arm == "gen":
            # arbitrary permute on an arbitrary shape, DRAM and L1 destinations.
            dims = tuple(int(d) for d in a.perm.split(","))
            xg = ttnn.from_torch(hg, layout=ttnn.TILE_LAYOUT, device=dev, dtype=BF16,
                                 memory_config=DRAM)
            td = timeit(dev, lambda: ttnn.permute(xg, dims, memory_config=DRAM), a.iters)
            tl = timeit(dev, lambda: ttnn.permute(xg, dims, memory_config=L1), a.iters)
            res.update(perm=a.perm, shape=list(hg.shape), MB=round(hg.numel() * 2 / 2 ** 20, 3),
                       dram_ms=td * 1e3, dram_gbps=gbps(hg.numel() * 2 / 2 ** 20 * 2, td),
                       l1_ms=tl * 1e3, l1_gbps=gbps(hg.numel() * 2 / 2 ** 20 * 2, tl),
                       speedup=td / tl)

        else:
            raise SystemExit(f"unknown arm {a.arm}")

    except Exception as e:  # a CB clash / OOM is a RESULT, not a crash
        res["error"] = f"{type(e).__name__}: {e}"[:400]

    ttnn.close_device(dev)
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
