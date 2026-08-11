#!/usr/bin/env python3
"""Exec step 4, stage 1: an off-fold MinimalMatmulConfig sweep at the three shapes the fold runs.

The census (perf/bigswing/mmcfg/census512_qb2c0.json) puts every minimal_matmul second at 512 aa on
three call sites, all DRAM-out:

    tenstorrent.py:1700  [1,512,512,256] x [256,128]   8384 calls   6.199 s screened
    tenstorrent.py:2097    [512,512,256] x [256,768]   1048 calls   2.561 s
    tenstorrent.py:2103    [512,512,256] x [256,256]   1048 calls   1.024 s

No config table exists, so this builds one: every legal (M_block, K_block, N_block, subblock_h,
subblock_w) at each shape, timed against the unconfigured default in the same process and
torch.equal-checked against it. Legality is mm_census._legal's rule, applied before the op sees it so
a sweep never dies on the table.

Screen only -- a per-call ratio here is not a fold gain and step 4's fold A/B is what decides.
"""
import argparse, itertools, json, statistics, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

SHAPES = [
    ("site1700", (1, 512, 512, 256), (256, 128)),
    ("site2097", (512, 512, 256), (256, 768)),
    ("site2103", (512, 512, 256), (256, 256)),
]
WARM, REPS = 2, 5


def divisors(n, cap=64):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def legal(M, K, N, sh, sw, mt, kt, nt):
    if M % sh or N % sw:
        return False
    if sh * sw > 8:
        return False
    return mt % M == 0 and nt % N == 0 and kt % K == 0


def med(dev, fn):
    for _ in range(WARM):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts, o = [], None
    for i in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i < REPS - 1:
            ttnn.deallocate(o)
    return statistics.median(ts), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="11,10")
    a = ap.parse_args()
    gx, gy = (int(v) for v in a.grid.split(","))
    dev = get_device()
    res = {"host": "qb2", "chip": 0, "ttnn": "0.68.0", "grid": [gx, gy],
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "warm": WARM, "reps": REPS, "note": "screen; per-call ratios, not fold gains",
           "sites": []}
    torch.manual_seed(0)

    for name, ish, wsh in SHAPES:
        mt = 1
        for d in ish[:-1]:
            mt *= d
        mt //= 32
        kt, nt = ish[-1] // 32, wsh[-1] // 32
        x = ttnn.from_torch(torch.randn(*ish, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(*wsh, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        base_ms, o = med(dev, lambda: ttnn.experimental.minimal_matmul(
            x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ref = ttnn.to_torch(o); ttnn.deallocate(o)
        site = {"site": name, "in": list(ish), "w": list(wsh), "m_tiles": mt, "k_tiles": kt,
                "n_tiles": nt, "base_ms": 1e3 * base_ms, "arms": []}
        print(f"\n== {name} {ish}x{wsh}  mt={mt} kt={kt} nt={nt}  base {1e3*base_ms:.4f} ms",
              flush=True)

        cands = []
        for M in divisors(mt, 32):
            for K in divisors(kt):
                for N in divisors(nt):
                    for sh in divisors(M, 8):
                        for sw in divisors(N, 8):
                            if legal(M, K, N, sh, sw, mt, kt, nt):
                                cands.append((M, K, N, sh, sw))
        # keep the sweep bounded: the block shape is what matters, so take every (M,K,N) with its
        # largest legal subblock rather than every subblock of every block.
        best_sub = {}
        for M, K, N, sh, sw in cands:
            best_sub[(M, K, N)] = max(best_sub.get((M, K, N), (0, 0)), (sh, sw), key=lambda t: t[0] * t[1])
        cands = [(M, K, N, sh, sw) for (M, K, N), (sh, sw) in best_sub.items()]
        cands.sort()
        site["n_candidates"] = len(cands)
        print(f"   {len(cands)} candidates", flush=True)

        for M, K, N, sh, sw in cands:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
            row = {"M": M, "K": K, "N": N, "sh": sh, "sw": sw}
            try:
                ms, o = med(dev, lambda: ttnn.experimental.minimal_matmul(
                    x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, config=cfg))
                got = ttnn.to_torch(o); ttnn.deallocate(o)
                row.update(ms=1e3 * ms, speedup=base_ms / ms, exact=bool(torch.equal(got, ref)))
            except Exception as e:                                          # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            site["arms"].append(row)
        ok = [r for r in site["arms"] if r.get("exact")]
        ok.sort(key=lambda r: -r["speedup"])
        site["best"] = ok[:5]
        for r in ok[:5]:
            print(f"   M={r['M']:3d} K={r['K']} N={r['N']:2d} sub={r['sh']}x{r['sw']}  "
                  f"{r['ms']:.4f} ms  {r['speedup']:.4f}x", flush=True)
        if not ok:
            print("   no bit-exact config beat nothing -- none legal or all threw", flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)
        res["sites"].append(site)

    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nwrote", a.out)


main()
