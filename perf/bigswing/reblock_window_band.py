#!/usr/bin/env python3
"""Where does the reblock_permute L1 window actually win on this grid? (exec runbook step 2)

Pass 18 measured two points and found 1.2999x at N=512 L1->L1, bit-exact, against a window that
closes at 352. The window's own docstring says it closes there because "Nt=12 puts 144 groups on
130 cores" -- a qb1 13x10 argument, and qb2 is 11x10. So the band is measured here rather than
raised to a round number: every N is forced through the kernel, timed against ttnn.permute in the
same process, and torch.equal-checked, and the window is then set to the contiguous winning run.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
        ~/tt-bio/env/bin/python3 perf/bigswing/reblock_window_band.py \
            --out perf/bigswing/reblock_window_band_qb2c0.json

Original pass-18 header follows.

Does the merged reblock_permute kernel serve the 512 aa trimul channel moves? (pass 18)

The 512 aa `--fast` census records 55 permute calls per Pairformer block and ZERO generic_op calls,
so `tt_bio/reblock_permute.py` -- this repo's own merged hand-written channel-move kernel -- serves
none of them. Three census rows are the forward/inverse trimul channel move at
[1,512,512,32] bf16 L1 -> L1, together 4.899 s/fold. `eligible()` opens its L1 leg only for
288 <= N <= 352, and the census shape is N=512.

This probe answers two questions with one process:
  1. the exact reject reason at the production shape (REJECTS), and
  2. what the kernel is worth there if the window is opened -- forced, timed against ttnn.permute
     in the same process, and torch.equal-checked.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
        ~/tt-bio/env/bin/python3 perf/bigswing/reblock_window_512.py --out perf/bigswing/reblock_window_512_qb2c0.json
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
import tt_bio.reblock_permute as rp                  # noqa: E402

WARM, REPS = 3, 7


def med(fn):
    """Median of REPS, every intermediate output freed.

    Freeing matters and is not hygiene: reblock_permute allocates its own output, so holding
    warm + timed results live grows the allocator's occupancy across the run and by a different
    amount in each arm. At [1,512,512,32] in L1 that is 16.8 MB per held result against a 157 MB
    bank total, which is enough to move the arm being measured. The last result is returned live
    for the torch.equal check and freed by the caller.
    """
    ts = []
    for _ in range(WARM + REPS):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(DEV)
        dt = time.perf_counter() - t0
        if _ >= WARM:
            ts.append(dt)
        if _ < WARM + REPS - 1:
            ttnn.deallocate(o)
    return statistics.median(ts), o


def mc(buf):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buf)


def run(n, c, buf_in, buf_out, results):
    tag = f"N={n} C={c} {buf_in.name}->{buf_out.name}"
    t = torch.randn(1, n, n, c, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                        memory_config=mc(buf_in))
    out_mc = mc(buf_out)

    rp.REJECTS.clear(); rp.STATS[0] = rp.STATS[1] = 0
    elig = rp.eligible(x, out_mc)
    rejects = {f"{k[0]}|{list(k[1])}": v for k, v in rp.REJECTS.items()}

    t_stock, o_stock = med(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=out_mc))
    ref = ttnn.to_torch(o_stock)
    ttnn.deallocate(o_stock)

    row = dict(tag=tag, n=n, c=c, buf_in=buf_in.name, buf_out=buf_out.name,
               eligible=bool(elig), rejects=rejects, ms_stock=1e3 * t_stock)
    try:
        t_kern, o_kern = med(lambda: rp.reblock_permute(x, out_mc))
        got = ttnn.to_torch(o_kern)
        ttnn.deallocate(o_kern)
        row.update(ms_kernel=1e3 * t_kern, speedup=t_stock / t_kern,
                   bit_exact=bool(torch.equal(ref, got)))
    except Exception as e:                                            # noqa: BLE001
        row.update(kernel_error=f"{type(e).__name__}: {e}"[:400])
    ttnn.deallocate(x)
    mb = n * n * c * 2 / 1e6
    row["MB"] = mb
    row["GBs_stock_rw"] = 2 * mb / 1e3 / (row["ms_stock"] / 1e3)
    if "ms_kernel" in row:
        row["GBs_kernel_rw"] = 2 * mb / 1e3 / (row["ms_kernel"] / 1e3)
    results.append(row)
    print(json.dumps(row)[:600], flush=True)


def run_inverse(n, c, buf, results):
    """(0,2,3,1) at [1,C,N,N] -- tenstorrent.py:1817. No custom kernel exists for this direction."""
    t = torch.randn(1, c, n, n, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                        memory_config=mc(buf))
    ms, o = med(lambda: ttnn.permute(x, (0, 2, 3, 1), memory_config=mc(buf)))
    ttnn.deallocate(o); ttnn.deallocate(x)
    mb = n * n * c * 2 / 1e6
    row = dict(tag=f"inverse (0,2,3,1) N={n} C={c} {buf.name}", n=n, c=c, buf_in=buf.name,
               buf_out=buf.name, ms_stock=ms * 1e3, MB=mb, GBs_stock_rw=2 * mb / 1e3 / ms)
    results.append(row)
    print(json.dumps(row)[:400], flush=True)


def run_clone(n, c, buf, results):
    t = torch.randn(1, c, n, n, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                        memory_config=mc(buf))
    ms, o = med(lambda: ttnn.clone(x, memory_config=mc(buf)))
    ttnn.deallocate(o); ttnn.deallocate(x)
    mb = n * n * c * 2 / 1e6
    row = dict(tag=f"clone roof N={n} C={c} {buf.name}", ms_stock=ms * 1e3, MB=mb,
               GBs_stock_rw=2 * mb / 1e3 / ms)
    results.append(row)
    print(json.dumps(row)[:300], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="320,352,384,416,448,480,512,544",
                    help="comma list of N to sweep, C=32, forced through the kernel at every N")
    ap.add_argument("--c", type=int, default=32)
    a = ap.parse_args()
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    DEV = get_device()
    g = DEV.compute_with_storage_grid_size()
    ncores = g.x * g.y
    print(f"grid {g.x}x{g.y} = {ncores} cores  loadavg: {up}", flush=True)
    res = []
    D, L = ttnn.BufferType.DRAM, ttnn.BufferType.L1
    sizes = [int(s) for s in a.sizes.split(",")]
    for n in sizes:
        nt = (n + rp.TILE_H - 1) // rp.TILE_H
        plan = rp._split_plan(DEV, nt * nt)
        print(f"--- N={n} Nt={nt} groups={nt*nt} split_plan={'ok' if plan else 'None'}", flush=True)
        run(n, a.c, L, L, res)
        res[-1].update(nt=nt, groups=nt * nt, split_plan=plan is not None, ncores=ncores)
        run(n, a.c, D, D, res)
        res[-1].update(nt=nt, groups=nt * nt, split_plan=plan is not None, ncores=ncores)
    json.dump(dict(host="qb2", card=0, ttnn="0.68.0", grid=[g.x, g.y], ncores=ncores,
                   uptime=up, warm=WARM, reps=REPS, sizes=sizes, c=a.c, rows=res),
              open(a.out, "w"), indent=1)
    print("wrote", a.out)

    l1 = [r for r in res if r["buf_out"] == "L1"]
    print("\nL1->L1 band:")
    for r in l1:
        print(f"  N={r['n']:4d} Nt={r['nt']:3d} groups={r['groups']:5d} "
              f"stock {r['ms_stock']:.4f} kernel {r.get('ms_kernel', float('nan')):.4f} "
              f"speedup {r.get('speedup', float('nan')):.4f} exact={r.get('bit_exact')}")
