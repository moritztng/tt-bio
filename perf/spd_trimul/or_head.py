#!/usr/bin/env python3
"""Compute headroom of RFD4's outer_residual (the pair write that produces the z trimul_out's T1 normalises).

LN statistics forwarding moves T1's row mean / rstd into the kernel that writes z. It only pays if that kernel has idle
compute: this times outer_residual on a bf16 z (the pair_bf16z stream) with its data movement on (DM 0) and with reads
and writes skipped (DM 7, outer_residual.DM bits 1 z reads, 2 pj rows, 4 writes), AICLK sampled during the timed calls.
DM 0 minus DM 7 is the time compute could absorb before the program gets longer.

    python perf/spd_trimul/or_head.py --ns 352,256 --c 384 --out or.jsonl

--stats adds the producer itself (outer_residual `stats_eps`): its time on the same z (stats_ms; minus dm0_ms is what
forwarding costs the producer) and its statistics against float64 ones of the bf16 output it wrote: mean error in units
of the row's std, rstd relative error, both rms and max, and the same for T1's own arithmetic emulated on the host
(bf16-truncated 1/K scaler) so the two can be compared. z gets a per-row offset (--mu, in std units) because real pair
rows have |mean| several times their std.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/ for clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--dm", default="0,1,4,7")
ap.add_argument("--stats", action="store_true")
ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--mu", type=float, default=3.0)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, outer_residual as OR
import clocksample

dev = T.get_device()


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


gen = torch.Generator().manual_seed(0)
for n in (int(v) for v in A.ns.split(",")):
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z = put(torch.randn(1, n, n, A.c, generator=gen) + A.mu * torch.randn(1, n, n, 1, generator=gen))
    pi = put(torch.randn(1, n, A.c, generator=gen))
    pj = put(torch.randn(1, n, A.c, generator=gen))
    rec = {"n": n, "c": A.c, "bytes_mb": round(2 * 2 * n * n * A.c / 1e6, 1)}
    clk = clocksample.during(period=0.5).__enter__()
    for dm in (int(v) for v in A.dm.split(",")):
        OR.DM = dm
        outs = []

        def f():
            r = OR.outer_residual(z, pi, pj)
            assert r is not None, "outer_residual declined"
            outs.append(r[0])
            while len(outs) > 1:
                ttnn.deallocate(outs.pop(0))
        rec[f"dm{dm}_ms"] = timed(f, A.reps)
        for t in outs:
            ttnn.deallocate(t)
    OR.DM = 0
    if A.stats:
        outs = []

        def g():
            r = OR.outer_residual(z, pi, pj, stats_eps=A.eps)
            assert r is not None and len(r) == 3, "outer_residual declined"
            outs.append(r)
            while len(outs) > 1:
                o = outs.pop(0)
                ttnn.deallocate(o[0]); ttnn.deallocate(o[2])
        rec["stats_ms"] = timed(g, A.reps)
        zo, _, st = outs.pop()
        x = ttnn.to_torch(zo).double().reshape(n, n, A.c)                  # [i, j, c], the bf16 z T1 reads
        s = ttnn.to_torch(st).double().reshape(n, n // 32, 64)              # [i, jt, 32 means | 32 rstd]
        ttnn.deallocate(zo); ttnn.deallocate(st)
        dm, dr = s[..., :32].reshape(n, n), s[..., 32:].reshape(n, n)
        mu, var = x.mean(-1), x.var(-1, unbiased=False)
        sd, rs = var.sqrt(), (var + A.eps).rsqrt()
        inv = torch.tensor([1.0 / A.c], dtype=torch.float32).view(torch.int32) & -65536
        k = float(inv.view(torch.float32))                                   # T1's bf16-truncated 1/K
        em = x.sum(-1) * k
        er = ((x - em[..., None]).pow(2).sum(-1) * k + A.eps).rsqrt()
        for tag, m_, r_ in (("dev", dm, dr), ("t1emu", em, er)):
            me, re = (m_ - mu) / sd, r_ / rs - 1
            rec[f"{tag}_mean_err_rms"] = float(me.pow(2).mean().sqrt())
            rec[f"{tag}_mean_err_max"] = float(me.abs().max())
            rec[f"{tag}_rstd_rel_rms"] = float(re.pow(2).mean().sqrt())
            rec[f"{tag}_rstd_rel_max"] = float(re.abs().max())
            rec[f"{tag}_rstd_rel_mean"] = float(re.mean())
    clk.__exit__(None, None, None)
    rec["aiclk"] = clk.summary()
    if "dm0_ms" in rec:
        rec["GBps"] = round(rec["bytes_mb"] / rec["dm0_ms"], 1)
    for t in (z, pi, pj):
        ttnn.deallocate(t)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as fh:
            fh.write(line + "\n")
