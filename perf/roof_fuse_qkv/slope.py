#!/usr/bin/env python3
"""The fused triangle-attention SDPA's time as a function of the bytes its readers move.

The Q fold cannot be priced by the ranked byte prize, because the prize has the wrong sign
(perf/roof_fuse_qkv/bytes.py). What it CAN be priced by is this kernel's own ms-per-MB: how much
time the program returns for a MB removed from, or charges for a MB added to, its input.

q_chunk is the axis. It moves k and v from 1 to 8 re-reads and the persistent mask read with it,
with the per-row arithmetic unchanged -- every row still attends over the same k and v in the same
k_chunk order, so this is a byte axis and not a math axis.

Interleaved round-robin over configs, warm, ttnn.synchronize_device on both edges of every timed
region, and the first config repeated as a second arm for the A/A floor.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ttnn
import torch

from tt_bio import tenstorrent as TT
from tt_bio import triatt_sdpa as TS
from tt_bio import sdpa_generic as SG

MB = 1e6
S = int(os.environ.get("ROOF_S", "512"))
H, DH = 4, 32
REPS = int(os.environ.get("ROOF_REPS", "5"))


def sdpa_bytes(p, q_chunk, elem=2):
    """DRAM bytes this program's readers and writer move, from the plan's own split."""
    B, NQH, Sq, Sk = p["B"], p["NQH"], p["Sq"], p["Sk"]
    cores = p["batch_pf"] * p["nh_pf"] * p["q_pf"]
    q = B * NQH * Sq * DH * elem
    kv = 2 * p["q_num_chunks"] * B * NQH * Sk * DH * elem
    # persistent mask: every core fills its own head's grid for its own q chunk, once.
    mask = cores * q_chunk * Sk * elem
    out = q
    return dict(q=q, kv=kv, mask=mask, out=out, total=q + kv + mask + out)


def main():
    dev = TT.get_device()
    grid = tuple(TT.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    print(f"host={os.uname().nodename} grid={grid} cores={cores} "
          f"measured={TT.COMPUTE_GRID_MEASURED} arch={dev.arch()}")

    torch.manual_seed(0)
    mk = lambda shape: ttnn.from_torch(  # noqa: E731
        torch.randn(*shape, dtype=torch.float32) * 0.1, layout=ttnn.TILE_LAYOUT,
        device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = mk([S, H, S, DH])
    k = mk([S, H, S, DH])
    v = mk([S, H, S, DH])
    bias = mk([1, H, S, S])
    scale = DH ** -0.5

    k_chunk = int(os.environ.get("ROOF_KCHUNK", "0")) or None
    configs = []
    for qc in (512, 256, 128, 64):
        if qc > S or S % qc:
            continue
        kc = k_chunk or qc
        if S % kc:
            continue
        q_pf = TS.q_parallel_factor(S, H, qc, cores)
        split = (cores // (H * q_pf), H, q_pf)
        if split[0] * H * q_pf > cores or split[0] < 1:
            continue
        p = SG.plan(q, k, v, bias, q, qc, kc, grid, (ttnn.MathFidelity.HiFi2, True, False, False),
                    scale, split)
        if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
            print(f"  skip q_chunk={qc}: q_per_core={p['q_per_core']} "
                  f"nh_per_core={p['nh_per_core']} padded={p['use_padded_mask']}")
            continue
        configs.append((qc, kc, p))

    if not configs:
        raise SystemExit("no servable config")

    arms = [(f"q{qc}k{kc}", qc, kc, p) for qc, kc, p in configs]
    # A/A: the first config again, under a second name, timed in the same round robin.
    arms.append((f"AA:{arms[0][0]}", arms[0][1], arms[0][2], arms[0][3]))

    def once(qc, kc):
        o = TS.sdpa(q, k, v, bias, scale, qc, kc)
        return o

    served = {}
    for name, qc, kc, _p in arms:
        o = once(qc, kc)
        served[name] = o is not None
        if o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    print("served:", {n: served[n] for n in served})
    if TS.REJECTS:
        print("gate declines:", {f"{k[0]}@{k[1]}": v for k, v in TS.REJECTS.items()})
    for kk, vv in TS.PM_L1_ERRORS.items():
        print("  L1 refusal", kk, str(vv)[:200])
    arms = [a for a in arms if served[a[0]]]
    if not arms:
        raise SystemExit("gate declined every config")

    # warm a second time: program cache populated, JIT done
    for _name, qc, kc, _p in arms:
        o = once(qc, kc)
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)

    samples = {a[0]: [] for a in arms}
    for r in range(REPS):
        for name, qc, kc, _p in arms:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = once(qc, kc)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            ttnn.deallocate(o)
            samples[name].append((t1 - t0) * 1e3)
        print(f"  rep {r + 1}/{REPS}: " + "  ".join(
            f"{n}={samples[n][-1]:.3f}" for n, *_ in arms), flush=True)

    rows = []
    for name, qc, kc, p in arms:
        b = sdpa_bytes(p, qc)
        ms = statistics.median(samples[name])
        rows.append(dict(name=name, q_chunk=qc, k_chunk=kc, q_pf=p["q_pf"],
                         q_num_chunks=p["q_num_chunks"], k_num_chunks=p["k_num_chunks"],
                         cores=p["batch_pf"] * p["nh_pf"] * p["q_pf"],
                         bytes=b, total_mb=b["total"] / MB, ms=ms,
                         samples=samples[name],
                         gbps=b["total"] / (ms * 1e-3) / 1e9))
    print()
    print(f"{'arm':14s} {'q_chunk':>7s} {'cores':>5s} {'MB':>9s} {'ms':>9s} {'GB/s':>7s}")
    for r in rows:
        print(f"{r['name']:14s} {r['q_chunk']:7d} {r['cores']:5d} {r['total_mb']:9.1f} "
              f"{r['ms']:9.3f} {r['gbps']:7.1f}")

    base = [r for r in rows if not r["name"].startswith("AA:")]
    aa = [r for r in rows if r["name"].startswith("AA:")]
    if aa:
        ref = [r for r in base if r["name"] == aa[0]["name"][3:]][0]
        print(f"\nA/A floor: {abs(aa[0]['ms'] - ref['ms']):.3f} ms "
              f"({abs(aa[0]['ms'] - ref['ms']) / ref['ms'] * 100:.2f} %)")

    if len(base) >= 2:
        n = len(base)
        mx = sum(r["total_mb"] for r in base) / n
        my = sum(r["ms"] for r in base) / n
        sxy = sum((r["total_mb"] - mx) * (r["ms"] - my) for r in base)
        sxx = sum((r["total_mb"] - mx) ** 2 for r in base)
        slope = sxy / sxx
        print(f"slope: {slope * 1000:.3f} us/MB  (intercept {my - slope * mx:.3f} ms)")
        print(f"       vs 435.2 GB/s Blackhole DRAM roof = 2.298 us/MB "
              f"-> {slope * 1000 / 2.298 * 100:.1f} % of roof rate")
    else:
        slope = None

    # What the shipped ladder itself picks at this size, so the sweep is anchored to production.
    TT.SDPA_K_CHUNK_STATS[0] = TT.SDPA_K_CHUNK_STATS[1] = 0
    o = TT._tri_att_sdpa(q, k, v, bias, scale)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)
    picks = {str(kk): vv for kk, vv in TT.SDPA_CHUNK_PICKS.items()}
    picks["routes"] = dict(TT.SDPA_ROUTE_COUNTS)
    print("shipped ladder picks:", picks or "(no SDPA_PICKS table)")

    out = dict(host=os.uname().nodename, grid=list(grid), S=S, H=H, DH=DH, reps=REPS,
               rows=rows, slope_us_per_mb=(slope * 1000 if slope else None),
               picks=picks, k_chunk_fixed=k_chunk)
    tag = f"{S}_k{k_chunk}" if k_chunk else str(S)
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"slope_{tag}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
