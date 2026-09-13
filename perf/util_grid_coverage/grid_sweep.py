#!/usr/bin/env python3
"""Measured core-grid scaling curve for the partial-grid op classes of the 512 aa Boltz-2 fold.

`grid_census.py` says which classes run on fewer than 110 cores. This says what the idle cores are
WORTH: each class is rebuilt at its shipped shape, dtype, fidelity, memory config and activation,
and the matmul program config is written out explicitly so the engaged core count is exact by
construction rather than inferred. The in-situ config is always one of the swept points, so the
curve carries its own check: that point must reproduce the census's device kernel time.

A core-count ratio is not a speedup. The whole reason this sweep exists is that the per-core work
quantum (per_core_M x per_core_N tiles) is what sets the critical path, and a wider grid that
leaves the quantum unchanged buys nothing.

Card: qb2 physical 2, one Blackhole of a p300c, 11x10 = 110 worker cores, ttnn 0.68.0 wheel.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
import ttnn

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
GRID_X, GRID_Y = 11, 10
WARM = 3
# Blackhole p300c idles at 800 MHz and boosts to 1350 under sustained load. A short isolated
# burst therefore measures the IDLE clock and reads 1.6875x slow against anything taken inside a
# real fold. Every timed point is preceded by a burn that holds the device busy, and the in-situ
# point is checked against the census to prove the clock actually boosted.
BURN_SHAPE = (1, 256, 1024, 1024)

# name, in0, in1, in0 mem, in1 mem, out mem, activation, kind(1d|2d), fuse_batch, in0_block_w,
# in-situ (cores, per_core_M, per_core_N), programs/block-or-step, census kernel ns (median),
# s/fold at the in-situ grid
CASES = [
    dict(name="B1-transition-fc1", in0=(1, 16, 512, 128), in1=(1, 1, 128, 512), m0=L1, m1=DRAM,
         mo=L1, act="silu", kind="1d", fb=True, ibw=1, situ=(86, 3, 16), n=64,
         census_ns=113587, sfold=1.177),
    dict(name="S1-dit-attn-proj", in0=(1, 1, 512, 768), in1=(1, 1, 768, 768), m0=DRAM, m1=DRAM,
         mo=DRAM, act=None, kind="2d", fb=True, ibw=4, situ=(64, 2, 3), n=97,
         census_ns=14723, sfold=0.330),
    dict(name="B2-trimul-batched", in0=(1, 128, 512, 512), in1=(1, 128, 512, 512), m0=DRAM,
         m1=DRAM, mo=DRAM, act=None, kind="2d", fb=False, ibw=8, situ=(64, 2, 2), n=2,
         census_ns=816791, sfold=0.422),
    dict(name="S4-dit-ff-up", in0=(1, 1, 512, 768), in1=(1, 1, 768, 1536), m0=L1, m1=DRAM,
         mo=L1, act=None, kind="2d", fb=True, ibw=4, situ=(80, 2, 5), n=76,
         census_ns=21921, sfold=0.338),
    dict(name="B3-transition-fc2", in0=(1, 16, 512, 512), in1=(1, 1, 512, 128), m0=L1, m1=DRAM,
         mo=DRAM, act=None, kind="1d", fb=True, ibw=1, situ=(86, 3, 4), n=32,
         census_ns=35714, sfold=0.302),
    dict(name="S5-dit-ff-up-4x", in0=(1, 1, 512, 768), in1=(1, 1, 768, 3072), m0=DRAM, m1=DRAM,
         mo=DRAM, act=None, kind="2d", fb=True, ibw=4, situ=(88, 2, 9), n=24,
         census_ns=39615, sfold=0.190),
    dict(name="S6-atom-proj", in0=(1, 140, 32, 128), in1=(1, 1, 128, 256), m0=DRAM, m1=DRAM,
         mo=DRAM, act=None, kind="1d", fb=True, ibw=1, situ=(70, 2, 8), n=18,
         census_ns=24594, sfold=0.088),
    dict(name="S3-dit-layernorm", in0=(1, 1, 512, 768), in1=None, m0=DRAM, m1=None, mo=DRAM,
         act=None, kind="ln", fb=None, ibw=None, situ=(16, None, None), n=100,
         census_ns=14449, sfold=0.318),
]


def tiles(n):
    return (n + 31) // 32


def subblocks(pcm, pcn):
    """Largest legal out-subblock. fp32_dest_acc halves DST, so h*w <= 4."""
    best = (1, 1)
    for h in range(1, pcm + 1):
        for w in range(1, pcn + 1):
            if pcm % h or pcn % w or h * w > 4:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def _build(ctor, kw, pcm, pcn):
    """0.68 takes out_block_h/w; older bindings do not. nanobind exposes no signature, so ask."""
    try:
        return ctor(out_block_h=pcm, out_block_w=pcn, **kw)
    except TypeError:
        return ctor(**kw)


def cfg2d(rows, cols, pcm, pcn, ibw, fb, act):
    sh, sw = subblocks(pcm, pcn)
    kw = dict(compute_with_storage_grid_size=(cols, rows), in0_block_w=ibw,
              out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn,
              transpose_mcast=False, fused_activation=act, fuse_batch=fb)
    return _build(ttnn.MatmulMultiCoreReuseMultiCastProgramConfig, kw, pcm, pcn)


def cfg1d(cores, pcm, pcn, ibw, fb, act):
    sh, sw = subblocks(pcm, pcn)
    kw = dict(compute_with_storage_grid_size=(GRID_X, GRID_Y), in0_block_w=ibw,
              out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn,
              fuse_batch=fb, fused_activation=act, mcast_in0=False)
    return _build(ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig, kw, pcm, pcn)


def ladder(case):
    """Every core count the op's own tile counts can be split into, in-situ point included."""
    Mt, Nt = tiles(case["in0"][2]), tiles(case["in1"][3])
    batch = case["in0"][0] * case["in0"][1]
    pts = []
    if case["kind"] == "1d":
        Mtot = Mt * batch if case["fb"] else Mt
        for pcm in range(1, Mtot + 1):
            cores = -(-Mtot // pcm)
            if cores > GRID_X * GRID_Y:
                continue
            if pts and pts[-1][0] == cores:
                continue
            pts.append((cores, pcm, Nt, None, None))
        pts = sorted({p[0]: p for p in pts}.values())
    else:
        for rows in range(1, GRID_Y + 1):
            for cols in range(1, GRID_X + 1):
                pcm, pcn = -(-Mt // rows), -(-Nt // cols)
                same = (rows > 1 and -(-Mt // (rows - 1)) == pcm) or \
                       (cols > 1 and -(-Nt // (cols - 1)) == pcn)
                # keep one padded point per (rows, cols) maximum: a wider grid whose quantum
                # cannot shrink is the decisive test of whether coverage alone buys anything
                if same and not (rows == GRID_Y or cols == GRID_X):
                    continue
                pts.append((rows * cols, pcm, pcn, rows, cols))
        best = {}
        for c, pcm, pcn, rows, cols in pts:
            if c not in best or (rows * cols > best[c][3] * best[c][4]):
                best[c] = (c, pcm, pcn, rows, cols)
        pts = sorted(best.values())
    return thin(pts, case["situ"][0])


def thin(pts, situ, keep=12):
    """A log-spaced sample of the ladder, with the in-situ point and both ends always kept."""
    if len(pts) <= keep:
        return pts
    idx = {0, len(pts) - 1}
    idx |= {i for i, p in enumerate(pts) if p[0] == situ}
    lo, hi = pts[0][0], pts[-1][0]
    import math
    for k in range(keep - len(idx)):
        target = lo * (hi / lo) ** ((k + 1) / (keep - len(idx) + 1))
        idx.add(min(range(len(pts)), key=lambda i: abs(math.log(pts[i][0] / target))))
    return [pts[i] for i in sorted(idx)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reps", type=int, default=0, help="fixed reps/group (profiler mode)")
    ap.add_argument("--groups", type=int, default=5)
    a = ap.parse_args()

    device = ttnn.open_device(device_id=0)
    res = []
    burn0 = ttnn.from_torch(torch.randn(*BURN_SHAPE) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    burn1 = ttnn.from_torch(torch.randn(1, 1, 1024, 1024) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    def boost(n=3):
        """Hold the device busy so AICLK is at 1350, not the 800 MHz idle."""
        for _ in range(n):
            ttnn.deallocate(ttnn.matmul(burn0, burn1, memory_config=DRAM,
                                        compute_kernel_config=HIFI4))
        ttnn.synchronize_device(device)
    try:
        for case in CASES:
            if a.only and a.only not in case["name"]:
                continue
            t0 = ttnn.from_torch(torch.randn(*case["in0"]) * 0.1, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device,
                                 memory_config=case["m0"])
            t1 = (ttnn.from_torch(torch.randn(*case["in1"]) * 0.1, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device,
                                  memory_config=case["m1"]) if case["in1"] else None)
            wt = ttnn.from_torch(torch.ones(1, 1, 32, case["in0"][3]), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
            act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if case["act"] else None
            print(f"\n== {case['name']}  in0={case['in0']} in1={case['in1']} "
                  f"in-situ {case['situ'][0]} cores, census {case['census_ns'] / 1e3:.2f} us")

            if case["kind"] == "ln":
                pts = [(16, None, None, None, None)]
            else:
                pts = ladder(case)
            print(f"   ladder: {[q[0] for q in pts]}")
            for (cores, pcm, pcn, rows, cols) in pts:
                def call():
                    if case["kind"] == "ln":
                        return ttnn.layer_norm(t0, weight=wt, epsilon=1e-5,
                                               memory_config=case["mo"],
                                               compute_kernel_config=HIFI4)
                    pc = (cfg1d(cores, pcm, pcn, case["ibw"], case["fb"], act)
                          if case["kind"] == "1d"
                          else cfg2d(rows, cols, pcm, pcn, case["ibw"], case["fb"], act))
                    return ttnn.matmul(t0, t1, program_config=pc, memory_config=case["mo"],
                                       dtype=ttnn.bfloat16, compute_kernel_config=HIFI4)
                try:
                    for _ in range(WARM):
                        ttnn.deallocate(call())
                    ttnn.synchronize_device(device)
                except Exception as e:                            # noqa: BLE001
                    msg = str(e).replace("\n", " ")[:120]
                    res.append(dict(case=case["name"], cores=cores, per_core_M=pcm,
                                    per_core_N=pcn, refused=msg))
                    print(f"   {cores:>4} cores  pc={pcm}x{pcn}  REFUSED {msg[:80]}")
                    continue
                # Back-to-back with ONE sync at the end of the region: per-call sync would
                # measure dispatch latency, not the device. The census's own numbers come from
                # back-to-back dispatch, so this is the comparable region.
                boost()
                s0 = time.perf_counter()
                ttnn.deallocate(call())
                ttnn.synchronize_device(device)
                one_us = (time.perf_counter() - s0) * 1e6
                reps = a.reps or max(3, min(40, int(20000 / max(one_us, 1))))
                laps = []
                for _ in range(a.groups):
                    boost()
                    s0 = time.perf_counter()
                    for _ in range(reps):
                        ttnn.deallocate(call())
                    ttnn.synchronize_device(device)
                    laps.append((time.perf_counter() - s0) * 1e6 / reps)
                med = st.median(laps)
                res.append(dict(case=case["name"], cores=cores, per_core_M=pcm, per_core_N=pcn,
                                rows=rows, cols=cols, us=med, us_min=min(laps), us_max=max(laps),
                                reps_per_group=reps, groups=5,
                                quantum=(pcm * pcn) if pcm else None,
                                in_situ=(cores == case["situ"][0]),
                                census_ns=case["census_ns"], sfold=case["sfold"], n=case["n"]))
                flag = "  <- in situ" if cores == case["situ"][0] else ""
                print(f"   {cores:>4} cores  pc={pcm}x{pcn} quantum={pcm * pcn if pcm else '-':>4}"
                      f"  {med:9.2f} us  [{min(laps):.2f}, {max(laps):.2f}] x{reps}{flag}")
            for t in (t0, t1, wt):
                if t is not None:
                    ttnn.deallocate(t)
            a.out.write_text(json.dumps(res, indent=1))
    finally:
        ttnn.close_device(device)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
