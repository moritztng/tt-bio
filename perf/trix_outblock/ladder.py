#!/usr/bin/env python3
"""The `out_block_h` ladder at the shapes a 512 aa fold actually runs.

Reads `census512.json`, takes the derived-config matmul keys that carry the FLOPs, and for each
one measures:

    derived      ttnn's own choice (core_grid=CORE_GRID_MAIN), twice -- the arm and its A/A twin
    mirror       an explicit config built by derive.py to equal ttnn's choice -- the IDENTITY
                 control. If this does not land on `derived` within the A/A floor, the mirror is
                 wrong and every other arm of that shape is uninterpretable.
    obh<h>       the same config with ONLY out_block_h moved, over the divisors of per_core_M
    ibw<w>       the same config with ONLY in0_block_w moved, over the divisors of k_tiles --
                 the neighbouring drain/contraction parameter, so the row can say what is alive
                 when out_block_h is not
    2d_obh<h>    for a shape ttnn routes to the 1D systolic path, the 2D alternative and its own
                 obh ladder. fixterm's key-A number compared a hand-built 2D config against a
                 derived 1D one, so this is the arm that separates "out_block_h" from
                 "config class".

Instrument: the n-ladder, never a single synced bracket. `sync; n calls; sync` costs L + n*c with
L = 0.05142 ms of host launch/drain that the chip never pays and that does not scale with the
shape (`synced-bracket-inflates-op-level-fixed-cost`). Every arm is measured at two bracket sizes
and the marginal is (t_hi - t_lo) / (hi - lo), which cancels L exactly with no assumption about
its size. Arms are interleaved rep by rep, because a sequential sweep bakes in compile and warm-up
drift (`op-ab-must-interleave-arms-compile-warmup-bias`).

Kernel config is identical across every arm of a shape. The same cube reads 1.40x apart on
fp32_dest_acc_en/packer_l1_acc, so an arm that moves those is measuring two things.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

from derive import build, derive, divisors                                     # noqa: E402

# name, batch, M, K, N, dtype, in0 residency, out residency, share of the fold's derived FLOPs
SHAPES = [
    ("SW1", 16, 512, 256, 1024, "bf16", "L1", "L1", 52.74),      # swiglu fc1/fc2, 33536 calls
    ("SW3", 16, 512, 1024, 256, "bf16", "L1", "DRAM", 26.37),    # swiglu fc3, 16768 calls
    ("DIT", 1, 512, 768, 1536, "fp32", "DRAM", "DRAM", 4.25),    # token DiT, 9600 calls
    ("OPM", 512, 512, 1024, 256, "bf16", "DRAM", "DRAM", 2.01),  # outer_product_mean, 40 calls
    ("KA", 16, 512, 128, 512, "bf16", "DRAM", "DRAM", 0.0),      # fixterm key A, the repro
    ("KB", 16, 512, 512, 128, "bf16", "DRAM", "DRAM", 0.0),      # fixterm key B, falsifier (a)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--shapes", default="SW1,SW3,DIT,OPM,KA,KB")
    ap.add_argument("--l1-budget-mb", type=float, default=100.0,
                    help="how much L1 one bracket may hold in live outputs. Too small a "
                         "budget collapses the n-ladder to hi-lo=1 and the A/A floor blows "
                         "out: SW1 first read a 12.81 %% floor at n=1/2.")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    want = set(a.shapes.split(","))

    import torch
    import ttnn
    import clk
    import tt_bio
    from tt_bio import tenstorrent as T
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT)

    device = T.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(a.clock, nodes)
    t0 = time.time()
    while time.time() - t0 < 10 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()):
        print(f"REFUSING to measure: clock {reached}")
        return 2
    print(f"nodes {held} forced to {a.clock} MHz, reads {reached}", flush=True)

    g = device.compute_with_storage_grid_size()
    GX, GY = T.COMPUTE_GRID_MAIN
    print(f"device grid {g.x}x{g.y}, CORE_GRID_MAIN {GX}x{GY}", flush=True)
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    MC = {"DRAM": DRAM, "L1": L1}
    torch.manual_seed(0)

    clocks: list[int] = []
    results: list[dict] = []

    for nm, b, M, K, N, dt, r0, ro, share in SHAPES:
        if nm not in want:
            continue
        tdt = ttnn.bfloat16 if dt == "bf16" else ttnn.float32
        eb = 2 if dt == "bf16" else 4
        mt, kt, nt = b * (M // 32), K // 32, N // 32
        c = derive(mt, kt, nt, GX, GY, True, M, K, N)
        c["fp32_dest_acc"] = True
        obytes = b * M * N * eb
        hi = max(2, min(8, int(a.l1_budget_mb * 1e6 // obytes) if ro == "L1" else 8))
        lo = 1 if hi <= 3 else 2
        x = ttnn.from_torch(torch.randn(1, b, M, K) * 0.05, dtype=tdt,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=MC[r0])
        w = ttnn.from_torch(torch.randn(K, N) * 0.05, dtype=tdt,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        print(f"\n== {nm}: b={b} M={M} K={K} N={N} {dt} in={r0} out={ro} mt={mt} kt={kt} nt={nt}"
              f"  ttnn -> {c['cls']} pcm={c['per_core_M']} pcn={c['per_core_N']} "
              f"ibw={c['in0_block_w']} obh={c['out_block_h']} n={lo}/{hi}", flush=True)

        arms: list[tuple[str, object, dict]] = []

        def add(tag, pc, **meta):
            arms.append((tag, pc, meta))

        add("derived", None)
        add("derived_aa", None)
        add("mirror", build(ttnn, c))
        for h in divisors(c["per_core_M"]):
            if h != c["out_block_h"]:
                add(f"obh{h:02d}", build(ttnn, c, out_block_h=h), obh=h,
                    blocks=c["per_core_M"] // h)
        for iw in divisors(kt):
            if iw != c["in0_block_w"] and iw <= 32:
                add(f"ibw{iw:02d}", build(ttnn, c, in0_block_w=iw), ibw=iw)
        if c["cls"] == "1d":
            alt = derive(mt, kt, nt, GX, GY, True, M, K, 32 * 1024)  # force the 2D branch
            alt = {**alt, "cls": "2d", "per_core_M": -(-mt // GY), "per_core_N": max(1, -(-nt // GX)),
                   "fp32_dest_acc": True, "grid": (GX, GY), "mcast_in0": False}
            iw = 4
            while kt % iw:
                iw -= 1
            alt["in0_block_w"] = iw
            alt["out_block_h"], alt["out_block_w"] = alt["per_core_M"], alt["per_core_N"]
            for h in divisors(alt["per_core_M"]):
                try:
                    pc = build(ttnn, alt, out_block_h=h)
                except ValueError:
                    continue
                add(f"2d_obh{h:02d}", pc, obh=h, cls2d=True, pcm=alt["per_core_M"],
                    pcn=alt["per_core_N"], ibw=alt["in0_block_w"])

        def call(pc, mc):
            kw = dict(compute_kernel_config=KC, memory_config=mc, dtype=tdt)
            if pc is None:
                kw["core_grid"] = ttnn.CoreGrid(x=GX, y=GY)
            else:
                kw["program_config"] = pc
            return ttnn.linear(x, w, **kw)

        # compile + legality screen, once, outside the timed loop
        live = []
        for tag, pc, meta in arms:
            try:
                r = call(pc, MC[ro])
                ttnn.synchronize_device(device)
                ttnn.deallocate(r)
                live.append((tag, pc, meta))
            except Exception as e:                                             # noqa: BLE001
                print(f"   {tag:12s} refused: {repr(e)[:110]}", flush=True)
        samples: dict[tuple[str, int], list[float]] = {}
        order = [(t, p, n) for (t, p, _) in live for n in (lo, hi)]
        for rep in range(a.reps + a.warm):
            random.Random(rep).shuffle(order)
            for tag, pc, n in order:
                ttnn.synchronize_device(device)
                t1 = time.perf_counter()
                outs = [call(pc, MC[ro]) for _ in range(n)]
                ttnn.synchronize_device(device)
                t2 = time.perf_counter()
                for o in outs:
                    ttnn.deallocate(o)
                if rep >= a.warm:
                    samples.setdefault((tag, n), []).append((t2 - t1) * 1e3)
            clocks.append(clk.aiclk(held[0]))

        rows = []
        for tag, pc, meta in live:
            slo, shi = samples[(tag, lo)], samples[(tag, hi)]
            marg = (st.median(shi) - st.median(slo)) / (hi - lo)
            rows.append({"tag": tag, "ms": round(marg, 6), "meta": meta,
                         "lo_ms": round(st.median(slo), 5), "hi_ms": round(st.median(shi), 5),
                         "hi_cv": round(st.stdev(shi) / st.mean(shi), 4)})
        base = next(r for r in rows if r["tag"] == "derived")
        aa = next(r for r in rows if r["tag"] == "derived_aa")
        aa_pct = 100 * abs(aa["ms"] - base["ms"]) / base["ms"]
        for r in sorted(rows, key=lambda r: r["ms"]):
            print(f"   {r['tag']:12s} {r['ms']:9.5f} ms  {base['ms'] / r['ms']:6.4f}x  "
                  f"cv {r['hi_cv'] * 100:4.1f}%  {r['meta']}", flush=True)
        print(f"   A/A floor {aa_pct:.2f} %", flush=True)
        results.append({"shape": nm, "b": b, "M": M, "K": K, "N": N, "dtype": dt, "in": r0,
                        "out": ro, "share_pct": share, "mt": mt, "kt": kt, "nt": nt,
                        "derived": c, "n_lo": lo, "n_hi": hi, "aa_pct": round(aa_pct, 3),
                        "rows": rows})
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"host": socket.gethostname(), "clock_target": a.clock,
                                     "aiclk_during": {"min": min(clocks), "max": max(clocks),
                                                      "n": len(clocks)},
                                     "grid": [GX, GY], "reps": a.reps,
                                     "loadavg1": round(os.getloadavg()[0], 2),
                                     "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
                                     "shapes": results}, indent=1))
    print(f"\nAICLK during: min {min(clocks)} max {max(clocks)} over {len(clocks)} samples",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
