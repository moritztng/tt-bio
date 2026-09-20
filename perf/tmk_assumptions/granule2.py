#!/usr/bin/env python3
"""tmk-assumptions pass 2: what does the SHIPPED channel move cost, at the shape production runs?

Pass 1 priced 'the shipped channel move' as ttnn.permute(zc, (0,3,1,2)) and read 63.3 GB/s. The
census (census_gated.py, cdk2x2_512, run 2026-09-20 on this card) says production does not call
that op at all: it calls reblock_permute_gated 1120 times per fold, every one at
xw=[1,512,512,512] slice_c=128 -> out=[1,128,512,512] in DRAM, with eligible_gated True on all 560
checks and reblock_permute (the non-gated forward kernel) firing zero times.

So every route that could be 'the shipped move' is timed here in ONE session, on ONE card, at ONE
pinned and during-sampled clock, on the production shape:

  * the production kernel itself, at its production shape and slices;
  * the sequence it replaces (chunk, sigmoid, multiply, move), with the move done both ways;
  * ttnn.permute -- pass 1's baseline, reproduced here so the 63.3 is comparable;
  * reblock_permute -- our non-gated kernel, the one perfwar read at 221 GB/s at N=1024;
  * reblock_permute_back -- the return move, 560 calls per fold;
  * the tile-granular control (last-two-axis transpose) and a straight clone, the same bytes.

Every time is an n-ladder slope, never a synced bracket.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("_rate_gap", Path(__file__).with_name("rate_gap.py"))
RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RG)

BF16 = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--D", type=int, default=128, help="slice_c, from the census")
    ap.add_argument("--out", default="perf/tmk_assumptions/granule2.json")
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import reblock_permute as RP
    from tt_bio.main import ensure_p300_mesh_descriptor
    print("  mgd:", ensure_p300_mesh_descriptor(), flush=True)

    N, D = a.N, a.D
    Z = N * N * D * BF16                       # one channel-slice plane, 67.11 MB at 512/128
    pin = RG.pin_clock(a.node, a.clock, REPO)
    clk = RG.ClockSampler(a.node)
    res, arms = {}, {}
    t_start = time.time()
    with clk:
        dev = ttnn.open_device(device_id=0)
        try:
            T.CORE_GRID_MAIN = dev.core_grid
            T.COMPUTE_GRID_MAIN = (dev.core_grid.x, dev.core_grid.y)
            print(f"  grid {T.COMPUTE_GRID_MAIN[0]}x{T.COMPUTE_GRID_MAIN[1]}", flush=True)
            DR = ttnn.DRAM_MEMORY_CONFIG

            def run(name, fn, byts, note=""):
                s = RG.ladder(dev, fn, reps=a.reps, serial=True)
                p = RG.ladder(dev, fn, reps=a.reps, serial=False)
                arms[name] = dict(serial_ms=round(s["ms"], 5), pipe_ms=round(p["ms"], 5),
                                  serial_intercept_ms=round(s["intercept_ms"], 5),
                                  serial_raw=s["raw"], pipe_raw=p["raw"],
                                  MB=round(byts / 1e6, 3),
                                  serial_GBs=round(byts / (s["ms"] * 1e-3) / 1e9, 2),
                                  pipe_GBs=round(byts / (p["ms"] * 1e-3) / 1e9, 2), note=note)
                print(f"  {name:26s} serial {s['ms']:8.4f} ms  pipe {p['ms']:8.4f} ms  "
                      f"{arms[name]['serial_GBs']:7.1f} GB/s  ({arms[name]['MB']} MB)  {note}",
                      flush=True)
                return arms[name]

            # ---- the production kernel, at the census shape --------------------
            gp = RG.dram(torch.randn(1, N, N, 4 * D, dtype=torch.bfloat16), dev)
            pa, ga = T.gp_off("p_a", D), T.gp_off("g_a", D)
            assert RP.eligible_gated(gp, D, DR), "eligible_gated refuses the census shape"
            print(f"  eligible_gated([1,{N},{N},{4*D}], slice_c={D}, DRAM) = True; "
                  f"p_a@{pa} g_a@{ga}", flush=True)

            def gated():
                o = RP.reblock_permute_gated(gp, pa, ga, D, memory_config=DR)
                ttnn.deallocate(o)

            n0 = int(RP.STATS_GATED[0])
            run("P_gated_production", gated, 3 * Z,
                "THE SHIPPED MOVE: reads p and g slices, writes the moved gated plane")
            run("P_gated_production_AA", gated, 3 * Z, "A/A twin")
            print(f"  reblock_permute_gated fired {int(RP.STATS_GATED[0]) - n0} times in those arms",
                  flush=True)

            # the sequence the fused kernel replaces, both ways of doing its move
            def unfused(move):
                def f():
                    parts = ttnn.chunk(gp, 4, -1)
                    v = ttnn.multiply(parts[0], ttnn.sigmoid(parts[1]), memory_config=DR)
                    o = move(v)
                    for t in list(parts) + [v, o]:
                        ttnn.deallocate(t)
                return f

            run("P_unfused_stock_permute", unfused(lambda v: ttnn.permute(v, (0, 3, 1, 2),
                                                                        memory_config=DR)),
                3 * Z, "chunk+sigmoid+mul then ttnn.permute -- bytes counted as the fused kernel's")
            run("P_unfused_reblock", unfused(lambda v: RP.reblock_permute(v, DR)), 3 * Z,
                "same, with our non-gated kernel as the move")
            gp.deallocate()

            # ---- the move alone, pair-major source ----------------------------
            zc = RG.dram(torch.randn(1, N, N, D, dtype=torch.bfloat16), dev)
            assert RP.eligible(zc, DR), "eligible refuses the pair-major shape"
            run("M_permute_stock_0312", lambda: ttnn.permute(zc, (0, 3, 1, 2), memory_config=DR),
                2 * Z, "pass 1's baseline op, reproduced in this session")
            run("M_permute_stock_0312_AA", lambda: ttnn.permute(zc, (0, 3, 1, 2), memory_config=DR),
                2 * Z, "A/A twin")
            run("M_reblock_fwd", lambda: RP.reblock_permute(zc, DR), 2 * Z,
                "our shipped non-gated kernel, same bytes, same move")
            run("M_reblock_fwd_AA", lambda: RP.reblock_permute(zc, DR), 2 * Z, "A/A twin")
            zc.deallocate()

            # ---- channel-major source: the return move and the tile controls ---
            zt = RG.dram(torch.randn(1, D, N, N, dtype=torch.bfloat16), dev)
            run("M_reblock_back", lambda: RP.reblock_permute_back(zt, DR), 2 * Z,
                "the return move, 560 calls per 512 aa fold")
            run("M_permute_back_stock", lambda: ttnn.permute(zt, (0, 2, 3, 1), memory_config=DR),
                2 * Z, "the stock op the return move replaces")
            run("C_tile_transpose", lambda: ttnn.transpose(zt, -2, -1, memory_config=DR), 2 * Z,
                "TILE-GRANULAR control: same bytes, no channel exchange")
            run("C_tile_transpose_AA", lambda: ttnn.transpose(zt, -2, -1, memory_config=DR), 2 * Z,
                "A/A twin")
            run("C_clone", lambda: ttnn.clone(zt, memory_config=DR), 2 * Z,
                "no-permute control: the same bytes copied")
            zt.deallocate()
        finally:
            ttnn.close_device(dev)
    t_end = time.time()
    res["clock"] = clk.summary(t_start, t_end); res["clock"]["target"] = a.clock
    res["shape"] = dict(N=N, D=D, Z_MB=round(Z / 1e6, 3), node=a.node)
    res["arms"] = arms
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["clock"], indent=1), flush=True)
    pin.terminate()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
