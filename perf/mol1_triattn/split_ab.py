#!/usr/bin/env python3
"""`triatt_sdpa.SPLIT_Q` on against off at every Wormhole (seq, heads) it moves, plus controls it must not move.

Random bf16 [S, H, S, 32] q/k/v and a [1, H, S, S] bias through `_tri_att_sdpa`, the arms interleaved round-robin,
first call of each discarded. Per cell: the pick each arm served, ms min/median, torch.equal between the arms, AICLK
sampled during the timed calls.

Usage: split_ab.py --cells 1024x10,1024x14,1024x12,800x8,672x14,672x16,512x10,512x14,1024x8 --out ab.jsonl
       [--margin 0.12] [--grid 8x8]   # the on arm at another SPLIT_Q_MARGIN; a smaller grid stands in for a 64-core chip
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))

import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_sdpa as TS
import clocksample

assert Path(T.__file__).resolve().is_relative_to(ROOT)


def arm(on):
    TS.SPLIT_Q = on
    TS.split_q_chunk.cache_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1024x10,1024x14,1024x12,800x8,672x14,672x16,512x10,512x14,1024x8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=TS.SPLIT_Q_MARGIN)
    ap.add_argument("--grid", default=None)
    a = ap.parse_args()
    TS.SPLIT_Q_MARGIN = a.margin
    dev = T.get_device()
    if a.grid:
        T.COMPUTE_GRID_MAIN = tuple(map(int, a.grid.split("x")))
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for cell in a.cells.split(","):
        S, H = map(int, cell.split("x"))
        g = torch.Generator().manual_seed(S * 100 + H)
        q, k, v = (put(torch.randn(S, H, S, 32, generator=g).bfloat16()) for _ in range(3))
        bias = put(torch.randn(1, H, S, S, generator=g).bfloat16())
        rec = {"S": S, "heads": H, "grid": list(T.COMPUTE_GRID_MAIN), "margin": a.margin}
        outs, ts = {}, {"on": [], "off": []}
        with clocksample.during(period=0.5) as clk:
            for rep in range(a.reps + 1):
                for name, on in (("on", True), ("off", False)):
                    arm(on)
                    T.SDPA_CHUNK_PICKS.clear()
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    o = T._tri_att_sdpa(q, k, v, bias, 32 ** -0.5)
                    ttnn.synchronize_device(dev)
                    dt = time.perf_counter() - t0
                    if rep:
                        ts[name].append(dt)
                    else:
                        outs[name] = ttnn.to_torch(o)
                        rec[f"pick_{name}"] = T.SDPA_CHUNK_PICKS.get((S, S))
                    ttnn.deallocate(o)
        arm(True)
        for name in ts:
            t = sorted(ts[name])
            rec[f"ms_min_{name}"] = round(t[0] * 1e3, 3)
            rec[f"ms_med_{name}"] = round(t[len(t) // 2] * 1e3, 3)
        rec["speedup_med"] = round(rec["ms_med_off"] / rec["ms_med_on"], 4)
        rec["equal"] = bool(torch.equal(outs["on"], outs["off"]))
        rec["aiclk"] = clk.summary()
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        print(json.dumps(rec), flush=True)
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    T.cleanup()


if __name__ == "__main__":
    main()
