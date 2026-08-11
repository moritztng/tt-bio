#!/usr/bin/env python3
"""`ttnn.chunk` along the last dim falls off a cliff when the source is wider than 4 tiles.

`perf/bigswing/trimul_inproj_split.py` found the mechanism behind the in-projection width lever's
fold regression: the mandatory split costs **5.797 ms** at G=1 and a flat **10.23-10.29 ms** at G=2,
4 and 8, on identical bytes (537 MB read, 537 MB written per trimul). 185 GB/s against 105 GB/s.

At G=1 each of the 4 pieces takes 1 tile of every 4 along a tile row. At G>=2 it takes 1 of every 8,
16 or 32, and the penalty saturates immediately. If the cliff is the *stride* and not the source
width, then splitting the same wide tensor **4 ways instead of 4G ways** -- each piece 32G wide and
therefore 4G contiguous tiles per row -- should run at the G=1 rate, and the lever is salvageable by
widening the downstream channel loop to match. If the cliff is the source width, the lever is dead
and no fold run is needed to say so.

Two arms per width: the production 4G-way split, and the 4-way split a role-major weight order would
allow. Same source tensor, same bytes out.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--groups", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio.tenstorrent import get_device

    torch.manual_seed(0)
    dev = get_device()
    S, C = a.seq, a.c
    groups = [int(x) for x in a.groups.split(",")]

    src = {g: ttnn.from_torch(torch.randn(1, S, S, 4 * C * g, dtype=torch.bfloat16),
                              layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG) for g in groups}
    # One matmul's worth of output at G=1 is 4*C wide; a trimul produces 8 of those either way, so
    # every arm is normalised to the same 8 pair-chunks of work.
    reps_per_trimul = {g: 8 // g for g in groups}

    keys = [(g, n) for g in groups for n in (4 * g, 4)]
    keys = sorted(set(keys))
    times = {k: [] for k in keys}

    def run(g, n):
        out = []
        for _ in range(reps_per_trimul[g]):
            out.extend(ttnn.chunk(src[g], chunks=n, dim=-1))
        return out

    for k in keys:                                       # warm
        for t in run(*k):
            ttnn.deallocate(t)
    ttnn.synchronize_device(dev)

    for _ in range(a.reps):
        for k in keys:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = run(*k)
            ttnn.synchronize_device(dev)
            times[k].append((time.perf_counter() - t0) * 1000.0)
            for t in outs:
                ttnn.deallocate(t)

    res = {"host": "qb2", "chip": 0, "seq": S, "C": C, "reps": a.reps, "arms": []}
    import importlib.metadata as im
    res["ttnn"] = im.version("ttnn")
    for (g, n) in keys:
        ms = sorted(times[(g, n)])
        med = st.median(ms)
        mb = 2 * (S * S * 4 * C * g * 2 / 2**20) * reps_per_trimul[g]   # read + write, per trimul
        res["arms"].append({
            "G": g, "src_width": 4 * C * g, "chunks": n, "piece_width": 4 * C * g // n,
            "piece_tiles_contig": 4 * C * g // n // 32, "calls_per_trimul": reps_per_trimul[g],
            "ms_median": round(med, 4), "spread_ms": round(ms[-1] - ms[0], 4),
            "traffic_MB": round(mb, 1), "GBps": round(mb / 2**10 / (med / 1000), 1)})
        print(f"G={g} src_w={4*C*g:5d} chunks={n:3d} piece={4*C*g//n:4d} "
              f"({4*C*g//n//32} contig tiles)  {med:8.3f} ms  {mb/2**10/(med/1000):7.1f} GB/s",
              flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
