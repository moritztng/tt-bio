"""Does ttnn clamp an oversized ``core_grid=`` to the device grid, or fall back?

Why this exists. ``tt_bio/rfd3/model.py`` binds ``CORE_GRID_MAIN`` by VALUE at import time, before
any device is open, so every one of its ~20 pinned ``core_grid=CORE_GRID_MAIN`` call sites carries
the 11x10 import-time default. ``_configure_active_compute_grid`` rebinds
``tenstorrent.CORE_GRID_MAIN`` at device open -- to 8x9 on the Wormhole Galaxy -- but rfd3's snapshot
does not follow. The module docstring says that snapshot is deliberate and measured on Blackhole,
where 11x10 is a valid subset of 13x10. On a 72-core Wormhole grid 11x10 does not exist, so the
question is what ttnn does with it: throw (RFD3 would not run on Wormhole at all), silently fall
back to the default heuristic (RFD3 loses the hint, worth up to 58.9x on the decoder attention pair),
or clamp to the device grid (harmless).

Answer, on Blackhole where the three outcomes are distinguishable: it CLAMPS. An oversized grid is
bit-exact against the device-grid arm and times identically, while an UNDERsized grid is genuinely
slower -- so the argument is respected, not ignored, and the top end is capped at the device.

Run:
    TT_VISIBLE_DEVICES=<card> PYTHONPATH=<worktree> python3 perf/whdesign/wh_core_grid_clamp.py
"""
import json
import pathlib
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

OUT = pathlib.Path("perf/whdesign/results/wh_core_grid_clamp.json")
WARMUP, ITERS, BLOCKS = 3, 20, 5


def bench(d, xt, wt, cg, ref):
    kw = {} if cg is None else {"core_grid": cg}
    for _ in range(WARMUP):
        ttnn.deallocate(ttnn.linear(xt, wt, **kw))
    ttnn.synchronize_device(d)
    ts = []
    for _ in range(BLOCKS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.deallocate(ttnn.linear(xt, wt, **kw))
        ttnn.synchronize_device(d)
        ts.append((time.perf_counter() - t0) / ITERS * 1e3)
    o = ttnn.linear(xt, wt, **kw)
    ttnn.synchronize_device(d)
    got = ttnn.to_torch(o)
    ttnn.deallocate(o)
    return sorted(ts)[BLOCKS // 2], min(ts), (None if ref is None else bool(torch.equal(ref, got))), got


def main():
    d = T.get_device()
    a = d.compute_with_storage_grid_size()
    gx, gy = int(a.x), int(a.y)
    res = {"grid": [gx, gy], "cores": gx * gy, "warmup": WARMUP, "iters": ITERS, "blocks": BLOCKS,
           "shapes": []}
    for m, k, n in [(1024, 512, 512), (4096, 512, 256)]:
        x = torch.randn(1, m, k, dtype=torch.bfloat16)
        w = torch.randn(k, n, dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
        wt = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
        ref = None
        pts = []
        # first arm is the device grid and is the bit-exactness reference for the rest
        for label, cg in [("device", (gx, gy)), ("rfd3_snapshot_11x10", (11, 10)),
                          ("oversized_14x12", (14, 12)), ("undersized_6x6", (6, 6)),
                          ("device_repeat", (gx, gy))]:
            med, mn, eq, got = bench(d, xt, wt, ttnn.CoreGrid(y=cg[1], x=cg[0]), ref)
            if ref is None:
                ref = got
            pts.append({"arm": label, "core_grid": list(cg), "median_ms": round(med, 5),
                        "min_ms": round(mn, 5), "bit_exact_vs_device_grid": eq})
        ttnn.deallocate(xt)
        ttnn.deallocate(wt)
        res["shapes"].append({"m": m, "k": k, "n": n, "points": pts})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for s in res["shapes"]:
        print("M=%d K=%d N=%d" % (s["m"], s["k"], s["n"]))
        for p in s["points"]:
            print("  %-20s %2dx%-2d  median %.5f ms  bit_exact=%s"
                  % (p["arm"], p["core_grid"][0], p["core_grid"][1], p["median_ms"],
                     p["bit_exact_vs_device_grid"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
