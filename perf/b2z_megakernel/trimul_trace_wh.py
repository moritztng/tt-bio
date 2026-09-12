#!/usr/bin/env python3
"""The trimul's DEVICE floor on Wormhole, by trace capture + replay.

Why this exists: `trimul_chain.py` times the chain eagerly, so its 16.971 ms carries whatever host
issue cost does not hide behind device work. The campaign's Blackhole cell (9.3975 ms/call,
`perf/b2x_op_cost/subunit_floor_512_qb2c0.json`) is a TRACE REPLAY and so carries none. Comparing
those two numbers directly -- which the first version of this workstream's state doc did -- is a
unit mismatch, not a cross-architecture result.

This measures the same quantity on Wormhole as the Blackhole cell measures: one replay of a
captured trace of `TriangleMultiplication.__call__`, no host issue in the loop. The gap between
it and the eager wall is this chain's host-dispatch cost, which is also the part a megakernel
deletes for free by being one op instead of thirteen.
"""
import argparse, json, statistics, time
from pathlib import Path
import torch, ttnn
from tt_bio import tenstorrent as TT
from perf.b2z_megakernel.trimul_chain import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--arm", default="start", choices=("start", "end"))
    ap.add_argument("--trace-mb", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = TT.get_device(trace_region_size=a.trace_mb << 20)
    grid = dev.compute_with_storage_grid_size()
    mods, z, mask = build(a.n, a.c_z, dev)
    res = {"_what": "trimul device floor on WH by trace replay, comparable to the BH cell",
           "arch": str(dev.arch()), "grid": f"{grid.x}x{grid.y}", "n": a.n, "c_z": a.c_z,
           "iters": a.iters, "arms": {}}

    picked = [(a.arm, mods[0] if a.arm == "start" else mods[1])]
    for name, m in picked:
        # warm: compile every program before the capture, or the capture records compilation
        for _ in range(3):
            ttnn.deallocate(m(z, mask))
        ttnn.synchronize_device(dev)

        # eager wall, no syncs inside
        ts = []
        for _ in range(a.iters):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = m(z, mask)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
        eager = statistics.median(ts)

        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = m(z, mask)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)

        # time N replays as one block: per-replay host cost is ~1 us, so the quotient is device time
        for _ in range(3):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(a.iters):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            reps.append((time.perf_counter() - t0) * 1e3 / a.iters)
        traced = statistics.median(reps)
        ttnn.release_trace(dev, tid)
        ttnn.deallocate(out)

        res["arms"][name] = {"eager_ms": eager, "traced_ms": traced,
                             "dispatch_ms": eager - traced,
                             "dispatch_frac": (eager - traced) / eager,
                             "replay_all_ms": reps}
        print(f"{name:5s}  eager {eager:8.3f} ms   traced {traced:8.3f} ms   "
              f"host dispatch {eager - traced:7.3f} ms ({(eager - traced) / eager * 100:.1f} %)")

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
