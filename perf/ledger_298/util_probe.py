#!/usr/bin/env python3
"""Core-utilisation instrument for the Protenix trunk org, without the device profiler.

The tt-metal device profiler does not work on these hosts (the ttnn wheel ships the `tracy` python
package but not the capture binaries) and `TT_METAL_WATCHER` is not a substitute: its dumps are
wall-clock periodic and land between programs, and it inflates op time ~10x. What does work is a
core-count sweep. Hold the problem size fixed, shrink the grid, and fit

    t(c) = max(launch_floor, W / (c * r))

Engaged cores is the largest `c` at which `t` still falls as 1/c. Calibrated on this card against
both known points: it reproduces DRAM read saturating at ~32 of 130 cores, and reports 110 of 110
for an op pinned to the full 10x11 grid.

    from perf.ledger_298.util_probe import engaged_cores, block_shard
    r = engaged_cores(dev, lambda mc: lambda: ttnn.mul(a(mc), 1.0001, memory_config=mc,
                                                       output_tensor=c(mc)), (1280, 1408))
    r["engaged"]        # -> 110
    r["floor_limited"]  # -> True means THROW THE ANSWER AWAY, see below

**Only valid for ops whose single-shot time is at least 3x the launch floor (>= 20 us on this
card).** Below that the op never leaves the 6.40 us floor, the sweep is flat, and the instrument
would report 1 core for an op using all 110. `floor_limited` is True exactly then; when it is set,
the engaged count is not a measurement.
"""
import time

import ttnn

TILE = 32
LAUNCH_FLOOR_US = 6.40          # measured on qb2 card 2, ttnn 0.68.0; see the T4 state doc
MIN_USABLE_US = 3 * LAUNCH_FLOOR_US
GRIDS = [(1, 1), (2, 2), (2, 4), (4, 4), (4, 8), (8, 4), (8, 8), (10, 8), (10, 11)]


def block_shard(rows, cols, gy, gx, buffer_type=ttnn.BufferType.L1):
    """Block-sharded memory config over a gy x gx core grid. Returns None if it does not tile."""
    if rows % gy or cols % gx or (rows // gy) % TILE or (cols // gx) % TILE:
        return None
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type,
                             ttnn.ShardSpec(crs, [rows // gy, cols // gx], ttnn.ShardOrientation.ROW_MAJOR))


def time_us(dev, fn, iters=60):
    """Device time per call, synchronised on both sides. An unsynced drain has inverted rankings."""
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def engaged_cores(dev, make_op, shape, grids=GRIDS, iters=60, tol=0.15):
    """Sweep the core grid and report how many cores the op actually engages.

    `make_op(memory_config)` returns a zero-argument callable that runs the op once with its
    tensors laid out on that grid. `shape` is (rows, cols) of the tensor being sharded.
    """
    rows, cols = shape
    times = {}
    for gy, gx in grids:
        mc = block_shard(rows, cols, gy, gx)
        if mc is None:
            continue
        try:
            fn = make_op(mc)
            times[gy * gx] = time_us(dev, fn, iters)
        except Exception as exc:                                        # noqa: BLE001
            times[gy * gx] = None
            print(f"  grid {gy}x{gx}: {str(exc)[:80]}")
    ok = {c: t for c, t in times.items() if t}
    if not ok:
        return {"times_us": times, "engaged": None, "floor_limited": None,
                "note": "every grid failed to allocate"}

    slowest = max(ok)
    fastest_t = min(ok.values())
    floor_limited = fastest_t < MIN_USABLE_US or (max(ok.values()) / fastest_t) < 1.5

    # Walk up the core counts; the op still scales while doubling cores still buys most of a halving.
    counts = sorted(ok)
    engaged = counts[0]
    for lo, hi in zip(counts, counts[1:]):
        ideal = ok[lo] * lo / hi
        gained = (ok[lo] - ok[hi]) / max(ok[lo] - ideal, 1e-9)
        if gained >= tol:
            engaged = hi
        else:
            break
    return {"times_us": {c: round(t, 3) if t else None for c, t in times.items()},
            "engaged": engaged, "max_grid_cores": slowest, "floor_limited": floor_limited,
            "launch_floor_us": LAUNCH_FLOOR_US,
            "note": ("op never left the launch floor -- the engaged count is NOT a measurement"
                     if floor_limited else "")}
