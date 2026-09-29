"""How this row times a dispatch, and why it drains before every call rather than after twenty.

An enqueue is only host work while the command queue has room. Let twenty of them pile up and
the twenty-first blocks on the device, so the number stops being a host measurement: the first
sitting read `ttnn.add` at 0.018 ms in one process and 0.030 in the next, on the same box at the
same loadavg, purely from what else was outstanding.

So the clock covers exactly one enqueue against an EMPTY queue. `synchronize_device` runs before
each rep, outside the clock. Deallocation also runs outside the clock, after the sync, which is
the only point at which freeing a buffer the device may still be reading is safe.
"""
from __future__ import annotations

import json
import os
import statistics
import time


def timed(name, fn, dev, ttnn, reps, spans=None, keep=False):
    """Median host ms for one call of `fn`, each against a drained queue.

    `fn` returns a tensor to free, or None. `keep=True` means the caller owns what comes back.
    """
    for _ in range(8):
        o = fn()
        ttnn.synchronize_device(dev)
        if o is not None and not keep:
            ttnn.deallocate(o)
    xs, t_lo = [], time.time()
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        xs.append(time.perf_counter() - t0)
        ttnn.synchronize_device(dev)
        if o is not None and not keep:
            ttnn.deallocate(o)
    if spans is not None:
        spans.append((t_lo, time.time()))
    xs.sort()
    r = {'median_ms': round(statistics.median(xs) * 1e3, 5),
         'p25_ms': round(xs[len(xs) // 4] * 1e3, 5),
         'p75_ms': round(xs[3 * len(xs) // 4] * 1e3, 5),
         'min_ms': round(xs[0] * 1e3, 5), 'reps': reps,
         'load': round(os.getloadavg()[0], 2)}
    print(f'{name:40s} {json.dumps(r)}', flush=True)
    return r
