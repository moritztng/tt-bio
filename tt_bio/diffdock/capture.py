"""Warm up, capture, replay, and time the diffusion loop on a card -- with its clock.

The device leg of the dfd-trace row is three things that are easy to get wrong separately:

* **Warm up before capturing.** `tests/test_diffdock_trace.py` found that the first pass
  issues ops the others do not -- `TensorProduct` casts its float64 Wigner 3j tables on first
  use -- and a buffer allocated inside `begin_trace_capture` is not a buffer the capture owns.
  So :func:`capture_loop` always runs the body once eagerly first.
* **Replay the whole loop, not one step.** A per-step capture pays a dispatch round trip per
  step; the point of a fixed-shape loop is that all 19 steps are one capture.
* **Sample the clock while it runs.** A Blackhole chip's AICLK moves with temperature and
  with what else is on the board, so a number taken before or after the run is not the clock
  the run had. :func:`sample_aiclk` polls in a thread for the duration and reports the median
  and the spread, and a dead-ARC reading (`runtime.aiclk_reading`) is dropped rather than
  averaged in as 4.3 GHz.

Nothing here is DiffDock-specific: it takes a callable and the inputs it mutates. That is
deliberate -- the loop body is still torch on this branch, and this is what turns a ttnn body
into a measurement the moment one exists.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from pathlib import Path


def sample_aiclk(stop: threading.Event, out: list, visible: str = "0", period: float = 0.5):
    """Poll ``tt-smi -s`` until *stop*, appending live AICLK readings in MHz to *out*."""
    from tt_bio.runtime import aiclk_reading
    import os
    import sys
    tt_smi = None
    for c in (Path(sys.executable).parent / "tt-smi", Path.home() / ".local/bin/tt-smi",
              Path("/usr/local/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            tt_smi = str(c)
            break
    if tt_smi is None:
        return
    while not stop.is_set():
        try:
            res = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=20)
            devs = json.loads(res.stdout).get("device_info", [])
            idx = min(int(visible), len(devs) - 1) if devs and visible.isdigit() else 0
            mhz = aiclk_reading(int((devs[idx].get("telemetry") or {}).get("aiclk", 0)))
            if mhz:
                out.append(mhz)
        except Exception:
            pass
        stop.wait(period)


def clock_summary(samples: list) -> dict:
    """What a timing has to carry beside its seconds: the clock it ran at, and its spread."""
    if not samples:
        return {"aiclk_mhz": None, "aiclk_samples": 0,
                "aiclk_note": "no live reading -- the timing is not comparable across boxes"}
    return {"aiclk_mhz": int(statistics.median(samples)), "aiclk_samples": len(samples),
            "aiclk_min": min(samples), "aiclk_max": max(samples)}


def capture_loop(device, body, n_steps: int):
    """Warm up *body* once, capture ``n_steps`` of it, and return ``(tid, warm_result)``.

    *body* takes the step index and issues one step's device work into the same buffers every
    time -- which is what `tests/test_diffdock_trace.py` proves about this loop on CPU.
    """
    from tt_bio.tenstorrent import capture_trace
    warm = body(0)                                    # eager: materialises every lazy constant
    tid, _ = capture_trace(device, lambda: [body(t) for t in range(n_steps)])
    return tid, warm


def timed_replay(device, tid, n_complexes: int = 1, repeats: int = 3,
                 visible: str = "0") -> dict:
    """Replay the capture *repeats* times, return the best seconds per complex and the clock.

    Best of *repeats* rather than mean: a slower replay is another process on the board or a
    thermal dip, and both are facts about the box that run, not about the loop.
    """
    import ttnn
    stop = threading.Event()
    samples: list = []
    t = threading.Thread(target=sample_aiclk, args=(stop, samples, visible), daemon=True)
    t.start()
    try:
        seconds = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            seconds.append(time.perf_counter() - t0)
    finally:
        stop.set()
        t.join(timeout=5)
    best = min(seconds)
    return {"s_per_complex": best / max(n_complexes, 1), "all_seconds": seconds,
            "n_complexes": n_complexes, **clock_summary(samples)}
