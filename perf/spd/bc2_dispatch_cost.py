#!/usr/bin/env python3
"""What one ttnn call costs on the host, split into wrapper and enqueue.

A W1 BindCraft 2 iteration spends 1.64 s of its ~6 s inside `ttnn/decorators.py::__call__`
(64,525 calls, 25 us each, cProfile tottime). cProfile cannot split that: the nanobind function
the wrapper calls is not a Python frame, so its time is inside the wrapper's own. This splits it
by timing the same op three ways on the same tensors:

  wrapper   ttnn.<op>(...)                      what tt-bio calls today
  raw       ttnn.<op>.function(...)             the nanobind binding, no Python wrapper
  probe     ttnn.graph.is_graph_capture_active()  the C++ call the wrapper makes per op

Host time only: every op is enqueued and nothing synchronises inside the loop, so a difference is
dispatch cost, not device time. One synchronize at the end of each block keeps the queue bounded.
"""
import argparse
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000, help="calls per timed block")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--tokens", type=int, default=224)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()
shape = (1, a.tokens, a.tokens, 128)
x = ttnn.from_torch(torch.ones(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)
y = ttnn.from_torch(torch.ones(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)


def block(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        if out is not None:
            ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n * 1e6


arms = {
    "wrapper_multiply": lambda: ttnn.multiply(x, y),
    "raw_multiply": lambda: ttnn.multiply.function(x, y),
    "graph_probe": lambda: (ttnn.graph.is_graph_capture_active(), None)[1],
    "wrapper_deallocate_only": lambda: (ttnn.deallocate(ttnn.multiply(x, y)), None)[1],
}
# Warm every arm once: the first call of an op compiles its program.
for fn in arms.values():
    out = fn()
    if out is not None:
        ttnn.deallocate(out)
ttnn.synchronize_device(dev)

print(f"== device {dev.arch()} tokens {a.tokens} n {a.n} reps {a.reps}", flush=True)
for name, fn in arms.items():
    us = [block(fn, a.n) for _ in range(a.reps)]
    print(f"{name:26} {statistics.median(us):8.2f} us/call  (min {min(us):.2f} max {max(us):.2f})", flush=True)
