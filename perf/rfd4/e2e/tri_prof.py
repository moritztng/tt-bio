"""RFD4's two triangle multiplications at the trunk's n under the device profiler, for per-sub-op kernel time.

    perf/rfd4/e2e/tri_prof.sh <out_dir>        (runs this under `python -m tracy` on the Tracy build)

The modules are the trunk's own (`DesignEngine(...).trunk.pair_updates[0]`), called standalone on a bf16
z of the trunk's shape with an all-ones pair mask, which is what a 128-token design passes (tri_gap.py
records the operands). Warm-up calls first, then a `tri_prof` signpost, then --calls calls per module,
each bracketed by `tri_out`/`tri_in` signposts so tri_prof_table.py can cut the ops report per call.
Only DEVICE KERNEL DURATION is read from the report; wall under the profiler is not a measurement.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
from tracy import signpost  # noqa: E402

import clocksample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ttuser/rfd4-bundle")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio.rfd4 import engine as E

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev = engine._device
    pu = list(engine.trunk.pair_updates.values())[0]
    n = a.n
    torch.manual_seed(0)
    z = ttnn.from_torch(torch.randn(1, n, n, 384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pm = ttnn.from_torch(torch.ones(1, n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mods = (("tri_out", pu.tri_out), ("tri_in", pu.tri_in))
    for _ in range(a.warm):
        for _, m in mods:
            ttnn.deallocate(m(z, pm))
    ttnn.synchronize_device(dev)
    signpost("tri_prof")
    with clocksample.during(period=0.5) as clk:
        for _ in range(a.calls):
            for name, m in mods:
                signpost(name)
                ttnn.deallocate(m(z, pm))
                ttnn.synchronize_device(dev)
    signpost("tri_end")
    ttnn.synchronize_device(dev)
    json.dump({"n": n, "calls": a.calls, "aiclk": clk.summary(), "load": os.getloadavg()[0],
               "card": os.environ.get("TT_VISIBLE_DEVICES")}, open(a.out, "w"), indent=2)
    ttnn.ReadDeviceProfiler(dev)


if __name__ == "__main__":
    main()
