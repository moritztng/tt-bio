#!/usr/bin/env python3
"""Is the token DiT layer's per-op floor on the device or on the host?

`dit_layer.py` measured a 20.6 us fixed cost per ttnn op on a [1,512,768] tensor, synced once
per block of enqueues. That construction reports max(host dispatch, device execution) per op,
so it cannot tell a device-side program-launch floor from a host-side enqueue floor -- and the
distinction decides whether the cost is removable. `b2z-gpu-roof-mirror` measured Blackhole's
small-op floor at 6.36 us with the host removed by trace replay, and 20.6 > 6.36.

Three readings of the same arm, same session, same warm program cache:

    t_sync    enqueue `reps`, synchronize, divide      = max(host, device) + tail
    t_host    enqueue `reps`, do NOT synchronize       = host dispatch alone
    t_trace   replay a captured trace of `reps` ops    = device alone, host removed

t_host near t_sync means the floor is dispatch. t_trace near t_sync means it is the device.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

sys.path.insert(0, str(HERE))
from dit_layer import DIM, HEADS, S, weights                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "dispatch_probe.json")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()

    dev = T.get_device(trace_region_size=1 << 28)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    layer = T.DiffusionTransformerLayer(DIM, HEADS, False, weights(), kc)

    def t(shape, sc=1.0):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16) * sc,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = t((1, S, DIM)); s = t((1, S, DIM)); z = t((1, HEADS, S, S), 0.1)
    x4 = t((1, 4 * S, DIM)); y = t((1, S, DIM))

    arms = {
        "ln_1x": lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=kc),
        "ln_4x": lambda: ttnn.layer_norm(x4, epsilon=1e-5, compute_kernel_config=kc),
        "add_1x": lambda: ttnn.add(x, y),
        "mm_768": lambda: ttnn.linear(x, layer.output_projection_weight,
                                      compute_kernel_config=kc, core_grid=T.CORE_GRID_MAIN),
        "adaln": lambda: layer.adaln(x, s),
        "layer_ship": lambda: layer(x, s, z),
    }

    out = {"host": platform.node(), "arch": str(dev.arch()), "reps": a.reps,
           "blocks": a.blocks, "loadavg": open("/proc/loadavg").read().split()[:3], "rows": {}}

    for name, fn in arms.items():
        row = {}
        for _ in range(3):                                   # warm the program cache
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)

        best = None
        for _ in range(a.blocks):
            outs = []
            t0 = time.perf_counter()
            for _ in range(a.reps):
                outs.append(fn())
                if len(outs) > 4:
                    ttnn.deallocate(outs.pop(0))
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / a.reps
            for o in outs:
                ttnn.deallocate(o)
            best = dt if best is None else min(best, dt)
        row["t_sync_us"] = best * 1e6

        # host dispatch alone: the same enqueues, clock stopped BEFORE the barrier. This number
        # is NOT an op cost and is never reported as one -- it is the dispatch half of the max.
        best = None
        for _ in range(a.blocks):
            outs = []
            t0 = time.perf_counter()
            for _ in range(a.reps):
                outs.append(fn())
                if len(outs) > 4:
                    ttnn.deallocate(outs.pop(0))
            dt = (time.perf_counter() - t0) / a.reps
            ttnn.synchronize_device(dev)
            for o in outs:
                ttnn.deallocate(o)
            best = dt if best is None else min(best, dt)
        row["t_host_us"] = best * 1e6

        # device alone: capture the same `reps` enqueues once, then replay.
        try:
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            held = [fn() for _ in range(a.reps)]
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            best = None
            for _ in range(a.blocks):
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / a.reps
                best = dt if best is None else min(best, dt)
            row["t_trace_us"] = best * 1e6
            ttnn.release_trace(dev, tid)
            for h in held:
                try:
                    ttnn.deallocate(h)
                except Exception:                                             # noqa: BLE001
                    pass
        except Exception as e:                                                # noqa: BLE001
            row["t_trace_us"] = None
            row["trace_error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200])
        ttnn.synchronize_device(dev)
        out["rows"][name] = row
        tr = row["t_trace_us"]
        print("%-12s sync %7.2f us   host %7.2f us   trace %s"
              % (name, row["t_sync_us"], row["t_host_us"],
                 ("%7.2f us" % tr) if tr else row.get("trace_error", "-")), flush=True)

    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
