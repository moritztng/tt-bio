#!/usr/bin/env python3
"""Device-side kernel cycle census of the Boltz-2 512 aa fold.

Every earlier pass in this campaign measured the block from outside the device: ttnn call
counts, host timers, byte counters, trace replay. They agree that one ``PairformerLayer``
costs 41.4152 ms and that its own op-cost model (428 ops x 6.36 us fixed + 6.651 GB at
445 GB/s) only accounts for 17.67 ms of it. This file measures the same block from *inside*
the device, with tt-metal's device profiler, so the missing 23.7 ms can be split into

  (a) cycles inside a kernel with the compute engine retiring instructions,
  (b) cycles inside a kernel waiting (NoC, circular buffers, semaphores),
  (c) cycles in no kernel at all -- the gap between one op's last core finishing and the
      next op's first core starting.

Run it under the profiler, one phase per process::

    python -m tracy -r -o OUT --op-support-count 20000 -- kernel_census.py --phase block

The precursor problem. Grabbing a settled ``PairformerLayer`` or ``Diffusion`` call needs a
real fold, and a real fold dispatches ~500k programs, which is three orders of magnitude over
the profiler's DRAM marker budget. So the precursor is cut down to the shortest run that
still produces a call with the *shipped shapes*: recycling is set to 1, the trunk's pairformer
stack is truncated to a couple of blocks, and the fold is aborted with a sentinel exception
the moment the wanted call has been grabbed. None of that touches the block being profiled --
only the values flowing into it, and a bf16 matmul's cycle count does not depend on its
values. The check that this is sound is external: the profiled block's summed device time
must reproduce the 41.4152 ms / 32.5179 ms trace-replay floors measured on the full fold.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None

FENCE_N = 3          # consecutive sentinel ops that bracket the profiled region
FENCE_DIM = 32       # sentinel tensor is FENCE_DIM x FENCE_DIM, a shape the block never uses


class Grabbed(Exception):
    """Raised to unwind out of the precursor fold as soon as the call has been captured."""


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def timed_reps(ttnn, dev, fn, args, kwargs, reps, fence):
    """Warm, fence, run `reps` profiled calls, fence. Returns the synced wall per call."""
    for _ in range(3):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    wall = (time.perf_counter() - t0) / reps
    fence()
    return round(1e3 * wall, 4)


def phase_probe(ttnn, dev, reps=20):
    """Instrument proof: one ttnn.matmul of known size, profiler cycles vs synced wall."""
    import torch
    n = 2048
    a = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    b = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    fence = make_fence(ttnn, dev)
    for _ in range(5):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    # isolated, one call per sync: the wall is this one matmul and nothing else
    solo = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.matmul(a, b)
        ttnn.synchronize_device(dev)
        solo.append(time.perf_counter() - t0)
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    btb = (time.perf_counter() - t0) / reps
    fence()
    return {"n": n, "reps": reps,
            "solo_synced_ms": round(1e3 * st.median(solo), 4),
            "back_to_back_ms": round(1e3 * btb, 4),
            "flops": 2 * n ** 3,
            "bytes": 3 * n * n * 2}


def truncate_stacks(T, keep):
    """Shorten every Pairformer/MSA block list so the precursor fold is cheap.

    Returns a restore callable. Shapes are untouched; only how many times a block runs.
    """
    import gc
    saved = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, (T.Pairformer, T.MSA)) and isinstance(
                    getattr(obj, "blocks", None), list) and len(obj.blocks) > keep:
                saved.append((obj, obj.blocks))
                obj.blocks = obj.blocks[:keep]
        except ReferenceError:
            continue

    def restore():
        for obj, blocks in saved:
            obj.blocks = blocks
    return restore, len(saved)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=("probe", "block", "step"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keep-blocks", type=int, default=2,
                    help="pairformer blocks left in the precursor stack (step phase only)")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "phase": a.phase, "reps": a.reps, "size": a.size,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "loadavg": loadavg()}
    dump()

    if a.phase == "probe":
        T.get_device()
        dev = T.get_device()
        OUT["probe"] = phase_probe(ttnn, dev)
        print("  probe " + json.dumps(OUT["probe"]), flush=True)
        dump()
        print("DONE", a.out, flush=True)
        return 0

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = 1 if a.phase == "step" else _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 30)
    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}", tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    cname = "PairformerLayer" if a.phase == "block" else "Diffusion"
    want = ((lambda o, ar: getattr(o, "transform_s", False)) if a.phase == "block"
            else (lambda o, ar: True))
    grabs: dict = {}
    counts: dict = {}
    cls = getattr(T, cname)
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts[cname] = counts.get(cname, 0) + 1
        out = orig(self_obj, *args, **kw)
        if cname not in grabs and counts[cname] >= 2 and want(self_obj, args):
            grabs[cname] = {"obj": self_obj,
                            "args": tuple(clone(x) for x in args),
                            "kwargs": {k: clone(v) for k, v in kw.items()}}
            print(f"  grabbed {cname} on call {counts[cname]}", flush=True)
            raise Grabbed
        return out
    cls.__call__ = wrapper

    restore = None
    if a.phase == "step":
        restore, n_trunc = truncate_stacks(T, a.keep_blocks)
        OUT["env"]["stacks_truncated"] = n_trunc
        OUT["env"]["keep_blocks"] = a.keep_blocks
        print(f"  truncated {n_trunc} block stacks to {a.keep_blocks} for the precursor",
              flush=True)

    print("=== precursor fold (aborted at the grab) ===", flush=True)
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
        if restore:
            restore()
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    OUT["class_calls_in_precursor"] = counts
    dump()
    if cname not in grabs:
        OUT["error"] = f"{cname} was never grabbed"
        dump()
        print("FAILED " + OUT["error"], flush=True)
        return 1

    g = grabs[cname]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else str(type(x).__name__)
                         for x in g["args"]]
    fence = make_fence(ttnn, dev)
    print(f"=== profiled region: {a.reps} x {cname} ===", flush=True)
    OUT["synced_wall_ms_per_call"] = timed_reps(ttnn, dev, g["obj"], g["args"], g["kwargs"],
                                                a.reps, fence)
    print(f"  {cname} synced wall {OUT['synced_wall_ms_per_call']:.4f} ms/call "
          f"(profiler={OUT['env']['profiler']})", flush=True)
    OUT["fence"] = {"op": "ttnn.exp", "n": FENCE_N, "dim": FENCE_DIM}
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
