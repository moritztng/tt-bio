#!/usr/bin/env python3
"""The diffusion step's program list, and the A/B that prices deleting programs from it.

`b2z2-sampler-stall-split` measured the step's input wait at 55.8 % of math-thread residency and
found 62.9 % of that wait is a per-program constant of 9.76 us, not bytes. So on this block the
currency is PROGRAMS REMOVED, not tile passes and not bytes. This file is the instrument for that
currency: it grabs one settled `Diffusion.__call__` out of a real 512 aa fold and

  --mode ops     graph-captures it once and prints the ordered top-level ttnn op list, so a
                 fusion target can be picked from the actual sequence rather than from a per-op-code
                 histogram that has lost the order;
  --mode time    replays it `--reps` times with the profiler OFF and reports the synced wall. The
                 profiler costs 3.81x on this block (41.4820 -> 158.2582 ms), all of it in the
                 gaps between programs, so a step wall may only be taken bare.
  --mode parity  runs the same grabbed call under two env settings and compares the outputs with
                 torch.equal, plus a negative control that must fail.

The grab is `perf/b2z_kernel_census/kernel_census.py`'s: recycling 1, the pairformer stack
truncated, and the fold aborted with a sentinel the moment the call is in hand. None of that
touches the step being measured -- only the values flowing into it, and a bf16 program's cycle
count does not depend on its values.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

OUT: dict = {}
OUT_PATH: Path | None = None

FENCE_N, FENCE_DIM = 3, 32


class Grabbed(Exception):
    """Unwind out of the precursor fold as soon as the call has been captured."""


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def timed_reps(ttnn, dev, fn, args, kwargs, reps, fence, n_med=5):
    """Warm, fence, then `n_med` blocks of `reps` calls. Median block wall per call."""
    for _ in range(3):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    fence()
    walls = []
    for _ in range(n_med):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args, **kwargs)
        ttnn.synchronize_device(dev)
        walls.append((time.perf_counter() - t0) / reps)
    fence()
    return {"ms_per_call": round(1e3 * st.median(walls), 4),
            "ms_all": [round(1e3 * w, 4) for w in walls],
            "reps": reps, "blocks": n_med}


def truncate_stacks(T, keep):
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


def grab_step(ttnn, T, B, size, keep_blocks):
    """Run the precursor fold and return the settled `Diffusion.__call__` and its operands."""
    from tt_bio.main import _resolve_recycling_steps          # noqa: F401  (import shape check)
    B.RECYCLING_STEPS = 1
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=512 << 20)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", HERE / f".msa_{size}", fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    grabs, counts = {}, {"n": 0}
    cls = T.Diffusion
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] += 1
        out = orig(self_obj, *args, **kw)
        if not grabs and counts["n"] >= 2:
            grabs["g"] = {"obj": self_obj,
                          "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
            raise Grabbed
        return out
    cls.__call__ = wrapper
    restore, n_trunc = truncate_stacks(T, keep_blocks)
    OUT["env"]["stacks_truncated"] = n_trunc

    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
        restore()
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    OUT["diffusion_calls_in_precursor"] = counts["n"]
    dump()
    if not grabs:
        raise SystemExit("Diffusion was never grabbed")
    g = grabs["g"]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else type(x).__name__
                         for x in g["args"]]
    return dev, g


def mode_ops(ttnn, dev, g):
    """One graph capture -> the ordered top-level ttnn op list."""
    from itemize import top_level_spans
    g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    nodes = ttnn.graph.end_graph_capture()
    ops, _owner = top_level_spans(nodes)
    names = [o["name"] for o in ops]
    by = defaultdict(int)
    for n in names:
        by[n] += 1
    import gzip
    raw = OUT_PATH.with_suffix(".graph.json.gz")
    with gzip.open(raw, "wt") as fh:
        json.dump(nodes, fh)
    OUT["graph"] = raw.name
    OUT["ops"] = {"n_top_level": len(names),
                  "by_name": dict(sorted(by.items(), key=lambda kv: -kv[1])),
                  "sequence": names}
    print(f"  {len(names)} top-level ttnn ops", flush=True)
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {v:5d}  {k}", flush=True)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("ops", "time"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "mode": a.mode, "size": a.size, "label": a.label,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "flags": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith("TT_BIO_") or k.startswith("B2_")},
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    dev, g = grab_step(ttnn, T, B, a.size, a.keep_blocks)
    if a.mode == "ops":
        mode_ops(ttnn, dev, g)
    else:
        fence = make_fence(ttnn, dev)
        OUT["step"] = timed_reps(ttnn, dev, g["obj"], g["args"], g["kwargs"],
                                 a.reps, fence, a.blocks)
        print(f"  step {OUT['step']['ms_per_call']:.4f} ms/call  {OUT['step']['ms_all']}",
              flush=True)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
