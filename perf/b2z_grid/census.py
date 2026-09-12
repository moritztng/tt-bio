#!/usr/bin/env python3
"""Core-grid census of one settled Boltz-2 unit, from the real dispatched programs.

Run under a Tracy-enabled tt-metal source build:

    TT_METAL_HOME=/home/moritz/tt-metal \
    PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME \
    python -m tracy -r -o OUT --op-support-count 40000 -- perf/b2z_grid/census.py --unit PairformerLayer

The fold is aborted the moment a settled call of the wanted class has been grabbed, so the profiler
only ever sees a few thousand programs instead of the ~487k a whole fold dispatches (the device
marker buffers overflow at ~1000 undrained programs). The grabbed call is then replayed `--repeats`
times warm; those repeats are the census rows, identified in the CSV by their GLOBAL CALL COUNT
being above the marker printed at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


class _Grabbed(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="PairformerLayer")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--marker", type=Path, default=HERE / "marker.json")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = a.recycles or _resolve_recycling_steps(None, "boltz2")
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
    dev = T.get_device()
    print("META " + json.dumps({k: str(meta[k]) for k in meta if k in
                                ("hardware", "grid", "card_type")}), flush=True)

    grab: dict = {}
    counts = {"n": 0}
    cls = getattr(T, a.unit)
    orig = cls.__dict__["__call__"]

    def want(o, ar):
        if a.unit == "PairformerLayer":
            return bool(getattr(o, "transform_s", False))
        return True

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def w(self_obj, *args, **kw):
        counts["n"] += 1
        # call 1 and 2 warm the program cache for this shape; call 3 is settled
        if counts["n"] >= 3 and want(self_obj, args) and not grab:
            out = orig(self_obj, *args, **kw)
            grab.update(obj=self_obj,
                        args=tuple(clone(x) for x in args),
                        kwargs={k: clone(v) for k, v in kw.items()})
            ttnn.synchronize_device(dev)
            raise _Grabbed
        return orig(self_obj, *args, **kw)

    cls.__call__ = w
    t0 = time.perf_counter()
    try:
        one_fold()
    except _Grabbed:
        pass
    finally:
        cls.__call__ = orig
    print(f"GRABBED {a.unit} on call {counts['n']} after {time.perf_counter()-t0:.1f}s", flush=True)

    obj, args, kwargs = grab["obj"], grab["args"], grab["kwargs"]
    # two more warm calls outside the census window, so the census rows are fully settled
    for _ in range(2):
        obj(*args, **kwargs)
    ttnn.synchronize_device(dev)

    marker = {"unit": a.unit, "size": a.size, "repeats": a.repeats,
              "grid": str(dev.compute_with_storage_grid_size())}
    walls = []
    for i in range(a.repeats):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        obj(*args, **kwargs)
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t)
        print(f"CENSUS-REP {i} wall_ms {1e3*walls[-1]:.3f}", flush=True)
    marker["wall_ms"] = [round(1e3 * x, 3) for x in walls]
    a.marker.write_text(json.dumps(marker, indent=1))
    print("MARKER " + json.dumps(marker), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
