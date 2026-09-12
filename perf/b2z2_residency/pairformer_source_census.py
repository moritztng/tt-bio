#!/usr/bin/env python3
"""Where do one PairformerLayer's operand tiles actually live before the op reads them?

`arrival_rate.py` prices a tile by its source. This counts how many tiles of each source the
block actually consumes, so the price can be multiplied by something real.

Method: grab a settled `PairformerLayer` out of a 512 aa precursor fold exactly the way
`b2z-kernel-cycle-census` does (wrap `__call__`, take the second call, unwind with a sentinel),
then replay that one call with every ttnn entry point the engine uses wrapped in a recorder. For
each call the recorder walks the arguments, and for every `ttnn.Tensor` it finds records the
buffer type (DRAM or L1), the memory layout (interleaved or sharded) and the tile count.

The unit is an operand-tile read: one tile, presented once to one op as an input. A matmul reads
its operands more than once when it multicasts, so this undercounts matmul traffic and is a lower
bound on the DRAM-sourced share, not an estimate of it. Stated that way on purpose.

No timing is taken here and none should be: the census is structural, so it is valid on any card.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


class Grabbed(Exception):
    """Unwind out of the precursor fold the moment the wanted call has been captured."""


WRAPPED = [
    "matmul", "linear", "add", "multiply", "subtract", "layer_norm", "softmax", "transpose",
    "permute", "concat", "slice", "reshape", "sigmoid", "exp", "silu", "clone", "typecast",
    "to_memory_config", "generic_op", "mul", "div", "sum", "where", "pad", "tilize", "untilize",
]


def tensor_facts(ttnn, t):
    try:
        mc = t.memory_config()
        buf = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
        layout = str(mc.memory_layout).rsplit(".", 1)[-1]
        tiles = math.prod(tuple(t.shape)) / 1024.0
        return buf, layout, tiles
    except Exception:  # noqa: BLE001
        return None


def walk(ttnn, obj, out, depth=0):
    if depth > 3:
        return
    if isinstance(obj, ttnn.Tensor):
        f = tensor_facts(ttnn, obj)
        if f:
            out.append(f)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            walk(ttnn, o, out, depth + 1)


def install_recorder(ttnn, log):
    originals = {}
    for name in WRAPPED:
        fn = getattr(ttnn, name, None)
        if fn is None or not callable(fn):
            continue
        originals[name] = fn

        def make(name, fn):
            def rec(*args, **kw):
                ins = []
                walk(ttnn, args, ins)
                walk(ttnn, list(kw.values()), ins)
                out = fn(*args, **kw)
                outs = []
                walk(ttnn, out, outs)
                log.append({"op": name, "in": ins, "out": outs})
                return out
            return rec
        setattr(ttnn, name, make(name, fn))

    def restore():
        for name, fn in originals.items():
            setattr(ttnn, name, fn)
    return restore, len(originals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 29)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}",
                                         fix / f"cdk2x2_{a.size}.yaml",
                                         fix / f"cdk2x2_{a.size}.a3m")

    grabs, counts = {}, {}
    cls = T.PairformerLayer
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] = counts.get("n", 0) + 1
        out = orig(self_obj, *args, **kw)
        if "g" not in grabs and counts["n"] >= 2 and getattr(self_obj, "transform_s", False):
            grabs["g"] = {"obj": self_obj, "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
            print(f"  grabbed PairformerLayer on call {counts['n']}", flush=True)
            raise Grabbed
        return out

    cls.__call__ = wrapper
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
    precursor_s = round(time.perf_counter() - t0, 3)
    if "g" not in grabs:
        print("FAILED PairformerLayer was never grabbed", flush=True)
        return 1

    g = grabs["g"]
    g["obj"](*g["args"], **g["kwargs"])  # warm, unrecorded
    ttnn.synchronize_device(T.get_device())

    log: list = []
    restore, nwrapped = install_recorder(ttnn, log)
    try:
        g["obj"](*g["args"], **g["kwargs"])
        ttnn.synchronize_device(T.get_device())
    finally:
        restore()

    by_src = defaultdict(float)
    by_layout = defaultdict(float)
    per_op = defaultdict(lambda: defaultdict(float))
    out_src = defaultdict(float)
    for rec in log:
        for buf, layout, tiles in rec["in"]:
            by_src[buf] += tiles
            by_layout[f"{buf}/{layout}"] += tiles
            per_op[rec["op"]][buf] += tiles
        for buf, _layout, tiles in rec["out"]:
            out_src[buf] += tiles

    total_in = sum(by_src.values())
    res = {
        "env": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
                "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                "hardware": meta.get("hardware"), "grid": meta.get("grid"),
                "precursor_s": precursor_s, "wrapped_entry_points": nwrapped},
        "recorded_calls": len(log),
        "operand_tile_reads_total": total_in,
        "operand_tile_reads_by_source": dict(by_src),
        "operand_tile_reads_by_source_and_layout": dict(by_layout),
        "dram_share_of_operand_tile_reads": by_src["DRAM"] / total_in if total_in else None,
        "output_tile_writes_by_destination": dict(out_src),
        "per_op": {k: dict(v) for k, v in sorted(
            per_op.items(), key=lambda kv: -sum(kv[1].values()))},
    }
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_op"}, indent=2), flush=True)
    print("top ops by operand tile reads:", flush=True)
    for k, v in list(res["per_op"].items())[:12]:
        tot = sum(v.values())
        print(f"  {k:18s} {tot:10.0f} tiles  DRAM {v.get('DRAM', 0)/tot*100:5.1f} %", flush=True)
    print("DONE " + str(a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
