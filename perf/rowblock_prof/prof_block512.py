#!/usr/bin/env python3
"""Device-profiled A/B of the row-blocked pair transpose, on one real 512 aa Pairformer block.

The predecessor leg (`protenix-trunk--z-rowblock`) proved the change bit-exact and left one number
open: does the blocked form make the two `ttnn.permute` sites faster *inside* the model, and with
what sign? Every host wall it tried has a session noise floor larger than the effect. This harness
answers it with the tt-metal device profiler instead, which times the op rather than the block that
contains it.

Run it UNDER `python -m tracy` (see the doc); bare, it still prints the host block wall and the
branch census, which is how you check the arm before spending a profiled run on it.

    TT_METAL_HOME=/home/ttuser/tt-metal
    PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME:$WT
    TT_VISIBLE_DEVICES=3 python3 -m tracy -r -o OUT --op-support-count 4000 -- \
        $WT/perf/rowblock_prof/prof_block512.py --arm rb_fit --n 512 --warm 1 --iters 3

Attribution is by MARKER, not by shape. Every `_pair_transpose` call is bracketed by a
`ttnn.sqrt` on a [1, 1, 32, 64] tensor, a shape and op that appear nowhere else in the block, so
everything the blocked path adds -- the row slices, the per-block permutes, the group flushes and
the final concat -- falls inside the bracket and cannot be missed. Shape-based attribution would
have missed it: at 512 aa the pair transpose lands as `TransposeDeviceOperation` HC, and its row
slices share `[512, 512, 256]` with the trimul chunker.

The clone roofs are taken in the SAME run and the SAME clock as the ops they score, at the same
[512, 512, 256] shape, to both destinations. A device-kernel duration scored against a host-wall
roof mixes two clocks (`ttnn-perf-profiling` §5) and that is how a roof gets mislabelled.
"""

import argparse
import importlib.util
import json
import os
import statistics as st
import time

import torch

import ttnn
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_H, MARK_W = 32, 64          # unique in the block; nothing else is 32x64
PAIR_BYTES = 512 * 512 * 256 * 2  # one 512 aa pair tensor, bf16


def load_build_layer():
    """pf_layer.py is a script, not a module: load it by path and reuse its checkpoint loader."""
    path = os.path.join(HERE, "..", "stage_split_298", "pf_layer.py")
    spec = importlib.util.spec_from_file_location("pf_layer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_layer


class Marker:
    def __init__(self, dev):
        self.t = ttnn.from_torch(torch.ones(1, 1, MARK_H, MARK_W), layout=ttnn.TILE_LAYOUT,
                                 device=dev, dtype=ttnn.bfloat16)

    def __call__(self):
        ttnn.deallocate(ttnn.sqrt(self.t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["on", "rb_fit"], required=True,
                    help="on = the unblocked permute byte for byte; rb_fit = the shipping blocked form")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--warm", type=int, default=1, help="unmeasured block executions (JIT + program cache)")
    ap.add_argument("--iters", type=int, default=3, help="measured block executions")
    ap.add_argument("--roof-reps", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    T._TRANSPOSE_ROWBLOCK = (args.arm == "rb_fit")

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = load_build_layer()(ckc)
    mark = Marker(dev)

    # Bracket every pair transpose, and record the branch it actually took rather than inferring it.
    census = []
    orig = T._pair_transpose

    def wrapped(x):
        mc = T._transpose_memory_config(x)
        plan = None if mc.buffer_type == ttnn.BufferType.L1 or not T._TRANSPOSE_ROWBLOCK \
            else T._rowblock_plan(x)
        census.append({"shape": [int(d) for d in x.shape], "dest": str(mc.buffer_type),
                       "plan": None if plan is None else list(plan)})
        mark()
        out = orig(x)
        mark()
        return out

    T._pair_transpose = wrapped

    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"ARM {args.arm} N={N} c_z={c_z} grid={T.COMPUTE_GRID_MAIN} "
          f"l1_budget={T._l1_budget_bytes()}", flush=True)

    for _ in range(args.warm):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)
    n_warm_calls = len(census)

    walls = []
    for _ in range(args.iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        walls.append((time.perf_counter() - t0) * 1e3)

    # Roofs: same run, same clock, same shape. Every buffer-type pair the two arms actually use --
    # the unblocked permute is DRAM to DRAM, the blocked one is DRAM to L1 then L1 to DRAM -- so all
    # three are needed and none of them may be inherited (charter §4.1).
    roof_dram = ttnn.from_torch(torch.randn(N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    roof_l1 = None
    try:
        roof_l1 = ttnn.clone(roof_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    except Exception as exc:                                                   # noqa: BLE001
        print(f"roof (L1 source) skipped, L1 allocation refused: {exc}", flush=True)
    plan = [("copy roof (DRAM), DRAM source", roof_dram, ttnn.DRAM_MEMORY_CONFIG),
            ("copy roof (L1), DRAM source", roof_dram, ttnn.L1_MEMORY_CONFIG)]
    if roof_l1 is not None:
        plan.append(("copy roof (DRAM), L1 source", roof_l1, ttnn.DRAM_MEMORY_CONFIG))
    roof_order = [name for name, _, _ in plan]
    for _, src, cfg in plan:
        for _ in range(args.roof_reps):
            mark()
            out = ttnn.clone(src, memory_config=cfg)
            mark()
            ttnn.deallocate(out)
    ttnn.synchronize_device(dev)

    blocked = [c for c in census if c["plan"]]
    out = {
        "arm": args.arm, "n": N, "c_z": c_z, "warm": args.warm, "iters": args.iters,
        "roof_reps": args.roof_reps, "grid": list(T.COMPUTE_GRID_MAIN),
        "l1_budget_bytes": T._l1_budget_bytes(),
        "pair_bytes": N * N * c_z * 2,
        "marker": {"op": "sqrt", "shape": [1, 1, MARK_H, MARK_W]},
        "roof_order": roof_order,
        "n_warm_transpose_calls": n_warm_calls,
        "n_measured_transpose_calls": len(census) - n_warm_calls,
        "census_unique": sorted({(str(c["shape"]), c["dest"], str(c["plan"])) for c in census}),
        "blocked_calls": len(blocked), "total_calls": len(census),
        "block_wall_ms": {"series": [round(w, 2) for w in walls],
                          "median": round(st.median(walls), 2) if walls else None},
    }
    print("CENSUS " + json.dumps(out["census_unique"]), flush=True)
    print(f"BLOCKED {len(blocked)}/{len(census)} calls   BLOCK_MS median "
          f"{out['block_wall_ms']['median']} series {out['block_wall_ms']['series']}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
