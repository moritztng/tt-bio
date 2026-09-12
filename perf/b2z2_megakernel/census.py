#!/usr/bin/env python3
"""The PairformerLayer block, op by op, in TILE PASSES -- the currency the fold is floored in.

Wave 1 measured that 57.0 % of the block's math-thread time is spent blocked on input tiles that
have not arrived, at a measured 71.3 ns a tile, and that cost scales with tile COUNT rather than
bytes or FLOPs (`state/b2z/FINDINGS.md`). Every byte-shaped lever this campaign priced therefore
mispriced itself. This file re-denominates the block in tiles.

A TILE PASS here is one 32x32 tile crossing the DRAM<->L1 boundary once: a write by the op that
allocated the buffer, a read by every op that consumes it. That is `real_traffic.counts`' byte
model with the bytes divided by the tile size of the buffer's dtype, so the two are the same
measurement in two units and can be checked against each other.

It is NOT a count of unpack/pack passes inside a kernel -- a matmul re-streams its operands from
L1 and nothing at the ttnn graph level sees that. Fusion deletes exactly the boundary passes
counted here, which is why this is the ledger a megakernel is priced against.

The block is grabbed from a real warm 512 aa fold, not reconstructed: the same hook
`perf/b2x_op_cost/device_floor.py` uses. Sampling steps are dropped because the census is a trunk
measurement and the sampler does not run inside a PairformerLayer; nothing here is a fold time.
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

from itemize import itemize                                                  # noqa: E402

# bytes per 32x32 tile, by the dtype name ttnn prints in a buffer node
TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088, "BFLOAT4_B": 576,
              "UINT32": 4096, "INT32": 4096, "UINT16": 2048, "UINT8": 1024}
NO_TRAFFIC = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate",
              "ttnn.to_memory_config", "ttnn.clone"}

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def tile_ledger(call, tile_bytes=2048):
    """Per-op tile reads and writes over the DRAM<->L1 boundary.

    `tile_bytes` is the fallback divisor: the graph's buffer nodes carry a size and a buffer type
    but not a dtype, and the Boltz-2 trunk is bf16 end to end, so 2048 is exact for every buffer
    whose size divides by it. Anything that does not divide is reported separately rather than
    silently rounded -- a buffer that is not a whole number of bf16 tiles is either a different
    dtype or not tiled, and guessing would put a fake number in the ledger.
    """
    ops, rows = itemize(call)
    dram = [r for r in rows if r["kind"] == "DRAM"]
    alloc_by_op = defaultdict(int)
    for r in dram:
        if r["alloc_op_i"] is not None:
            alloc_by_op[r["alloc_op_i"]] += r["size"]

    def moves(i):
        return ops[i]["name"] not in NO_TRAFFIC and (alloc_by_op[i] > 0
                                                     or ops[i]["name"].endswith("_"))

    w_b, r_b = defaultdict(int), defaultdict(int)
    odd = 0
    for r in dram:
        if r["size"] % tile_bytes:
            odd += r["size"]
        readers = [i for i in r["consumers"] if moves(i)]
        if r["alloc_op_i"] is not None:
            w_b[r["alloc_op_i"]] += r["size"]
        for i in readers:
            r_b[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc_by_op[i] == 0:
                w_b[i] += r["size"]
        if not readers:
            r_b[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]

    per_op = []
    for i, op in enumerate(ops):
        if not (w_b[i] + r_b[i]):
            continue
        per_op.append({"i": i, "op": op["name"],
                       "tiles_r": r_b[i] // tile_bytes, "tiles_w": w_b[i] // tile_bytes,
                       "tiles": (r_b[i] + w_b[i]) // tile_bytes,
                       "MB": round((r_b[i] + w_b[i]) / 1e6, 3)})
    per_op.sort(key=lambda d: -d["tiles"])
    by_name = defaultdict(lambda: {"n": 0, "tiles": 0})
    for d in per_op:
        by_name[d["op"]]["n"] += 1
        by_name[d["op"]]["tiles"] += d["tiles"]
    tot_r, tot_w = sum(r_b.values()), sum(w_b.values())
    return {"n_ops": len(ops),
            "tiles_r": tot_r // tile_bytes, "tiles_w": tot_w // tile_bytes,
            "tile_passes": (tot_r + tot_w) // tile_bytes,
            "MB": round((tot_r + tot_w) / 1e6, 3),
            "non_tile_bytes": odd,
            "per_op": per_op,
            "by_op_name": {k: v for k, v in sorted(by_name.items(),
                                                   key=lambda kv: -kv[1]["tiles"])}}


def capture(ttnn, dev, fn, *args, **kw):
    """One graph capture of one settled call. Warm first: a first call compiles programs."""
    out = fn(*args, **kw)
    _free(ttnn, out)
    ttnn.synchronize_device(dev)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = fn(*args, **kw)
    ttnn.synchronize_device(dev)
    g = ttnn.graph.end_graph_capture()
    _free(ttnn, out)
    return {"nodes": g, "sig": getattr(fn, "__name__", "call")}


PROTECT: set = set()


def _addr(t):
    try:
        return t.buffer_address()
    except Exception:                                                        # noqa: BLE001
        return None


def _free(ttnn, out):
    """Free a stage's result -- unless it IS one of the stage's inputs.

    `PairformerLayer.__call__` updates z with `ttnn.add_` and returns that same buffer, so a
    blanket deallocate of the result frees the block's own input and every later stage dies on
    `Buffer is not allocated`. The protected set is the addresses of the operands the census
    holds.
    """
    for t in (out if isinstance(out, (tuple, list)) else [out]):
        if isinstance(t, ttnn.Tensor) and _addr(t) not in PROTECT:
            ttnn.deallocate(t)


def trace_ms(ttnn, dev, fn, args, kwargs, reps=16, n_med=5):
    """Device ms per call by trace replay -- host removed, the same protocol as b2x_op_cost."""
    for _ in range(2):
        _free(ttnn, fn(*args, **kwargs))
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn(*args, **kwargs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    meds = []
    for _ in range(n_med):
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    ttnn.release_trace(dev, tid)
    return round(1e3 * st.median(meds), 4)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4, help="sampler steps; trunk census only")
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    OUT_PATH = a.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = a.steps
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 29)
    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "host": os.uname().nodename,
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "size": a.size, "sampling_steps": a.steps,
                  "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn")}
    dump()
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}", tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    # The census runs INSIDE the fold, on the live call. A device clone does not survive the
    # fold it was taken in, and neither do the module's own weights: the first attempt restored
    # its arguments from the host and still died in `layer_norm` on an unallocated norm weight.
    # Whatever frees them runs at fold exit, so the only place every operand and every weight is
    # certainly alive is inside the call itself.
    class _Done(Exception):
        pass

    counts: dict = {}
    cls = T.PairformerLayer
    orig = cls.__dict__["__call__"]

    def run_census(layer, args, kwargs):
        s_, z_ = args[0], args[1]
        for t in list(args) + list(kwargs.values()):
            if isinstance(t, ttnn.Tensor):
                PROTECT.add(_addr(t))
        mask = args[2] if len(args) > 2 else kwargs.get("mask")
        am_s = args[3] if len(args) > 3 else kwargs.get("attn_mask_start")
        am_e = args[4] if len(args) > 4 else kwargs.get("attn_mask_end")
        extra = kwargs.get("extra_attn_bias")
        OUT["shapes"] = {"s": list(s_.shape), "z": list(z_.shape),
                         "s_dtype": str(s_.dtype), "z_dtype": str(z_.dtype),
                         "hidden_z": list(layer.transition_z.fc1_weight.shape),
                         "hidden_s": list(layer.transition_s.fc1_weight.shape)}
        dump()

        def blk():
            return orig(layer, *args, **kwargs)

        def s_norm_apb():
            sn = ttnn.layer_norm(s_, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                                 epsilon=1e-5,
                                 compute_kernel_config=layer.compute_kernel_config)
            u = layer.attention_pair_bias(sn, z_, seq_mask=extra if extra is not None else am_s)
            ttnn.deallocate(sn)
            return u

        stages = [
            ("block", blk, (), {}),
            ("tri_mul_start", layer.triangle_multiplication_start, (z_, mask), {}),
            ("tri_mul_end", layer.triangle_multiplication_end, (z_, mask), {}),
            ("tri_att_start", layer.triangle_attention_start, (z_, am_s), {}),
            ("tri_att_end", layer.triangle_attention_end, (z_, am_e), {}),
            ("transition_z", layer.transition_z, (z_,), {}),
            ("attn_pair_bias", s_norm_apb, (), {}),
            ("transition_s", layer.transition_s, (s_,), {}),
        ]
        OUT["stages"] = {}
        for name, fn, fa, fk in stages:
            rec: dict = {}
            try:
                g = capture(ttnn, dev, fn, *fa, **fk)
                rec.update(tile_ledger(g))
                (HERE / "out" / f"graph_{name}_{a.size}.json").write_text(json.dumps(g))
            except Exception as e:                                          # noqa: BLE001
                rec["capture_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if not a.no_trace:
                try:
                    rec["device_ms"] = trace_ms(ttnn, dev, fn, fa, fk)
                except Exception as e:                                      # noqa: BLE001
                    rec["trace_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            OUT["stages"][name] = rec
            print("  %-16s %8s tiles  %9s MB  %s ms  (%s ops)"
                  % (name, rec.get("tile_passes"), rec.get("MB"), rec.get("device_ms"),
                     rec.get("n_ops")), flush=True)
            dump()

    def w(self_obj, *args, **kw):
        counts["n"] = counts.get("n", 0) + 1
        if counts["n"] >= 3 and getattr(self_obj, "transform_s", False):
            OUT["grabbed_on_call"] = counts["n"]
            print(f"  census on PairformerLayer call {counts['n']}", flush=True)
            run_census(self_obj, args, kw)
            raise _Done
        return orig(self_obj, *args, **kw)

    cls.__call__ = w
    print("=== cold fold (compiles, no census) ===", flush=True)
    cls.__call__ = orig
    one_fold()
    print("=== fold 2: census on the live call ===", flush=True)
    cls.__call__ = w
    try:
        one_fold()
    except _Done:
        print("  census complete", flush=True)
    finally:
        cls.__call__ = orig
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
