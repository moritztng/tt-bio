#!/usr/bin/env python3
"""Whole-fold FLOP and DRAM-byte census, per stage, from real shapes.

Same stage boundaries as perf/stage_split_298/stage_split.py, but instead of timing
it counts arithmetic: every ttnn matmul/linear/SDPA call contributes 2 * output
elements * contraction length, and every op contributes the bytes of its DRAM-resident
operands and results. The result is the model's essential arithmetic (shape-derived,
not estimated) and the traffic today's implementation actually moves.

Instrumented, so NEVER quote a duration from this run.

    TT_MESH_GRAPH_DESC_PATH=... TT_VISIBLE_DEVICES=3 python3 perf/ceiling/flopfold.py \
      --model protenix-v2 --target examples/prot300.yaml \
      --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --out census_fold_298.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STAGE = ["other"]
TALLY = OrderedDict()
SHAPES = OrderedDict()
DT_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1, "BFLOAT4_B": 1,
            "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1}


def _dtb(t):
    return DT_BYTES.get(str(t.dtype).split(".")[-1].upper(), 2)


def _acc(stage, op, kind, f, di, do, li, lo):
    e = TALLY.setdefault((stage, op, kind), [0, 0, 0, 0, 0, 0])
    e[0] += 1
    e[1] += f
    e[2] += di
    e[3] += do
    e[4] += li
    e[5] += lo


def _wrap(mod, name, kind):
    fn = getattr(mod, name, None)
    if fn is None:
        return

    def w(*a, **k):
        out = fn(*a, **k)
        ins, outs = [], []
        try:
            for x in a:
                if isinstance(x, ttnn.Tensor):
                    ins.append(x)
            for v in k.values():
                if isinstance(v, ttnn.Tensor):
                    ins.append(v)
            for o in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(o, ttnn.Tensor):
                    outs.append(o)
            f = 0
            if kind == "matmul" and outs and len(ins) >= 2:
                f = 2 * math.prod(list(outs[0].shape)) * list(ins[0].shape)[-1]
            elif kind == "sdpa" and len(ins) >= 2:
                q, kk = list(ins[0].shape), list(ins[1].shape)
                f = 4 * q[0] * q[1] * q[2] * kk[2] * q[3]
            di = do = li = lo = 0
            for t in ins:
                b = math.prod(list(t.shape)) * _dtb(t)
                if "DRAM" in str(t.memory_config().buffer_type):
                    di += b
                else:
                    li += b
            for t in outs:
                b = math.prod(list(t.shape)) * _dtb(t)
                if "DRAM" in str(t.memory_config().buffer_type):
                    do += b
                else:
                    lo += b
            _acc(STAGE[0], name, kind, f, di, do, li, lo)
            if kind in ("matmul", "sdpa"):
                sig = (STAGE[0], name,
                       "x".join(map(str, list(ins[0].shape))),
                       "x".join(map(str, list(ins[1].shape))) if len(ins) > 1 else "-")
                s = SHAPES.setdefault(sig, [0, 0])
                s[0] += 1
                s[1] += f
        except Exception:
            pass
        return out

    setattr(mod, name, w)


def install_op_patches():
    for n in ("matmul", "linear"):
        _wrap(ttnn, n, "matmul")
    _wrap(ttnn.experimental, "minimal_matmul", "matmul")
    _wrap(ttnn.transformer, "scaled_dot_product_attention", "sdpa")
    for n in ("add", "add_", "multiply", "multiply_", "subtract", "sigmoid", "silu", "gelu", "exp"):
        _wrap(ttnn, n, "eltwise")
    for n in ("layer_norm", "rms_norm", "softmax"):
        _wrap(ttnn, n, "norm")
    for n in ("permute", "concat", "chunk", "clone", "slice", "transpose", "to_layout",
              "pad", "embedding", "reallocate", "repeat", "typecast"):
        _wrap(ttnn, n, "move")


def staged(name, fn):
    def wrapper(*a, **k):
        prev = STAGE[0]
        STAGE[0] = name if prev == "other" else prev
        try:
            return fn(*a, **k)
        finally:
            STAGE[0] = prev
    return wrapper


def install_stage_patches():
    import tt_bio.protenix as P
    import tt_bio.opendde as O
    P.edm_sample = staged("diffusion", P.edm_sample)
    P.Trunk.__call__ = staged("trunk", P.Trunk.__call__)
    P.Protenix._diffusion_pair_cond = staged("diff_pair_cond", P.Protenix._diffusion_pair_cond)
    P.ConfidenceHead.confidence = staged("confidence", P.ConfidenceHead.confidence)
    if hasattr(P.ConfidenceHead, "confidence_device"):
        P.ConfidenceHead.confidence_device = staged("confidence", P.ConfidenceHead.confidence_device)
    O.OpenDDE.expand_and_refine = staged("expand_refine", O.OpenDDE.expand_and_refine)
    # nested detail inside the trunk
    import tt_bio.tenstorrent as T
    T.Pairformer.__call__ = _sub("trunk.pairformer", T.Pairformer.__call__)
    T.MSA.__call__ = _sub("trunk.msa", T.MSA.__call__)
    T.TriangleMultiplication.__call__ = _sub("pf.trimul", T.TriangleMultiplication.__call__)
    T.TriangleAttention.__call__ = _sub("pf.triatt", T.TriangleAttention.__call__)
    T.Transition.__call__ = _sub("pf.transition", T.Transition.__call__)
    T.AttentionPairBias.__call__ = _sub("pf.attnpairbias", T.AttentionPairBias.__call__)


def _sub(name, fn):
    """Override the stage label unconditionally (finer grain wins over the outer stage)."""
    def wrapper(*a, **k):
        prev = STAGE[0]
        STAGE[0] = name
        try:
            return fn(*a, **k)
        finally:
            STAGE[0] = prev
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    install_op_patches()
    install_stage_patches()

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    try:
        # repeat=0: exactly ONE fold (the cold one). Cold and warm run identical
        # arithmetic, and measure() only trips over the empty warm list afterwards.
        tb.measure(args.model, 0, msa_dir, Path("/tmp/_flopfold_side.json"),
                   args.target, args.msa_a3m, "census")
    except IndexError:
        pass

    rows = []
    for (st, op, kind), (n, f, di, do, li, lo) in TALLY.items():
        rows.append({"stage": st, "op": op, "kind": kind, "n": n, "flops": f,
                     "dram_in": di, "dram_out": do, "l1_in": li, "l1_out": lo})
    by_stage = OrderedDict()
    for r in rows:
        e = by_stage.setdefault(r["stage"], {"n": 0, "flops": 0, "dram_in": 0, "dram_out": 0})
        e["n"] += r["n"]
        e["flops"] += r["flops"]
        e["dram_in"] += r["dram_in"]
        e["dram_out"] += r["dram_out"]
    shp = sorted(({"stage": k[0], "op": k[1], "a": k[2], "b": k[3], "n": v[0], "flops": v[1]}
                  for k, v in SHAPES.items()), key=lambda r: -r["flops"])[:60]
    out = {"model": args.model, "target": str(args.target),
           "total_flops": sum(r["flops"] for r in rows),
           "total_dram_in": sum(r["dram_in"] for r in rows),
           "total_dram_out": sum(r["dram_out"] for r in rows),
           "by_stage": by_stage, "rows": rows, "top_matmul_shapes": shp}
    args.out.write_text(json.dumps(out, indent=1))

    print("\n=== FOLD CENSUS (one fold; instrumented, do not quote durations) ===", flush=True)
    tf = out["total_flops"]
    for st, e in sorted(by_stage.items(), key=lambda kv: -kv[1]["flops"]):
        print(f"  {st:20s} calls={e['n']:8d}  FLOPs={e['flops']/1e12:9.3f} T ({100*e['flops']/tf:5.1f}%)"
              f"  DRAM r={e['dram_in']/1e9:8.2f} GB w={e['dram_out']/1e9:8.2f} GB", flush=True)
    print(f"  TOTAL                          FLOPs={tf/1e12:9.3f} T   "
          f"DRAM r={out['total_dram_in']/1e9:.2f} GB w={out['total_dram_out']/1e9:.2f} GB", flush=True)
    print("wrote", args.out, flush=True)


main()
