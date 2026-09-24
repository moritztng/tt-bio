#!/usr/bin/env python3
"""bcx-bytes: how many bytes does one real AF2 block move, and how many does its arithmetic need?

`bcx-realcensus` timed every op of one checkpointed block backward at n=256; this counts what
those ops read and write. Every ttnn op call (`FastOperation.__call__`, depth 0 only, so a
composite is counted once) is recorded with its operands' padded and logical shapes, dtypes,
layouts, buffer types and DRAM addresses, plus the model-code line and the tape-code line that
issued it. The address stream gives the dataflow (each input's producer is the last op that wrote
its address), so an intermediate that only ever feeds elementwise ops is visible as one.

  trace  one device open: `--arms` x `--stacks` x `--ns`, one block, checkpointed as the step runs
         it, forward and backward traced separately, K=1. Writes trace_<arm>_<stack>_n<n>.json.
  census reduce traces to bytes moved per class, and the avoidable part of it by reason:
           typecast   the whole op; it moves no arithmetic
           layout     permute/transpose/concat/slice/copy/tilize/untilize/reshape kernels and head
                      splits; avoidable only by an operand layout the consumer reads directly
           padding    padded minus logical bytes of every non-layout operand (tile padding)
           fusible    write + reads of an intermediate produced by an eltwise op whose every
                      consumer is an eltwise op (what a fused eltwise chain would not store)
           fp32       the half of an fp32 operand's bytes a bf16 operand would not move; a
                      precision question, listed apart and never summed into "avoidable"
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "perf" / "bcx_bytes"
ITEM = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "UINT32": 4,
        "INT32": 4, "UINT16": 2, "UINT8": 1}
LAYOUT_OPS = ("permute", "transpose", "concat", "slice", "copy", "clone", "tilize", "untilize",
              "to_layout", "reshape", "nlp_", "pad", "repeat", "split", "move", "reallocate",
              "to_memory_config", "chunk", "fill")
ELTWISE_OPS = ("add", "subtract", "multiply", "mul", "div", "sub", "exp", "relu", "sigmoid",
               "neg", "where", "square", "sqrt", "rsqrt", "recip", "gelu", "silu", "mac",
               "addcmul", "abs", "clamp", "pow", "log", "tanh", "maximum", "minimum", "gt", "lt",
               "ge", "le", "eq", "ne", "logical", "zeros_like", "ones_like", "full_like")


def op_class(name):
    n = name.split(".")[-1].lower()
    if n == "typecast":
        return "typecast"
    if "matmul" in n or n == "linear":
        return "matmul"
    if "softmax" in n:
        return "softmax"
    if "norm" in n:
        return "layernorm"
    if n in ("sum", "mean", "max", "min", "var", "std", "prod", "argmax") or "reduce" in n:
        return "reduction"
    if any(n.startswith(k) or k in n for k in LAYOUT_OPS):
        return "layout"
    if any(n == k or n.startswith(k + "_") or n.startswith(k) for k in ELTWISE_OPS):
        return "eltwise"
    if n in ("deallocate", "reallocate_"):
        return "free"
    if n in ("zeros", "ones", "full", "from_torch", "to_torch", "empty", "arange"):
        return "alloc"
    return "other"


# ------------------------------------------------------------------------------ trace


def _desc(t):
    import ttnn
    try:
        mem = t.memory_config()
        buf = str(mem.buffer_type).split(".")[-1]
        addr = int(t.buffer_address()) if t.is_allocated() else None
    except Exception:
        buf, addr = "HOST", None
    return {"addr": addr, "buf": buf, "shape": [int(d) for d in t.shape],
            "padded": [int(d) for d in t.padded_shape],
            "dtype": str(t.dtype).split(".")[-1].upper(),
            "layout": "TILE" if t.layout == ttnn.TILE_LAYOUT else "RM"}


def _tensors(obj, out):
    import ttnn
    if isinstance(obj, ttnn.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out)
    return out


def _sites():
    """(model line, tt_bio stack): innermost af2.py/tenstorrent.py frame, and up to eight tt_bio
    and harness frames innermost first (the harness reproduces the `stack` arm's old add_grad)."""
    f = sys._getframe(2)
    model, st = None, []
    while f is not None and len(st) < 8:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn or "/perf/bcx_stack/" in fn:
            name = f"{pathlib.Path(fn).name}:{f.f_lineno}:{f.f_code.co_name}"
            st.append(name)
            if model is None and pathlib.Path(fn).name in ("af2.py", "tenstorrent.py"):
                model = name
        f = f.f_back
    return model, st


class Recorder:
    """Every depth-0 ttnn op while `on`, and for a backward op the forward model line whose tape
    node's closure issued it (`node`), which a backward frame stack cannot show."""

    def __init__(self, ag):
        import ttnn.decorators as D
        self.on, self.depth, self.ops, self.nested, self.node = False, 0, [], 0, None
        real = D.FastOperation.__call__
        rec = self

        def call(op, *a, **k):
            if not rec.on:
                return real(op, *a, **k)
            if rec.depth:
                rec.nested += 1
                return real(op, *a, **k)
            ins = [_desc(t) for t in _tensors((a, k), [])]
            model, st = _sites()
            rec.depth += 1
            try:
                res = real(op, *a, **k)
            finally:
                rec.depth -= 1
            rec.ops.append({"op": op.python_fully_qualified_name, "model": model, "stack": st,
                            "node": rec.node, "ins": ins,
                            "outs": [_desc(t) for t in _tensors(res, [])]})
            return res

        D.FastOperation.__call__ = call
        real_init = ag._Node.__init__

        def init(node, fn, parents, group=None):
            site, _ = _sites()

            def run(g, _fn=fn, _site=site):
                prev, rec.node = rec.node, _site
                try:
                    return _fn(g)
                finally:
                    rec.node = prev
            real_init(node, run, parents, group)

        ag._Node.__init__ = init

    def take(self):
        ops, n = self.ops, self.nested
        self.ops, self.nested = [], 0
        return ops, n


def cmd_trace(args):
    from perf.bcx_stack import stack as S
    lv, dev, ref = S.open_all(args)
    rec = Recorder(dev.ag)
    ag = dev.ag
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = S.inputs(ref, n, args.seed)
        for arm in args.arms.split(","):
            lv.arm(arm)
            for stack_name in args.stacks.split(","):
                for _ in range(args.warm):
                    S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=1, ckpt=True)
                gc.collect()
                ke, kv = (1, 0) if stack_name == "extra" else (0, 1)
                ml, zl = dev.leaf(m0), dev.leaf(z0)
                dev.sync()
                rec.on = True
                with dev.tt.tape():
                    mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
                fwd, fn_ = rec.take()
                roots = [zo] if stack_name == "extra" else [mo, zo]
                seeds = ([dev.seed(wz, zo)] if stack_name == "extra"
                         else [dev.seed(wm, mo), dev.seed(wz, zo)])
                rec.take()                                   # the seeds' uploads are not the block
                ag.backward(roots, seeds)
                dev.sync()
                bwd, bn_ = rec.take()
                rec.on = False
                ag.release_pins()
                del mo, zo, ml, zl, roots, seeds
                blob = {"arm": arm, "stack": stack_name, "n": n, "K": 1, "ckpt": True,
                        "fwd": fwd, "bwd": bwd, "nested": {"fwd": fn_, "bwd": bn_}}
                OUT.mkdir(parents=True, exist_ok=True)
                p = OUT / f"trace_{arm}_{stack_name}_n{n}.json"
                p.write_text(json.dumps(blob))
                print(f"{p.name}: fwd {len(fwd)} ops, bwd {len(bwd)} ops", flush=True)


# ------------------------------------------------------------------------------ census


def _bytes(d, logical=False):
    if d["buf"] != "DRAM":
        return 0.0
    s = d["shape"] if logical else d["padded"]
    return float(np.prod(s)) * ITEM.get(d["dtype"], 2)


def why(o):
    """Why a typecast exists, from the frames that issued it. None for every other op."""
    if o["op"].split(".")[-1] != "typecast":
        return None
    st = o.get("stack") or []
    names = [x.rsplit(":", 1)[-1] for x in st]
    if "_join_slices" in names:
        return "slice-join: mixed-dtype slice gradients widened before one concat"
    if any(n in ("add_grad", "_old_add_grad", "add_grad_") for n in names):
        return "fan-in: add_grad widens both operands to fp32 for the sum"
    if st and st[0].startswith("autograd.py") and names[0] == "backward":
        return "cast-back: backward() narrows an fp32 gradient to the value's dtype"
    if st and st[0].startswith("taped_ttnn.py") and names[0] == "bw" and o.get("node"):
        return "verb-bw: a forward typecast's gradient cast back to its source dtype"
    if o.get("model"):
        return "forward island: " + o["model"].rsplit(":", 1)[-1]
    return "other: " + (st[0] if st else "?")


def census(ops):
    """Per op: DRAM bytes moved and the avoidable part by reason. A slice reads what it writes."""
    last = {}                                    # addr -> index of the op that last wrote it
    consumers = collections.defaultdict(list)    # producer index -> consumer indices
    for i, o in enumerate(ops):
        for d in o["ins"]:
            if d["addr"] is not None and d["addr"] in last:
                consumers[last[d["addr"]]].append(i)
        for d in o["outs"]:
            if d["addr"] is not None:
                last[d["addr"]] = i
    rows = []
    for i, o in enumerate(ops):
        cls = op_class(o["op"])
        slice_ = "slice" in o["op"].lower()
        ins = o["ins"]
        if slice_ and o["outs"]:
            ins = [dict(ins[0], padded=o["outs"][0]["padded"], shape=o["outs"][0]["shape"])] + ins[1:]
        io = ins + o["outs"]
        src = {d["addr"] for d in o["ins"] if d["addr"] is not None}
        view = (bool(o["outs"]) and all(d["addr"] in src for d in o["outs"])) or cls == "free"
        moved = 0.0 if view else sum(_bytes(d) for d in io)
        pad = 0.0 if view else sum(_bytes(d) - _bytes(d, logical=True) for d in io) if cls not in ("layout", "typecast") else 0.0
        fp32 = 0.0 if view else sum(_bytes(d) / 2 for d in io if d["dtype"] == "FLOAT32") if cls not in ("typecast",) else 0.0
        fus = 0.0
        if cls == "eltwise" and o["outs"]:
            cs = consumers.get(i, [])
            if cs and all(op_class(ops[j]["op"]) == "eltwise" for j in cs):
                out = _bytes(o["outs"][0])
                fus = out * (1 + len(cs))              # its write, and each consumer's read
        rows.append({"i": i, "op": o["op"].split(".")[-1], "cls": cls, "view": view, "model": o["model"],
                     "tape": (o.get("stack") or [None])[0], "node": o.get("node"), "why": why(o),
                     "nodeop": f"{o['op'].split('.')[-1]} @ {o.get('node') or o['model']} <- {(o.get('stack') or ['?'])[0]}", "moved": moved,
                     "typecast": moved if cls == "typecast" else 0.0,
                     "layout": moved if cls == "layout" else 0.0,
                     "padding": pad, "fusible": fus, "fp32": fp32,
                     "sig": " x ".join(f"{d['dtype'][:4]}{d['padded']}" for d in ins[:2])
                            + " -> " + " ".join(f"{d['dtype'][:4]}" for d in o["outs"][:1])})
    return rows


def summarize(rows):
    def tot(key, rs=rows):
        return sum(r[key] for r in rs)
    moved = tot("moved")
    avoid = tot("typecast") + tot("layout") + tot("padding") + min(tot("fusible"), moved)
    by_cls = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        by_cls[r["cls"]][0] += 1
        by_cls[r["cls"]][1] += r["moved"]
    def group(key, field):
        g = collections.defaultdict(lambda: [0, 0.0])
        for r in rows:
            if r[field]:
                g[r[key]][0] += 1
                g[r[key]][1] += r[field]
        return sorted(([k, n, b / 1e6] for k, (n, b) in g.items()), key=lambda x: -x[2])
    return {"ops": len(rows), "kernels": sum(not r["view"] for r in rows), "moved_MB": moved / 1e6,
            "kernels_by_op": dict(collections.Counter(r["op"] for r in rows if not r["view"]).most_common()),
            "typecast_MB": tot("typecast") / 1e6, "layout_MB": tot("layout") / 1e6,
            "padding_MB": tot("padding") / 1e6, "fusible_MB": tot("fusible") / 1e6,
            "fp32_excess_MB": tot("fp32") / 1e6, "avoidable_MB": avoid / 1e6,
            "by_class": {k: [n, b / 1e6] for k, (n, b) in sorted(by_cls.items(), key=lambda x: -x[1][1])},
            "typecast_by_why": group("why", "typecast")[:30],
            "typecast_by_node": group("node", "typecast")[:30],
            "typecast_by_site": group("tape", "typecast")[:30],
            "typecast_by_model": group("model", "typecast")[:30],
            "layout_by_sig": group("sig", "layout")[:30],
            "layout_by_node": group("nodeop", "layout")[:40],
            "layout_by_model": group("model", "layout")[:30],
            "padding_by_sig": group("sig", "padding")[:20],
            "fusible_by_model": group("model", "fusible")[:20]}


def cmd_census(args):
    out = {}
    for p in sorted(OUT.glob("trace_*.json")):
        t = json.loads(p.read_text())
        key = f"{t['arm']} {t['stack']} n={t['n']}"
        out[key] = {ph: summarize(census(t[ph])) for ph in ("fwd", "bwd")}
        s = out[key]["bwd"]
        print(f"{key} bwd: {s['ops']} ops, moved {s['moved_MB']:.0f} MB, typecast "
              f"{s['typecast_MB']:.0f}, layout {s['layout_MB']:.0f}, padding {s['padding_MB']:.0f}, "
              f"fusible {s['fusible_MB']:.0f}, fp32 excess {s['fp32_excess_MB']:.0f}", flush=True)
    (OUT / "census.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------ chunk


def cmd_chunk(args):
    """Taped `_fp32_softmax_attention` block height, swept in one process on the real block.

    The block cap there comes from the L1 plan, and `shard_for` refuses the shard whenever a tape
    is open, so under the tape the cap buys no residency. An arm replaces the plan's row count
    ONLY while taping (the checkpoint's untaped forward keeps the shipped plan): `ship` is the
    shipped plan, `whole` no cap at all, a number that many rows. Arms rotate each step; each
    arm's gradients are compared with `ship`'s bit for bit.
    """
    from perf.bcx_stack import stack as S
    from tt_bio import ops, tenstorrent as tn
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    clock = S.Clock()
    real = tn._fp32_softmax_l1_plan
    ovr = {"rows": None}
    seen = collections.Counter()

    def plan(*a, **k):
        r = real(*a, **k)
        if ops.taping():
            seen[(ovr["rows"], r[0])] += 1
            if ovr["rows"] is not None:
                return (0, 0) if ovr["rows"] == 0 else (ovr["rows"], r[1])
        return r

    tn._fp32_softmax_l1_plan = plan
    arms = args.chunks.split(",")
    rows = {a: (None if a == "ship" else 0 if a == "whole" else int(a)) for a in arms}
    blob = {"stamp": S.stamp(args, clock), "arm": args.arm, "arms": arms, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = S.inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            res = {a: [] for a in arms}
            spans = {a: [] for a in arms}
            grads = {}
            for a in arms:
                ovr["rows"] = rows[a]
                for _ in range(args.warm):
                    S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=1, ckpt=True)
            for step in range(args.steps):
                order = arms[step % len(arms):] + arms[:step % len(arms)]
                for a in order:
                    ovr["rows"] = rows[a]
                    t, g = S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=1, ckpt=True)
                    spans[a] += t.pop("spans")
                    res[a].append(t)
                    if a not in grads:
                        grads[a] = [x.float() for x in g if x is not None]
            base = grads[arms[0]]
            pt = {"n": n, "stack": stack_name, "load1": os.getloadavg()[0], "arms": {}}
            for a in arms:
                d = {k: S.dist([r[k] for r in res[a]]) for k in ("fwd", "bwd", "bwd_cpu")}
                d["aiclk"] = clock.window(spans[a])
                d["bits_vs_" + arms[0]] = [
                    {"identical": bool((x == y).all()),
                     "rel_l2": float((x - y).norm() / y.norm())} for x, y in zip(grads[a], base)]
                pt["arms"][a] = d
                print(f"n={n} {stack_name} {a}: bwd {d['bwd']['median']:.4f} s, "
                      f"cpu {d['bwd_cpu']['median']:.4f}, fwd {d['fwd']['median']:.4f}, "
                      f"aiclk {d['aiclk']}, bits {d['bits_vs_' + arms[0]]}", flush=True)
            blob["points"].append(pt)
            blob["plan_calls"] = {f"override={k[0]} shipped_rows={k[1]}": v for k, v in seen.items()}
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / (args.out or "chunk.json")).write_text(json.dumps(blob, indent=1, default=str))
    blob["plan_calls"] = {f"override={k[0]} shipped_rows={k[1]}": v for k, v in seen.items()}
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.out or "chunk.json")).write_text(json.dumps(blob, indent=1, default=str))


def main():
    from perf.bcx_afgrad import afgrad as A
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "census", "chunk"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arms", default="stack,bwd")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--ns", default="256,128")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="bwd", help="chunk: the lever arm every chunk arm runs on")
    ap.add_argument("--chunks", default="ship,whole,128,64,32")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"trace": cmd_trace, "census": cmd_census, "chunk": cmd_chunk}[args.cmd](args)


if __name__ == "__main__":
    main()
