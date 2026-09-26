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
                blob = {"arm": arm + args.tag, "stack": stack_name, "n": n, "K": 1, "ckpt": True,
                        "fwd": fwd, "bwd": bwd, "nested": {"fwd": fn_, "bwd": bn_}}
                OUT.mkdir(parents=True, exist_ok=True)
                p = OUT / f"trace_{arm}{args.tag}_{stack_name}_n{n}.json"
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
        if cls == "other" and o["op"].split(".")[-1] == "generic_op" \
                and any("reblock_permute.py" in x for x in (o.get("stack") or [])):
            cls = "layout"                 # a reblock move is a layout kernel, not an unknown one
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


# ------------------------------------------------------------------------------ join


def _csv_shape(r, pre):
    out = []
    for ax in "WZYX":
        v = (r.get(f"{pre}_{ax}_PAD[LOGICAL]") or "").split("[")[0].strip()
        if not v:
            return None
        out.append(int(v))
    return out


def cmd_join(args):
    """Device ms per avoidable reason: each traced kernel matched, in order, to the next ops-report
    row of the same backward whose first input has the same padded shape (right-aligned to 4-D).
    The report is `bcx-realcensus`'s, arm `stack`, so only the `stack` traces can be joined."""
    import csv
    rows = list(csv.DictReader(open(args.report)))
    seg, segs = None, collections.defaultdict(list)
    for r in rows:
        if (r.get("OP TYPE") or "").lower() == "signpost":
            seg = r["OP CODE"]
            continue
        if seg and r.get("DEVICE KERNEL DURATION [ns]"):
            segs[seg].append(r)
    out = {}
    for stack_name in ("evo", "extra"):
        t = json.loads((OUT / f"trace_{args.trace_arm}_{stack_name}_n256.json").read_text())
        cr = segs[args.seg.format(stack=stack_name)]
        cen = census(t["bwd"])
        j, matched, ms = 0, 0, collections.defaultdict(float)
        per_row = []
        for r, o in zip(cen, t["bwd"]):
            if r["view"] or r["cls"] in ("alloc", "free") or not o["ins"]:
                continue
            want = ([1] * 4 + o["ins"][0]["padded"])[-4:]
            for k in range(j, min(j + 12, len(cr))):
                if _csv_shape(cr[k], "INPUT_0") == want:
                    d = float(cr[k]["DEVICE KERNEL DURATION [ns]"]) * 1e-6
                    matched += 1
                    j = k + 1
                    frac = {key: (r[key] / r["moved"] if r["moved"] else 0.0)
                            for key in ("typecast", "layout", "padding", "fusible")}
                    for key, f in frac.items():
                        ms[key] += d * min(f, 1.0)
                    ms["all"] += d
                    ms["cls:" + r["cls"]] += d
                    if r["why"]:
                        ms["why:" + r["why"]] += d
                    per_row.append([r["i"], r["op"], cr[k]["OP CODE"], d, r["nodeop"]])
                    site = r["node"] or r["model"] or "tape:" + str(r["tape"])
                    ms["site:" + site.split(":")[0] + ":" + site.rsplit(":", 1)[-1]] += d
                    break
        total = sum(float(x["DEVICE KERNEL DURATION [ns]"]) for x in cr) * 1e-6
        out[stack_name] = {"csv_rows": len(cr), "csv_ms": total, "matched": matched,
                           "matched_ms": ms["all"], "ms": dict(ms),
                           "top_nodes": sorted(per_row, key=lambda x: -x[3])[:60]}
        print(f"{stack_name}: matched {matched}/{len(cr)} rows, {ms['all']:.1f}/{total:.1f} ms; "
              + ", ".join(f"{k} {v:.1f}" for k, v in sorted(ms.items(), key=lambda x: -x[1])), flush=True)
    (OUT / args.out).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------ arms


class Arms:
    """This row's levers as switches, installed BEFORE any taped call (the shim caches verbs).

      perm  a permute's backward goes through the reblock kernels when its inverse is one of
            their two index moves: bit-exact against `ttnn.permute`, and what the generic tape
            backward never reached. `bcx-bwbytes` moved this into the engine as
            `taped_ttnn.PERMUTE_BW_REBLOCK`, so the arm is now that flag and not a second copy
      tree  a leading-axis sum served by `autograd._pairwise_sum0` instead of `ttnn.sum(dim=0)`,
            above `--tree-rows` rows (also `bcx-bwbytes`)
      smbf16  the softmax backward on bf16 operands with fp32 accumulation in DST instead of
            fp32 tensors in DRAM -- 25.7 % of an Evoformer block's backward bytes live in that
            one expression. A PRECISION arm
      fanin the fan-in add taking a bf16 contribution directly, fp32 accumulator unchanged.
            A PRECISION arm; grade it WITH smbf16, not beside it
      moreh the softmax backward through `ttnn.moreh_softmax_backward` with the renorm kept by
            `dx = S * moreh(y/S, g)`: 8 passes of the score tensor against the chain's 10, and
            zero build, since the op ships in the wheel `pyproject.toml` already pins
      chunk the shipped taped triangle-attention blocking: the L1 plan's rows, which
            `tenstorrent.py` no longer applies under a tape (`shard_for` refuses the shard there),
            pinned back through `_FP32_SOFTMAX_DRAM_ROW_CAP`, which does apply
      bf16res  every trunk residual as the plain bf16 `ttnn.add_` instead of the fp32 round trip
            `AF2PairBlock.rne_residual` takes for torch-bf16 bit parity: a precision arm
    An arm is `+`-joined switches on top of the lever arm `--arm` (e.g. `base`, `perm+tri`).
    """

    def __init__(self):
        import ttnn
        from tt_bio import ops, taped_ttnn as T, tenstorrent as tn
        self.perm, self.chunk = False, False
        self.served = collections.Counter()
        arms = self

        from tt_bio import autograd as _ag, reblock_permute as R

        # The permute arm is the ENGINE's path, counted here rather than reimplemented: a second
        # copy of a lever is a second thing to keep in step with the one that ships.
        self.ag, self.tree_rows = _ag, 256
        for mod, name, tag in ((R, "reblock_permute_back", "perm:back"),
                               (R, "reblock_permute", "perm:fwd"),
                               (_ag, "_pairwise_sum0", "tree")):
            real_fn = getattr(mod, name)

            def counted(*a, _f=real_fn, _t=tag, **k):
                arms.served[_t] += 1
                return _f(*a, **k)
            setattr(mod, name, counted)
        real = tn._fp32_softmax_l1_plan
        self.tn, self.pinned = tn, set()

        def plan(per_row, height_per_row, width, *a, **k):
            r = real(per_row, height_per_row, width, *a, **k)
            if ops.taping() and r[0]:
                arms.served[f"tri:{'chunk' if arms.chunk else 'whole'}:plan={r[0]}"] += 1
                if arms.chunk:
                    tn._FP32_SOFTMAX_DRAM_ROW_CAP[(height_per_row, width)] = r[0]
                    arms.pinned.add((height_per_row, width))
            return r

        tn._fp32_softmax_l1_plan = plan

    def set(self, name):
        from tt_bio import taped_ttnn as T
        sw = set(name.split("+")) - {"base"}
        self.perm = "perm" in sw
        self.chunk = "chunk" in sw
        T.PERMUTE_BW_REBLOCK = self.perm
        self.ag.LEADING_SUM_TREE_ROWS = self.tree_rows if "tree" in sw else 1 << 30
        self.ag.SOFTMAX_BW_DTYPE = "bf16" if "smbf16" in sw else "keep"
        self.ag.SOFTMAX_BW_ROUTE = "moreh" if "moreh" in sw else "chain"
        self.ag.FANIN_WIDEN_INCOMING = "fanin" not in sw
        from tt_bio import af2
        af2.AF2PairBlock.rne_residual = "bf16res" not in sw
        for key in self.pinned:
            self.tn._FP32_SOFTMAX_DRAM_ROW_CAP.pop(key, None)
        self.pinned.clear()


def _run_block(dev, lv, m0, z0, wm, wz, stack_name, sign=None):
    """One checkpointed taped forward and backward of one block; signposts around the backward."""
    ag = dev.ag
    ke, kv = (1, 0) if stack_name == "extra" else (0, 1)
    gc.collect()
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
    dev.sync()
    t1 = time.time()
    roots = [zo] if stack_name == "extra" else [mo, zo]
    seeds = [dev.seed(wz, zo)] if stack_name == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
    dev.sync()
    if sign:
        sign(f"bwd {sign.tag}")
    t2, c2 = time.time(), time.process_time()
    ag.backward(roots, seeds)
    dev.sync()
    t3, c3 = time.time(), time.process_time()
    if sign:
        sign(f"end {sign.tag}")
    g = (dev.grad(ml, m0.shape), dev.grad(zl, z0.shape))
    ag.release_pins()
    del mo, zo, ml, zl, roots, seeds
    return {"fwd": t1 - t0, "bwd": t3 - t2, "bwd_cpu": c3 - c2, "spans": [(t0, t1), (t2, t3)]}, g


def cmd_arms(args):
    """Arms interleaved step by step in one process: walls, process CPU, AICLK inside the windows,
    and each arm's block gradients against the first arm's, bit for bit. With `--prof` (run under
    `python -m tracy` on the Tracy build) the backwards are signposted for `psum`."""
    arms_ = Arms()
    from perf.bcx_stack import stack as S
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    clock = S.Clock()
    names = args.arms.split(",")
    sign = None
    if args.prof:
        import ttnn
        from perf.bcx_realcensus import realcensus as RC
        sign = RC.signpost
    blob = {"stamp": S.stamp(args, clock), "lever_arm": args.arm, "arms": names, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = S.inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            res, spans, grads = ({a: [] for a in names} for _ in range(3))
            for a in names:
                arms_.set(a)
                for _ in range(args.warm):
                    _run_block(dev, lv, m0, z0, wm, wz, stack_name)
            if args.prof:
                ttnn.ReadDeviceProfiler(dev.device)
            for step in range(args.steps):
                order = names[step % len(names):] + names[:step % len(names)]
                for a in order:
                    arms_.set(a)
                    if sign:
                        def sg(msg, _tag=f"{a} {stack_name} n={n} rep={step}"):
                            RC.signpost(msg)
                        sg.tag = f"{a} {stack_name} n={n} rep={step}"
                    t, g = _run_block(dev, lv, m0, z0, wm, wz, stack_name, sg if sign else None)
                    spans[a] += t.pop("spans")
                    res[a].append(t)
                    if not grads[a]:
                        grads[a] = [x.float() for x in g]
                if args.prof:
                    ttnn.ReadDeviceProfiler(dev.device)
            base = grads[names[0]]
            pt = {"n": n, "stack": stack_name, "load1": os.getloadavg()[0], "arms": {}}
            if args.f64:
                pt["f64"] = f64_grade(ref, m0, z0, wm, wz, stack_name, grads)
                print(f"n={n} {stack_name} vs float64: {pt['f64']}", flush=True)
            for a in names:
                d = {k: S.dist([r[k] for r in res[a]]) for k in ("fwd", "bwd", "bwd_cpu")}
                d["aiclk"] = clock.window(spans[a])
                d["bits_vs_" + names[0]] = [
                    {"identical": bool(torch_equal(x, y)), "rel_l2": rel(x, y)}
                    for x, y in zip(grads[a], base)]
                pt["arms"][a] = d
                print(f"n={n} {stack_name} {a}: bwd {d['bwd']['median']:.4f} s, "
                      f"cpu {d['bwd_cpu']['median']:.4f}, fwd {d['fwd']['median']:.4f}, "
                      f"aiclk {d['aiclk']}, bits {d['bits_vs_' + names[0]]}", flush=True)
            blob["points"].append(pt)
            blob["served"] = dict(arms_.served)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    clock.stop()
    print("served", dict(arms_.served), flush=True)


def f64_grade(ref, m0, z0, wm, wz, stack_name, grads):
    """Rel L2 of each arm's block input gradients against a float64 VJP of the same block on the
    same bf16-rounded inputs and cotangents, with torch bf16 and fp32 on the reference beside it."""
    from perf.bcx_afgrad import afgrad as A
    import torch
    m, z = A.bf(m0), A.bf(z0)
    while m.dim() > 3:
        m = m.squeeze(0)
    gm, gz = A.bf(wm).reshape(m.shape), A.bf(wz).reshape(z.shape)

    def vjp(arm):
        mod = ref[arm]
        dt = mod.trunk_dtype
        if stack_name == "extra":
            g, _ = A.ref_vjp(lambda a: A.ref_extra(mod, 0, a), [z.to(dt)], [gz])
            return [None, g[0].double()]
        g, _ = A.ref_vjp(lambda a, b: A.ref_evo(mod, 0, a, b), [m.to(dt), z.to(dt)], [gm, gz])
        return [g[0].double(), g[1].double()]

    with torch.enable_grad():
        r = {arm: vjp(arm) for arm in ("f64", "f32", "bf16")}
    out = {}
    names = ("dm", "dz")

    def d(x, y):
        return float((x.double().reshape(y.shape) - y).norm() / y.norm())

    for arm in ("bf16", "f32"):
        out["torch " + arm] = {k: d(x, y) for k, x, y in zip(names, r[arm], r["f64"]) if y is not None}
    for a, g in grads.items():
        out[a] = {k: d(x, y) for k, x, y in zip(names, g, r["f64"]) if y is not None}
    return out


def torch_equal(x, y):
    return bool((x == y).all()) if x.shape == y.shape else False


def rel(x, y):
    d = float(y.norm())
    return float((x - y).norm()) / d if d else float((x - y).norm())


def cmd_psum(args):
    """Device kernel ms per arm from a `--prof` run's ops report: per backward, by class, and the
    op signatures that moved most against the first arm."""
    import csv
    from perf.bcx_realcensus import realcensus as RC
    rep = args.report
    seg, segs = None, collections.defaultdict(list)
    for r in csv.DictReader(open(rep)):
        if (r.get("OP TYPE") or "").lower() == "signpost":
            seg = r["OP CODE"][4:] if r["OP CODE"].startswith("bwd ") else None
            continue
        if seg is not None and r.get("DEVICE KERNEL DURATION [ns]"):
            segs[seg].append(r)
    by = collections.defaultdict(list)
    for tag, rs in segs.items():
        arm, rest = tag.split(" ", 1)
        key = (arm, rest.rsplit(" ", 1)[0])
        sig = collections.Counter()
        cls = collections.Counter()
        for r in rs:
            d = float(r["DEVICE KERNEL DURATION [ns]"]) * 1e-6
            sig[f"{r['OP CODE']} {RC._dims(r, 'INPUT_0')} cores={r.get('CORE COUNT')}"] += d
            cls[RC.op_class(r["OP CODE"])] += d
        by[key].append({"ms": sum(sig.values()), "ops": len(rs), "sig": sig, "cls": cls})
    names = args.arms.split(",")
    out = {"report": rep, "points": {}}
    for (arm, where), reps in sorted(by.items()):
        b = by.get((names[0], where))
        pt = {"ms": [round(x["ms"], 3) for x in reps], "ops": [x["ops"] for x in reps],
              "median_ms": float(np.median([x["ms"] for x in reps])),
              "by_class_ms": {c: round(float(np.median([x["cls"][c] for x in reps])), 3)
                              for c in sorted({c for x in reps for c in x["cls"]})}}
        if b and arm != names[0]:
            bs, as_ = b[0]["sig"], reps[0]["sig"]
            pt["moved_vs_" + names[0]] = [[round(bs[s] - as_[s], 3), s] for s in
                                          sorted(set(bs) | set(as_), key=lambda s: -(abs(bs[s] - as_[s])))[:args.top]]
        out["points"][f"{arm} {where}"] = pt
        print(f"{arm} {where}: {pt['median_ms']:.2f} ms {pt['ms']} ops {pt['ops']}", flush=True)
    (OUT / args.out).write_text(json.dumps(out, indent=1))


def cmd_whole(args):
    """`stack.py whole` with this row's switches: `--arms bwd+chunk,bwd` runs the lever arm `bwd`
    with the shipped taped blocking pinned, then without it, reps interleaved in one process."""
    arms_ = Arms()
    from perf.bcx_stack import stack as S
    real = S.open_all

    def open_all(a):
        lv, dev, ref = real(a)
        lever = lv.arm

        def arm(name):
            parts = name.split("+")
            mine = [p for p in parts if p in ("chunk", "perm", "bf16res")]
            lever("+".join(p for p in parts if p not in mine))
            arms_.set("+".join(mine) or "base")

        lv.arm = arm
        return lv, dev, ref

    S.open_all = open_all
    S.cmd_whole(args)
    print("served", dict(arms_.served), flush=True)


def set_levers(args):
    from tt_bio import autograd as ag, taped_ttnn as T
    on = args.levers == "on"
    T.PERMUTE_BW_REBLOCK = on
    ag.LEADING_SUM_TREE_ROWS = args.tree_rows if on else 1 << 30
    ag.SOFTMAX_BW_DTYPE = "bf16" if on and args.precision else "keep"
    ag.SOFTMAX_BW_ROUTE = "moreh" if on and args.moreh else "chain"
    ag.FANIN_WIDEN_INCOMING = not (on and args.precision)
    print(f"levers {args.levers} precision={args.precision}: "
          f"PERMUTE_BW_REBLOCK={T.PERMUTE_BW_REBLOCK} "
          f"LEADING_SUM_TREE_ROWS={ag.LEADING_SUM_TREE_ROWS} "
          f"SOFTMAX_BW_DTYPE={ag.SOFTMAX_BW_DTYPE} "
          f"SOFTMAX_BW_ROUTE={ag.SOFTMAX_BW_ROUTE} "
          f"FANIN_WIDEN_INCOMING={ag.FANIN_WIDEN_INCOMING}", flush=True)


def main():
    from perf.bcx_afgrad import afgrad as A
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "census", "chunk", "join", "arms", "psum", "whole"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arms", default="stack,bwd",
                    help="trace: lever arms; arms/psum: this row's arms, e.g. chunk,base,perm")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--ns", default="256,128")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="bwd", help="chunk: the lever arm every chunk arm runs on")
    ap.add_argument("--chunks", default="ship,whole,128,64,32")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--n", type=int, default=256, help="whole: tokens")
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="", help="trace: suffix on the arm name, for the same lever "
                    "arm on a different tt_bio tree (e.g. -fix)")
    ap.add_argument("--trace-arm", default="stack", help="join: which trace_<arm>_* to join")
    ap.add_argument("--seg", default="bwd {stack} K=1 rep=0", help="join: report segment")
    ap.add_argument("--f64", action="store_true", help="arms: grade each arm against float64")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--report", default="/dev/shm/bcx-rc-out/ops_perf_results_n256.csv")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--levers", default="off", choices=["off", "on"],
                    help="bcx-bwbytes' levers; `off` reproduces the tree this branch forked "
                         "from. Four of the five are already on main under other names -- only "
                         "the bf16 softmax backward is new (perf/bcx_bwbytes/DISPOSITION.md)")
    ap.add_argument("--tree-rows", type=int, default=256)
    ap.add_argument("--moreh", action="store_true",
                    help="with --levers on, route the softmax backward through the wheel's "
                         "moreh_softmax_backward, renorm kept by rescaling")
    ap.add_argument("--precision", action="store_true",
                    help="with --levers on, also take the two PRECISION levers (the bf16 softmax "
                         "backward and the bf16 fan-in). Off by default: they move the gradient "
                         "and are graded as a stack before they are counted as a byte win")
    args = ap.parse_args()
    if args.cmd not in ("census", "join", "psum"):
        set_levers(args)
    {"trace": cmd_trace, "census": cmd_census, "chunk": cmd_chunk, "join": cmd_join, "arms": cmd_arms, "psum": cmd_psum, "whole": cmd_whole}[args.cmd](args)


if __name__ == "__main__":
    main()
