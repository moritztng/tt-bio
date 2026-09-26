#!/usr/bin/env python3
"""bcx-p10-bfp8: the bfp8 arm against bf16 on the taped AF2 blocks BindCraft 2 differentiates.

The arm is `TT_BIO_TRIATT_B8` (`tenstorrent.py:1679`), the narrow region: triangle attention`s
q/k/v/gate/bias and nothing else. The residual accumulator, the stored weights and the normed
pair tensor stay bf16, and the region holds no typecast.

Two subcommands, one device open each, both arms inside ONE process:

  bytes  every ttnn verb`s operand and result bytes, per arm, on the padded shape the device
         actually holds, differenced K=2 minus K=1 so everything outside the blocks cancels.
         Typecasts are counted like any other verb: a bfp8 arm that inserts them can move MORE
         bytes than the bf16 one (`bcx-bwbytes`).
  time   fwd and bwd per block, arms alternating with the order rotating every step, AICLK read
         from the card`s own sysfs node every 0.25 s DURING each timed window, loadavg per point.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
from perf.bcx_stack import stack as S   # noqa: E402
from perf.bcx_afgrad import afgrad as A  # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_bfp8"

#: bytes per element on the card. bfp8_b is a 16-element block sharing one exponent byte,
#: so 16 mantissa bytes + 1 exponent byte over 16 elements.
ELEM = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 17.0 / 16.0, "BFLOAT4_B": 9.0 / 16.0,
        "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0}


def set_arm(name):
    """`bf16` or `b8`: the narrow triangle-attention region, flipped between taped calls."""
    from tt_bio import tenstorrent as tn
    tn._TRIATT_B8 = (name == "b8")


# ------------------------------------------------------------------------------ bytes


VERBS = [
    "matmul", "linear", "add", "add_", "subtract", "multiply", "multiply_", "div",
    "layer_norm", "softmax", "sigmoid", "relu", "typecast", "clone", "permute", "transpose",
    "concat", "reshape", "to_layout", "slice", "pad", "sum", "max", "mul", "neg", "exp",
    "reciprocal", "rsqrt", "sqrt", "bcast", "repeat", "repeat_interleave", "embedding",
    "experimental.minimal_matmul", "transformer.scaled_dot_product_attention",
    "transformer.concatenate_heads", "transformer.split_query_key_value_and_split_heads",
    "experimental.nlp_concat_heads", "experimental.nlp_create_qkv_heads",
]


def _nbytes(t):
    try:
        dt = str(t.dtype).rsplit(".", 1)[-1].upper()
    except Exception:
        return 0.0
    per = ELEM.get(dt)
    if per is None:
        return 0.0
    try:
        shape = list(t.padded_shape)
    except Exception:
        try:
            shape = list(t.shape)
        except Exception:
            return 0.0
    n = 1
    for d in shape:
        n *= int(d)
    return n * per


class ByteCounter:
    """Sum operand and result bytes of every ttnn verb, by verb and by dtype.

    Installed on the REAL ttnn module before any taped call, so `taped_ttnn``s shim reaches
    these wrappers as its `shipped` verb rather than around them.
    """

    def __init__(self, ttnn):
        self.ttnn = ttnn
        self.read = collections.Counter()
        self.written = collections.Counter()
        self.read_dtype = collections.Counter()
        self.written_dtype = collections.Counter()
        # (phase, verb, dtype) -> bytes read. The cross-tab is what splits the backward's
        # float32 by what CONSUMES it: a matmul operand is free to narrow because ttnn
        # truncates both operands to bfloat16 whatever dtype the tensor carries, while an
        # eltwise accumulation is a real accuracy question.
        self.read_vd = collections.Counter()
        self.written_vd = collections.Counter()
        self.calls = collections.Counter()
        self.phase = "fwd"
        self._saved = []

    def _dt(self, t):
        return str(t.dtype).rsplit(".", 1)[-1].upper()

    def _walk(self, obj, out):
        T = self.ttnn.Tensor
        if isinstance(obj, T):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self._walk(o, out)
        elif isinstance(obj, dict):
            for o in obj.values():
                self._walk(o, out)

    def install(self):
        for path in VERBS:
            parent, _, leaf = path.rpartition(".")
            mod = self.ttnn
            ok = True
            for part in parent.split(".") if parent else []:
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if not ok or not hasattr(mod, leaf):
                continue
            real = getattr(mod, leaf)
            if not callable(real):
                continue
            self._saved.append((mod, leaf, real))
            setattr(mod, leaf, self._wrap(path, real))
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()
        return self

    def _wrap(self, path, real):
        def w(*args, **kwargs):
            ins = []
            self._walk(args, ins)
            self._walk(kwargs, ins)
            out = real(*args, **kwargs)
            outs = []
            self._walk(out, outs)
            key = (self.phase, path)
            self.calls[key] += 1
            for t in ins:
                b = _nbytes(t)
                self.read[key] += b
                self.read_dtype[(self.phase, self._dt(t))] += b
                self.read_vd[(self.phase, path, self._dt(t))] += b
            for t in outs:
                b = _nbytes(t)
                self.written[key] += b
                self.written_dtype[(self.phase, self._dt(t))] += b
                self.written_vd[(self.phase, path, self._dt(t))] += b
            return out
        w.__name__ = getattr(real, "__name__", path)
        return w

    def uninstall(self):
        for mod, leaf, real in self._saved:
            setattr(mod, leaf, real)
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()

    def take(self):
        snap = {"calls": dict(self.calls), "read": dict(self.read), "written": dict(self.written),
                "read_dtype": dict(self.read_dtype), "written_dtype": dict(self.written_dtype),
                "read_vd": dict(self.read_vd), "written_vd": dict(self.written_vd)}
        self.read_vd = collections.Counter()
        self.written_vd = collections.Counter()
        self.calls = collections.Counter()
        self.read = collections.Counter()
        self.written = collections.Counter()
        self.read_dtype = collections.Counter()
        self.written_dtype = collections.Counter()
        return snap


def _fold(snap):
    """`{phase: {read, written, total, by_dtype, calls}}` in bytes."""
    out = {}
    for phase in ("fwd", "bwd"):
        r = sum(v for (p, _), v in snap["read"].items() if p == phase)
        w = sum(v for (p, _), v in snap["written"].items() if p == phase)
        out[phase] = {
            "read": r, "written": w, "total": r + w,
            "calls": sum(v for (p, _), v in snap["calls"].items() if p == phase),
            "read_by_dtype": {d: v for (p, d), v in snap["read_dtype"].items() if p == phase},
            "written_by_dtype": {d: v for (p, d), v in snap["written_dtype"].items() if p == phase},
            "read_by_verb": {k: v for (p, k), v in snap["read"].items() if p == phase},
            "read_by_verb_dtype": {f"{k}|{d}": v for (p, k, d), v in snap["read_vd"].items()
                                   if p == phase},
            "written_by_verb_dtype": {f"{k}|{d}": v for (p, k, d), v in snap["written_vd"].items()
                                      if p == phase},
        }
    out["total"] = out["fwd"]["total"] + out["bwd"]["total"]
    return out


def _diff(a, b):
    """b minus a, per phase, so everything outside the K extra blocks cancels."""
    out = {}
    for phase in ("fwd", "bwd"):
        d = {}
        for key in ("read", "written", "total", "calls"):
            d[key] = b[phase][key] - a[phase][key]
        for key in ("read_by_dtype", "written_by_dtype", "read_by_verb",
                    "read_by_verb_dtype", "written_by_verb_dtype"):
            names = set(a[phase][key]) | set(b[phase][key])
            d[key] = {n: b[phase][key].get(n, 0) - a[phase][key].get(n, 0) for n in names}
            d[key] = {n: v for n, v in d[key].items() if abs(v) > 0.5}
        out[phase] = d
    out["total"] = out["fwd"]["total"] + out["bwd"]["total"]
    return out


def cmd_bytes(args):
    import ttnn
    lv, dev, ref = S.open_all(args)
    # Levers keeps its constructed state, which is main’s shipped code with every lever
    # site switched to the path main takes. `arm()` is deliberately not called: `arm("stack")`
    # would turn bcx-bwdplan’s three backward levers OFF, which main does not.
    lv.mask = True   # the design path hands the Evoformer an MSA mask
    bc = ByteCounter(ttnn).install()
    blob = {"stamp": S.stamp(args), "n": args.n, "arms": {}, "loadavg": os.getloadavg()}
    try:
        for arm in args.arms.split(","):
            set_arm(arm)
            per = {}
            for stack_name in args.stacks.split(","):
                m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
                S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=1)   # warm, dropped
                bc.take()
                snaps = {}
                for k in (1, 2):
                    bc.phase = "fwd"
                    _run(bc, dev, lv, m0, z0, wm, wz, stack_name, k)
                    snaps[k] = _fold(bc.take())
                per[stack_name] = {"K1": snaps[1], "K2": snaps[2],
                                   "per_block": _diff(snaps[1], snaps[2])}
                pb = per[stack_name]["per_block"]
                print(json.dumps({"arm": arm, "stack": stack_name, "n": args.n,
                                  "fwd_GB": round(pb["fwd"]["total"] / 1e9, 4),
                                  "bwd_GB": round(pb["bwd"]["total"] / 1e9, 4),
                                  "total_GB": round(pb["total"] / 1e9, 4),
                                  "fwd_dtype_GB": {d: round(v / 1e9, 4) for d, v in
                                                   pb["fwd"]["read_by_dtype"].items()},
                                  "bwd_dtype_GB": {d: round(v / 1e9, 4) for d, v in
                                                   pb["bwd"]["read_by_dtype"].items()}}),
                      flush=True)
            blob["arms"][arm] = per
            _save(args.out or f"bytes_n{args.n}.json", blob)
    finally:
        bc.uninstall()
        set_arm("bf16")


def _run(bc, dev, lv, m0, z0, wm, wz, stack_name, k):
    """`S.block_step` with the counter told which phase it is in."""
    real = S.block_step
    ag = dev.ag
    import gc
    import torch
    ke, kv = (k, 0) if stack_name == "extra" else (0, k)
    gc.collect()
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    lv.phase = bc.phase = "fwd"
    mask = dev.up(torch.ones(1, m0.shape[-2])) if lv.mask else None
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=False, msa_mask=mask)
    dev.sync()
    roots = [zo] if stack_name == "extra" else [mo, zo]
    seeds = [dev.seed(wz, zo)] if stack_name == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
    dev.sync()
    lv.phase = bc.phase = "bwd"
    ag.backward(roots, seeds)
    dev.sync()
    bc.phase = "fwd"
    lv.phase = "fwd"
    del mo, zo, ml, zl, roots, seeds
    gc.collect()


# ------------------------------------------------------------------------------ time


def cmd_time(args):
    lv, dev, ref = S.open_all(args)
    # Levers keeps its constructed state, which is main’s shipped code with every lever
    # site switched to the path main takes. `arm()` is deliberately not called: `arm("stack")`
    # would turn bcx-bwdplan’s three backward levers OFF, which main does not.
    lv.mask = True   # the design path hands the Evoformer an MSA mask
    clock = S.Clock()
    arms = args.arms.split(",")
    blob = {"stamp": S.stamp(args, clock), "arms": arms, "steps": args.steps, "warm": args.warm,
            "order": "arms rotate by one position every step", "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = S.inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            for k in [int(x) for x in args.ks.split(",")]:
                rec = {a: [] for a in arms}
                t_start = time.time()
                for step in range(args.warm + args.steps):
                    order = arms[step % len(arms):] + arms[:step % len(arms)]
                    for arm in order:
                        set_arm(arm)
                        r, _ = S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k)
                        if step >= args.warm:
                            rec[arm].append(r)
                    lv.take()
                pt = {"n": n, "stack": stack_name, "K": k, "loadavg": os.getloadavg(),
                      "wall": [t_start, time.time()], "arms": {},
                      "aiclk_point": clock.window([(t_start, time.time())])}
                for arm in arms:
                    rs = rec[arm]
                    pt["arms"][arm] = {
                        "fwd": S.dist([r["fwd"] for r in rs]),
                        "bwd": S.dist([r["bwd"] for r in rs]),
                        "step": S.dist([r["fwd"] + r["bwd"] for r in rs]),
                        # Host CPU of the backward beside its wall. A block-level wall on this
                        # path is host-bound (5.57 s against a 0.23 s device proxy at n=256), so
                        # a byte lever is invisible in it; the pair says so on its face.
                        "bwd_host_cpu": S.dist([r["bwd_cpu"] for r in rs]),
                        "aiclk": clock.window([s for r in rs for s in r["spans"]])}
                base = pt["arms"][arms[0]]
                for arm in arms:
                    a = pt["arms"][arm]
                    a["x_vs_" + arms[0]] = {m: base[m]["median"] / a[m]["median"]
                                            for m in ("fwd", "bwd", "step")}
                    print(json.dumps({"n": n, "stack": stack_name, "K": k, "arm": arm,
                                      "fwd": round(a["fwd"]["median"], 4),
                                      "bwd": round(a["bwd"]["median"], 4),
                                      "bwd_p10_p90": [round(a["bwd"]["p10"], 4),
                                                      round(a["bwd"]["p90"], 4)],
                                      "bwd_cpu": round(a["bwd_host_cpu"]["median"], 4),
                                      "x": a["x_vs_" + arms[0]], "aiclk": a["aiclk"],
                                      "loadavg1": pt["loadavg"][0]}), flush=True)
                blob["points"].append(pt)
                _save(args.out or f"time_n{n}.json", blob)
    clock.stop()
    set_arm("bf16")


def _save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bytes", "time"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ns", default="256")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--arms", default="bf16,b8")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    {"bytes": cmd_bytes, "time": cmd_time}[args.cmd](args)


if __name__ == "__main__":
    main()
