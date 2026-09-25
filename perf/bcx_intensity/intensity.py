#!/usr/bin/env python3
"""bcx-intensity: which roof binds each part of BindCraft 2's gradient step.

Arithmetic intensity is FLOP per byte, and a part below the machine's balance point cannot
reach the compute roof no matter how well its kernel is written. This campaign has twice
priced a bandwidth-bound part against a compute roof and lost passes to it, so every number
here is FLOP and bytes from the SHAPES THE STEP ACTUALLY RAN, recorded call by call, and
every roof is named beside the number it bounds.

What it measures, all in one device open:

  ai    one Evoformer block and one extra-MSA block, taped forward + backward, checkpointed
        exactly as `perf/bcx_stack/stack.py whole` runs the 4+48 step. Every ttnn call is
        recorded with its operand shapes, dtypes and layouts. FLOP and bytes are computed
        from those shapes, attributed to the block component that issued the call
        (`AF2PairBlock._update`'s own names) and to the phase (forward, backward, the
        checkpoint's recompute).

  time  the same blocks with a device synchronize at every component boundary, and a second
        arm that returns without syncing. The synced wall bounds the component's device time
        above; synced minus enqueue bounds it below. A component whose enqueue exceeds its
        device share is host-bound and says so.

The recorder patches the `ttnn` MODULE, before `tt_bio.taped_ttnn._Ttnn.__getattr__` caches
its `shipped` closure and before the first tape, so one patch covers both the shimmed forward
and the backward (which calls the real verbs, `taped_ttnn._NEVER_SHIM`).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import types

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "perf" / "bcx_intensity"

# ---------------------------------------------------------------- roofs, from BACKWARD.md

COMPUTE_ROOF = 115.685e12          # FLOP/s, measured dense 4096 cube, Blackhole p150a/p300c
BALANCE_LO, BALANCE_HI = 247.0, 338.0   # FLOP/byte, the measured pair
BW_ROOF_LO = COMPUTE_ROOF / BALANCE_HI  # 342.3 GB/s
BW_ROOF_HI = COMPUTE_ROOF / BALANCE_LO  # 468.4 GB/s

# ---------------------------------------------------------------- bytes and FLOP per call

ITEMSIZE = {"bfloat16": 2.0, "float32": 4.0, "bfloat8_b": 1.0625, "bfloat4_b": 0.5625,
            "uint32": 4.0, "int32": 4.0, "uint16": 2.0, "uint8": 1.0, "float64": 8.0}

# FLOP per output element, for the ops that are not matmuls. 1.0 unless the op is a
# transcendental or a multi-pass reduction; 0.0 for pure data movement, which is the whole
# point of this row -- a layout op has an arithmetic intensity of exactly zero and its
# %-of-compute-peak is not a number.
PER_ELEM = {
    "softmax": 5.0, "softmax_in_place": 5.0, "scale_mask_softmax": 6.0,
    "layer_norm": 8.0, "rms_norm": 6.0, "sigmoid": 4.0, "silu": 5.0, "gelu": 6.0,
    "exp": 4.0, "log": 4.0, "sqrt": 3.0, "rsqrt": 3.0, "tanh": 6.0, "relu": 1.0,
    "add": 1.0, "sub": 1.0, "multiply": 1.0, "mul": 1.0, "div": 1.0, "divide": 1.0,
    "add_": 1.0, "multiply_": 1.0, "sub_": 1.0,
    "mean": 1.0, "sum": 1.0, "max": 1.0, "min": 1.0,
}
# Not ops: they touch no data and produce no kernel. Charging them bytes is how a census
# invents traffic -- `is_tensor_storage_on_device` alone booked 2.18 GB of an Evoformer
# backward in this harness's first run, against a device trace that has no such kernel.
NON_OPS = {"is_tensor_storage_on_device", "allocate_tensor_on_device", "zeros", "ones", "full",
           "zeros_like", "ones_like", "empty", "get_memory_config", "is_sharded", "buffer_address",
           "is_tensor_on_device_or_multidevice", "has_storage_type_of", "dump_device_memory_state",
           "get_device", "num_cores_to_corerangeset", "create_sharded_memory_config"}

MOVE_ONLY = {"typecast", "permute", "transpose", "reshape", "to_layout", "clone", "copy",
             "unsqueeze", "squeeze", "chunk", "view",
             "concat", "pad", "slice", "narrow", "untilize", "tilize", "to_memory_config",
             "reallocate", "move", "sharded_to_interleaved", "interleaved_to_sharded",
             "experimental.view", "reshard", "nlp_concat_heads", "nlp_create_qkv_heads",
             "experimental.nlp_concat_heads", "experimental.nlp_create_qkv_heads"}
MATMULS = {"matmul", "linear", "bmm", "experimental.matmul", "moreh_matmul", "sparse_matmul"}

# Families, for the roofline table. A part is a component crossed with an op family: the
# component says where in the block it sits and the family says what kind of roof it can
# even reach.
def family(op: str) -> str:
    base = op.split(".")[-1]
    if op in MATMULS or base in ("matmul", "linear", "bmm", "moreh_matmul"):
        return "matmul"
    if base in ("softmax", "softmax_in_place", "scale_mask_softmax"):
        return "softmax"
    if base in ("layer_norm", "rms_norm"):
        return "norm"
    if base in ("sum", "mean", "max", "min", "moreh_sum", "prod"):
        return "reduce"
    if op in MOVE_ONLY or base in {o.split(".")[-1] for o in MOVE_ONLY}:
        return "layout" if base != "typecast" else "typecast"
    return "eltwise"


def _dtype_name(t) -> str:
    try:
        return str(t.dtype).rsplit(".", 1)[-1].lower()
    except Exception:
        return "bfloat16"


def _shape(t):
    try:
        return [int(d) for d in t.shape]
    except Exception:
        return None


def _padded_numel(shape, tiled: bool) -> float:
    if not shape:
        return 0.0
    s = list(shape)
    if tiled and len(s) >= 2:
        s[-1] = -(-s[-1] // 32) * 32
        s[-2] = -(-s[-2] // 32) * 32
    n = 1.0
    for d in s:
        n *= max(int(d), 0)
    return n


def _logical_numel(shape) -> float:
    n = 1.0
    for d in shape or []:
        n *= max(int(d), 0)
    return n


def _is_tensor(x, TT) -> bool:
    return isinstance(x, TT)


def _tensor_bytes(t, TT):
    """Padded bytes, which is what the DRAM controller moves, and logical bytes beside it."""
    sh = _shape(t)
    if sh is None:
        return 0.0, 0.0
    try:
        tiled = "TILE" in str(t.layout)
    except Exception:
        tiled = True
    it = ITEMSIZE.get(_dtype_name(t), 2.0)
    return _padded_numel(sh, tiled) * it, _logical_numel(sh) * it


def _matmul_flop(ins, kwargs):
    """2*B*M*N*K from the two operands' own shapes, honouring transpose_a/transpose_b."""
    if len(ins) < 2:
        return 0.0
    a, b = ins[0], ins[1]
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    a, b = list(a), list(b)
    if kwargs.get("transpose_a"):
        a[-1], a[-2] = a[-2], a[-1]
    if kwargs.get("transpose_b"):
        b[-1], b[-2] = b[-2], b[-1]
    m, k, k2, n = a[-2], a[-1], b[-2], b[-1]
    if k != k2:
        k = min(k, k2)
    batch = 1
    for i in range(max(len(a), len(b)) - 2):
        da = a[len(a) - 3 - i] if len(a) - 3 - i >= 0 else 1
        db = b[len(b) - 3 - i] if len(b) - 3 - i >= 0 else 1
        batch *= max(da, db)
    return 2.0 * batch * m * n * k


# ---------------------------------------------------------------- the recorder

class Recorder:
    """Patches the ttnn module's tensor verbs, once, before anything binds them."""

    SKIP_PREFIX = ("Device", "Mesh", "Tensor", "Shape", "CoreGrid", "CoreRange", "Config")
    SKIP = {"open_device", "close_device", "synchronize_device", "from_torch", "to_torch",
            "from_device", "to_device", "deallocate", "as_tensor", "copy_host_to_device_tensor",
            "copy_device_to_host_tensor", "dump_tensor", "load_tensor", "begin_trace_capture",
            "end_trace_capture", "execute_trace", "release_trace", "GetMemoryConfig"}

    def __init__(self, ttnn):
        self.ttnn = ttnn
        self.TT = ttnn.Tensor
        self.on = False
        self.component = "?"
        self.phase = "fwd"
        self.calls = collections.defaultdict(
            lambda: {"n": 0, "flop": 0.0, "bytes": 0.0, "lbytes": 0.0})
        self.sig = collections.defaultdict(
            lambda: {"n": 0, "flop": 0.0, "bytes": 0.0})
        self._patch(ttnn, "")
        for name in ("experimental", "operations"):
            mod = getattr(ttnn, name, None)
            if isinstance(mod, types.ModuleType):
                self._patch(mod, name + ".")

    def _patch(self, mod, prefix):
        for name in dir(mod):
            if name.startswith("_") or name in self.SKIP or name.startswith(self.SKIP_PREFIX):
                continue
            try:
                attr = getattr(mod, name)
            except Exception:
                continue
            if not callable(attr) or isinstance(attr, type) or isinstance(attr, types.ModuleType):
                continue
            try:
                setattr(mod, name, self._wrap(prefix + name, attr))
            except Exception:
                pass

    def _wrap(self, qual, fn):
        TT = self.TT

        def call(*args, **kwargs):
            if not self.on or qual.split(".")[-1] in NON_OPS:
                return fn(*args, **kwargs)
            ins = [a for a in args if isinstance(a, TT)]
            ins += [v for v in kwargs.values() if isinstance(v, TT)]
            in_shapes = [_shape(t) for t in ins]
            out = fn(*args, **kwargs)
            outs = [out] if isinstance(out, TT) else (
                [o for o in out if isinstance(o, TT)] if isinstance(out, (list, tuple)) else [])
            # A ttnn shape op can hand back the INPUT's own buffer (`autograd._tape` carries the
            # same check for the opposite reason). No kernel runs and no byte moves, so a census
            # that charges in+out here manufactures traffic: `reshape` was 4.83 GB of an
            # Evoformer backward in this harness's first run against 5.46 GB for the device
            # trace's WHOLE layout class.
            def _addr(t):
                try:
                    return t.buffer_address() if t.is_allocated() else None
                except Exception:
                    return None
            in_addrs = {a for a in (_addr(t) for t in ins) if a is not None}
            view = bool(outs) and all(_addr(o) in in_addrs for o in outs) and bool(in_addrs)
            b = lb = 0.0
            if not view:
                for t in ins + outs:
                    pb, l = _tensor_bytes(t, TT)
                    b += pb
                    lb += l
            fam = "view" if view else family(qual)
            if view:
                fl = 0.0
            elif fam == "matmul":
                fl = _matmul_flop(in_shapes, kwargs)
                if qual.split(".")[-1] == "linear" and outs:
                    fl += _padded_numel(_shape(outs[0]), True)
            elif fam in ("layout", "typecast"):
                fl = 0.0
            else:
                per = PER_ELEM.get(qual.split(".")[-1], 1.0)
                tgt = outs[0] if outs else (ins[0] if ins else None)
                if fam == "reduce" and ins:
                    tgt = ins[0]
                fl = per * _padded_numel(_shape(tgt), True) if tgt is not None else 0.0
            r = self.calls[(self.component, self.phase, fam, qual)]
            r["n"] += 1
            r["flop"] += fl
            r["bytes"] += b
            r["lbytes"] += lb
            key = (qual, tuple(tuple(s or ()) for s in in_shapes),
                   tuple(_dtype_name(t) for t in ins),
                   tuple(tuple(_shape(o) or ()) for o in outs))
            s = self.sig[key]
            s["n"] += 1
            s["flop"] += fl
            s["bytes"] += b
            return out

        return call

    def reset(self):
        self.calls.clear()
        self.sig.clear()

    def table(self):
        rows = []
        for (comp, phase, fam, qual), r in sorted(self.calls.items()):
            rows.append({"component": comp, "phase": phase, "family": fam, "op": qual,
                         **r, "ai": (r["flop"] / r["bytes"]) if r["bytes"] else 0.0})
        return sorted(rows, key=lambda r: -r["bytes"])

    def top_sigs(self, k=40):
        rows = []
        for (qual, ins, dts, outs), r in self.sig.items():
            rows.append({"op": qual, "in": [list(s) for s in ins], "dtypes": list(dts),
                         "out": [list(s) for s in outs], **r})
        return sorted(rows, key=lambda r: -r["bytes"])[:k]


# ---------------------------------------------------------------- component labelling

@contextlib.contextmanager
def labelled(rec, name):
    prev = rec.component
    rec.component = name
    try:
        yield
    finally:
        rec.component = prev


def install_labels(rec):
    """`AF2PairBlock._update` names every component of a block; `_residual` is the rest.

    The backward has no such stack -- it runs closures out of `autograd._Node` long after
    the module returned -- so each node records the component that CREATED it and restores
    it while its closure runs. A checkpoint's recompute re-enters the forward and relabels
    itself live, which is what makes the recompute column separable below.
    """
    from tt_bio import af2, autograd as ag

    up = af2.AF2PairBlock._update

    def _update(self, name, device, x, *a):
        with labelled(rec, name):
            return up(self, name, device, x, *a)
    af2.AF2PairBlock._update = _update

    res = af2.AF2PairBlock._residual

    def _residual(self, x, update):
        with labelled(rec, "residual" if rec.component == "?" else rec.component):
            return res(self, x, update)
    af2.AF2PairBlock._residual = _residual

    base = ag._Node

    class Labelled(base):
        __slots__ = ()

        def __init__(self, fn, parents, group=None):
            label = rec.component
            def wrapped(g):
                prev_c, prev_p = rec.component, rec.phase
                rec.component = label
                if rec.phase == "fwd":
                    rec.phase = "bwd"
                try:
                    return fn(g)
                finally:
                    rec.component, rec.phase = prev_c, prev_p
            super().__init__(wrapped, parents, group)
    ag._Node = Labelled


# ---------------------------------------------------------------- stamp

def sysfs_node(visible=None):
    visible = visible or os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    root = "/sys/class/tenstorrent"
    nodes = sorted(os.listdir(root),
                   key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
    node = nodes[int(visible)]
    return f"{root}/{node}", os.path.basename(os.path.realpath(f"{root}/{node}/device"))


class Clock:
    def __init__(self, dt=0.25):
        import threading
        node, self.pci = sysfs_node()
        self.path, self.dt, self.samples = f"{node}/tt_aiclk", dt, []
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), int(open(self.path).read().split()[0])))
            except Exception:
                pass
            self._stop.wait(self.dt)

    def stop(self):
        self._stop.set()

    def window(self, spans):
        xs = sorted(c for t, c in self.samples if any(a <= t <= b for a, b in spans))
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}


def stamp(args, clock=None):
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "?"
    node, pci = sysfs_node()
    return {"host": socket.gethostname(), "card": args.card, "pci": pci, "sysfs": node,
            "commit": sha, "loadavg": os.getloadavg(), "nproc": os.cpu_count(),
            "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aiclk_now": int(open(f"{node}/tt_aiclk").read().split()[0]),
            "roofs": {"compute_flops": COMPUTE_ROOF, "balance": [BALANCE_LO, BALANCE_HI],
                      "bw_gbs": [BW_ROOF_LO / 1e9, BW_ROOF_HI / 1e9]}}


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {p}", flush=True)


# ---------------------------------------------------------------- the step

def build(args, rec):
    from perf.bcx_afgrad import afgrad as A
    dm, ref = A.load_models(args.params, device_arm=True)
    dm.to_device()
    from perf.bcx_afgrad.afgrad import Dev
    dev = Dev(dm)
    return A, dev, ref


def block_pass(dev, ag, stack_name, n, seed, ckpt=True, sync_each=None):
    """One taped forward + one backward of ONE block, checkpointed as the step runs it."""
    torch.manual_seed(seed)
    from perf.bcx_afgrad import afgrad as A
    m0, z0 = A.embed(_REF["bf16"], torch.randn(n, 20), torch.arange(n))
    m0, z0 = m0.detach(), z0.detach()
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    ke, kv = (1, 0) if stack_name == "extra" else (0, 1)
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=ckpt)
    dev.sync()
    t1 = time.time()
    roots = [zo] if stack_name == "extra" else [mo, zo]
    seeds = ([dev.seed(wz, zo)] if stack_name == "extra"
             else [dev.seed(wm, mo), dev.seed(wz, zo)])
    dev.sync()
    t2 = time.time()
    ag.backward(roots, seeds)
    dev.sync()
    t3 = time.time()
    if ckpt:
        ag.release_pins()
    del mo, zo, ml, zl, roots, seeds
    gc.collect()
    return {"fwd": t1 - t0, "bwd": t3 - t2, "spans": [(t0, t1), (t2, t3)]}


_REF = None


def cmd_ai(args):
    import ttnn
    rec = Recorder(ttnn)
    global _REF
    A, dev, ref = build(args, rec)
    _REF = ref
    install_labels(rec)
    from tt_bio import autograd as ag
    clock = Clock()
    out = {"stamp": stamp(args), "n": args.n, "blocks": {}}
    for stack_name in args.stacks.split(","):
        # warm: compile every kernel, so the recorded pass is the steady-state one
        for _ in range(args.warm):
            block_pass(dev, ag, stack_name, args.n, args.seed)
        rec.reset()
        rec.on = True
        rec.phase, rec.component = "fwd", "?"
        w = block_pass(dev, ag, stack_name, args.n, args.seed)
        rec.on = False
        out["blocks"][stack_name] = {
            "wall": w, "aiclk": clock.window(w["spans"]),
            "loadavg": os.getloadavg(),
            "table": rec.table(), "top_sigs": rec.top_sigs(args.sigs)}
        print(f"{stack_name}: {len(rec.calls)} rows, "
              f"{sum(r['n'] for r in rec.calls.values())} calls, "
              f"{sum(r['flop'] for r in rec.calls.values())/1e12:.3f} TFLOP, "
              f"{sum(r['bytes'] for r in rec.calls.values())/1e9:.3f} GB", flush=True)
    clock.stop()
    save(args.out or f"ai_n{args.n}.json", out)


def cmd_time(args):
    """Component-boundary synced wall against enqueue-only wall, arms interleaved."""
    import ttnn
    rec = Recorder(ttnn)
    rec.on = False
    global _REF
    A, dev, ref = build(args, rec)
    _REF = ref
    install_labels(rec)
    from tt_bio import af2, autograd as ag

    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    mode = {"sync": True, "rep": 0}
    device = dev.device

    up = af2.AF2PairBlock._update

    def timed_update(self, name, dev_fn, x, *a):
        prev = rec.component
        rec.component = name
        t0 = time.time()
        try:
            r = up(self, name, dev_fn, x, *a)
        finally:
            if mode["sync"]:
                ttnn.synchronize_device(device)
            acc[(name, rec.phase, "sync" if mode["sync"] else "enq")][mode["rep"]] += \
                time.time() - t0
            rec.component = prev
        return r
    af2.AF2PairBlock._update = timed_update

    base = ag._Node

    class Timed(base):
        __slots__ = ()

        def __init__(self, fn, parents, group=None):
            label = rec.component

            def wrapped(g):
                prev_c, prev_p = rec.component, rec.phase
                rec.component, rec.phase = label, "bwd"
                t0 = time.time()
                try:
                    return fn(g)
                finally:
                    if mode["sync"]:
                        ttnn.synchronize_device(device)
                    acc[(label, "bwd", "sync" if mode["sync"] else "enq")][mode["rep"]] += \
                        time.time() - t0
                    rec.component, rec.phase = prev_c, prev_p
            super().__init__(wrapped, parents, group)
    ag._Node = Timed

    clock = Clock()
    spans = []
    for _ in range(args.warm):
        block_pass(dev, ag, args.stacks.split(",")[0], args.n, args.seed)
    acc.clear()
    for rep in range(args.reps):
        for stack_name in args.stacks.split(","):
            for sync in (True, False) if rep % 2 == 0 else (False, True):
                mode["sync"], mode["rep"] = sync, rep
                rec.phase = "fwd"
                w = block_pass(dev, ag, stack_name, args.n, args.seed)
                spans += w["spans"]
                key = (stack_name, "sync" if sync else "enq")
                acc[("__whole__" + stack_name, "fwd", "sync" if sync else "enq")][rep] += w["fwd"]
                acc[("__whole__" + stack_name, "bwd", "sync" if sync else "enq")][rep] += w["bwd"]
    clock.stop()
    rows = []
    for (name, phase, arm), byrep in sorted(acc.items()):
        xs = sorted(byrep.values())
        rows.append({"component": name, "phase": phase, "arm": arm, "n": len(xs),
                     "median": float(np.median(xs)), "min": float(min(xs)),
                     "max": float(max(xs)), "total": float(sum(xs))})
    save(args.out or f"time_n{args.n}.json",
         {"stamp": stamp(args), "n": args.n, "reps": args.reps,
          "aiclk": clock.window(spans), "loadavg": os.getloadavg(), "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ai", "time"])
    ap.add_argument("--params", default=os.path.expanduser(
        "~/.boltz/af2/params/params_model_1_ptm.npz"))
    ap.add_argument("--card", type=int,
                    default=int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sigs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"ai": cmd_ai, "time": cmd_time}[args.cmd](args)


if __name__ == "__main__":
    main()
