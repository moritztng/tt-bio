#!/usr/bin/env python3
"""A census of the tape's own verbs, and each verb's OWN backward error, measured inside it.

Two instruments, one install, nothing in `tt_bio/` edited.

CENSUS. `autograd._tape` is the single place every taped verb builds its node, so wrapping it
counts every verb that RUNS, keyed by the calling function's name and the output shape read off
the tensor. The backward count comes from wrapping the node's own `fn`. D196 is the defect where
a route read in source acquired a measured share and then turned out never to execute; a counter
inside the function cannot make that mistake. Nothing is inferred from a constructor.

INJECTION. For a selected node the wrapper snapshots the operands and the incoming cotangent,
runs the SHIPPED backward, snapshots what it added to each parent, and differences that against
the same vector-Jacobian product computed in float64 on the host from the SAME operands. That
difference is the verb's own injected error at that call. Two references where they differ:

  own   the device's own forward output, differentiated exactly. Isolates the backward's
        arithmetic. For every verb whose backward does not read its own output this is the
        only reference there is.
  true  the mathematically exact forward recomputed in float64 from the device's inputs, then
        differentiated. Adds the forward's own error to the reading. Only `softmax` has both,
        because only its backward reads `y`.

Selection is by SHAPE, not by name: the trunk's single track is [1, 16, N, *] (16 heads), its pair
track is [N, 4, N, *]. So "the single track's attention" is a property of the tensors that arrive,
which is what the row is allowed to assume.

The instrument changes no arithmetic. It pins the softmax input so the float64 forward can be
rebuilt at backward time, which moves memory and not numbers, and `arm.sh` proves that by
comparing an instrumented run's cotangent output against an uninstrumented one byte for byte.
"""
from __future__ import annotations

import sys

import torch
import ttnn

from tt_bio import autograd as ag
import tt_bio.taped_ttnn as tt

STATE = {
    "forward": {},          # "<verb> <out shape>" -> forward nodes taped
    "backward": {},         # "<verb> <out shape>" -> backward closures fired
    "nodes": [],            # per captured single-track node
    "selected_heads": None,
    "captured": 0,
    "capture_errors": {},
}


def _bump(d, k):
    d[k] = d.get(k, 0) + 1


def _shape(v):
    try:
        return [int(d) for d in v.shape]
    except Exception:                                            # noqa: BLE001
        return []


def _t(v):
    """A device tensor as float64 on the host, or None if it is gone."""
    try:
        return ttnn.to_torch(v).to(torch.float64)
    except Exception:                                            # noqa: BLE001
        return None


def _triple(dev, ref):
    if dev is None or ref is None:
        return None
    d = dev.reshape(-1)
    r = ref.reshape(-1)
    if d.numel() != r.numel():
        return {"shape_mismatch": [list(dev.shape), list(ref.shape)]}
    nr = float(torch.linalg.vector_norm(r))
    nd = float(torch.linalg.vector_norm(d))
    ae = float(torch.linalg.vector_norm(d - r))
    return {"abs_err": ae,
            "rel_l2": (ae / nr) if nr else None,
            "norm_ratio": (nd / nr) if nr else None,
            "cos": (float((d @ r) / (nd * nr)) if nd and nr else None),
            "ref_norm": nr, "dev_norm": nd}


# --- the float64 references ----------------------------------------------------------------

def _grad(fwd, inputs, g):
    xs = [x.clone().requires_grad_(True) for x in inputs]
    out = fwd(*xs)
    gs = torch.autograd.grad(out, xs, g, allow_unused=True)
    return list(gs)


def _ref_matmul(desc, ops, g):
    ta, tb = desc["ta"], desc["tb"]
    def fwd(A, B):
        return (A.transpose(-2, -1) if ta else A) @ (B.transpose(-2, -1) if tb else B)
    return {"own": _grad(fwd, [ops[0], ops[1]], g)}


_BIN = {
    "add": lambda A, B: A + B,
    "add_": lambda A, B: A + B,
    "subtract": lambda A, B: A - B,
    "multiply": lambda A, B: A * B,
    "multiply_": lambda A, B: A * B,
    "divide": lambda A, B: A / B,
}


def _ref_binary(desc, ops, g):
    f = _BIN[desc["op"]]
    if desc.get("scalar") is not None:
        s = desc["scalar"]
        return {"own": _grad(lambda A: f(A, torch.tensor(s, dtype=torch.float64)), [ops[0]], g)}
    return {"own": _grad(f, [ops[0], ops[1]], g)}


def _ref_softmax(desc, ops, g):
    dim = desc["dim"]
    y = desc["y"]
    out = {}
    if y is not None:
        inner = (g * y).sum(dim=dim, keepdim=True)
        if desc["renorm"]:
            inner = inner / y.sum(dim=dim, keepdim=True)
        out["own"] = [y * (g - inner)]
    x = ops[0]
    if x is not None:
        out["true"] = _grad(lambda X: torch.softmax(X, dim=dim), [x], g)
    return out


def _ref_sliced(desc, ops, g):
    out = torch.zeros(desc["shape"], dtype=torch.float64)
    idx = tuple(slice(a, b) for a, b in zip(desc["starts"], desc["ends"]))
    out[idx] = g
    return {"own": [out]}


def _ref_qkv_heads(desc, ops, g):
    B, L, H, dh, s = desc["B"], desc["L"], desc["H"], desc["dh"], desc["slot"]
    rows = g.permute(0, 2, 1, 3).reshape(B, L, 1, H * dh)
    zero = torch.zeros_like(rows)
    parts = [rows if i == s else zero for i in range(3)]
    return {"own": [torch.cat(parts, dim=2).reshape(B, 1, L, 3 * H * dh)]}


def _ref_transpose(desc, ops, g):
    return {"own": [g.transpose(desc["d0"], desc["d1"])]}


def _ref_permute(desc, ops, g):
    return {"own": [g.permute(desc["inv"])]}


def _ref_triatt(desc, ops, g):
    q, k, v = ops[0], ops[1], ops[2]
    bias = ops[3] if len(ops) > 3 else None
    sc = desc["scale"]
    def fwd(*xs):
        Q, K, V = xs[0], xs[1], xs[2]
        s = (Q @ K.transpose(-2, -1)) * sc
        if len(xs) > 3:
            s = s + xs[3]
        return torch.softmax(s, dim=-1) @ V
    ins = [q, k, v] + ([bias] if bias is not None else [])
    return {"own": _grad(fwd, ins, g)}


_REF = {"_v_matmul": _ref_matmul, "impl": _ref_binary, "_v_softmax": _ref_softmax,
        "_v_transpose": _ref_transpose, "_v_permute": _ref_permute,
        "triangle_attention": _ref_triatt, "_sliced": _ref_sliced,
        "_v_create_qkv_heads": _ref_qkv_heads}


# The binary verbs are registered as six closures over one `impl`, so the op's NAME is not in
# the frame -- `shipped` is a nanobind object with no usable `__name__`. Identify it the way the
# dispatch table itself does, by the rule functions the closure holds. Built at install time from
# `_VERBS`, so a verb added later is reported as unidentified rather than guessed.
_BINMAP = {}


def _build_binmap():
    for name in ("add", "add_", "subtract", "multiply", "multiply_", "divide"):
        f = tt._VERBS.get(name)
        if f is None or not getattr(f, "__closure__", None):
            continue
        cells = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
        _BINMAP[(id(cells.get("grad_a")), id(cells.get("grad_b")),
                 id(cells.get("out_of_place")))] = name


# --- the wrapper ----------------------------------------------------------------------------

def _describe(caller, frame, parents):
    """Everything the float64 reference needs, read out of the verb's OWN frame."""
    lv = frame.f_locals
    if caller == "_v_matmul":
        return {"verb": "matmul", "ta": bool(lv.get("ta")), "tb": bool(lv.get("tb"))}
    if caller == "impl":                                  # taped_ttnn._binary's inner
        op = _BINMAP.get((id(lv.get("grad_a")), id(lv.get("grad_b")),
                          id(lv.get("out_of_place"))))
        if op is None or op not in _BIN:
            return None
        raw_b = lv.get("raw_b")
        sc = None
        if not isinstance(raw_b, (ag.Tensor, ttnn.Tensor)):
            try:
                sc = float(raw_b)
            except Exception:                                     # noqa: BLE001
                return None
        return {"verb": op, "op": op, "scalar": sc}
    if caller == "_v_softmax":
        return {"verb": "softmax", "dim": int(lv.get("dim", -1)),
                "renorm": bool(ag.SOFTMAX_BW_RENORM)}
    if caller == "_v_transpose":
        return {"verb": "transpose", "d0": int(lv.get("d0", 0)), "d1": int(lv.get("d1", 1))}
    if caller == "_v_permute":
        dims = [int(d) for d in lv.get("dims", [])]
        inv = [0] * len(dims)
        for i, d in enumerate(dims):
            inv[d] = i
        return {"verb": "permute", "inv": inv}
    if caller == "triangle_attention":
        return {"verb": "triangle_attention", "scale": float(lv.get("scale", 1.0))}
    if caller == "_sliced":
        return {"verb": "sliced", "shape": [int(d) for d in lv["shape"]],
                "starts": [int(v) for v in lv["starts"]], "ends": [int(v) for v in lv["ends"]]}
    if caller == "_v_create_qkv_heads":
        slot = None
        for d in range(1, 6):
            try:
                f = sys._getframe(d)
            except ValueError:
                break
            if f.f_code.co_name == "<genexpr>" and "s" in f.f_locals:
                slot = int(f.f_locals["s"])
                break
        if slot is None:
            return None
        return {"verb": "create_qkv_heads", "B": int(lv["B"]), "L": int(lv["L"]),
                "H": int(lv["H"]), "dh": int(lv["dh"]), "slot": slot}
    return None


def install(heads: int = 16, capture: bool = True):
    """Wrap `_tape`. `heads` selects the track: 16 is the trunk's single track."""
    STATE["selected_heads"] = heads if capture else None
    _build_binmap()
    STATE["binmap"] = len(_BINMAP)
    real_tape = ag._tape
    sm_seen = [0]
    seq = [0]

    def _tape(out_value, parents, make_fn, reads=None):
        # A comprehension or a generator expression gets its own frame, so `_v_create_qkv_heads`
        # would be reported as `<genexpr>` and its locals would be the comprehension's. Walk up
        # to the first real function; it is the verb.
        frame, depth = sys._getframe(1), 1
        while frame.f_code.co_name.startswith("<") and depth < 6:
            depth += 1
            frame = sys._getframe(depth)
        caller = frame.f_code.co_name
        out = real_tape(out_value, parents, make_fn, reads=reads)
        shp = _shape(out_value)
        _bump(STATE["forward"], f"{caller} {shp}")
        if out.node is None:
            return out
        desc = None
        if capture and len(shp) == 4 and shp[0] == 1 and shp[1] == heads:
            desc = _describe(caller, frame, parents)
        if desc is not None and desc["verb"] == "softmax":
            # The float64 forward needs x, and the shipped chain deallocates it right after
            # the softmax. Pinning keeps the bytes (in DRAM) without touching any arithmetic.
            for p in parents:
                p.pinned = True
        realfn = out.node.fn
        plist = list(parents)

        def fn(g, realfn=realfn, caller=caller, shp=shp, desc=desc, plist=plist, out=out):
            _bump(STATE["backward"], f"{caller} {shp}")
            if desc is None:
                return realfn(g)
            return _run(desc, caller, shp, g, realfn, plist, out, sm_seen, seq)

        out.node.fn = fn
        return out

    ag._tape = _tape
    tt._tape = _tape
    return STATE


def _run(desc, caller, shp, g, realfn, plist, out, sm_seen, seq):
    rec = {"seq": seq[0], "verb": desc["verb"], "out_shape": shp,
           "sm_ord": sm_seen[0], "parents": []}
    seq[0] += 1
    try:
        gt = _t(g)
        ops = [_t(p.value) for p in plist]
        if desc["verb"] == "softmax":
            desc = dict(desc)
            desc["y"] = _t(out.box[0] if out.box is not None else out.value)
        before = [(_t(p.grad) if p.grad is not None else None) for p in plist]
    except Exception as e:                                        # noqa: BLE001
        _bump(STATE["capture_errors"], f"pre {desc['verb']} {type(e).__name__}")
        return realfn(g)
    realfn(g)
    if desc["verb"] == "softmax":
        sm_seen[0] += 1
    try:
        after = [(_t(p.grad) if p.grad is not None else None) for p in plist]
        refs = _REF[caller](desc, ops, gt)
        for i, p in enumerate(plist):
            if after[i] is None:
                continue
            dev = after[i] if before[i] is None else (after[i] - before[i])
            ent = {"i": i, "shape": _shape(p.value)}
            for nm, gs in refs.items():
                r = gs[i] if i < len(gs) else None
                ent[nm] = _triple(dev, r)
            rec["parents"].append(ent)
        STATE["nodes"].append(rec)
        STATE["captured"] += 1
    except Exception as e:                                        # noqa: BLE001
        _bump(STATE["capture_errors"], f"post {desc['verb']} {type(e).__name__}: {e}"[:160])
    return None


def summary():
    return {"forward": dict(sorted(STATE["forward"].items())),
            "backward": dict(sorted(STATE["backward"].items())),
            "captured_nodes": STATE["captured"],
            "capture_errors": STATE["capture_errors"],
            "selected_heads": STATE["selected_heads"],
            "binary_verbs_identified": STATE.get("binmap"),
            "nodes": STATE["nodes"]}
