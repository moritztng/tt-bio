#!/usr/bin/env python3
"""Pin a named verb's backward to its float64 VJP, inside chosen blocks, and re-score the ladder.

`of3t-tapeattn` MEASURED every verb's own injected error. That census could only rank injectors;
it could not say which one the cotangent step at blocks 45 and 44 is made of, because an
injector's size at a call says nothing about how much of it survives to the block boundary. A
SUBSTITUTION can: replace one verb's backward result with the exact one and read the ladder again.

Three things this has to get right, because each has already cost this campaign a pass.

REACH. A pin that never fires and a pin that fires and is inert are different results and no
output comparison tells them apart (D121). Every pin counts its own firings per block, and the
`identity` mode is the control that fires exactly as often and must be BIT-EXACT.

THE BLOCK. Rung k is the gradient of block k's input, so the step is created by the backwards of
blocks 45 and 44. The block index comes from wrapping `ops.checkpoint_segment` and tagging the
node the segment returns, so "inside block 45" is read off the machinery that defines a block and
not inferred from a firing ordinal.

THE WRITE-BACK. The exact gradient is computed in float64 on the host and has to go back to a
device tensor, so the pin's own floor is one float32 rounding per pinned contribution. It is
written in the parent's OWN dtype and layout -- whatever `add_grad` had just produced -- so the
storage format is untouched and only the value moves. The float32 cast is checked elementwise,
not assumed.

Selection is by SHAPE, the way `tapecensus.py` does it, because the tracks are shapes:

    sm16          the single track's softmax, `[1, 16, N, N]`
    sm4           the pair track's TriangleAttention softmax, `[N, 4, N, N]`
    pairattn      that softmax AND both of the pair track's attention matmuls
    paircontract  TriangleMultiplication's contraction, `pair_contract`'s per-channel matmul
                  over the token axis, out `[1, C, N, N]`
    identity      every node any of the above selects, round-tripped through host float64 and
                  written back UNCHANGED. The inertness control.
"""
from __future__ import annotations

import sys

import torch
import ttnn

from tt_bio import autograd as ag
import tt_bio.ops as ops
import tt_bio.taped_ttnn as tt

STATE = {
    "forward": {},
    "backward": {},
    "blocks_seen": 0,
    "pin": None,
    "pin_blocks": None,
    "fired": {},              # "<verb> block<k>" -> substitutions applied
    "selected": 0,            # nodes the predicate matched, fired or not
    "errors": {},
    "cast_lossy": 0,
    "cast_tensors": 0,
    "moved": [],              # per substitution: how far the pin moved the gradient
}
CUR = [None]


def _bump(d, k):
    d[k] = d.get(k, 0) + 1


def _shape(v):
    try:
        return [int(d) for d in v.shape]
    except Exception:                                            # noqa: BLE001
        return []


def _t(v):
    try:
        return ttnn.to_torch(v).to(torch.float64)
    except Exception:                                            # noqa: BLE001
        return None


def _rel(a, b):
    if a is None or b is None or a.numel() != b.numel():
        return None
    a = a.reshape(-1)
    b = b.reshape(-1)
    nb = float(torch.linalg.vector_norm(b))
    if nb == 0.0:
        return None
    return float(torch.linalg.vector_norm(a - b) / nb)


# --- the float64 references ------------------------------------------------------------------

def _grad(fwd, inputs, g):
    xs = [x.clone().requires_grad_(True) for x in inputs]
    out = fwd(*xs)
    return list(torch.autograd.grad(out, xs, g, allow_unused=True))


def _ref_matmul(desc, ops_, g):
    ta, tb = desc["ta"], desc["tb"]

    def fwd(A, B):
        return (A.transpose(-2, -1) if ta else A) @ (B.transpose(-2, -1) if tb else B)

    return _grad(fwd, [ops_[0], ops_[1]], g)


def _ref_softmax(desc, ops_, g):
    """The backward's OWN arithmetic, exactly: the device's own y, differentiated in float64.

    `own` and not `true`: pinning to a float64 recomputation of the FORWARD would fold the
    forward's error into the reading and the arm would no longer be about the backward.
    """
    dim = desc["dim"]
    y = desc["y"]
    if y is None:
        return None
    inner = (g * y).sum(dim=dim, keepdim=True)
    if desc["renorm"]:
        inner = inner / y.sum(dim=dim, keepdim=True)
    return [y * (g - inner)]


def _ref_triatt(desc, ops_, g, chunk=16):
    """TriangleAttention's VJP in float64, chunked over the leading axis.

    The scores are `[B, 4, N, N]`; at padded 384 that is 1.8 GB in float64 for one call, so the
    analytic form is used rather than `torch.autograd.grad`, chunked over B the same way the
    shipped forward chunks it. Chunking is exact here for the same reason it is exact on device:
    each leading slice is an independent attention problem, so no softmax denominator is split.
    """
    q, k, v = ops_[0], ops_[1], ops_[2]
    bias = ops_[3] if len(ops_) > 3 else None
    if q is None or k is None or v is None:
        return None
    sc = desc["scale"]
    B = int(q.shape[0])
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    bcast = bias is not None and int(bias.shape[0]) == 1
    dbias = torch.zeros_like(bias) if bias is not None else None
    for b0 in range(0, B, chunk):
        b1 = min(b0 + chunk, B)
        Q, K, V, G = q[b0:b1], k[b0:b1], v[b0:b1], g[b0:b1]
        s = (Q @ K.transpose(-2, -1)) * sc
        if bias is not None:
            s = s + (bias if bcast else bias[b0:b1])
        p = torch.softmax(s, dim=-1)
        dv[b0:b1] = p.transpose(-2, -1) @ G
        dp = G @ V.transpose(-2, -1)
        ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
        del s, p, dp
        dq[b0:b1] = (ds @ K) * sc
        dk[b0:b1] = (ds.transpose(-2, -1) @ Q) * sc
        if dbias is not None:
            if bcast:
                dbias += ds.sum(dim=0, keepdim=True)
            else:
                dbias[b0:b1] = ds
        del ds
    out = [dq, dk, dv]
    if bias is not None:
        out.append(dbias)
    return out


# --- what a node IS --------------------------------------------------------------------------

def _describe(caller, frame):
    lv = frame.f_locals
    if caller == "_v_matmul":
        return {"verb": "matmul", "ta": bool(lv.get("ta")), "tb": bool(lv.get("tb"))}
    if caller == "_v_softmax":
        return {"verb": "softmax", "dim": int(lv.get("dim", -1)),
                "renorm": bool(ag.SOFTMAX_BW_RENORM)}
    if caller == "triangle_attention":
        return {"verb": "triangle_attention", "scale": float(lv.get("scale", 1.0))}
    return None


_REF = {"_v_matmul": _ref_matmul, "_v_softmax": _ref_softmax,
        "triangle_attention": _ref_triatt}


def _sel_sm16(caller, shp, desc):
    """The single track's softmax, `[1, 16, N, N]`. 48 backward firings, one per block."""
    return caller == "_v_softmax" and len(shp) == 4 and shp[0] == 1 and shp[1] == 16


def _sel_sm4(caller, shp, desc):
    """The pair track's TriangleAttention softmax, `[N, 4, N, N]`, two per block.

    `autograd.triangle_attention` -- the fused node with its own chunked-recompute backward --
    NEVER RUNS on this path. The census at padded 64 finds `_v_softmax [64, 4, 64, 64]` 288
    times and `triangle_attention` zero times, so the pair track's attention is on the tape as
    ordinary verbs. A selector written against the fused node would have been silently dead,
    which is D196's shape exactly, and is why this is read off a census and not off the source.
    """
    return caller == "_v_softmax" and len(shp) == 4 and shp[1] == 4 and shp[0] > 1


def _sel_pairattn(caller, shp, desc):
    """The pair track's whole attention: its softmax and both of its matmuls."""
    if _sel_sm4(caller, shp, desc):
        return True
    return caller == "_v_matmul" and len(shp) == 4 and shp[1] == 4 and shp[0] > 1


def _sel_paircontract(caller, shp, desc):
    """`pair_contract`, TriangleMultiplication's contraction over the token axis.

    `permute -> matmul -> permute`, so the contraction is the only matmul on the tape whose
    output is `[1, C, N, N]`: a channel-sized second axis and two equal token extents. c_z is
    128, the single track's head count is 16, so the `shp[1] >= 32` clause separates them.
    """
    return (caller == "_v_matmul" and len(shp) == 4 and shp[0] == 1
            and shp[1] >= 32 and shp[2] == shp[3])


SELECT = {"sm16": _sel_sm16, "sm4": _sel_sm4, "pairattn": _sel_pairattn,
          "paircontract": _sel_paircontract}
SELECT["identity"] = lambda c, s, d: any(f(c, s, d) for f in
                                         (_sel_sm16, _sel_pairattn, _sel_paircontract))
# `all` is the same union predicate in EXACT mode: every node any selector sees, pinned to
# its float64 VJP in one arm. The saturation test the three single-pin refutations point at --
# if R44 does not move with every taped verb in blocks 45 and 44 exact, the carrier is not a
# taped verb at all and the next instrument has to be the forward activations they consume.
SELECT["all"] = SELECT["identity"]


def _sel_allref(caller, shp, desc):
    """Every node this instrument can make exact, whatever its shape: the CEILING, not the union.

    `all` is the union of the four shape selectors and it reaches 816 of the 11856 backward
    firings in one taped backward, 6.88 %. That is NOT "every taped verb in the block", which is
    what a saturation test has to be, and the gap is not harmless: 480 firings per backward have
    an exact float64 VJP available and no selector matches them, among them BOTH of the single
    track's attention matmuls (`_v_matmul [1, 16, N, N]` and `[1, 16, N, 32]`, 48 each) -- on the
    one track `R44` can see at all. `sm16` pinned that attention's softmax and left its two
    matmuls shipped.

    So this predicate selects on the CALLER instead of the shape: every `_v_matmul`, every
    `_v_softmax`, every `triangle_attention`, which is exactly the key set of `_REF`. 1296 of
    11856, 10.93 %, and there is no larger arm available without writing a new float64 VJP.
    The remaining 89.07 % are verbs this instrument has no exact reference for at all
    (`_taped_linear`, `_taped_layer_norm`, `silu`, the reshape/permute/slice family, `impl`),
    and that residue is the bound this arm reports rather than erases.
    """
    return caller in _REF


SELECT["allref"] = _sel_allref


# --- the block tag -----------------------------------------------------------------------------

def _install_blocks():
    real = ops.checkpoint_segment

    def cs(fn, *inputs):
        out = real(fn, *inputs)
        outs = out if isinstance(out, (tuple, list)) else [out]
        taped = [t for t in outs if isinstance(t, ag.Tensor) and t.node is not None]
        if not taped:
            return out
        k = STATE["blocks_seen"]
        STATE["blocks_seen"] += 1
        for t in taped:
            orig = t.node.fn

            def fn2(g, orig=orig, k=k):
                prev = CUR[0]
                CUR[0] = k
                try:
                    return orig(g)
                finally:
                    CUR[0] = prev

            t.node.fn = fn2
        return out

    ops.checkpoint_segment = cs


# --- the wrapper --------------------------------------------------------------------------------

def install(pin: str = "identity", blocks=None, chunk: int = 16):
    """`pin` names the verb to substitute; `blocks` restricts it to those block indices."""
    if pin not in SELECT:
        raise ValueError(f"unknown pin {pin!r}; have {sorted(SELECT)}")
    STATE["pin"] = pin
    STATE["pin_blocks"] = None if blocks is None else sorted(blocks)
    want = None if blocks is None else set(blocks)
    predicate = SELECT[pin]
    _install_blocks()
    real_tape = ag._tape

    def _tape(out_value, parents, make_fn, reads=None):
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
        desc = _describe(caller, frame) if predicate(caller, shp, None) else None
        if desc is not None and desc["verb"] == "softmax":
            for p in parents:
                p.pinned = True
        realfn = out.node.fn
        plist = list(parents)

        def fn(g, realfn=realfn, caller=caller, shp=shp, desc=desc, plist=plist, out=out):
            _bump(STATE["backward"], f"{caller} {shp}")
            if desc is None or (want is not None and CUR[0] not in want):
                return realfn(g)
            STATE["selected"] += 1
            return _run(desc, caller, shp, g, realfn, plist, out, pin, chunk)

        out.node.fn = fn
        return out

    ag._tape = _tape
    tt._tape = _tape
    return STATE


def _run(desc, caller, shp, g, realfn, plist, out, pin, chunk):
    blk = CUR[0]
    try:
        gt = _t(g)
        ops_ = [_t(p.value) for p in plist]
        if desc["verb"] == "softmax":
            desc = dict(desc)
            desc["y"] = _t(out.box[0] if out.box is not None else out.value)
        before = [(_t(p.grad) if p.grad is not None else None) for p in plist]
    except Exception as e:                                        # noqa: BLE001
        _bump(STATE["errors"], f"pre {desc['verb']} {type(e).__name__}")
        return realfn(g)
    realfn(g)
    try:
        after = [(_t(p.grad) if p.grad is not None else None) for p in plist]
        refs = None
        if pin != "identity":
            if desc["verb"] == "triangle_attention":
                refs = _ref_triatt(desc, ops_, gt, chunk=chunk)
            else:
                refs = _REF[caller](desc, ops_, gt)
        dev = _device()
        moved = []
        for i, p in enumerate(plist):
            if after[i] is None:
                continue
            contrib = after[i] if before[i] is None else (after[i] - before[i])
            if pin == "identity":
                new = after[i]
                mv = 0.0
            else:
                r = refs[i] if (refs is not None and i < len(refs)) else None
                if r is None:
                    continue
                r = r.reshape(contrib.shape)
                new = r if before[i] is None else (before[i] + r)
                mv = _rel(contrib, r)
            tgt = p.grad
            f32 = new.to(torch.float32)
            STATE["cast_tensors"] += 1
            if not torch.equal(f32.to(torch.float64), new):
                STATE["cast_lossy"] += 1
            p.grad = ttnn.from_torch(f32.reshape([int(d) for d in tgt.shape]),
                                     dtype=tgt.dtype, layout=tgt.layout, device=dev)
            moved.append(mv)
        _bump(STATE["fired"], f"{desc['verb']} block{blk}")
        if moved:
            STATE["moved"].append({"verb": desc["verb"], "block": blk, "rel_moved": moved})
    except Exception as e:                                        # noqa: BLE001
        _bump(STATE["errors"], f"post {desc['verb']} {type(e).__name__}: {e}"[:200])
    return None


def _device():
    from tt_bio.tenstorrent import get_device
    return get_device()


def summary():
    return {"pin": STATE["pin"], "pin_blocks": STATE["pin_blocks"],
            "blocks_seen": STATE["blocks_seen"],
            "nodes_selected": STATE["selected"],
            "substitutions": dict(sorted(STATE["fired"].items())),
            "substitutions_total": sum(STATE["fired"].values()),
            "float32_writeback_tensors": STATE["cast_tensors"],
            "float32_writeback_lossy": STATE["cast_lossy"],
            "errors": STATE["errors"],
            "moved": STATE["moved"][:400],
            "forward": dict(sorted(STATE["forward"].items())),
            "backward": dict(sorted(STATE["backward"].items()))}
