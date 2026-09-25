#!/usr/bin/env python3
"""Capture a LayerNorm's OWN operands and cotangent at named sites, changing no arithmetic.

`of3t-lnaffine` located the trunk's LayerNorm-affine error in the inherited cotangent rather
than in the leaf, using three measurements on the device's own operands in float64. Two of the
three (conditioning and isolation) live in `perf/of3t_bwdaccum/dev_cot.py`, which is trunk-only:
its `--ln-capture` is keyed to pairformer blocks. This is the same idea at diffusion scope.

The shipped rule runs FIRST and unmodified -- this wraps the tape node's backward closure after
`_taped_layer_norm` has built it, so the forward is byte-identical and the gradient the model
computes is the shipped one. All this adds is a read of `x` and of the incoming cotangent `g`.

What it accumulates, per site, over every call the backward makes (48 structures x however many
times the site fires), all in float64 on the host:

    gsum  += sum over leading dims of (g * xhat)   the float64 contraction of the DEVICE's own
                                                   operands -- what the LayerNorm's affine
                                                   gradient WOULD be if its arithmetic were exact
    gabs  += sum over leading dims of |g * xhat|   the same sum without cancellation
    gcot  += sum over leading dims of g            the raw cotangent, so the cotangent itself can
                                                   be compared and not only its contraction
    per_call                                       (xhat, g) for every call, kept so the reference
                                                   operands can be substituted into our own
                                                   contraction offline instead of by another
                                                   device run. It has to be per call: `s` is the
                                                   conditioning and the conditioning carries the
                                                   noise level, so `xhat` is NOT the same matrix
                                                   on every structure at the DiT sites
                                                   (`xhat_maxdiff` 2.55). It is at the atom ones
                                                   (0.0), and the field says which.

KAPPA = ||gabs|| / ||gsum|| is `of3t-lnaffine`'s conditioning number: it bounds the relative
error a bf16-class evaluation of this contraction can reach at about 2^-9 * KAPPA. A site whose
reading exceeds that bound is not explained by cancellation in its own sum.

A zero call count on a site that was asked for is a hard failure, the same rule `host_f64.report`
applies: a probe that silently captured nothing reads exactly like a site with no error.
"""
from __future__ import annotations

import torch
import ttnn

from tt_bio import taped_ttnn as TT

NAMES: dict = {}        # id(raw ttnn weight) -> checkpoint name, installed by the harness
WANT: tuple = ()        # name suffixes to capture
ACC: dict = {}          # name -> accumulators
CALLS: dict = {}        # name -> number of backward calls seen
_ORIG = None
_ARMED = False
KEEP_PER_CALL = True


def _slot(nm, c):
    a = ACC.get(nm)
    if a is None:
        a = ACC[nm] = {"gsum": torch.zeros(c, dtype=torch.float64),
                       "gabs": torch.zeros(c, dtype=torch.float64),
                       "gcot": torch.zeros(c, dtype=torch.float64),
                       "rows": 0, "first": None, "per_call": [],
                       "G": None, "xhat": None, "xhat_maxdiff": 0.0}
    return a


def _record(nm, xv, g, eps):
    x64 = ttnn.to_torch(xv).double()
    g64 = ttnn.to_torch(g).double()
    c = x64.shape[-1]
    x64 = x64.reshape(-1, c)
    g64 = g64.reshape(-1, c)
    mu = x64.mean(dim=-1, keepdim=True)
    var = (x64 - mu).pow(2).mean(dim=-1, keepdim=True)
    xhat = (x64 - mu) * torch.rsqrt(var + eps)
    prod = g64 * xhat
    a = _slot(nm, c)
    a["gsum"] += prod.sum(dim=0)
    a["gabs"] += prod.abs().sum(dim=0)
    a["gcot"] += g64.sum(dim=0)
    a["rows"] += prod.shape[0]
    if KEEP_PER_CALL:
        a["per_call"].append((xhat.clone(), g64.clone()))
    if a["G"] is None:
        a["G"] = g64.clone()
        a["xhat"] = xhat.clone()
    elif a["G"].shape == g64.shape:
        a["G"] += g64
        a["xhat_maxdiff"] = max(a["xhat_maxdiff"],
                                float((xhat - a["xhat"]).abs().max()))
    else:
        a["G"] = None          # row count moved between calls; the factorisation does not hold
    CALLS[nm] = CALLS.get(nm, 0) + 1
    if a["first"] is None:
        # the first structure alone, so the conditioning can be read per structure as well as
        # over the whole 48-structure sum the leaf's gradient actually is
        a["first"] = {"gsum": a["gsum"].clone(), "gabs": a["gabs"].clone()}


def _rule(shipped, args, kwargs):
    out = _ORIG(shipped, args, kwargs)
    if not _ARMED:
        return out
    aa = list(args) + [None] * (3 - len(args))
    graw = aa[1] if aa[1] is not None else kwargs.get("weight")
    if graw is None:
        return out
    nm = NAMES.get(id(graw))
    if nm is None or not any(nm.endswith(s) for s in WANT):
        return out
    node = getattr(out, "node", None)
    if node is None:                       # not differentiated on this path
        return out
    xw = TT._wrap(aa[0])
    eps = float(kwargs.get("epsilon", 1e-5))
    inner = node.fn

    def bw(g, _nm=nm, _xw=xw, _eps=eps, _inner=inner):
        _record(_nm, _xw.value, g, _eps)
        return _inner(g)

    node.fn = bw
    return out


def install(names, want):
    """Arm the probe. `names` is id(raw weight) -> checkpoint name; `want` is name suffixes."""
    global _ORIG, _ARMED, WANT
    NAMES.clear()
    NAMES.update(names)
    WANT = tuple(want)
    if _ORIG is None:
        _ORIG = TT._VERBS["layer_norm"]
        TT._VERBS["layer_norm"] = _rule
        # the shim caches the wrapped verb on first use; without this the replacement is a
        # silent no-op and the arm reads exactly like the unprobed one
        TT._SHIM.__dict__.pop("layer_norm", None)
    _ARMED = True
    return sum(1 for n in names.values() if n and any(n.endswith(s) for s in WANT))


def report():
    out = {}
    for nm, a in ACC.items():
        f = a["first"]
        out[nm] = {"calls": CALLS.get(nm, 0), "rows": a["rows"],
                   "gsum": a["gsum"], "gabs": a["gabs"], "gcot": a["gcot"],
                   "G": a["G"], "xhat": a["xhat"], "xhat_maxdiff": a["xhat_maxdiff"],
                   "per_call": a["per_call"],
                   "first_gsum": None if f is None else f["gsum"],
                   "first_gabs": None if f is None else f["gabs"]}
    return out
