"""of3t-stackexact: every LayerNorm in the trunk step computed in float64 on the host.

The sibling of `tt_bio.autograd.exact_softmax()`, built the same way and for the same step-wide
scope, so the ladder's third rung differs from the second by this block alone:

  the VERB  `taped_ttnn._VERBS["layer_norm"]` and `autograd._TAPED["layer_norm"]`, so every
            differentiated layer norm has an exact forward AND an exact backward (dx, dgamma,
            dbeta all in float64, from x re-read off the card).
  the RAW   `ttnn.layer_norm` itself, so a frozen, `no_grad` or never-shimmed site gets the
            exact forward too, and a checkpointed block's recompute sees the same activations
            the taped forward produced.

The rounded copy goes back to the card in the dtype, layout and memory config the shipped op
would have returned. A perf-script lever, off unless the block is open, never imported by an
inference path. Counted at the call, as EXACT_SOFTMAX_STATS is.
"""
from __future__ import annotations

import contextlib

import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt

STATS = {"verb": 0, "raw": 0, "elements": 0, "bw": 0}
_KNOWN = {"weight", "bias", "epsilon", "compute_kernel_config", "memory_config", "program_config"}


def _vec(t, d):
    v = ttnn.to_torch(t).double().reshape(-1)
    assert v.numel() == d, f"layer_norm affine of {v.numel()} elements for a last dim of {d}"
    return v


def _values(v, gamma, beta, eps, memory_config=None):
    """Raw ttnn in; (x64, n64, rstd64, y64, y on the card) out."""
    x64 = ttnn.to_torch(v).double()
    d = x64.shape[-1]
    xc = x64 - x64.mean(-1, keepdim=True)
    rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)
    n64 = xc * rstd
    y64 = n64 if gamma is None else n64 * _vec(gamma, d)
    if beta is not None:
        y64 = y64 + _vec(beta, d)
    STATS["elements"] += int(x64.numel())
    y = ttnn.from_torch(y64.float(), layout=v.layout, device=v.device(), dtype=v.dtype,
                        memory_config=memory_config or v.memory_config())
    return n64, rstd, y


def _split(args, kwargs):
    args = list(args) + [None] * (3 - len(args))
    x, gamma, beta = args[0], args[1], args[2]
    kw = dict(kwargs)
    unknown = set(kw) - _KNOWN - {"l1_headroom"}
    if unknown:
        raise NotImplementedError(f"exact layer_norm does not know {sorted(unknown)}")
    gamma = kw.pop("weight", None) if kw.get("weight") is not None else gamma
    beta = kw.pop("bias", None) if kw.get("bias") is not None else beta
    return x, gamma, beta, kw.get("epsilon", 1e-5), kw.get("memory_config")


def _raw(*args, **kwargs):
    x, gamma, beta, eps, mc = _split(args, kwargs)
    STATS["raw"] += 1
    return _values(x, gamma, beta, eps, mc)[2]


def _verb(shipped, args, kwargs):
    x, gamma, beta, eps, mc = _split(args, kwargs)
    x, gamma, beta = ag._wrap(x), ag._wrap(gamma), ag._wrap(beta)
    STATS["verb"] += 1
    y = _values(x.value, None if gamma is None else gamma.value,
                None if beta is None else beta.value, eps, mc)[2]
    # n and rstd are re-derived in the backward from x, not held on the host across the tape
    parents = [t for t in (x, gamma, beta) if t is not None]

    def make():
        def bw(g):
            STATS["bw"] += 1
            g64 = ttnn.to_torch(g).double()
            x64 = ttnn.to_torch(x.value).double()
            d = x64.shape[-1]
            xc = x64 - x64.mean(-1, keepdim=True)
            rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)
            n = xc * rstd
            lead = tuple(range(g64.dim() - 1))

            def put(t64, like):
                return ttnn.from_torch(t64.float(), layout=like.layout, device=g.device(),
                                       dtype=like.dtype, memory_config=like.memory_config())

            if gamma is not None and gamma.requires_grad:
                gamma.add_grad(put((g64 * n).sum(lead).reshape(ttnn.to_torch(gamma.value).shape),
                                   gamma.value))
            if beta is not None and beta.requires_grad:
                beta.add_grad(put(g64.sum(lead).reshape(ttnn.to_torch(beta.value).shape),
                                  beta.value))
            if x.requires_grad:
                dn = g64 if gamma is None else g64 * _vec(gamma.value, d)
                dx = rstd * (dn - dn.mean(-1, keepdim=True) - n * (dn * n).mean(-1, keepdim=True))
                x.add_grad(put(dx, g))
        return bw

    return ag._tape(y, parents, make)


@contextlib.contextmanager
def exact_layer_norm():
    saved = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"], ttnn.layer_norm)
    tt._VERBS["layer_norm"] = _verb
    ag._TAPED["layer_norm"] = _verb
    ttnn.layer_norm = _raw
    tt.forget_shim_bindings("layer_norm")
    try:
        yield
    finally:
        tt._VERBS["layer_norm"], ag._TAPED["layer_norm"], ttnn.layer_norm = saved
        tt.forget_shim_bindings("layer_norm")
