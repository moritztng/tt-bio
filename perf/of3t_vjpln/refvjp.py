#!/usr/bin/env python3
"""Float64 VJPs for `_taped_layer_norm` and `_taped_linear`, written against the SHIPPED
backward's own arithmetic.

`of3t-blk4544` pinned every node `pinvjp._REF` could make exact -- 1296 of 11856 backward
firings, 10.93 % -- and `R44` did not move. The residue it could not reach is 89.07 %, and the
two largest entries that are real arithmetic rather than data movement are `_taped_linear`
(1584 firings, 13.36 %) and `_taped_layer_norm` (816, 6.88 %). This module is those two VJPs.

WHICH FUNCTION IS BEING DIFFERENTIATED. Not the one a textbook would write. The SDPA trap is on
record in this campaign and cost a verification pass: the shipped attention adds the mask BEFORE
the scale, so a reference written the usual way agrees with a wrong gradient while both disagree
with the forward. Both functions here are read off `tt_bio/autograd.py` and reproduced term for
term:

  layer_norm (`_taped_layer_norm`, autograd.py:1805). The backward RECOMPUTES its statistics
  from `x.value` with a two-pass `E[(x - mean)^2]`, then applies the analytic form
  `dx = (dnorm - mean(dnorm) - norm * mean(dnorm * norm)) * rstd`. So the function this VJP is
  the exact gradient of is that two-pass normaliser, evaluated in float64 -- the backward's OWN
  arithmetic, exactly, the same choice `_ref_softmax` makes and for the same reason. Pinning to
  a float64 recomputation of the ttnn FORWARD instead would fold the forward's error into a
  reading that is supposed to be about the backward.

  linear (`_taped_linear`, autograd.py:1737). `out = x @ w (+ bias)`: `dx = g @ w^T` reduced to
  x's shape, `dw = flat2d(x)^T @ flat2d(g)`, `dbias = sum over every leading coordinate`. The
  fused activation is NOT part of this node -- `_taped_linear` composes it as a separate taped
  verb because `ttnn.linear` folds it into the packer and silu is not invertible -- so the
  cotangent arriving here is already pre-activation and no activation derivative belongs in it.

ROLES, NOT POSITIONS. `parents` is `[t for t in (x, w, bias) if t is not None]`, so a call with
no bias shifts every later index. The role of each parent is resolved by OBJECT IDENTITY against
the shipped frame's own locals at tape time, never by counting. A VJP that returns dW where the
consumer expects dbias is wrong in a way no shape check catches when both are rank-1.

This module is pure torch. It imports neither ttnn nor tt_bio, so `fdcheck.py` can validate it
against central finite differences on the host with no device and no tape.
"""
from __future__ import annotations

import torch


def _flat2d(t):
    """(prod(leading), last). `autograd._flat2d` without the placement move, which is not
    arithmetic: the move exists to keep ttnn's K-block partials in fp32 and float64 has no
    such regime."""
    return t.reshape(-1, int(t.shape[-1]))


def _sum_leading(t, shape):
    """`autograd._sum_leading`: sum every leading coordinate down to a trailing shape."""
    return _flat2d(t).sum(dim=0).reshape(list(shape))


def _reduce_to(g, shape):
    """`autograd._reduce_to`: same shape passes, same volume reshapes, a genuinely stretched
    leading axis is summed. The third branch does not arise for `linear` on this model -- a
    matmul does not broadcast its batch against the operand it reduces -- but it is written
    because the shipped helper writes it, and a reference that silently omits a branch the
    shipped op has is a reference for a different function."""
    gs = [int(d) for d in g.shape]
    ws = [int(d) for d in shape]
    if gs == ws:
        return g
    gv, wv = 1, 1
    for d in gs:
        gv *= d
    for d in ws:
        wv *= d
    if gv == wv:
        return g.reshape(ws)
    pad = [1] * (len(gs) - len(ws)) + ws
    out = g
    for ax in range(len(gs)):
        if pad[ax] == 1 and gs[ax] != 1:
            out = out.sum(dim=ax, keepdim=True)
    return out.reshape(ws)


# --- the forwards, exactly as the shipped backwards model them ---------------------------------

def layer_norm_forward(x, gamma, beta, eps):
    """The two-pass normaliser `_taped_layer_norm`'s backward differentiates."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    norm = centered * torch.rsqrt(var + eps)
    if gamma is not None:
        norm = norm * gamma
    if beta is not None:
        norm = norm + beta
    return norm


def linear_forward(x, w, bias):
    out = x @ w
    if bias is not None:
        out = out + bias
    return out


# --- the VJPs ----------------------------------------------------------------------------------

def layer_norm_vjp(x, gamma, beta, g, eps):
    """Returns (dx, dgamma, dbeta); any of the three may be None when the operand is absent.

    Term for term against `autograd.py:1830-1860`. `dnorm` uses `gamma is not None`, not
    `gamma.requires_grad`, because that is what the shipped branch tests."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    norm = centered * rstd

    dgamma = _sum_leading(g * norm, gamma.shape) if gamma is not None else None
    dbeta = _sum_leading(g, beta.shape) if beta is not None else None

    dnorm = (g * gamma) if gamma is not None else g
    dn_mean = dnorm.mean(dim=-1, keepdim=True)
    dn_norm_mean = (dnorm * norm).mean(dim=-1, keepdim=True)
    dx = (dnorm - dn_mean - norm * dn_norm_mean) * rstd
    return dx, dgamma, dbeta


def linear_vjp(x, w, bias, g):
    """Returns (dx, dw, dbias). `dw` contracts over EVERY leading coordinate at once, which is
    the reduction the shipped op has to ask for fp32 output to survive; in float64 it is just a
    matmul."""
    dx = _reduce_to(g @ w.transpose(-2, -1), x.shape)
    dw = (_flat2d(x).transpose(0, 1) @ _flat2d(g)).reshape(list(w.shape))
    dbias = _sum_leading(g, bias.shape) if bias is not None else None
    return dx, dw, dbias
