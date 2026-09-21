#!/usr/bin/env python3
"""Float64 BOUNDS on the tape's verbs, of3t-adaln's `host_f64_rule` generalised.

of3t-trajectory bounded the softmax by replacing `taped_ttnn._VERBS["softmax"]` with a rule
that computes the op, forward and backward, on the host in float64. The replacement is on the
VERB, so it is scope-agnostic: every softmax the tape sees is bounded, wherever it sits. This
module does the same for the other verbs, so the question "what does the conditioned
transition contribute" can be asked the same way the softmax question was.

Two things it does differently, both deliberate:

**The backward comes from torch, not from a hand-written rule.** The softmax rule spells its
own `y*(g - (g*y).sum())`, which is fine for one op. Spelling five more by hand would put six
new hand-derived backward rules between the measurement and the answer, and a bound whose own
derivative is wrong is worse than no bound. Here the float64 forward runs under torch autograd
and the backward is `torch.autograd.grad`. The cost is that the float64 graph stays alive until
the backward runs, which is the same cost the softmax rule already pays by retaining `y64`.

**Every verb carries a fire counter and zero is a hard failure.** The shim caches its wrapper,
so a patch that does not drop the cached attribute is a silent no-op -- the arm then reads
exactly like the shipped one and looks like a bound that found nothing. That is how the first
softmax sandwich reported two identical columns. `install` drops the cache and `report` refuses
a zero count.

A bound is not a lever. Nothing here is shippable: it moves activations to the host one op at a
time. `of3t-softmax` already returned NO-GO on every shipped precision lever.
"""
from __future__ import annotations

import torch
import ttnn

from tt_bio import taped_ttnn as TT

# Only the activations this tree actually fuses. Anything else raises rather than being
# silently differentiated as the identity, which is `_activation`'s rule in taped_ttnn and
# the reason the TriangleAttention gate did not read 4.88x high a second time.
_ACT = {
    ttnn.UnaryOpType.SIGMOID: torch.sigmoid,
    ttnn.UnaryOpType.SILU: torch.nn.functional.silu,
}
_LINEAR_ACT = {"silu": torch.nn.functional.silu, None: lambda x: x}

CALLS: dict[str, int] = {}
_ORIG: dict[str, object] = {}


def _t64(v):
    return ttnn.to_torch(v).double()


def _back(t, g64):
    """Hand a float64 host gradient back to a taped Tensor in its own layout and dtype."""
    t.add_grad(ttnn.from_torch(g64.float().contiguous(), layout=t.value.layout,
                               device=t.value.device(), dtype=t.value.dtype))


def _taped_f64(name, parents, fn, out_like):
    """Run `fn` on the parents' float64 values under autograd and tape the result."""
    xs = []
    for p in parents:
        x = _t64(p.value)
        x.requires_grad_(True)
        xs.append(x)
    y = fn(*xs)
    out_v = ttnn.from_torch(y.detach().float().contiguous(), layout=out_like.layout,
                            device=out_like.device(), dtype=out_like.dtype)

    def make():
        def bw(g):
            g64 = _t64(g)
            need = [i for i, p in enumerate(parents) if p.requires_grad]
            if not need:
                return
            gs = torch.autograd.grad(y, [xs[i] for i in need], g64,
                                     allow_unused=True, retain_graph=True)
            for i, gr in zip(need, gs):
                if gr is not None:
                    _back(parents[i], gr)
        return bw

    return TT._tape(out_v, parents, make)


def _act_of(kwargs, key):
    acts = list(kwargs.get(key) or ())
    if not acts:
        return lambda x: x
    if len(acts) > 1:
        raise NotImplementedError(f"{key} carries {len(acts)} fused activations")
    op = acts[0]
    if op not in _ACT:
        raise NotImplementedError(
            f"perf/of3t_residual/host_f64.py has no float64 form for the fused activation "
            f"{op!r}. Add it to _ACT rather than dropping it: the bound would then "
            f"differentiate a different function than the arm it is bounding.")
    return _ACT[op]


def _binary(name, op):
    def rule(shipped, args, kwargs):
        CALLS[name] = CALLS.get(name, 0) + 1
        a = TT._wrap(args[0])
        raw_b = args[1]
        fa, fb = _act_of(kwargs, "input_tensor_a_activations"), _act_of(
            kwargs, "input_tensor_b_activations")
        if not isinstance(raw_b, (TT.Tensor, ttnn.Tensor)):
            f = float(raw_b)
            out = _taped_f64(name, [a], lambda x: op(fa(x), f), a.value)
        else:
            b = TT._wrap(raw_b)
            out = _taped_f64(name, [a, b], lambda x, y: op(fa(x), fb(y)), a.value)
        if name.endswith("_"):
            # taped OUT of place, exactly as taped_ttnn does: the destination's PLACE is
            # handed back so the tuned forward's L1 budget still holds, and `free` evicts
            # rather than releases because the backward reads the value.
            a.free()
        return out
    return rule


def _layer_norm(shipped, args, kwargs):
    CALLS["layer_norm"] = CALLS.get("layer_norm", 0) + 1
    args = list(args) + [None] * (3 - len(args))
    x = TT._wrap(args[0])
    gamma = TT._wrap(args[1] if args[1] is not None else kwargs.get("weight"))
    beta = TT._wrap(args[2] if args[2] is not None else kwargs.get("bias"))
    eps = float(kwargs.get("epsilon", 1e-5))
    parents = [t for t in (x, gamma, beta) if t is not None]

    def fn(*ts):
        it = iter(ts)
        xv = next(it)
        g = next(it) if gamma is not None else None
        b = next(it) if beta is not None else None
        mu = xv.mean(dim=-1, keepdim=True)
        var = (xv - mu).pow(2).mean(dim=-1, keepdim=True)
        y = (xv - mu) * torch.rsqrt(var + eps)
        if g is not None:
            y = y * g
        if b is not None:
            y = y + b
        return y

    return _taped_f64("layer_norm", parents, fn, x.value)


def _linear(shipped, args, kwargs):
    CALLS["linear"] = CALLS.get("linear", 0) + 1
    args = list(args) + [None] * (3 - len(args))
    x, w = TT._wrap(args[0]), TT._wrap(args[1])
    bias = TT._wrap(args[2] if args[2] is not None else kwargs.get("bias"))
    act = _LINEAR_ACT.get(kwargs.get("activation"))
    if act is None:
        raise NotImplementedError(
            f"host_f64 linear has no float64 form for activation {kwargs.get('activation')!r}")
    parents = [t for t in (x, w, bias) if t is not None]

    def fn(*ts):
        it = iter(ts)
        xv, wv = next(it), next(it)
        y = torch.matmul(xv, wv)
        if bias is not None:
            y = y + next(it)
        return act(y)

    return _taped_f64("linear", parents, fn, x.value)


def _matmul(shipped, args, kwargs):
    """`ttnn.matmul` and `experimental.minimal_matmul`, the pair taped_ttnn._v_matmul covers.

    `linear` was already bounded and `matmul` is a DIFFERENT tape verb, so an arm that bounds
    `linear` leaves 2928 matmul calls on device -- which is what made the five-class residual an
    upper bound rather than a floor. `transpose_a`/`transpose_b` transpose the OPERAND in the
    product, so they are applied to the operand here; the autograd backward then follows from
    the expression instead of being hand-derived, which is this module's whole point.
    """
    CALLS["matmul"] = CALLS.get("matmul", 0) + 1
    kw = dict(kwargs)
    a = TT._wrap(args[0] if args else kw.get("input_tensor"))
    b = TT._wrap(args[1] if len(args) > 1 else kw.get("weight_tensor"))
    bias = TT._wrap(kw.get("bias_tensor"))
    ta, tb = bool(kw.get("transpose_a", False)), bool(kw.get("transpose_b", False))
    if kw.get("activation") is not None:
        raise NotImplementedError(
            f"host_f64 matmul has no float64 form for the fused activation "
            f"{kw['activation']!r}; taped_ttnn._v_matmul refuses it too")
    parents = [t for t in (a, b, bias) if t is not None]

    def fn(*ts):
        it = iter(ts)
        xv, wv = next(it), next(it)
        y = torch.matmul(xv.transpose(-2, -1) if ta else xv,
                         wv.transpose(-2, -1) if tb else wv)
        return y + next(it) if bias is not None else y

    return _taped_f64("matmul", parents, fn, a.value)


_RULES = {
    "layer_norm": _layer_norm,
    "linear": _linear,
    "multiply": _binary("multiply", lambda a, b: a * b),
    "multiply_": _binary("multiply_", lambda a, b: a * b),
    "add": _binary("add", lambda a, b: a + b),
    "add_": _binary("add_", lambda a, b: a + b),
    "subtract": _binary("subtract", lambda a, b: a - b),
    "matmul": _matmul,
}


def install(verbs):
    """Replace the named verbs with their float64 forms. Returns the list installed."""
    unknown = [v for v in verbs if v not in _RULES]
    if unknown:
        raise SystemExit(f"host_f64: no float64 rule for {unknown}; known: {sorted(_RULES)}")
    for v in verbs:
        _ORIG[v] = TT._VERBS[v]
        TT._VERBS[v] = _RULES[v]
        CALLS.setdefault(v, 0)
        # The shim caches the wrapped verb on first use. Without this the replacement is a
        # silent no-op and the arm is the shipped arm under another name.
        TT._SHIM.__dict__.pop(v, None)
    return list(verbs)


def report(verbs):
    """The intercept count per verb. A zero on any installed verb is a hard failure."""
    counts = {v: CALLS.get(v, 0) for v in verbs}
    return counts, [v for v, n in counts.items() if n == 0]


def census_install():
    """Count every verb without changing any arithmetic, so the cost of a bound is known
    before it is paid."""
    for v, impl in list(TT._VERBS.items()):
        def wrap(v=v, impl=impl):
            def call(shipped, args, kwargs):
                CALLS[v] = CALLS.get(v, 0) + 1
                return impl(shipped, args, kwargs)
            return call
        TT._VERBS[v] = wrap()
        TT._SHIM.__dict__.pop(v, None)
