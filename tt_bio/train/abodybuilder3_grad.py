"""The tape behind `tt_bio/abodybuilder3_ops.py`. Training module: installs, never imported by one.

`abodybuilder3_ops` holds a hook slot and offers every call to it before running the production op.
This file fills that slot. The direction matters more than the mechanism: nothing on the inference
path imports this module, so importing the ABodyBuilder3 model cannot reach the tape, and a fold
that never calls `install()` pays one `is None` test per op.

**A taped op's forward is the shipped function, called.** The hook receives the production callable
alongside the arguments and computes its value with it, so there is one forward implementation and
the training path cannot drift from the served one. It is not a claim about two implementations
agreeing; there is one.

Three things the hook does that a substitution could not, each borrowed from
`train-a1-defork`'s `_Hook` because the reasoning is the same:

* It declines outright when no operand is on the tape, so the call is the shipped one.
* It runs the shipped op, not a taped one, for an operand that is on the tape but not being
  differentiated -- under `no_grad`, or a frozen block. `_GRAD_ENABLED` alone cannot do that: it
  prunes the tape node after the caller has already computed the forward.
* It keeps the site's own kernel config, dtype and core grid, because it never re-specifies them.

The gradients are scored against torch float64 by `scripts/abb3_port/op_gradcheck.py`, forward and
backward, under a fixed random cotangent.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import ttnn

from .. import abodybuilder3_ops as ops
from .. import autograd as ag

Tensor = ag.Tensor

__all__ = ["install", "uninstall", "installed", "backward", "param", "const", "Tensor"]


def param(value) -> Tensor:
    """A leaf that wants a gradient."""
    return Tensor(value, requires_grad=True)


def const(value) -> Tensor:
    """A leaf that does not. Masks, one-hots and the rigid-group tables are all this."""
    return Tensor(value, requires_grad=False)


def _grad_enabled() -> bool:
    """`ag.no_grad`'s flag. `autograd.py` exposes it only through the context manager that sets it;
    `ag.is_grad_enabled()` is in `train-a1-defork`'s brief and this reads the private name until it
    lands, rather than adding it to a file this row does not own."""
    return ag._GRAD_ENABLED


def _on_tape(*ts) -> bool:
    return any(isinstance(t, Tensor) for t in ts)


def _differentiating(*ts) -> bool:
    return _grad_enabled() and any(isinstance(t, Tensor) and t.requires_grad for t in ts)


def _unwrap(t):
    return t.value if isinstance(t, Tensor) else t


def _wrap(t):
    """A raw ttnn tensor joins the tape as an untracked leaf; a `Tensor` passes through."""
    return t if t is None or isinstance(t, Tensor) else Tensor(t)


def backward(roots: Sequence[Tensor], seeds: Sequence) -> None:
    """Replay the tape from several roots at once, each with its own incoming gradient.

    `ag.Tensor.backward` seeds one root with ones, which is right for a scalar loss. The geometry
    losses here are computed on the host in torch from several device outputs -- the frames, the
    angles and the single representation of all 8 blocks -- so what comes back is a vector-Jacobian
    seed per output. Eight separate `backward` calls would replay the shared trunk eight times and
    double-count the fan-in.

    Same reverse post-order as `ag._reverse_topo`, over a virtual root owning every seeded tensor,
    so each closure fires once and only after all of its gradients have landed.
    `ag.backward(roots, seeds)` is in `train-a1-defork`'s brief; this is the shape it should take.
    """
    order: list = []
    seen: set = set()
    stack: list = [(t, False) for t in reversed(list(roots))]
    while stack:
        t, expanded = stack.pop()
        if expanded:
            order.append(t)
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        stack.append((t, True))
        if t.node is not None:
            for p in t.node.parents:
                if id(p) not in seen:
                    stack.append((p, False))
    order.reverse()
    for t, seed in zip(roots, seeds):
        t.grad = seed if t.grad is None else ttnn.add(t.grad, seed)
    for t in order:
        if t.node is not None and t.grad is not None:
            t.node.fn(t.grad)


#: The tape's own node type, not a second one. An earlier version of this file duck-typed it --
#: the replay reads only `fn` and `parents` -- and `train-a1-defork`'s
#: `test_the_training_package_defines_no_forward` was right to reject it: two node classes for one
#: tape is the ambiguity that check exists to catch, whatever their fields are.
_Node = ag._Node


def _tape(out_value, parents: Sequence[Optional[Tensor]], make_fn) -> Tensor:
    """`ag._tape`, with the same reason for the argument-free `make_fn`.

    A closure that reads its own output makes the cycle out -> node -> fn -> out, which refcounting
    cannot collect; the measured consequence was OOM at step 2 of the hallucination loop. `make_fn`
    takes nothing, so no closure can capture the output.
    """
    live = [p for p in parents if isinstance(p, Tensor)]
    out = Tensor(out_value, requires_grad=True)
    out.node = _Node(make_fn(), live)
    return out


def _shape(t) -> list[int]:
    return [int(d) for d in t.shape]


def _flat2d(t):
    s = _shape(t)
    return ttnn.reshape(t, [int(math.prod(s[:-1])), s[-1]])


def _sum_to(g, shape: Sequence[int]):
    """Reduce `g` onto `shape`, summing every axis `shape` has as 1. The backward of a broadcast."""
    want = [int(d) for d in shape]
    got = _shape(g)
    if want == got:
        return g
    # A matmul that folds the sample axis into the width hands a rank-3 gradient back to a rank-2
    # weight. Dropping leading 1s is a reshape, not a reduction, so it happens first and the
    # broadcast reduction below still sees equal ranks.
    while len(got) > len(want) and got[0] == 1:
        got = got[1:]
        g = ttnn.reshape(g, got)
    assert len(want) == len(got), f"cannot reduce {got} onto {want}"
    for dim, (w, d) in enumerate(zip(want, got)):
        if w == d:
            continue
        assert w == 1, f"cannot reduce {got} onto {want} at axis {dim}"
        g = ttnn.sum(g, dim=dim, keepdim=True)
    return g


def _accumulate(t, g) -> None:
    if isinstance(t, Tensor):
        t.add_grad(_sum_to(g, _shape(t.value)))


# --------------------------------------------------------------------------- taped ops
# Each takes the production callable first and computes its forward with it. The backward is the
# only thing written here.

def _linear(shipped, x, w, bias=None, **kw):
    """Backward is the two-matmul pair, for `ag.linear`'s reasons: the moreh family is
    interleaved-only at our pin and `moreh_linear_backward` is these two matmuls plus a reduction
    anyway. `dW` flattens both operands first, because a batched matmul would give one `dW` per
    sample instead of their sum.

    A fused activation is refused rather than dropped: it changes what the backward gates on, and a
    caller that wants one on a taped path should ask for `relu` explicitly so the gating is visible.
    """
    if kw.get("activation") is not None:
        raise ValueError("fused activation on a taped linear: call relu() explicitly instead")
    xv, wv, bv = _unwrap(x), _unwrap(w), _unwrap(bias)
    out_v = shipped(xv, wv, bv, **kw)
    cfg = ops.kernel_config()

    def make():
        def bw(g):
            if isinstance(x, Tensor) and x.requires_grad:
                x.add_grad(ttnn.matmul(g, wv, transpose_b=True, compute_kernel_config=cfg))
            if isinstance(w, Tensor) and w.requires_grad:
                w.add_grad(ttnn.matmul(_flat2d(xv), _flat2d(g), transpose_a=True,
                                       compute_kernel_config=cfg))
            if isinstance(bias, Tensor) and bias.requires_grad:
                bias.add_grad(ttnn.reshape(_sum_to(_flat2d(g), [1, _shape(bv)[-1]]), _shape(bv)))
        return bw

    return _tape(out_v, [x, w, bias], make)


def _matmul(shipped, a, b, *, transpose_a=False, transpose_b=False):
    """The four backward cases are `ag.matmul`'s, unchanged.

    The part worth restating: a transposed operand does not get a transposed gradient. If the
    forward used `A^T` then `dA` is the transpose of the gradient of the transposed operand, so the
    product re-associates instead of being post-transposed -- and a shape check cannot catch the
    mistake when both dims are equal, which for a pair tensor they always are.
    """
    av, bv = _unwrap(a), _unwrap(b)
    out_v = shipped(av, bv, transpose_a=transpose_a, transpose_b=transpose_b)
    cfg = ops.kernel_config()

    def make():
        def bw(g):
            if isinstance(a, Tensor) and a.requires_grad:
                if not transpose_a:
                    a.add_grad(_sum_to(ttnn.matmul(g, bv, transpose_b=not transpose_b,
                                                   compute_kernel_config=cfg), _shape(av)))
                else:
                    a.add_grad(_sum_to(ttnn.matmul(bv, g, transpose_a=transpose_b,
                                                   transpose_b=True, compute_kernel_config=cfg),
                                       _shape(av)))
            if isinstance(b, Tensor) and b.requires_grad:
                if not transpose_b:
                    b.add_grad(_sum_to(ttnn.matmul(av, g, transpose_a=not transpose_a,
                                                   compute_kernel_config=cfg), _shape(bv)))
                else:
                    b.add_grad(_sum_to(ttnn.matmul(g, av, transpose_a=True,
                                                   transpose_b=transpose_a,
                                                   compute_kernel_config=cfg), _shape(bv)))
        return bw

    return _tape(out_v, [a, b], make)


def _add(shipped, a, b):
    out_v = shipped(_unwrap(a), _unwrap(b))

    def make():
        def bw(g):
            _accumulate(a, g)
            _accumulate(b, g)
        return bw

    return _tape(out_v, [a, b], make)


def _sub(shipped, a, b):
    out_v = shipped(_unwrap(a), _unwrap(b))

    def make():
        def bw(g):
            _accumulate(a, g)
            _accumulate(b, ttnn.neg(g))
        return bw

    return _tape(out_v, [a, b], make)


def _sub_square(shipped, a, b):
    """`d/da = 2 (a - b) g`, `d/db = -2 (a - b) g`, with `(a - b)` RECOMPUTED in the backward.

    Retaining the difference would hold 150 MB per block across the whole backward pass for a
    tensor one subtract reproduces. The recompute is the point of the op as much as the fusion is.
    """
    av, bv = _unwrap(a), _unwrap(b)
    out_v = shipped(av, bv)

    def make():
        def bw(g):
            twice_diff = ttnn.multiply(ttnn.subtract(av, bv), 2.0)
            term = ttnn.multiply(g, twice_diff)
            _accumulate(a, term)
            _accumulate(b, ttnn.neg(term))
        return bw

    return _tape(out_v, [a, b], make)


def _mul(shipped, a, b):
    av, bv = _unwrap(a), _unwrap(b)
    out_v = shipped(av, bv)

    def make():
        def bw(g):
            _accumulate(a, ttnn.multiply(g, bv))
            _accumulate(b, ttnn.multiply(g, av))
        return bw

    return _tape(out_v, [a, b], make)


def _div(shipped, a, b):
    av, bv = _unwrap(a), _unwrap(b)
    out_v = shipped(av, bv)

    def make():
        def bw(g):
            inv = ttnn.reciprocal(bv)
            _accumulate(a, ttnn.multiply(g, inv))
            # -g * a / b^2, from the retained output rather than a second reciprocal.
            _accumulate(b, ttnn.neg(ttnn.multiply(ttnn.multiply(g, out_v), inv)))
        return bw

    return _tape(out_v, [a, b], make)


def _scale(shipped, x, factor):
    out_v = shipped(_unwrap(x), factor)
    f = float(factor)

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(g, f))
        return bw

    return _tape(out_v, [x], make)


def _shift(shipped, x, offset):
    out_v = shipped(_unwrap(x), offset)

    def make():
        def bw(g):
            _accumulate(x, g)
        return bw

    return _tape(out_v, [x], make)


def _sqrt_plus(shipped, x, eps):
    """Backward `g / (2 sqrt(x + eps))`, read off the retained output."""
    out_v = shipped(_unwrap(x), eps)

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(ttnn.reciprocal(ttnn.multiply(out_v, 2.0)), g))
        return bw

    return _tape(out_v, [x], make)


def _softplus(shipped, x):
    """Backward is the sigmoid, cheaper to recompute than to retain."""
    xv = _unwrap(x)
    out_v = shipped(xv)

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(g, ttnn.sigmoid(xv)))
        return bw

    return _tape(out_v, [x], make)


def _minimum(shipped, x, cap):
    """`torch.minimum`'s gradient: it goes to whichever operand is smaller.

    The second operand is a constant in this port's only use -- FAPE's clamp pattern comes from the
    region labels -- so it would have been enough to give `x` a gradient and stop. The gradcheck
    disagreed, and it was right to: an op that silently drops one operand's gradient is a trap for
    the next caller, and the harness reported it as an infinite error rather than a small one
    precisely because the gradient was absent rather than wrong. Ties are measure-zero on a
    continuous distance, so they are not split.
    """
    xv, capv = _unwrap(x), _unwrap(cap)
    out_v = shipped(xv, capv)

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(g, ttnn.lt(xv, capv)))
            _accumulate(cap, ttnn.multiply(g, ttnn.lt(capv, xv)))
        return bw

    return _tape(out_v, [x, cap], make)


def _clamp_min(shipped, x, value):
    """Gradient passes where `x` is above the floor and nowhere else, which is `clamp`'s."""
    xv = _unwrap(x)
    out_v = shipped(xv, value)

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(g, ttnn.gtz(ttnn.subtract(xv, float(value)))))
        return bw

    return _tape(out_v, [x], make)


def _norm_from_sq(shipped, x, eps):
    """`g * gtz(x) / (2 sqrt(x + eps))`: the 2-norm's derivative, zero at the origin."""
    xv = _unwrap(x)
    out_v = shipped(xv, eps)

    def make():
        def bw(g):
            denom = ttnn.multiply(ttnn.sqrt(ttnn.add(ttnn.relu(xv), float(eps))), 2.0)
            _accumulate(x, ttnn.multiply(ttnn.multiply(g, ttnn.gtz(xv)),
                                         ttnn.reciprocal(denom)))
        return bw

    return _tape(out_v, [x], make)


def _relu(shipped, x):
    """Gated on the OUTPUT: `relu(x) > 0` exactly where `x > 0`, and the output is already
    materialised, so the input need not be retained."""
    out_v = shipped(_unwrap(x))

    def make():
        def bw(g):
            _accumulate(x, ttnn.multiply(g, ttnn.gtz(out_v)))
        return bw

    return _tape(out_v, [x], make)


def _sum_last(shipped, x, *, keepdim=True):
    xv = _unwrap(x)
    src = _shape(xv)
    out_v = shipped(xv, keepdim=keepdim)

    def make():
        def bw(g):
            gg = g if keepdim else ttnn.reshape(g, src[:-1] + [1])
            _accumulate(x, ttnn.multiply(ttnn.ones_like(xv), gg))
        return bw

    return _tape(out_v, [x], make)


def _sum_dim(shipped, x, dim, *, keepdim=True):
    xv = _unwrap(x)
    src = _shape(xv)
    dim = dim % len(src)
    out_v = shipped(xv, dim, keepdim=keepdim)

    def make():
        def bw(g):
            gg = g if keepdim else ttnn.reshape(g, src[:dim] + [1] + src[dim + 1:])
            _accumulate(x, ttnn.multiply(ttnn.ones_like(xv), gg))
        return bw

    return _tape(out_v, [x], make)


def _softmax(shipped, x, dim=-1):
    """Backward `y * (dy - sum(dy * y))`, which needs only `y`."""
    out_v = shipped(_unwrap(x), dim=dim)

    def make():
        def bw(g):
            inner = ttnn.sum(ttnn.multiply(g, out_v), dim=dim, keepdim=True)
            _accumulate(x, ttnn.multiply(out_v, ttnn.subtract(g, inner)))
        return bw

    return _tape(out_v, [x], make)


def _layer_norm(shipped, x, gamma, beta, *, eps=1e-5):
    """The production kernel, taped by recomputing mean and rstd in the backward.

    This is the op a differentiable implementation did not have to fork. The kernel returns neither
    statistic, but both are two reductions away from the input the backward retains regardless, so
    the forward stays the shipped one and the cost lands on the backward pass, where there is
    already a reduction per op.
    """
    xv, gv, bv = _unwrap(x), _unwrap(gamma), _unwrap(beta)
    out_v = shipped(xv, gv, bv, eps=eps)

    def make():
        def bw(g):
            mean = ttnn.mean(xv, dim=-1, keepdim=True)
            centred = ttnn.subtract(xv, mean)
            var = ttnn.mean(ttnn.multiply(centred, centred), dim=-1, keepdim=True)
            rstd = ttnn.rsqrt(ttnn.add(var, eps))
            norm = ttnn.multiply(centred, rstd)
            if isinstance(gamma, Tensor) and gamma.requires_grad:
                gamma.add_grad(ttnn.reshape(
                    _sum_to(_flat2d(ttnn.multiply(g, norm)), [1, _shape(gv)[-1]]), _shape(gv)))
            if isinstance(beta, Tensor) and beta.requires_grad:
                beta.add_grad(ttnn.reshape(_sum_to(_flat2d(g), [1, _shape(bv)[-1]]), _shape(bv)))
            if isinstance(x, Tensor) and x.requires_grad:
                dnorm = ttnn.multiply(g, gv)
                dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
                dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1, keepdim=True)
                dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean), ttnn.multiply(norm, dn_norm_mean))
                x.add_grad(ttnn.multiply(dx, rstd))
        return bw

    return _tape(out_v, [x, gamma, beta], make)


def _reshape(shipped, x, shape):
    xv = _unwrap(x)
    src = _shape(xv)
    out_v = shipped(xv, shape)

    def make():
        def bw(g):
            _accumulate(x, ttnn.reshape(g, src))
        return bw

    return _tape(out_v, [x], make)


def _permute(shipped, x, dims):
    dims = [int(d) for d in dims]
    inverse = [0] * len(dims)
    for i, d in enumerate(dims):
        inverse[d] = i
    out_v = shipped(_unwrap(x), dims)

    def make():
        def bw(g):
            _accumulate(x, ttnn.permute(g, inverse))
        return bw

    return _tape(out_v, [x], make)


def _transpose_last(shipped, x):
    out_v = shipped(_unwrap(x))
    rank = len(_shape(_unwrap(x)))
    dims = list(range(rank))
    dims[-2], dims[-1] = dims[-1], dims[-2]

    def make():
        def bw(g):
            _accumulate(x, ttnn.permute(g, dims))
        return bw

    return _tape(out_v, [x], make)


def _slice_dim(shipped, x, dim, start, end):
    """The backward puts the zero extents back by concatenation.

    `ttnn.pad` refuses front padding on a tiled device tensor ("on device tile padding does not
    support front padding"), so a padded gradient is not available; a concatenation of zeros is the
    same tensor and works on the axis this op is used on.
    """
    xv = _unwrap(x)
    src = _shape(xv)
    dim = dim % len(src)
    out_v = shipped(xv, dim, start, end)

    def make():
        def bw(g):
            pieces = []
            if int(start) > 0:
                lead = list(src)
                lead[dim] = int(start)
                pieces.append(ttnn.zeros(lead, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                         device=g.device()))
            pieces.append(g)
            if src[dim] - int(end) > 0:
                tail = list(src)
                tail[dim] = src[dim] - int(end)
                pieces.append(ttnn.zeros(tail, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                         device=g.device()))
            _accumulate(x, pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=dim))
        return bw

    return _tape(out_v, [x], make)


def _concat(shipped, xs, dim=-1):
    xs = list(xs)
    src = [_shape(_unwrap(t)) for t in xs]
    dim = dim % len(src[0])
    out_v = shipped([_unwrap(t) for t in xs], dim=dim)
    shape = _shape(out_v)

    def make():
        def bw(g):
            at = 0
            for t, s in zip(xs, src):
                width = s[dim]
                if isinstance(t, Tensor) and t.requires_grad:
                    starts = [0] * len(shape)
                    ends = list(shape)
                    starts[dim], ends[dim] = at, at + width
                    t.add_grad(ttnn.slice(g, starts, ends))
                at += width
        return bw

    return _tape(out_v, xs, make)


_TAPED = {
    "linear": _linear, "matmul": _matmul, "add": _add, "sub": _sub, "sub_square": _sub_square, "mul": _mul, "div": _div,
    "scale": _scale, "shift": _shift, "sqrt_plus": _sqrt_plus, "softplus": _softplus,
    "clamp_min": _clamp_min, "minimum": _minimum, "norm_from_sq": _norm_from_sq, "relu": _relu, "sum_last": _sum_last, "sum_dim": _sum_dim, "softmax": _softmax, "layer_norm": _layer_norm,
    "reshape": _reshape, "permute": _permute, "transpose_last": _transpose_last,
    "slice_dim": _slice_dim, "concat": _concat,
}


def _operands(args):
    """Every argument that could be on the tape, including the members of a list argument."""
    flat = []
    for a in args:
        flat.extend(a) if isinstance(a, (list, tuple)) else flat.append(a)
    return flat


def _hook(name, shipped, args, kwargs):
    """`abodybuilder3_ops`'s grad hook. Returns None to decline, which falls through to production."""
    operands = _operands(args)
    if not _on_tape(*operands):
        return None
    if not _differentiating(*operands):
        return Tensor(shipped(*[_unwrap(a) if not isinstance(a, (list, tuple))
                                else [_unwrap(x) for x in a] for a in args], **kwargs))
    taped = _TAPED.get(name)
    if taped is None:
        raise NotImplementedError(
            f"abodybuilder3_ops.{name} has no backward. Add one to _TAPED -- declining here would "
            f"silently drop the gradient of an op the forward uses.")
    return taped(shipped, *args, **kwargs)


def install() -> None:
    """Make the ABodyBuilder3 op surface differentiable. Idempotent."""
    ops.set_grad_hook(_hook)


def uninstall() -> None:
    """Put the inference path back. Idempotent."""
    ops.set_grad_hook(None)


def installed() -> bool:
    return ops.grad_hook() is _hook
