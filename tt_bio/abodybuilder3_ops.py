"""The tape-aware op surface the ABodyBuilder3 port runs on. Nothing here is ABodyBuilder3's.

`tt_bio/autograd.py` already has the tape: `ag.Tensor`, the fan-in sum in `add_grad`, and the
reverse topological replay in `backward`. This file reuses all of that and differs from `ag`'s own
ops in one structural way, which is the point of it and is feedback for `train-a1-defork` rather
than a local preference:

**The tape wraps the production call; it does not re-implement it.** `ag.linear` computes its
forward with `ttnn.linear(..., compute_kernel_config=precise_config())` and drops `activation`,
`dtype` and `core_grid`, so grad-on and grad-off compute different numbers and a shipped forward
cannot be taped in place without moving inference. Here every op calls exactly the op the
inference path calls, with the same config and grid, and then optionally records a closure. One
forward implementation, one set of numbers, and `no_grad` costs a branch.

The same rule fixes layer norm, which is the case `ag` had to fork over: `ttnn.layer_norm` returns
neither mean nor rstd, so `ag.layer_norm` replaces it with a composite that has them to hand. This
file keeps the production kernel and **recomputes mean and rstd in the backward** from the input it
retains anyway. Two extra reductions on the backward pass, and the forward is untouched.

Three ops here have no `ag` equivalent because only a geometry model needs them: `pairwise_sub`
(the `[*, N, 1]` against `[*, 1, N]` difference Alg. 22's point term is built from, whose backward
is a pair of reductions), `mul_reduce` (a multiply against a broadcastable constant-shaped
parameter, e.g. the 12 IPA head weights against a `[B, H, N, N]` logit tensor), and `sum_last`.

Everything is fp32. Measured on qb1 card 3: eltwise is fp32-exact at 3.0e-07 while a matmul on
fp32 operands keeps ~11 mantissa bits at 1.25e-03 relative whatever the kernel config says
(`scripts/abb3_port/precision_probe.py`). That asymmetry is not a detail, it decides which algebraic
form each op takes, so it is recorded next to the ops rather than in a doc.

**Reductions round like the matmul, and that reaches the backward of every broadcast.** A
`ttnn.sum` carries ~1e-3 relative, so `layer_norm`, `softmax` and `sum_last` land at ~1e-3 on the
forward, and so does the gradient of any op whose backward reduces a broadcast axis -- which is
`add`, `sub`, `mul`, every bias and every weight gradient. That is the precision upstream trained
at anyway: `stages/train.py:20` sets `torch.set_float32_matmul_precision("medium")`, i.e. bf16
matmul math at ~4e-3, so a 1e-3 device gradient is tighter than the recipe's own.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import ttnn

from . import autograd as ag
from .tenstorrent import CORE_GRID_MAIN, get_device

Tensor = ag.Tensor

__all__ = ["Tensor", "param", "const", "grad_enabled", "backward", "kernel_config",
           "linear", "matmul", "add", "sub", "mul", "mul_reduce", "scale", "shift", "add_const",
           "mul_const", "sum_last", "sqrt_plus", "softplus", "relu", "softmax", "layer_norm",
           "reshape", "permute", "slice_dim", "concat", "transpose_last", "pairwise_sub"]

_KERNEL_CONFIG = None


def kernel_config():
    """HiFi4 with an fp32 accumulator: the repo's trunk config, and the best this card offers.

    `precision_probe.py` measured what "best" means here: 1.25e-03 relative on a matmul against
    7.05e-03 at HiFi2 and 2.85e-02 at LoFi, so the knob is real and its ceiling is ~11 mantissa
    bits. `fp32_dest_acc_en` is worth 1.5x of that (1.25e-03 against 1.88e-03) and is the whole of
    what keeps a gradient from accumulating into noise.
    """
    global _KERNEL_CONFIG
    if _KERNEL_CONFIG is None:
        device = get_device()
        cls = (ttnn.types.WormholeComputeKernelConfig
               if device.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
        _KERNEL_CONFIG = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                             fp32_dest_acc_en=True, packer_l1_acc=True)
    return _KERNEL_CONFIG


def param(value) -> Tensor:
    """A leaf that wants a gradient."""
    return Tensor(value, requires_grad=True)


def const(value) -> Tensor:
    """A leaf that does not. Masks, one-hots and the rigid-group tables are all this."""
    return Tensor(value, requires_grad=False)


def grad_enabled() -> bool:
    """Whether `ag.no_grad` is in force.

    Reads `ag._GRAD_ENABLED` because `autograd.py` exposes the flag only through the context
    manager that sets it. `train-a1-defork` owns that file; an `ag.is_grad_enabled()` there would
    let every dispatching op ask the question without reaching into a private name.
    """
    return ag._GRAD_ENABLED


def backward(roots: Sequence[Tensor], seeds: Sequence) -> None:
    """Replay the tape from several roots at once, each with its own incoming gradient.

    `ag.Tensor.backward` seeds one root with ones, which is the right shape for a scalar loss. The
    geometry losses here are computed on the host in torch from several device outputs -- the
    frames, the angles and the single representation of all 8 blocks -- so what comes back is a
    vector-Jacobian seed per output, not a scalar. Doing it as 8 separate `backward` calls would
    replay the shared trunk 8 times and, worse, would double-count the fan-in.

    Same reverse post-order as `ag._reverse_topo`, over a virtual root that owns every seeded
    tensor, so each closure still fires exactly once and only after all of its gradients have
    landed. `train-a1-defork`: this belongs in `autograd.py` as `ag.backward(roots, seeds)`.
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


class _Node:
    """Duck-typed to `ag._Node`: `ag`'s replay reads only `fn` and `parents`."""

    __slots__ = ("fn", "parents")

    def __init__(self, fn, parents):
        self.fn = fn
        self.parents = parents


def _tape(out_value, parents: Sequence[Optional[Tensor]], make_fn) -> Tensor:
    """`ag._tape`, with the same reason for the argument-free `make_fn`.

    A closure that reads its own output makes the cycle out -> node -> fn -> out, which refcounting
    cannot collect, and the measured consequence was OOM at step 2 of the hallucination loop
    (`ag._tape`'s comment). `make_fn` takes nothing, so no closure can capture the output.
    """
    live = [p for p in parents if p is not None]
    needs = grad_enabled() and any(p.requires_grad for p in live)
    out = Tensor(out_value, requires_grad=needs)
    if needs:
        out.node = _Node(make_fn(), live)
    return out


def _shape(t) -> list[int]:
    return [int(d) for d in t.shape]


def _sum_to(g, shape: Sequence[int]):
    """Reduce `g` onto `shape`, summing every axis `shape` has as 1. The backward of a broadcast."""
    want = [int(d) for d in shape]
    got = _shape(g)
    if want == got:
        return g
    assert len(want) == len(got), f"cannot reduce {got} onto {want}"
    for dim, (w, d) in enumerate(zip(want, got)):
        if w == d:
            continue
        assert w == 1, f"cannot reduce {got} onto {want} at axis {dim}"
        g = ttnn.sum(g, dim=dim, keepdim=True)
    return g


# --------------------------------------------------------------------------------- matmul family

def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None, *, activation: Optional[str] = None,
           core_grid=CORE_GRID_MAIN, dtype=ttnn.float32) -> Tensor:
    """`x @ w (+ b)` with `w` in `(in, out)` layout, on the production call.

    Backward is the two-matmul pair `ag.linear` uses and for its reasons: the moreh family is
    interleaved-only at our pin, and `moreh_linear_backward` is these two matmuls plus a reduction
    anyway. `dW` flattens both operands first, because a batched matmul would give one `dW` per
    sample instead of their sum.

    `activation` is refused when the op is taped rather than silently dropped. A fused ReLU changes
    what the backward has to gate on, and a caller that wants one on a taped path should ask for
    `relu` explicitly so the gating is visible.
    """
    cfg = kernel_config()
    taped = grad_enabled() and any(t is not None and t.requires_grad for t in (x, w, b))
    if activation is not None and taped:
        raise ValueError("fused activation on a taped linear: call relu() explicitly instead")
    out_v = ttnn.linear(x.value, w.value, bias=(b.value if b is not None else None),
                        compute_kernel_config=cfg, dtype=dtype, core_grid=core_grid,
                        activation=activation)

    def make():
        def bw(g):
            if x.requires_grad:
                x.add_grad(ttnn.matmul(g, w.value, transpose_b=True, compute_kernel_config=cfg))
            if w.requires_grad:
                w.add_grad(ttnn.matmul(_flat2d(x.value), _flat2d(g), transpose_a=True,
                                       compute_kernel_config=cfg))
            if b is not None and b.requires_grad:
                b.add_grad(_sum_to(_flat2d(g), [1, _shape(b.value)[-1]]).reshape(_shape(b.value)))
        return bw

    return _tape(out_v, [x, w, b], make)


def _flat2d(t):
    s = _shape(t)
    return ttnn.reshape(t, [int(math.prod(s[:-1])), s[-1]])


def matmul(a: Tensor, b: Tensor, *, transpose_a: bool = False, transpose_b: bool = False) -> Tensor:
    """`op(a) @ op(b)`. The four backward cases are `ag.matmul`'s, unchanged.

    The part worth restating: a transposed operand does not get a transposed gradient. If the
    forward used `A^T` then `dA` is the transpose of the gradient of the transposed operand, so the
    product re-associates instead of being post-transposed, and a shape check cannot catch the
    mistake when both dims are equal -- which for a pair tensor they always are.
    """
    cfg = kernel_config()
    out_v = ttnn.matmul(a.value, b.value, transpose_a=transpose_a, transpose_b=transpose_b,
                        compute_kernel_config=cfg)

    def make():
        def bw(g):
            if a.requires_grad:
                if not transpose_a:
                    a.add_grad(_sum_to(ttnn.matmul(g, b.value, transpose_b=not transpose_b,
                                                   compute_kernel_config=cfg), _shape(a.value)))
                else:
                    a.add_grad(_sum_to(ttnn.matmul(b.value, g, transpose_a=transpose_b,
                                                   transpose_b=True, compute_kernel_config=cfg),
                                       _shape(a.value)))
            if b.requires_grad:
                if not transpose_b:
                    b.add_grad(_sum_to(ttnn.matmul(a.value, g, transpose_a=not transpose_a,
                                                   compute_kernel_config=cfg), _shape(b.value)))
                else:
                    b.add_grad(_sum_to(ttnn.matmul(g, a.value, transpose_a=True,
                                                   transpose_b=transpose_a,
                                                   compute_kernel_config=cfg), _shape(b.value)))
        return bw

    return _tape(out_v, [a, b], make)


# ------------------------------------------------------------------------------ eltwise, fp32

def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum, broadcasting allowed. Each gradient is reduced back onto its own shape."""
    out_v = ttnn.add(a.value, b.value)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(_sum_to(g, _shape(a.value)))
            if b.requires_grad:
                b.add_grad(_sum_to(g, _shape(b.value)))
        return bw

    return _tape(out_v, [a, b], make)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise difference, broadcasting allowed."""
    out_v = ttnn.subtract(a.value, b.value)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(_sum_to(g, _shape(a.value)))
            if b.requires_grad:
                b.add_grad(_sum_to(ttnn.neg(g), _shape(b.value)))
        return bw

    return _tape(out_v, [a, b], make)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise product, broadcasting allowed on either side."""
    out_v = ttnn.multiply(a.value, b.value)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(_sum_to(ttnn.multiply(g, b.value), _shape(a.value)))
            if b.requires_grad:
                b.add_grad(_sum_to(ttnn.multiply(g, a.value), _shape(b.value)))
        return bw

    return _tape(out_v, [a, b], make)


#: The IPA head weights against a `[B, H, N, N]` logit tensor is the one broadcast in this model
#: where the small side is a parameter, so it is the one that needs the reducing backward. `mul`
#: already does it; the alias exists so the call site reads as what it is.
mul_reduce = mul


def scale(x: Tensor, factor: float) -> Tensor:
    """Multiply by a python scalar. The backward is the same scalar."""
    f = float(factor)
    out_v = ttnn.multiply(x.value, f)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, f))
        return bw

    return _tape(out_v, [x], make)


def shift(x: Tensor, offset: float) -> Tensor:
    """Add a python scalar. The backward is the identity."""
    out_v = ttnn.add(x.value, float(offset))

    def make():
        def bw(g):
            x.add_grad(g)
        return bw

    return _tape(out_v, [x], make)


def add_const(x: Tensor, c) -> Tensor:
    """Add a raw ttnn tensor that carries no gradient: the attention mask, a one-hot, a table."""
    out_v = ttnn.add(x.value, c)

    def make():
        def bw(g):
            x.add_grad(_sum_to(g, _shape(x.value)))
        return bw

    return _tape(out_v, [x], make)


def mul_const(x: Tensor, c) -> Tensor:
    """Multiply by a raw ttnn tensor that carries no gradient."""
    out_v = ttnn.multiply(x.value, c)

    def make():
        def bw(g):
            x.add_grad(_sum_to(ttnn.multiply(g, c), _shape(x.value)))
        return bw

    return _tape(out_v, [x], make)


def sum_last(x: Tensor, *, keepdim: bool = True) -> Tensor:
    """Sum over the last axis. The backward broadcasts the gradient back over it.

    Reductions round like the matmul, not like eltwise: 1.2e-02 on 96 fp32 terms
    (`precision_probe.py`). That is why the point term sums 12 channels by `add` rather than by
    laying them out along the last axis and calling this.
    """
    src = _shape(x.value)
    out_v = ttnn.sum(x.value, dim=-1, keepdim=True)
    if not keepdim:
        out_v = ttnn.reshape(out_v, src[:-1])

    def make():
        def bw(g):
            gg = g if keepdim else ttnn.reshape(g, src[:-1] + [1])
            x.add_grad(ttnn.multiply(ttnn.ones_like(x.value), gg))
        return bw

    return _tape(out_v, [x], make)


def sqrt_plus(x: Tensor, eps: float) -> Tensor:
    """`sqrt(x + eps)`. Backward `g / (2 sqrt(x + eps))`, read off the retained output.

    The epsilon is the argument and not a default because it is the difference between a gradient
    and a NaN: Alg. 22's point norm is exactly zero wherever a value point collapses, and
    upstream's own 1e-7 (`params.yaml` `epsilon`) is what keeps the derivative finite there.
    """
    y = ttnn.sqrt(ttnn.add(x.value, float(eps)))

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(ttnn.reciprocal(ttnn.multiply(y, 2.0)), g))
        return bw

    return _tape(y, [x], make)


def softplus(x: Tensor) -> Tensor:
    """`log(1 + exp(x))`. Backward is the sigmoid, which is cheaper to recompute than to retain."""
    out_v = ttnn.softplus(x.value, beta=1.0, threshold=20.0)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.sigmoid(x.value)))
        return bw

    return _tape(out_v, [x], make)


def relu(x: Tensor) -> Tensor:
    """ReLU, gated on the OUTPUT: `relu(x) > 0` exactly where `x > 0`, and the output is already
    materialised, so the input need not be retained."""
    out_v = ttnn.relu(x.value)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.gtz(out_v)))
        return bw

    return _tape(out_v, [x], make)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax over `dim`. Backward `y * (dy - sum(dy * y))`, which needs only `y`."""
    y = ttnn.softmax(x.value, dim=dim, compute_kernel_config=kernel_config())

    def make():
        def bw(g):
            inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
            x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
        return bw

    return _tape(y, [x], make)


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, *, eps: float = 1e-5) -> Tensor:
    """The production `ttnn.layer_norm`, taped by recomputing mean and rstd in the backward.

    This is the op `ag.layer_norm` had to fork the forward over, and it did not need to. The
    kernel returns neither mean nor rstd, but both are two reductions away from the input the
    backward retains regardless, so the forward stays the shipped one and the cost lands on the
    backward pass where there is already a reduction per op.
    """
    out_v = ttnn.layer_norm(x.value, weight=gamma.value, bias=beta.value, epsilon=eps,
                            compute_kernel_config=kernel_config())

    def make():
        def bw(g):
            mean = ttnn.mean(x.value, dim=-1, keepdim=True)
            centred = ttnn.subtract(x.value, mean)
            var = ttnn.mean(ttnn.multiply(centred, centred), dim=-1, keepdim=True)
            rstd = ttnn.rsqrt(ttnn.add(var, eps))
            norm = ttnn.multiply(centred, rstd)
            if gamma.requires_grad:
                gamma.add_grad(_sum_to(_flat2d(ttnn.multiply(g, norm)),
                                       [1, _shape(gamma.value)[-1]]).reshape(
                                           _shape(gamma.value)))
            if beta.requires_grad:
                beta.add_grad(_sum_to(_flat2d(g), [1, _shape(beta.value)[-1]]).reshape(
                    _shape(beta.value)))
            if x.requires_grad:
                dnorm = ttnn.multiply(g, gamma.value)
                dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
                dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1, keepdim=True)
                dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                   ttnn.multiply(norm, dn_norm_mean))
                x.add_grad(ttnn.multiply(dx, rstd))
        return bw

    return _tape(out_v, [x, gamma, beta], make)


# ---------------------------------------------------------------------------------- shape ops

def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshape. The backward of a shape op is the inverse shape op: no kernel work."""
    src = _shape(x.value)
    out_v = ttnn.reshape(x.value, [int(d) for d in shape])

    def make():
        def bw(g):
            x.add_grad(ttnn.reshape(g, src))
        return bw

    return _tape(out_v, [x], make)


def permute(x: Tensor, dims: Sequence[int]) -> Tensor:
    """Permute. The backward is the inverse permutation."""
    dims = [int(d) for d in dims]
    inverse = [0] * len(dims)
    for i, d in enumerate(dims):
        inverse[d] = i
    out_v = ttnn.permute(x.value, dims)

    def make():
        def bw(g):
            x.add_grad(ttnn.permute(g, inverse))
        return bw

    return _tape(out_v, [x], make)


def transpose_last(x: Tensor) -> Tensor:
    """Swap the last two axes, through `permute` and deliberately not through `transpose`.

    Measured, and the answer is not the one the name suggests: moving the last two axes rounds at
    4.95e-04 relative on fp32 data whichever call you make it through, `ttnn.transpose` or
    `ttnn.permute`, while a permute of the LEADING axes is exact at 5.3e-08
    (`scripts/abb3_port/op_gradcheck.py`). The last two axes are the tiled ones, so this is the
    tile transpose and it is arithmetic, not data movement.

    Nothing on this port's precision-critical path calls it. The attention logits take k^T through
    `matmul(transpose_b=True)`, inside a matmul that rounds at 1.25e-03 anyway, and the one
    reordering the IPA output needs -- the head axis past the residue axis -- is a leading-axis
    permute. Kept, measured and documented so that stays a decision.
    """
    rank = len(_shape(x.value))
    dims = list(range(rank))
    dims[-2], dims[-1] = dims[-1], dims[-2]
    return permute(x, dims)


def slice_dim(x: Tensor, dim: int, start: int, end: int) -> Tensor:
    """Slice one axis. The backward pads the gradient back with zeros.

    The port slices the head axis and never the tiled last two, which is why this is cheap: the
    coordinate index of a point is folded into the head axis exactly so that the rotation, which
    mixes coordinates, mixes slices of an untiled axis instead of sub-tile channel ranges.
    """
    src = _shape(x.value)
    dim = dim % len(src)
    starts = [0] * len(src)
    ends = list(src)
    starts[dim], ends[dim] = int(start), int(end)
    out_v = ttnn.slice(x.value, starts, ends)

    def make():
        def bw(g):
            # `ttnn.pad` refuses front padding on a tiled device tensor ("on device tile padding
            # does not support front padding"), so the zero extents are concatenated instead of
            # padded. Same result on the axis this op is used on.
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
            x.add_grad(pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=dim))
        return bw

    return _tape(out_v, [x], make)


def concat(xs: Sequence[Tensor], dim: int = -1) -> Tensor:
    """Concatenate. The backward slices the gradient back into each input's extent."""
    xs = list(xs)
    src = [_shape(t.value) for t in xs]
    dim = dim % len(src[0])
    out_v = ttnn.concat([t.value for t in xs], dim=dim)
    shape = _shape(out_v)

    def make():
        def bw(g):
            at = 0
            for t, s in zip(xs, src):
                width = s[dim]
                if t.requires_grad:
                    starts = [0] * len(shape)
                    ends = list(shape)
                    starts[dim], ends[dim] = at, at + width
                    t.add_grad(ttnn.slice(g, starts, ends))
                at += width
        return bw

    return _tape(out_v, [t for t in xs], make)


def pairwise_sub(q: Tensor, k: Tensor) -> Tensor:
    """`q[..., i, 0] - k[..., 0, j]`: the two-sided broadcast difference, `[*, N, 1]` x `[*, 1, N]`.

    Alg. 22's point term is 12 of these per head, squared and accumulated, and this file exists
    partly to make that form cheap enough to prefer. The alternative -- one matmul via
    `|q-k|^2 = |q|^2 + |k|^2 - 2 q.k` -- was measured and refused: at 1.25e-03 relative on a
    matmul and global coordinates reaching 2e4 A^2, the identity lands 14.8 A^2 out on pairs under
    100 A^2, which is a 0.70 error on a softmax logit. This form lands 2.7e-05 A^2, because every
    op in it is eltwise and eltwise on this card is fp32 (`precision_probe.py`).

    The backward is the pair of reductions the broadcast implies, and `_sum_to` gets them right by
    construction rather than by hand.
    """
    out_v = ttnn.subtract(q.value, k.value)

    def make():
        def bw(g):
            if q.requires_grad:
                q.add_grad(_sum_to(g, _shape(q.value)))
            if k.requires_grad:
                k.add_grad(_sum_to(ttnn.neg(g), _shape(k.value)))
        return bw

    return _tape(out_v, [q, k], make)
