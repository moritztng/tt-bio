"""Reverse-mode autograd over on-device ttnn tensors.

The design is tt-train's. In ``tt-metal/tt-train/sources/ttml/autograd/graph.cpp`` a
differentiated op pushes a closure onto a tape (``Graph::add_node(GradFunction&&,
span<NodeId>)``, :30) and ``Tensor::backward`` (tensor.cpp:65) topologically sorts from the
root, reverses, seeds the root gradient with ones and calls each closure in turn. Every
closure reads its output's gradient and accumulates into its inputs' via ``add_grad``
(tensor.cpp:35), which sums on fan-in. This module is that structure in Python over raw
ttnn tensors.

Why not torch.autograd, which is the obvious alternative: torch's engine can only route
torch tensors. A ttnn device handle returned from ``torch.autograd.Function.forward``
gets no ``grad_fn`` at all -- torch drops it from the graph silently -- and a handle
smuggled as an attribute on a real-tensor carrier is lost by the engine's fan-in sum and
absent on ``grad_output`` in backward (all three measured, see docs/hallgrad.md). Making
the carrier hold real data means a host round-trip per tape node, which the activation
budget does not allow. The engine would only be scheduling for us anyway; the reverse
topological sort and the fan-in sum below are the whole of what it would have supplied.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import ttnn

__all__ = [
    "Tensor", "precise_config", "no_grad",
    "linear", "matmul", "layer_norm", "softmax", "mul", "add", "sigmoid", "reshape",
]


def precise_config():
    """HiFi4 with fp32 destination accumulation, per ``ComputeKernelConfig::precise()``.

    tt-train sets exactly this for backward math (core/compute_kernel_config.cpp:9-16:
    ``fp32_dest_acc_en = true``, ``math_approx_mode = false``, ``MathFidelity::HiFi4``,
    ``packer_l1_acc = true``). A backward accumulates over the reduction axis and again
    over fan-in, and bf16 accumulation is how a gradient turns into noise. Every op here
    defaults to it.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


_GRAD_ENABLED = True


class no_grad:
    """Run a forward without taping it. This is the stop-gradient on earlier recycles."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


class _Node:
    __slots__ = ("fn", "parents")

    def __init__(self, fn, parents):
        self.fn = fn
        self.parents = parents


class Tensor:
    """A ttnn tensor plus its place on the tape. The value stays on device throughout."""

    __slots__ = ("value", "grad", "requires_grad", "node")

    def __init__(self, value, requires_grad: bool = False):
        self.value = value
        self.grad = None
        self.requires_grad = requires_grad
        self.node = None

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def add_grad(self, grad) -> None:
        """Accumulate one contribution. A tensor read by k consumers gets k calls."""
        if not self.requires_grad:
            return
        want, got = tuple(self.value.shape), tuple(grad.shape)
        if want != got:
            raise ValueError(f"gradient shape {got} does not match value shape {want}")
        self.grad = grad if self.grad is None else ttnn.add(self.grad, grad)

    def backward(self, seed=None) -> None:
        """Replay the tape from here. ``seed`` defaults to ones, i.e. d(sum(self))/d(self)."""
        order = _reverse_topo(self)
        self.grad = ttnn.ones_like(self.value) if seed is None else seed
        for t in order:
            if t.node is not None:
                t.node.fn()


def _reverse_topo(root: Tensor) -> list:
    """Reverse post-order DFS over tape edges (output -> inputs).

    Reverse post-order is a topological order of a DAG, so every consumer of a tensor runs
    before the tensor's own closure. That ordering is what makes the fan-in sum in
    ``add_grad`` complete rather than partial: a tensor's closure only fires once all of
    its gradient contributions have landed.
    """
    order: list = []
    seen: set = set()
    stack = [(root, False)]
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
    return order


def _tape(out_value, parents: Sequence[Tensor], make_fn) -> Tensor:
    """Wrap ``out_value`` and tape ``make_fn(out)`` if any parent wants a gradient.

    Skipping the node when no parent requires one is tt-train's branch pruning
    (autograd/graph_utils.hpp:74) and it is what makes stop-gradient free: an untaped
    subgraph costs nothing to carry.
    """
    needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
    out = Tensor(out_value, requires_grad=needs)
    if needs:
        out.node = _Node(make_fn(out), list(parents))
    return out


def _flat2d(t):
    """Collapse every leading dim, leaving (prod(leading), last). Shape bookkeeping only."""
    s = [int(d) for d in t.shape]
    return ttnn.reshape(t, [int(math.prod(s[:-1])), s[-1]])


def _sum_leading(t, out_shape):
    """Sum ``t`` down to ``out_shape``, which must be its trailing dims. Used for bias/gamma."""
    flat = _flat2d(t)
    summed = ttnn.sum(flat, dim=0, keepdim=True)
    return ttnn.reshape(summed, [int(d) for d in out_shape])


def matmul(a: Tensor, b: Tensor, *, config=None) -> Tensor:
    """``a @ b``. Backward is the two transposed products, which is a matmul's own shape."""
    cfg = config or precise_config()
    out_v = ttnn.matmul(a.value, b.value, compute_kernel_config=cfg)

    def make(out):
        def bw():
            g = out.grad
            if a.requires_grad:
                a.add_grad(ttnn.matmul(g, b.value, transpose_b=True, compute_kernel_config=cfg))
            if b.requires_grad:
                b.add_grad(ttnn.matmul(a.value, g, transpose_a=True, compute_kernel_config=cfg))
        return bw

    return _tape(out_v, [a, b], make)


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None, *, config=None) -> Tensor:
    """``x @ w (+ b)``, with ``w`` in ttnn's (in, out) layout -- tt-bio's own convention.

    Backward is two matmuls rather than ``ttnn.moreh_linear_backward``. tt-train ships both
    side by side in ops/linear_op.cpp (``ttnn_linear_backward`` :21,
    ``moreh_linear_backward`` :52) and wires only the two-matmul one into ``linear_op``
    (:115), which is the answer to why they kept both: moreh is the reference, the explicit
    pair is what runs. Two reasons it is also right for us. The moreh family is
    interleaved-only at our pin, so routing a gradient through it forfeits the L1-residency
    levers tt-bio ships; and ``moreh_linear_backward`` is itself just these two matmuls plus
    a reduction, so there is no kernel to gain. Our linears are thin-K, where a matmul the
    production path already tunes beats a generic one.
    """
    cfg = config or precise_config()
    out_v = ttnn.linear(x.value, w.value,
                        bias=(b.value if b is not None else None),
                        compute_kernel_config=cfg)
    parents = [p for p in (x, w, b) if p is not None]

    def make(out):
        def bw():
            g = out.grad
            if x.requires_grad:
                x.add_grad(ttnn.matmul(g, w.value, transpose_b=True, compute_kernel_config=cfg))
            if w.requires_grad:
                # dW = X^T @ dY, summed over every leading dim, so flatten both first:
                # a batched matmul would give one dW per batch instead of their sum.
                w.add_grad(ttnn.matmul(_flat2d(x.value), _flat2d(g),
                                       transpose_a=True, compute_kernel_config=cfg))
            if b is not None and b.requires_grad:
                b.add_grad(_sum_leading(g, b.value.shape))
        return bw

    return _tape(out_v, parents, make)


def layer_norm(x: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None,
               *, eps: float = 1e-6, config=None) -> Tensor:
    """Layer norm over the last dim, composite so the backward has mean and rstd to hand.

    Composite on purpose. ``ttnn.layer_norm`` does not return mean/rstd and
    ``moreh_layer_norm_backward`` requires both, so the moreh route needs the forward taught
    to emit them and then refuses bfloat8_b in the kernel. Computing them here costs two
    reductions and keeps the whole op on production eltwise.

    One deliberate departure from tt-train: ``composite_layernorm``
    (ops/layernorm_op.cpp:144) takes the variance as E[x^2] - E[x]^2, which cancels
    catastrophically once the mean dominates the spread. This uses the two-pass
    E[(x - mean)^2] instead, for one extra pass over the row.
    """
    cfg = config or precise_config()
    xv = x.value
    mean = ttnn.mean(xv, dim=-1, keepdim=True)
    centered = ttnn.subtract(xv, mean)
    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True)
    rstd = ttnn.rsqrt(ttnn.add(var, eps))
    norm = ttnn.multiply(centered, rstd)
    out_v = norm
    if gamma is not None:
        out_v = ttnn.multiply(out_v, gamma.value)
    if beta is not None:
        out_v = ttnn.add(out_v, beta.value)
    parents = [p for p in (x, gamma, beta) if p is not None]
    width = float(int(xv.shape[-1]))

    def make(out):
        def bw():
            g = out.grad
            if gamma is not None and gamma.requires_grad:
                gamma.add_grad(_sum_leading(ttnn.multiply(g, norm), gamma.value.shape))
            if beta is not None and beta.requires_grad:
                beta.add_grad(_sum_leading(g, beta.value.shape))
            if x.requires_grad:
                dnorm = ttnn.multiply(g, gamma.value) if gamma is not None else g
                # dx = (dnorm - mean(dnorm) - norm * mean(dnorm * norm)) * rstd
                dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
                dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1, keepdim=True)
                dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                   ttnn.multiply(norm, dn_norm_mean))
                x.add_grad(ttnn.multiply(dx, rstd))
        return bw

    _ = width  # row width enters only through the means above
    return _tape(out_v, parents, make)


def softmax(x: Tensor, dim: int = -1, *, config=None) -> Tensor:
    """Softmax over ``dim``. Backward is ``y * (dy - sum(dy * y))``, which needs only y."""
    cfg = config or precise_config()
    y = ttnn.softmax(x.value, dim=dim, compute_kernel_config=cfg)

    def make(out):
        def bw():
            g = out.grad
            inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
            x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
        return bw

    return _tape(y, [x], make)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise product. Same shapes only; broadcasting would need a reducing backward."""
    out_v = ttnn.multiply(a.value, b.value)

    def make(out):
        def bw():
            g = out.grad
            if a.requires_grad:
                a.add_grad(ttnn.multiply(g, b.value))
            if b.requires_grad:
                b.add_grad(ttnn.multiply(g, a.value))
        return bw

    return _tape(out_v, [a, b], make)


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum. Both gradients are the incoming one, which is why fan-in must sum."""
    out_v = ttnn.add(a.value, b.value)

    def make(out):
        def bw():
            g = out.grad
            if a.requires_grad:
                a.add_grad(g)
            if b.requires_grad:
                b.add_grad(g)
        return bw

    return _tape(out_v, [a, b], make)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid. Backward ``y * (1 - y)``, computed from the retained output."""
    y = ttnn.sigmoid(x.value)

    def make(out):
        def bw():
            x.add_grad(ttnn.multiply(out.grad, ttnn.multiply(y, ttnn.rsub(y, 1.0))))
        return bw

    return _tape(y, [x], make)


def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshape. The backward of a shape op is the inverse shape op: no kernel work."""
    src = [int(d) for d in x.value.shape]
    out_v = ttnn.reshape(x.value, [int(d) for d in shape])

    def make(out):
        def bw():
            x.add_grad(ttnn.reshape(out.grad, src))
        return bw

    return _tape(out_v, [x], make)
