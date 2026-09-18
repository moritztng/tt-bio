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
    "linear", "matmul", "layer_norm", "softmax", "mul", "add", "scale", "sigmoid",
    "reshape",
    "triangle_attention", "permute", "pair_contract", "checkpoint",
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
                t.node.fn(t.grad)


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
        # `make_fn` takes no arguments and the closure it returns takes the output gradient,
        # so no backward closure ever captures `out`. That matters for more than style: a
        # closure that reads `out.grad` makes the cycle out -> node -> fn -> out, which
        # CPython's refcounting cannot collect, so every intermediate of every step stays
        # resident until the cyclic collector happens to run. Measured: the hallucination
        # loop died of OOM at step 2 at 256 aa and inside 410 steps at 128 aa, on a 34.23 GB
        # card, with a per-step tape that fits several times over.
        out.node = _Node(make_fn(), list(parents))
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


def matmul(a: Tensor, b: Tensor, *, transpose_a: bool = False, transpose_b: bool = False,
           config=None) -> Tensor:
    """``op(a) @ op(b)`` where op is transpose-or-not on the last two dims.

    The four backward cases are tt-train's, worked out in ttnn_fixed/matmuls.cpp:54-79. The
    part worth restating is why a transposed operand does not simply get a transposed
    gradient: if the forward used A^T, then dA_effective is the gradient of the transposed
    operand and dA is its transpose, so the product has to be re-associated rather than
    post-transposed. Writing `ttnn.transpose` on the result instead is the classic way to
    get a silently wrong gradient on a non-square operand, and a shape check will not catch
    it when both dims are equal, which for a pair tensor they always are.
    """
    cfg = config or precise_config()
    out_v = ttnn.matmul(a.value, b.value, transpose_a=transpose_a, transpose_b=transpose_b,
                        compute_kernel_config=cfg)

    def make():
        def bw(g):
            if a.requires_grad:
                if not transpose_a:
                    # dA = g @ op(b)^T
                    a.add_grad(ttnn.matmul(g, b.value, transpose_b=not transpose_b,
                                           compute_kernel_config=cfg))
                else:
                    # A entered as A^T, so dA = (dA_eff)^T = op(b) @ g^T
                    a.add_grad(ttnn.matmul(b.value, g, transpose_a=transpose_b,
                                           transpose_b=True, compute_kernel_config=cfg))
            if b.requires_grad:
                if not transpose_b:
                    # dB = op(a)^T @ g
                    b.add_grad(ttnn.matmul(a.value, g, transpose_a=not transpose_a,
                                           compute_kernel_config=cfg))
                else:
                    # B entered as B^T, so dB = (dB_eff)^T = g^T @ op(a)
                    b.add_grad(ttnn.matmul(g, a.value, transpose_a=True,
                                           transpose_b=transpose_a, compute_kernel_config=cfg))
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

    def make():
        def bw(g):
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

    def make():
        def bw(g):
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

    def make():
        def bw(g):
            inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
            x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
        return bw

    return _tape(y, [x], make)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise product. Same shapes only; broadcasting would need a reducing backward."""
    out_v = ttnn.multiply(a.value, b.value)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(ttnn.multiply(g, b.value))
            if b.requires_grad:
                b.add_grad(ttnn.multiply(g, a.value))
        return bw

    return _tape(out_v, [a, b], make)


def scale(x: Tensor, factor: float) -> Tensor:
    """Multiply by a python scalar. The backward is the same scalar.

    `mul` cannot do this: it takes two taped tensors of equal shape, and a scalar has
    neither a shape nor a gradient. LoRA needs it for the `alpha/rank` factor, which
    tt-train applies as `lora_update * scaling` (modules/lora.py:106).
    """
    f = float(factor)
    out_v = ttnn.multiply(x.value, f)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, f))
        return bw

    return _tape(out_v, [x], make)


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum. Both gradients are the incoming one, which is why fan-in must sum."""
    out_v = ttnn.add(a.value, b.value)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(g)
            if b.requires_grad:
                b.add_grad(g)
        return bw

    return _tape(out_v, [a, b], make)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid. Backward ``y * (1 - y)``, computed from the retained output."""
    y = ttnn.sigmoid(x.value)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.multiply(y, ttnn.rsub(y, 1.0))))
        return bw

    return _tape(y, [x], make)


def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshape. The backward of a shape op is the inverse shape op: no kernel work."""
    src = [int(d) for d in x.value.shape]
    out_v = ttnn.reshape(x.value, [int(d) for d in shape])

    def make():
        def bw(g):
            x.add_grad(ttnn.reshape(g, src))
        return bw

    return _tape(out_v, [x], make)


def triangle_attention(q: Tensor, k: Tensor, v: Tensor, bias: Optional[Tensor] = None,
                       *, scale: Optional[float] = None, chunk: Optional[int] = None,
                       q_chunk: Optional[int] = None, config=None) -> Tensor:
    """Triangle attention with a chunked-recompute backward that never holds the scores.

    ``q``/``k``/``v`` are ``[B, H, N, d]`` head-major and ``bias`` is ``[1, H, N, N]``,
    broadcast over B. This is the shape tt-bio's ``TriangleAttention`` produces: for a pair
    tensor ``[S, S, c]`` the leading axis B is S, so triangle attention is S independent
    attention problems rather than one big one.

    That leading axis is why this needs no log-sum-exp bookkeeping. A flash kernel chunks
    KEYS, so each block sees a partial softmax denominator and has to carry running row
    statistics to rescale. Here the only reason to chunk is to keep the score tensor out of
    DRAM, and chunking the leading axis and the QUERY axis while keeping every key does
    that, so each chunk's softmax is already exact and complete. The retained set is q, k,
    v and bias; scores and probabilities are recomputed in the backward and freed per chunk.

    The object being avoided is concrete: at 800 aa with 8 heads the full
    ``[S, H, S, S]`` bf16 score tensor is 8.19 GB. One leading-axis chunk of 1 is 10.24 MB.

    ``chunk`` bounds the leading axis, ``q_chunk`` the query axis; both default to the whole
    extent, which materialises the scores and is only appropriate at small N. Gradients are
    invariant to both, which ``perf/hallgrad/gradcheck.py --cases triatt_chunked`` checks by
    differencing two chunkings rather than trusting one.
    """
    cfg = config or precise_config()
    qs = [int(d) for d in q.value.shape]
    ks = [int(d) for d in k.value.shape]
    if len(qs) != 4 or len(ks) != 4:
        raise ValueError(f"expected [B, H, N, d] q and k, got {qs} and {ks}")
    B, H, n_q, head_dim = qs
    n_k = ks[2]
    if scale is None:
        scale = head_dim ** -0.5
    cB = B if chunk is None else min(int(chunk), B)
    cQ = n_q if q_chunk is None else min(int(q_chunk), n_q)

    def _scores(qb, b0, b1, i0, i1):
        """Recompute one score block and its softmax. The only place the scores exist."""
        s = ttnn.matmul(qb, k.value[b0:b1], transpose_b=True, compute_kernel_config=cfg)
        s = ttnn.multiply(s, scale)
        if bias is not None:
            # bias is [1, H, n_q, n_k] and broadcasts over the leading axis, so the row
            # slice follows the query chunk and the leading slice is dropped.
            s = ttnn.add(s, bias.value[:, :, i0:i1, :])
        return ttnn.softmax(s, dim=-1, compute_kernel_config=cfg)

    out_blocks = []
    for b0 in range(0, B, cB):
        b1 = min(b0 + cB, B)
        row_blocks = []
        for i0 in range(0, n_q, cQ):
            i1 = min(i0 + cQ, n_q)
            p = _scores(q.value[b0:b1, :, i0:i1, :], b0, b1, i0, i1)
            row_blocks.append(ttnn.matmul(p, v.value[b0:b1], compute_kernel_config=cfg))
            ttnn.deallocate(p)
        out_blocks.append(row_blocks[0] if len(row_blocks) == 1
                          else ttnn.concat(row_blocks, dim=2))
    out_v = out_blocks[0] if len(out_blocks) == 1 else ttnn.concat(out_blocks, dim=0)
    parents = [p for p in (q, k, v, bias) if p is not None]

    def make():
        def bw(g):
            dq_blocks, dk_blocks, dv_blocks, dbias_rows = [], [], [], None
            for b0 in range(0, B, cB):
                b1 = min(b0 + cB, B)
                dk_acc = dv_acc = None
                dq_rows, dbias_acc = [], []
                for i0 in range(0, n_q, cQ):
                    i1 = min(i0 + cQ, n_q)
                    # Recompute, rather than retain. This is the whole memory argument:
                    # one forward matmul plus one softmax per block, traded against the
                    # 8.19 GB the retained scores would cost at 800 aa.
                    p = _scores(q.value[b0:b1, :, i0:i1, :], b0, b1, i0, i1)
                    go = g[b0:b1, :, i0:i1, :]
                    # dV = P^T @ dO, summed over the query chunks that share these keys.
                    dv_part = ttnn.matmul(p, go, transpose_a=True, compute_kernel_config=cfg)
                    dv_acc = dv_part if dv_acc is None else ttnn.add(dv_acc, dv_part)
                    # dS = P * (dP - rowsum(dP * P)), the softmax backward on the block.
                    dp = ttnn.matmul(go, v.value[b0:b1], transpose_b=True,
                                     compute_kernel_config=cfg)
                    inner = ttnn.sum(ttnn.multiply(dp, p), dim=-1, keepdim=True)
                    ds = ttnn.multiply(p, ttnn.subtract(dp, inner))
                    ttnn.deallocate(p)
                    if bias is not None:
                        # dBias is dS summed over the leading axis, since bias broadcast over it.
                        dbias_acc.append(ttnn.sum(ds, dim=0, keepdim=True))
                    dq_rows.append(ttnn.multiply(
                        ttnn.matmul(ds, k.value[b0:b1], compute_kernel_config=cfg), scale))
                    dk_part = ttnn.multiply(
                        ttnn.matmul(ds, q.value[b0:b1, :, i0:i1, :], transpose_a=True,
                                    compute_kernel_config=cfg), scale)
                    dk_acc = dk_part if dk_acc is None else ttnn.add(dk_acc, dk_part)
                    ttnn.deallocate(ds)
                dq_blocks.append(dq_rows[0] if len(dq_rows) == 1
                                 else ttnn.concat(dq_rows, dim=2))
                dk_blocks.append(dk_acc)
                dv_blocks.append(dv_acc)
                if bias is not None:
                    rows = dbias_acc[0] if len(dbias_acc) == 1 else ttnn.concat(dbias_acc, dim=2)
                    dbias_rows = rows if dbias_rows is None else ttnn.add(dbias_rows, rows)
            def cat0(blocks):
                return blocks[0] if len(blocks) == 1 else ttnn.concat(blocks, dim=0)
            if q.requires_grad:
                q.add_grad(cat0(dq_blocks))
            if k.requires_grad:
                k.add_grad(cat0(dk_blocks))
            if v.requires_grad:
                v.add_grad(cat0(dv_blocks))
            if bias is not None and bias.requires_grad:
                bias.add_grad(dbias_rows)
        return bw

    return _tape(out_v, parents, make)


def permute(x: Tensor, dims: Sequence[int]) -> Tensor:
    """Permute axes. The backward is the inverse permutation, so no kernel work but a real
    data movement, which is why the trimul contraction below pays for two of them."""
    dims = [int(d) for d in dims]
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    out_v = ttnn.permute(x.value, dims)

    def make():
        def bw(g):
            x.add_grad(ttnn.permute(g, inv))
        return bw

    return _tape(out_v, [x], make)


def pair_contract(a: Tensor, b: Tensor, *, incoming: bool = False, config=None) -> Tensor:
    """TriangleMultiplication's contraction, composed from `permute` and `matmul`.

    Outgoing is ``out[i,j,c] = sum_k a[i,k,c] * b[j,k,c]``, incoming is
    ``out[i,j,c] = sum_k a[k,i,c] * b[k,j,c]``. Both are one per-channel matmul once the
    channel axis is moved to the front, so this needs no new differentiated op at all: the
    tape's own `permute` and `matmul` with transpose flags cover it, and its gradient is
    therefore already verified by their gradchecks rather than needing its own.

    Inputs are ``[N, N, C]``, which is tt-bio's pair layout.
    """
    ap = permute(a, (2, 0, 1))
    bp = permute(b, (2, 0, 1))
    if incoming:
        # sum over the FIRST index: A_c^T @ B_c
        prod = matmul(ap, bp, transpose_a=True, config=config)
    else:
        # sum over the SECOND index: A_c @ B_c^T
        prod = matmul(ap, bp, transpose_b=True, config=config)
    return permute(prod, (1, 2, 0))


def checkpoint(fn, *inputs: Tensor) -> Tensor:
    """Run ``fn`` untaped, and re-run it taped inside its own backward.

    Trades one extra forward for dropping every intermediate ``fn`` produced. The
    feasibility study called per-block checkpointing mandatory for a 48-block trunk; the
    measurement is sharper than that. ONE pairformer block at 512 aa with c_z=256 exhausts
    all 34.23 GB without it (perf/hallgrad/e2e_distogram.py at --n 512 dies in
    bank_manager.cpp:439 with 34.0 GB allocated and 0.03 GB free), because a tape retains
    both every op's saved intermediate AND a gradient per op. So checkpointing is not an
    optimisation that buys depth, it is what makes a single block fit.

    Only the ``inputs`` named here receive gradients. Anything ``fn`` closes over -- weights,
    typically -- does not, which is right for hallucination, where the design variable is the
    input and the weights are frozen. A training use would have to pass the weights in too.
    """
    with no_grad():
        produced = fn(*inputs)
    out_value = produced.value if isinstance(produced, Tensor) else produced

    def make():
        def bw(g):
            # Re-run the segment on fresh tape nodes over the same input VALUES, then
            # backprop through that inner tape and forward the results to the real inputs.
            inner = [Tensor(t.value, requires_grad=t.requires_grad) for t in inputs]
            y = fn(*inner)
            if y.node is None:
                raise RuntimeError("checkpoint(fn): the recomputed segment built no tape; "
                                   "fn must use taped ops and at least one input must "
                                   "require a gradient")
            y.backward(seed=g)
            for src, dup in zip(inputs, inner):
                if dup.grad is not None:
                    src.add_grad(dup.grad)
        return bw

    return _tape(out_value, list(inputs), make)
