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

import contextlib
import math
import sys
from typing import Optional, Sequence

import ttnn

__all__ = [
    "Tensor", "precise_config", "no_grad",
    "linear", "matmul", "layer_norm", "softmax", "mul", "add", "scale", "sigmoid",
    "relu", "silu", "reshape",
    "triangle_attention", "permute", "pair_contract", "checkpoint",
    "install", "uninstall", "installed", "is_grad_enabled", "backward", "tape",
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


def is_grad_enabled() -> bool:
    """Whether taping is on. ``no_grad`` is the only thing that turns it off, and a caller
    that needs to branch on it should not have to reach into the context manager to find out."""
    return _GRAD_ENABLED


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
    """A ttnn tensor plus its place on the tape. The value stays on device throughout.

    It answers everything a ttnn tensor answers. The shipped modules do far more with a
    tensor than compute on it -- they read ``padded_shape`` to size a program config,
    ``memory_config()`` to decide whether an operand is already in L1, ``layout`` and
    ``device()`` to pick a kernel -- and a taped tensor that could not answer those would
    make the tuned path unreachable from a gradient. ``__getattr__`` forwards the rest.

    What it deliberately does NOT answer to is ``isinstance(x, ttnn.Tensor)``. That is the
    discriminator the tape needs: a raw handle is nobody's activation and may be freed, a
    taped one may not.
    """

    __slots__ = ("value", "grad", "requires_grad", "node", "pinned")

    def __init__(self, value, requires_grad: bool = False):
        self.value = value
        self.grad = None
        self.requires_grad = requires_grad
        self.node = None
        # Set by `_tape` the moment a closure is built that can read this value. It is the
        # whole of the lifetime rule: `free` refuses a pinned tensor and nothing else.
        self.pinned = False

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __getattr__(self, name):
        # Only reached on a miss, so slots and properties above keep their own meaning.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "value"), name)

    def __getitem__(self, index):
        return _getitem(self, index)

    def free(self) -> None:
        """Release the device buffer, unless a backward can still read it.

        This is what ``ttnn.deallocate`` becomes while a tape is open. Inference frees an
        activation the moment its consumer has read it; a backward reads it again, much
        later, and the crash that follows a premature free surfaces in an unrelated op
        several layers away. A leaf parameter is never freed either -- it is not an
        intermediate, and its gradient lands on it.
        """
        if not self.pinned and not self.requires_grad:
            ttnn.deallocate(self.value)

    def add_grad(self, grad) -> None:
        """Accumulate one contribution. A tensor read by k consumers gets k calls.

        The second contribution promotes the accumulator to fp32, and that is not a
        precaution. Fan-in here is not two or three: a chunked forward calls this once per
        chunk, so a weight in the row-blocked transition takes one contribution per row
        block -- 64 of them on a 64-row pair block, 512 at 512 aa. bfloat16 carries 8
        mantissa bits, and a 64-term bf16 running sum measured 6.5e-02 relative L2 against
        the float64 reference where the same gradient unchunked measured 6.5e-03. The
        first contribution is stored as it arrives, so a tensor with one consumer -- most
        of the tape -- pays nothing in memory or in a cast.
        """
        if not self.requires_grad:
            return
        want, got = tuple(self.value.shape), tuple(grad.shape)
        if want != got:
            raise ValueError(f"gradient shape {got} does not match value shape {want}")
        if self.grad is None:
            self.grad = grad
            return
        if self.grad.dtype != ttnn.float32:
            self.grad = ttnn.typecast(self.grad, ttnn.float32)
        self.grad = ttnn.add(self.grad, grad if grad.dtype == ttnn.float32
                             else ttnn.typecast(grad, ttnn.float32))

    def backward(self, seed=None) -> None:
        """Replay the tape from here. ``seed`` defaults to ones, i.e. d(sum(self))/d(self)."""
        backward([self], [seed])


def backward(roots, seeds=None) -> None:
    """Replay one tape from SEVERAL roots at once, seeding each.

    A loss assembled on the host from k device outputs comes back as k seeds, and calling
    ``Tensor.backward`` k times is wrong twice over rather than merely slow. It replays every
    shared ancestor k times -- for ABodyBuilder3 that is the whole trunk, eight times -- and
    because each replay runs a shared node's closure with only the gradient that had arrived
    by then, the fan-in sums land partial and then get summed again on the next pass. One
    reverse topological order over the union of the roots' ancestors fixes both: every node's
    closure fires exactly once, after every contribution to it has landed.

    ``seeds`` is one per root, ``None`` for ones. A root that is also an ancestor of another
    root is handled correctly, because its seed is accumulated with ``add_grad`` semantics
    before the traversal and its own closure does not fire until the topological order says
    every consumer has run.

    Requested by ``train-b2-abb3-port``, whose eight geometry-loss outputs share one trunk.
    The eight-term ``ProtenixLoss`` has the same shape the moment it reads a second head.
    """
    if isinstance(roots, Tensor):
        roots = [roots]
        seeds = [seeds] if not isinstance(seeds, (list, tuple)) else seeds
    roots = list(roots)
    if seeds is None:
        seeds = [None] * len(roots)
    seeds = list(seeds)
    if len(seeds) != len(roots):
        raise ValueError(f"{len(roots)} roots but {len(seeds)} seeds")
    order = _reverse_topo(roots)
    for r, sd in zip(roots, seeds):
        g = ttnn.ones_like(r.value) if sd is None else sd
        r.grad = g if r.grad is None else ttnn.add(r.grad, g)
    for t in order:
        if t.node is not None:
            g = t.grad
            if g.dtype != t.value.dtype:
                g = ttnn.typecast(g, t.value.dtype)
            t.node.fn(g)


def _reverse_topo(roots) -> list:
    """Reverse post-order DFS over tape edges (output -> inputs), from one root or many.

    Reverse post-order is a topological order of a DAG, so every consumer of a tensor runs
    before the tensor's own closure. That ordering is what makes the fan-in sum in
    ``add_grad`` complete rather than partial: a tensor's closure only fires once all of
    its gradient contributions have landed.
    """
    order: list = []
    seen: set = set()
    if isinstance(roots, Tensor):
        roots = [roots]
    # Reversed, so that with an explicit LIFO stack the first root is expanded first and the
    # single-root order is unchanged from what this function produced before it took a list.
    stack = [(r, False) for r in reversed(list(roots))]
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
        # A closure is now live over these values, so nothing may free them. Pinning the
        # output too costs nothing -- an intermediate is some later node's parent anyway --
        # and covers the backwards that read their own output (softmax, sigmoid, relu).
        out.pinned = True
        for p in parents:
            p.pinned = True
    return out


def _flat2d(t):
    """Collapse every leading dim, leaving (prod(leading), last), DRAM-interleaved.

    The reshape is shape bookkeeping. The move is not, and it is the difference between a
    gradient and a direction.

    Every caller here is setting up a reduction over EVERY token at once -- K is 4096 on a
    64x64 pair block and 262144 at 512 aa -- and a long-K matmul only keeps fp32
    accumulation across its K blocks when it can use L1 accumulation for the partials. An
    L1-INTERLEAVED operand has already spent that L1, so ttnn plans the same matmul without
    it and the reduction falls back to bf16. Measured on a p300c, the shipped Transition's
    dW3, identical operands, identical kernel config, the only difference being where the
    left operand lived: 6.65e-02 relative L2 from L1, 5.41e-04 from DRAM. 123x, and the
    Wormhole and Blackhole kernel configs read the same to four digits, so this is
    placement and not fidelity.

    The forward never sees it: its matmuls reduce over a channel, 128 or 256, which fits in
    one K block. It is the backward that contracts over the token axis, and the tuned
    forward is exactly the code that leaves its activations in L1.
    """
    s = [int(d) for d in t.shape]
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(t, [int(math.prod(s[:-1])), s[-1]])


def _sum_leading(t, out_shape):
    """Sum ``t`` down to ``out_shape``, which must be its trailing dims. Used for bias/gamma.

    fp32 destination accumulation, not for style: this reduces every leading coordinate at
    once -- 4096 of them on a 64x64 pair block, 262144 at 512 aa -- and bf16 carries 8
    mantissa bits, so the default fidelity returns a direction rather than a gradient.
    Measured at the 64x64 block before the kernel config was passed: cosine 0.379 against
    the float64 reference, 0.999995 after.
    """
    flat = _flat2d(t)
    summed = ttnn.sum(flat, dim=0, keepdim=True, compute_kernel_config=precise_config())
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


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None, *, dtype=None, core_grid=None,
           config=None, backward_config=None, **kw) -> Tensor:
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

    ``dtype`` and ``core_grid`` exist here because the shipped call site passes them and
    this op used to drop them: the training forward is meant to be the served forward, and
    an output dtype or a core grid silently changed is a deviation nobody measured.
    ``backward_config`` separates the two halves, because they want opposite things. The
    forward wants whatever the production site measured; the backward accumulates over the
    reduction axis and again over fan-in, so it wants ``precise_config()``. Default is the
    forward's, which is what every caller predating the split already got.

    No ``activation``. ``ttnn.linear`` fuses one into the packer and the result is not
    generally invertible -- silu's is not -- so a fused activation here would be a silently
    wrong gradient. The attach point composes a taped activation on the output instead.
    """
    cfg = config or precise_config()
    bwcfg = backward_config or cfg
    out_v = ttnn.linear(x.value, w.value,
                        bias=(b.value if b is not None else None),
                        dtype=dtype, core_grid=core_grid,
                        compute_kernel_config=cfg, **kw)
    parents = [p for p in (x, w, b) if p is not None]

    def make():
        def bw(g):
            if x.requires_grad:
                x.add_grad(ttnn.matmul(g, w.value, transpose_b=True, compute_kernel_config=bwcfg))
            if w.requires_grad:
                # dW = X^T @ dY, summed over every leading dim, so flatten both first:
                # a batched matmul would give one dW per batch instead of their sum.
                w.add_grad(ttnn.matmul(_flat2d(x.value), _flat2d(g),
                                       transpose_a=True, compute_kernel_config=bwcfg))
            if b is not None and b.requires_grad:
                b.add_grad(_sum_leading(g, b.value.shape))
        return bw

    return _tape(out_v, parents, make)


def layer_norm(x: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None,
               *, eps: float = 1e-6, config=None, backward_config=None,
               memory_config=None) -> Tensor:
    """Layer norm over the last dim. The forward is ``ttnn.layer_norm``, production's own.

    This op used to be a composite five-op forward, justified by ``ttnn.layer_norm`` returning
    neither mean nor rstd while ``moreh_layer_norm_backward`` needs both and then refuses
    bfloat8_b. That argument rules out MOREH; it does not rule out keeping the production
    forward. The backward already retains ``x`` -- ``dx`` is a function of it -- so mean and
    rstd can be recomputed there for two reductions on a pass that carries one per op anyway.
    The composite forward was an unforced concession and it cost the one thing worth having:
    with the kernel back, a taped layer norm's forward is BIT-IDENTICAL to the served one, not
    merely close. Found by ``train-b2-abb3-port``'s ``abodybuilder3_ops.layer_norm``, which
    reached the same shape from the other end.

    Note what is NOT recoverable and why it does not matter: ``ttnn.layer_norm``'s internal
    mean and rstd are whatever its kernel computed, and the two the backward recomputes here
    are a two-pass ``E[(x - mean)^2]``, so they can differ in the last bits. That is a
    backward-side approximation of the backward's own coefficients, which is the half of the
    op where precision is cheap and where ``precise_config`` already applies. It is not a
    forward difference, and the forward is the half that has to match what we serve.

    ``eps`` defaults to 1e-6 while every tt-bio layer norm on the inference path uses 1e-5, a
    factor of ten. The default stays because the diagnostics under ``perf/hallgrad`` are
    calibrated against it; the attach point passes the site's own epsilon, so nothing
    dispatched through ``tt_bio.ops`` can inherit this default.
    """
    cfg = config or precise_config()
    bwcfg = backward_config or precise_config()
    xv = x.value
    kw = {} if memory_config is None else {"memory_config": memory_config}
    out_v = ttnn.layer_norm(xv, weight=(gamma.value if gamma is not None else None),
                            bias=(beta.value if beta is not None else None),
                            epsilon=eps, compute_kernel_config=cfg, **kw)
    parents = [p for p in (x, gamma, beta) if p is not None]

    def make():
        def bw(g):
            # Recomputed here, not retained from the forward: two reductions, and it is what
            # buys the production kernel above. The two-pass E[(x - mean)^2] rather than
            # tt-train's E[x^2] - E[x]^2 (ops/layernorm_op.cpp:144), which cancels
            # catastrophically once the mean dominates the spread.
            mean = ttnn.mean(xv, dim=-1, keepdim=True)
            centered = ttnn.subtract(xv, mean)
            var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                            compute_kernel_config=bwcfg)
            rstd = ttnn.rsqrt(ttnn.add(var, eps))
            norm = ttnn.multiply(centered, rstd)
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


def relu(x: Tensor) -> Tensor:
    """ReLU. The denoiser applies one between the atom encoder and the token aggregation
    (``protenix.py:1110``), and it is the only activation in that path the tape lacked.

    The backward gates on the OUTPUT rather than the input: relu(x) > 0 exactly where
    x > 0, and the output is already materialised, so the input need not be retained.
    """
    out_v = ttnn.relu(x.value)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.gtz(out_v)))
        return bw

    return _tape(out_v, [x], make)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid. Backward ``y * (1 - y)``, computed from the retained output."""
    y = ttnn.sigmoid(x.value)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.multiply(y, ttnn.rsub(y, 1.0))))
        return bw

    return _tape(y, [x], make)


def silu(x: Tensor) -> Tensor:
    """SiLU, ``x * sigmoid(x)``. Eight shipped linears fuse this activation into the packer.

    The backward needs the INPUT, not the output: ``y = x*s`` is not invertible, so unlike
    relu and sigmoid there is no way back from what was materialised. ``dy/dx = s*(1 + x*(1
    - s))``, and the sigmoid is retained rather than recomputed because it is the expensive
    half.
    """
    sig = ttnn.sigmoid(x.value)
    out_v = ttnn.multiply(x.value, sig)
    xv = x.value

    def make():
        def bw(g):
            d = ttnn.multiply(sig, ttnn.add(ttnn.multiply(xv, ttnn.rsub(sig, 1.0)), 1.0))
            x.add_grad(ttnn.multiply(g, d))
        return bw

    return _tape(out_v, [x], make)


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
    # A bias with a leading extent of 1 broadcasts over the B independent attention
    # problems, which is what the pair tracks pass. The atom transformer's local windows
    # pass a DIFFERENT bias per trunk, so its gradient is the score gradient itself
    # rather than a sum over that axis, and summing it there would quietly average the
    # windows together.
    bias_bcast = bias is None or int(bias.value.shape[0]) == 1
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
            dbias_blocks = []
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
                        # CLONE on the per-trunk path. The broadcast path appends the
                        # result of a sum, a fresh tensor, but ds itself is deallocated
                        # a few lines down, so keeping a reference to it hands the bias
                        # gradient freed storage -- which surfaces much later, in an
                        # unrelated permute backward.
                        dbias_acc.append(ttnn.sum(ds, dim=0, keepdim=True)
                                         if bias_bcast else ttnn.clone(ds))
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
                    if bias_bcast:
                        dbias_rows = rows if dbias_rows is None else ttnn.add(dbias_rows, rows)
                    else:
                        dbias_blocks.append(rows)
            def cat0(blocks):
                return blocks[0] if len(blocks) == 1 else ttnn.concat(blocks, dim=0)
            if q.requires_grad:
                q.add_grad(cat0(dq_blocks))
            if k.requires_grad:
                k.add_grad(cat0(dk_blocks))
            if v.requires_grad:
                v.add_grad(cat0(dv_blocks))
            if bias is not None and bias.requires_grad:
                bias.add_grad(dbias_rows if bias_bcast else cat0(dbias_blocks))
        return bw

    return _tape(out_v, parents, make)


def _axis(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def narrow(x: Tensor, dim: int, start: int, length: int) -> Tensor:
    """``x[..., start:start+length, ...]`` along ``dim``, with a zero-padding backward.

    Added for the atom transformer's local attention window, which upstream builds with
    ``unfold(dim, size=n_keys, step=n_queries)`` (primitives.py:371). Because n_keys is
    exactly 4 x n_queries, that unfold is four shifted slices concatenated rather than a
    general gather, so a slice and a concat are the only two ops the tape was missing.

    KEEP THE SLICED AXIS TILE-ALIGNED. Every use here slices the leading trunk axis or a
    32-multiple of the key axis; a sub-tile slice of the LAST axis is the documented
    Blackhole wedge and is not what this is for.
    """
    shape = [int(d) for d in x.value.shape]
    ax = _axis(dim, len(shape))
    starts = [0] * len(shape)
    ends = list(shape)
    starts[ax], ends[ax] = start, start + length
    out_v = ttnn.slice(x.value, starts, ends)
    before, after = start, shape[ax] - (start + length)

    def make():
        def bw(g):
            parts = []
            if before:
                z = list(shape)
                z[ax] = before
                parts.append(ttnn.zeros(z, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                        device=g.device()))
            parts.append(g)
            if after:
                z = list(shape)
                z[ax] = after
                parts.append(ttnn.zeros(z, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                        device=g.device()))
            x.add_grad(parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=ax))
        return bw

    return _tape(out_v, [x], make)


def concat(xs: Sequence[Tensor], dim: int = -2) -> Tensor:
    """Concatenate along ``dim``. The backward is the slice this op is the inverse of.

    A tensor may appear more than once in ``xs`` -- the window build concatenates four
    overlapping slices of the same padded key tensor -- so each occurrence adds its own
    contribution rather than overwriting, which ``add_grad`` already does.
    """
    xs = list(xs)
    shape = [int(d) for d in xs[0].value.shape]
    ax = _axis(dim, len(shape))
    sizes = [int(x.value.shape[ax]) for x in xs]
    out_v = ttnn.concat([x.value for x in xs], dim=ax)

    def make():
        def bw(g):
            off = 0
            gs = [int(d) for d in g.shape]
            for x, n in zip(xs, sizes):
                if x.requires_grad:
                    starts = [0] * len(gs)
                    ends = list(gs)
                    starts[ax], ends[ax] = off, off + n
                    x.add_grad(ttnn.slice(g, starts, ends))
                off += n
        return bw

    return _tape(out_v, xs, make)


def windows(x: Tensor, n_queries: int, n_keys: int, n_trunks: int,
            pad_left: int) -> Tensor:
    """The local-attention key window: ``[n_trunks, n_keys, c]`` out of ``[n_pad, c]``.

    ``x`` must ALREADY be padded to ``n_trunks * n_queries + pad_left + pad_right`` rows,
    and ``n_keys`` must be a multiple of ``n_queries`` -- upstream runs 128 and 32
    (primitives.py:366-372), so window i is exactly the four consecutive query-blocks
    starting at block i of the padded tensor. That is why this needs no gather.
    """
    if n_keys % n_queries:
        raise ValueError(f"n_keys {n_keys} is not a multiple of n_queries {n_queries}")
    per = n_keys // n_queries
    c = int(x.value.shape[-1])
    n_blocks = int(x.value.shape[-2]) // n_queries
    if n_blocks < n_trunks + per - 1:
        raise ValueError(f"padded length {n_blocks * n_queries} is too short for "
                         f"{n_trunks} trunks of {n_keys} keys")
    blocks = reshape(x, [n_blocks, n_queries, c])
    return concat([narrow(blocks, 0, i, n_trunks) for i in range(per)], dim=1)


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


def checkpoint(fn, *inputs: Tensor, params: Sequence[Tensor] = ()) -> Tensor:
    """Run ``fn`` untaped, and re-run it taped inside its own backward.

    Trades one extra forward for dropping every intermediate ``fn`` produced. The
    feasibility study called per-block checkpointing mandatory for a 48-block trunk; the
    measurement is sharper than that. ONE pairformer block at 512 aa with c_z=256 exhausts
    all 34.23 GB without it (perf/hallgrad/e2e_distogram.py at --n 512 dies in
    bank_manager.cpp:439 with 34.0 GB allocated and 0.03 GB free), because a tape retains
    both every op's saved intermediate AND a gradient per op. So checkpointing is not an
    optimisation that buys depth, it is what makes a single block fit.

    ``inputs`` are re-taped as duplicates, so their gradients arrive on the duplicates and
    are forwarded. Parameters ``fn`` closes over need no duplication: they are LEAVES, so
    the recomputed tape calls ``add_grad`` on the original objects and the gradient lands
    where it belongs without any plumbing.

    ``params`` therefore exists for one reason, and it is not plumbing -- it is the
    PRUNING decision. ``_tape`` only builds a node when some parent requires a gradient,
    so a checkpointed segment whose input is frozen (the first adapted block, reading a z
    the production path produced) would get NO node, its backward would never run, and
    every weight gradient inside it would be silently absent. Naming the parameters as
    parents is what keeps that from happening. Fine-tuning is exactly the case where the
    input can be frozen while the weights are not, so the old signature was correct for
    hallucination and wrong here.
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

    return _tape(out_value, list(inputs) + list(params), make)


# ---------------------------------------------------------------------------------------
# The attach point.
#
# `tt_bio/ops.py` is the one linear and layer-norm call every device module makes, and it
# offers each call to a hook before running it. This is that hook, and installing it is the
# whole of what makes the SHIPPED forward differentiable -- there is no taped copy of a
# production module to drift from what we serve.
#
# `ops` holds a slot and `install` fills it. Nothing under `tt_bio/` imports this module, so
# importing `tt_bio` cannot reach the tape and an inference path that never calls `install`
# pays one `is None` test per linear.
#
# Three things the hook does that a substitution could not, each of them a measured gap:
#
#   * It declines outright when no operand is on the tape, so the call is the shipped one,
#     byte for byte, including the tuned narrow-projection and L1-resident-norm routings.
#   * It runs the SHIPPED op, not the composite one, for an operand that is on the tape but
#     not being differentiated -- under `no_grad`, or for a frozen block in a fine-tune.
#     `_GRAD_ENABLED` alone cannot do this: it prunes the tape node after the caller has
#     already computed the forward.
#   * It forwards `dtype`, `core_grid` and the site's own compute kernel config and epsilon,
#     all four of which the tape's ops used to drop.
#
# With `layer_norm` now running `ttnn.layer_norm` in the forward and recomputing mean and
# rstd in the backward, the taped forward is the production forward op for op. The one
# remaining difference is a fused activation, which is composed rather than fused because
# `ttnn.linear` fuses into the packer and silu's output cannot be inverted. That is measured,
# not assumed, and it only exists where a gradient is actually wanted.
# ---------------------------------------------------------------------------------------

_ACTIVATIONS = {"relu": relu, "sigmoid": sigmoid, "silu": silu}


def _on_tape(*ts):
    return any(isinstance(t, Tensor) for t in ts)


def _differentiating(*ts):
    return _GRAD_ENABLED and any(isinstance(t, Tensor) and t.requires_grad for t in ts)


def _wrap(t):
    """A raw ttnn tensor joins the tape as an untracked leaf; a `Tensor` passes through."""
    return t if t is None or isinstance(t, Tensor) else Tensor(t)


def _unwrap(t):
    return t.value if isinstance(t, Tensor) else t


def _walk(args, kwargs):
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, (list, tuple)):
            yield from v
        else:
            yield v


def _on_tape(args, kwargs):
    return any(isinstance(v, Tensor) for v in _walk(args, kwargs))


def _differentiating(args, kwargs):
    return _GRAD_ENABLED and any(isinstance(v, Tensor) and v.requires_grad
                                 for v in _walk(args, kwargs))


def _wrap(t):
    """A raw ttnn tensor joins the tape as an untracked leaf; a `Tensor` passes through."""
    return t if t is None or isinstance(t, Tensor) else Tensor(t)


def _unwrap(t):
    return t.value if isinstance(t, Tensor) else t


def _deep_unwrap(v):
    if isinstance(v, (list, tuple)):
        return type(v)(_unwrap(u) for u in v)
    return _unwrap(v)


def _raw(args, kwargs):
    return ([_deep_unwrap(v) for v in args],
            {k: _deep_unwrap(v) for k, v in kwargs.items()})


def _taped_linear(shipped, args, kwargs):
    """`ops.linear` with a gradient. The VALUE comes from `shipped`, never recomputed here."""
    args = list(args) + [None] * (3 - len(args))
    x, w, bias = (_wrap(args[0]), _wrap(args[1]), _wrap(args[2]))
    if bias is None and "bias" in kwargs:
        bias = _wrap(kwargs["bias"])
    kw = {k: v for k, v in kwargs.items() if k != "bias"}
    activation = kw.pop("activation", None)
    act = None
    if activation is not None:
        act = _ACTIVATIONS.get(activation)
        if act is None:
            raise NotImplementedError(
                f"tt_bio.autograd has no backward for the fused activation {activation!r}. "
                f"Add one to _ACTIVATIONS -- dropping it would train against a forward we do "
                f"not serve.")
    cfg = kw.pop("compute_kernel_config", None)
    # The activation is composed rather than fused, because `ttnn.linear` fuses it into the
    # packer and silu's output cannot be inverted back to its input. That is a real deviation
    # of the training forward from the served one, and it is measured, not assumed away.
    out_v = shipped(x.value, w.value, bias=(bias.value if bias is not None else None),
                    activation=None, compute_kernel_config=cfg, **kw)
    cfg = cfg or precise_config()
    bwcfg = precise_config()
    parents = [t for t in (x, w, bias) if t is not None]

    def make():
        def bw(g):
            if x.requires_grad:
                # dX reduces over the OUTPUT channel -- 128 to 512 terms -- and is
                # activation-shaped, so it stays in the forward dtype.
                x.add_grad(ttnn.matmul(g, w.value, transpose_b=True,
                                       compute_kernel_config=bwcfg))
            if w.requires_grad:
                # dW reduces over every token at once: 4096 terms on a 64x64 pair block,
                # 262144 at 512 aa. `fp32_dest_acc_en` does not cover that, because
                # `packer_l1_acc` accumulates the per-K-block partials at the OUTPUT
                # dtype, so a bf16 result means a bf16 running sum however precise the
                # destination register is. Measured on the shipped Transition: 6.5e-02
                # relative L2 at K=4096 against 6.5e-03 at K=64, the sqrt(K) signature of
                # a bf16 reduction. Asking for fp32 out fixes it and costs nothing that
                # matters -- a weight gradient is weight-shaped, and the optimiser wants
                # it in fp32 anyway.
                w.add_grad(ttnn.matmul(_flat2d(x.value), _flat2d(g), transpose_a=True,
                                       compute_kernel_config=bwcfg, dtype=ttnn.float32))
            if bias is not None and bias.requires_grad:
                bias.add_grad(_sum_leading(g, bias.value.shape))
        return bw

    out = _tape(out_v, parents, make)
    return act(out) if act is not None else out


def _taped_layer_norm(shipped, args, kwargs):
    """`ops.layer_norm` with a gradient. The VALUE comes from `shipped`.

    `l1_headroom` is dropped on purpose. It asks for an L1-resident RESULT, and a backward
    holding a tape's worth of activations cannot be priced against a budget sized for one
    tensor. It is a placement lever and not a numeric one -- the values `ttnn.layer_norm`
    computes do not depend on where it writes them -- so the forward stays bit-identical to
    production either way.
    """
    args = list(args) + [None] * (3 - len(args))
    x, gamma, beta = (_wrap(args[0]), _wrap(args[1]), _wrap(args[2]))
    kw = dict(kwargs)
    for nm, slot in (("weight", 1), ("bias", 2)):
        if kw.get(nm) is not None:
            v = _wrap(kw.pop(nm))
            gamma, beta = (v, beta) if slot == 1 else (gamma, v)
        else:
            kw.pop(nm, None)
    kw.pop("l1_headroom", None)
    eps = kw.pop("epsilon", 1e-5)
    cfg = kw.pop("compute_kernel_config", None)
    xv = x.value
    out_v = shipped(xv, weight=(gamma.value if gamma is not None else None),
                    bias=(beta.value if beta is not None else None),
                    epsilon=eps, compute_kernel_config=cfg, **kw)
    bwcfg = precise_config()
    parents = [t for t in (x, gamma, beta) if t is not None]

    def make():
        def bw(g):
            # Recomputed here, not retained: two reductions, and it is what buys the
            # production kernel above. Two-pass E[(x - mean)^2] rather than tt-train's
            # E[x^2] - E[x]^2 (ops/layernorm_op.cpp:144), which cancels catastrophically
            # once the mean dominates the spread.
            mean = ttnn.mean(xv, dim=-1, keepdim=True)
            centered = ttnn.subtract(xv, mean)
            var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                            compute_kernel_config=bwcfg)
            rstd = ttnn.rsqrt(ttnn.add(var, eps))
            norm = ttnn.multiply(centered, rstd)
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

    return _tape(out_v, parents, make)


_TAPED = {"linear": _taped_linear, "layer_norm": _taped_layer_norm}


def _hook(name, shipped, args, kwargs):
    """`tt_bio.ops`'s grad hook. Returns None to decline, which falls through to production."""
    if not _on_tape(args, kwargs):
        return None
    if not _differentiating(args, kwargs):
        # On the tape but frozen, or inside `no_grad`: the SHIPPED op, then rewrapped.
        ra, rk = _raw(args, kwargs)
        return Tensor(shipped(*ra, **rk))
    impl = _TAPED.get(name)
    if impl is None:
        raise NotImplementedError(
            f"tt_bio.ops.{name} has no taped implementation. Add one to "
            f"tt_bio.autograd._TAPED; declining here would silently drop the gradient.")
    return impl(shipped, args, kwargs)


def install():
    """Route `tt_bio.ops` through the tape. Idempotent; returns the hook it replaced.

    This is the narrow seam -- two verbs. `tape()` is the whole shipped forward.
    """
    from . import ops
    return ops.set_grad_hook(_hook)


def uninstall() -> None:
    """Put the inference path back. Idempotent."""
    from . import ops
    ops.set_grad_hook(None)


def installed() -> bool:
    from . import ops
    return ops.grad_hook() is _hook


# ---------------------------------------------------------------------------------------
# The shipped forward, taped where it computes.
#
# `tt_bio.ops` routes two verbs, and the Pairformer chain calls 380 ttnn verbs across 56
# names (`perf/ptx_fastpath/census.py` counts them and re-counts them on demand). Routing
# every one by hand would mean editing the most heavily tuned file in the repo in 380
# places, and then keeping every future perf lever in step with the tape by hand. It also
# puts the gradient in the call site rather than in the op, and a call site is what goes
# stale: a lever landed next week in the trimul would take the gradient with it silently.
#
# So the seam moves one level out. `tape()` rebinds the name `ttnn` inside tt-bio's own
# modules to the object below, which is ttnn with the differentiable verbs taped and every
# other attribute ttnn's own. One rebinding per module, restored on exit, live only while a
# tape is open. It is not a monkeypatch of ttnn's namespace: nothing outside tt-bio sees a
# different ttnn, and `tt_bio.autograd` itself is excluded, so the backward closures below
# call the real verbs and cannot re-enter the tape.
#
# The mechanism is `dispatch.OpSurface`'s, one level up. A taped verb computes its value BY
# CALLING the shipped verb on unwrapped operands, so there is still exactly one forward and
# the grad-off column is bit-identical rather than nearly equal. The tuned path is reached
# the same way it always was, because the tuning reads shapes and memory configs off the
# tensor and a taped tensor answers those.
#
# A verb with no entry that is handed a taped tensor raises. That is deliberate: an op the
# tape cannot follow is loud, not a silently dropped gradient.
# ---------------------------------------------------------------------------------------

_VERBS: dict = {}


def _verb(*names):
    """Register one taped verb under each ttnn name, dotted for a nested namespace."""
    def register(fn):
        for n in names:
            _VERBS[n] = fn
        return fn
    return register


def _identity_grad(shipped, args, kwargs, cast=False):
    """A verb that moves or retypes bytes without changing what they mean.

    typecast, to_layout, to_memory_config, clone and reallocate are all this: the value is
    whatever the shipped verb produced and the gradient passes straight through. They are
    the ops a tuned forward is mostly made of, and none of them is a no-op on device.
    """
    x = _wrap(args[0])
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)
    src_dtype = x.value.dtype

    def make():
        def bw(g):
            x.add_grad(ttnn.typecast(g, src_dtype) if cast and g.dtype != src_dtype else g)
        return bw

    return _tape(out_v, [x], make)


_VERBS["linear"] = _taped_linear
_VERBS["layer_norm"] = _taped_layer_norm


@_verb("matmul", "experimental.minimal_matmul")
def _v_matmul(shipped, args, kwargs):
    """Both shipped matmuls. `transpose_a`/`transpose_b` reassociate rather than transpose
    the result -- see `matmul` above for why writing the post-transpose is the classic way
    to get a wrong gradient that a shape check cannot catch."""
    a, b = _wrap(args[0]), _wrap(args[1])
    kw = dict(kwargs)
    ta = bool(kw.get("transpose_a", False))
    tb = bool(kw.get("transpose_b", False))
    cfg = kw.get("compute_kernel_config") or precise_config()
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            if a.requires_grad:
                a.add_grad(ttnn.matmul(g, b.value, transpose_b=not tb,
                                       compute_kernel_config=cfg) if not ta else
                           ttnn.matmul(b.value, g, transpose_a=tb, transpose_b=True,
                                       compute_kernel_config=cfg))
            if b.requires_grad:
                b.add_grad(ttnn.matmul(a.value, g, transpose_a=not ta,
                                       compute_kernel_config=cfg) if not tb else
                           ttnn.matmul(g, a.value, transpose_a=True, transpose_b=ta,
                                       compute_kernel_config=cfg))
        return bw

    return _tape(out_v, [a, b], make)


@_verb("softmax", "softmax_in_place")
def _v_softmax(shipped, args, kwargs):
    """`softmax_in_place` is taped out of place. The backward reads y, which the in-place
    kernel has written over its own input, so the input is gone either way; what the tape
    cannot afford is the CALLER's copy being gone, and that is what in-place destroys."""
    x = _wrap(args[0])
    dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
    ra, rk = _raw(args, kwargs)
    y = ttnn.softmax(*ra, **rk) if shipped is ttnn.softmax_in_place else shipped(*ra, **rk)

    def make():
        def bw(g):
            inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
            x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
        return bw

    return _tape(y, [x], make)


def _unary(fn):
    """Register a unary eltwise verb whose backward is `fn(x_value, out_value) -> dy/dx`.

    An `output_tensor=` argument is dropped, which is the in-place case: the shipped verb
    writes its result over its input and the tape needs the input to still be there.
    """
    def impl(shipped, args, kwargs):
        x = _wrap(args[0])
        kw = {k: v for k, v in kwargs.items() if k != "output_tensor"}
        out_v = shipped(x.value, *[_unwrap(a) for a in args[1:]], **kw)
        xv = x.value

        def make():
            def bw(g):
                x.add_grad(ttnn.multiply(g, fn(xv, out_v)))
            return bw

        return _tape(out_v, [x], make)
    return impl


_VERBS["silu"] = _unary(
    # sigma * (1 + x * (1 - sigma)). The output is not invertible, so the input is read.
    lambda xv, y: (lambda s: ttnn.multiply(
        s, ttnn.add(ttnn.multiply(xv, ttnn.rsub(s, 1.0)), 1.0)))(ttnn.sigmoid(xv)))
_VERBS["sigmoid"] = _unary(lambda xv, y: ttnn.multiply(y, ttnn.rsub(y, 1.0)))
_VERBS["relu"] = _unary(lambda xv, y: ttnn.gtz(xv))
_VERBS["exp"] = _unary(lambda xv, y: y)


def _binary(grad_a, grad_b, scalar, out_of_place=None):
    """Register a binary eltwise verb. The second operand may be a python scalar, which the
    shipped chain does often enough (`ttnn.multiply(s, 1 / sqrt(d))`) that treating it as a
    tensor would be wrong rather than merely slow.

    An in-place verb (`add_`, `multiply_`) is taped OUT of place. That is the memory cost
    the tape pays for a residual: the pre-update tensor stays live because the producer's
    backward reads it. It is one tensor per residual, not per layer.
    """
    def impl(shipped, args, kwargs):
        a = _wrap(args[0])
        raw_b = args[1]
        kw = {k: v for k, v in kwargs.items() if k != "output_tensor"}
        # THE in-place trap, and it is silent: `ttnn.multiply_(a, b)` writes the product
        # over a's buffer, and a is exactly what that product's backward reads for db.
        # Taping the call while still invoking the in-place kernel gives a forward that is
        # right and a gradient that is anti-correlated with the truth. Measured before this
        # line existed: d(fc2) cosine -0.028 on the shipped Transition.
        if out_of_place is not None:
            shipped = out_of_place
        if not isinstance(raw_b, (Tensor, ttnn.Tensor)):
            f = float(raw_b)
            out_v = shipped(a.value, raw_b, **kw)

            def make_s():
                def bw(g):
                    a.add_grad(scalar(g, f))
                return bw

            return _tape(out_v, [a], make_s)
        b = _wrap(raw_b)
        out_v = shipped(a.value, b.value, **kw)
        av, bv = a.value, b.value

        def make():
            def bw(g):
                if a.requires_grad:
                    a.add_grad(grad_a(g, av, bv))
                if b.requires_grad:
                    b.add_grad(grad_b(g, av, bv))
            return bw

        return _tape(out_v, [a, b], make)
    return impl


_ADD = (lambda g, av, bv: g, lambda g, av, bv: g, lambda g, f: g)
_MUL = (lambda g, av, bv: ttnn.multiply(g, bv), lambda g, av, bv: ttnn.multiply(g, av),
        lambda g, f: ttnn.multiply(g, f))
_VERBS["add"] = _binary(*_ADD)
_VERBS["add_"] = _binary(*_ADD, out_of_place=ttnn.add)
_VERBS["subtract"] = _binary(
    lambda g, av, bv: g, lambda g, av, bv: ttnn.multiply(g, -1.0), lambda g, f: g)
_VERBS["multiply"] = _binary(*_MUL)
_VERBS["multiply_"] = _binary(*_MUL, out_of_place=ttnn.multiply)
_VERBS["divide"] = _binary(
    lambda g, av, bv: ttnn.divide(g, bv),
    lambda g, av, bv: ttnn.multiply(ttnn.divide(g, ttnn.multiply(bv, bv)),
                                    ttnn.multiply(av, -1.0)),
    lambda g, f: ttnn.multiply(g, 1.0 / f))


# --- shape ------------------------------------------------------------------------------
# Every one of these is its own inverse applied to the gradient. They carry no arithmetic,
# so nothing here can lose precision; what they can lose is an axis, which is why each
# reads the source shape at forward time rather than inferring it in the backward.

@_verb("reshape", "unsqueeze", "squeeze")
def _v_reshape(shipped, args, kwargs):
    """reshape, unsqueeze and squeeze differ only in how they name the target shape, and
    the backward of all three is the source shape, read here rather than inferred there."""
    x = _wrap(args[0])
    src = [int(d) for d in x.value.shape]
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            x.add_grad(ttnn.reshape(g, src))
        return bw

    return _tape(out_v, [x], make)


@_verb("permute")
def _v_permute(shipped, args, kwargs):
    x = _wrap(args[0])
    dims = [int(d) for d in (kwargs.get("dims") if len(args) < 2 else args[1])]
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            x.add_grad(ttnn.permute(g, inv))
        return bw

    return _tape(out_v, [x], make)


@_verb("transpose")
def _v_transpose(shipped, args, kwargs):
    """A transpose is its own inverse on the two axes it names, so the backward is the
    same call. It is NOT free on device -- it moves every byte -- but it is exact."""
    x = _wrap(args[0])
    d0 = kwargs.get("dim0", args[1] if len(args) > 1 else 0)
    d1 = kwargs.get("dim1", args[2] if len(args) > 2 else 1)
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            x.add_grad(ttnn.transpose(g, d0, d1))
        return bw

    return _tape(out_v, [x], make)


@_verb("concat")
def _v_concat(shipped, args, kwargs):
    xs = [_wrap(t) for t in args[0]]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
    ax = _axis(int(dim), len(xs[0].value.shape))
    sizes = [int(t.value.shape[ax]) for t in xs]
    kw = {k: v for k, v in kwargs.items() if k != "dim"}
    out_v = shipped([t.value for t in xs], dim=ax, **kw)

    def make():
        def bw(g):
            off = 0
            gs = [int(d) for d in g.shape]
            for t, n in zip(xs, sizes):
                if t.requires_grad:
                    starts, ends = [0] * len(gs), list(gs)
                    starts[ax], ends[ax] = off, off + n
                    t.add_grad(ttnn.slice(g, starts, ends))
                off += n
        return bw

    return _tape(out_v, xs, make)


def _sliced(x: "Tensor", out_v, starts, ends):
    """Tape one slice of ``x`` whose value is already computed.

    Shared by `slice`, `chunk` and `__getitem__`, which are the same op three ways: the
    backward pads the gradient back out with zeros on every axis that was cut. `chunk`
    reads its cut points off the shipped call rather than recomputing them, so the tape
    cannot disagree with the kernel about where the blocks begin.
    """
    shape = [int(d) for d in x.value.shape]

    def make():
        def bw(g):
            for ax in range(len(shape)):
                before, after = starts[ax], shape[ax] - ends[ax]
                if not before and not after:
                    continue

                def pad(n):
                    z = [int(d) for d in g.shape]
                    z[ax] = n
                    return ttnn.zeros(z, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                      device=g.device())

                parts = ([pad(before)] if before else []) + [g] + \
                        ([pad(after)] if after else [])
                g = ttnn.concat(parts, dim=ax)
            x.add_grad(g)
        return bw

    return _tape(out_v, [x], make)


@_verb("slice")
def _v_slice(shipped, args, kwargs):
    x = _wrap(args[0])
    starts = [int(v) for v in (kwargs.get("slice_start") or args[1])]
    ends = [int(v) for v in (kwargs.get("slice_end") or args[2])]
    ra, rk = _raw(args, kwargs)
    return _sliced(x, shipped(*ra, **rk), starts, ends)


@_verb("chunk")
def _v_chunk(shipped, args, kwargs):
    """n blocks along ``dim``. Each is a slice and each gets its own node, so a consumer
    that reads only some of the blocks -- which the row-chunked transition does, one
    block at a time -- contributes only those, and the fan-in sum does the rest."""
    x = _wrap(args[0])
    n = int(kwargs.get("chunks", args[1]))
    dim = int(kwargs.get("dim", args[2] if len(args) > 2 else 0))
    shape = [int(d) for d in x.value.shape]
    ax = _axis(dim, len(shape))
    outs, off, taped = shipped(x.value, n, dim=ax), 0, []
    for o in outs:
        m = int(o.shape[ax])
        starts, ends = [0] * len(shape), list(shape)
        starts[ax], ends[ax] = off, off + m
        taped.append(_sliced(x, o, starts, ends))
        off += m
    return taped


def _getitem(x: Tensor, index):
    """``x[...]`` on a taped tensor, in terms of `slice`. The shipped chain slices row
    blocks out of the pair tensor with exactly this syntax, and a taped tensor that did
    not answer to it would make the chunked paths -- which is all of the tuned ones --
    unreachable from a gradient."""
    shape = [int(d) for d in x.value.shape]
    idx = index if isinstance(index, tuple) else (index,)
    starts, ends = [0] * len(shape), list(shape)
    for ax, sl in enumerate(idx):
        if isinstance(sl, slice):
            s, e, step = sl.indices(shape[ax])
            if step != 1:
                raise NotImplementedError(f"strided slice on axis {ax} has no tape entry")
            starts[ax], ends[ax] = s, e
        elif isinstance(sl, int):
            s = sl + shape[ax] if sl < 0 else sl
            starts[ax], ends[ax] = s, s + 1
        else:
            raise NotImplementedError(f"index {sl!r} has no tape entry")
    return _sliced(x, ttnn.slice(x.value, starts, ends), starts, ends)


# --- placement and lifetime --------------------------------------------------------------

_VERBS["clone"] = _VERBS["reallocate"] = _VERBS["to_layout"] = \
    _VERBS["to_memory_config"] = lambda s, a, k: _identity_grad(s, a, k)
_VERBS["typecast"] = lambda s, a, k: _identity_grad(s, a, k, cast=True)


@_verb("deallocate")
def _v_deallocate(shipped, args, kwargs):
    """Inference frees an activation as soon as its consumer has read it. A backward reads
    it again, so the tape decides: `Tensor.free` releases what no closure can reach and
    refuses the rest. A raw handle never reaches here -- the wrapper below passes it
    straight to ttnn -- so nothing an inference run frees today stays live."""
    t = args[0]
    if isinstance(t, Tensor):
        t.free()
    else:
        shipped(*args, **kwargs)
    return _FREED


class _Freed:
    """`deallocate` returns None, and None is how a hook declines. This is neither."""
    __slots__ = ()
    def __repr__(self):                                                    # pragma: no cover
        return "<freed>"


_FREED = _Freed()


# --- the ttnn tt-bio sees while a tape is open --------------------------------------------

class _Ttnn:
    """ttnn, with the differentiable verbs taped. Every other attribute is ttnn's own.

    Lookups are cached into the instance dict on first use, so a hot call site pays one
    ordinary attribute load. A verb that never meets a taped tensor -- every config class,
    every device query -- costs one `_on_tape` scan of its arguments.
    """

    def __init__(self, real, prefix: str = ""):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        real = object.__getattribute__(self, "_real")
        attr = getattr(real, name)
        qual = object.__getattribute__(self, "_prefix") + name
        if isinstance(attr, type(ttnn)):
            out = _Ttnn(attr, qual + ".")
        elif callable(attr) and not isinstance(attr, type):
            out = _taped_verb(qual, attr)
        else:
            return attr
        object.__setattr__(self, name, out)
        return out


def _taped_verb(qual, shipped):
    impl = _VERBS.get(qual)

    def call(*args, **kwargs):
        if not _on_tape(args, kwargs):
            return shipped(*args, **kwargs)
        if impl is None:
            raise NotImplementedError(
                f"ttnn.{qual} has no tape entry, and it was handed a taped tensor. Add one "
                f"to tt_bio.autograd._VERBS -- unwrapping here would drop the gradient of "
                f"everything upstream of this call, silently.")
        out = impl(shipped, args, kwargs)
        return None if out is _FREED else out

    return call


_SHIM = _Ttnn(ttnn)
_SHIMMED: list = []


def _swap(to_shim: bool) -> None:
    """Rebind the name `ttnn` in every tt-bio module that holds one.

    Every module, rather than a named list, because the chain reaches nine of them today
    and a tenth added next week would otherwise be the one place the tape stops. `this`
    module is excluded: the backward closures above must call the real verbs or they would
    tape their own gradients.
    """
    global _SHIMMED
    if to_shim:
        _SHIMMED = []
        for name, mod in list(sys.modules.items()):
            if not name.startswith("tt_bio") or mod is None or name == __name__:
                continue
            if getattr(mod, "ttnn", None) is ttnn:
                mod.ttnn = _SHIM
                _SHIMMED.append(mod)
    else:
        for mod in _SHIMMED:
            mod.ttnn = ttnn
        _SHIMMED = []


@contextlib.contextmanager
def tape():
    """Make the shipped forward differentiable for the duration of the block.

        with tt_bio.autograd.tape():
            out = model(x)          # the production module, the production kernels
        out.backward()

    One line, and every knob the shipped path has is still reachable, because this changes
    no call site: the modules run the code they always ran and the tensors they are handed
    answer to everything a ttnn tensor answers to.

    Reentrant, and restores on the way out even if the forward raises. Modules imported
    INSIDE the block are not shimmed -- tt-bio imports its device modules at import time,
    so this only matters to a caller doing something unusual, and it fails loudly (a raw
    ttnn verb handed a taped tensor) rather than quietly.
    """
    if _SHIMMED:
        yield                       # already open; the outermost block owns the swap
        return
    prev = install()
    _swap(True)
    try:
        yield
    finally:
        _swap(False)
        from . import ops
        ops.set_grad_hook(prev)
