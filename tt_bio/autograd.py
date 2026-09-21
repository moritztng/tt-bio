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

import gc
import contextlib
import math
import sys
from typing import Optional, Sequence

import ttnn

from tt_bio.envflags import env_flag

__all__ = [
    "Tensor", "precise_config", "softmax_bw_inner", "no_grad", "parameter",
    "forget_parameters",
    "release_pins",
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


# `ttnn.softmax` does not return rows that sum to one. Measured over the OF3 trunk's own
# shapes (`perf/of3t_d116/rowsum.json`, float64-referenced): mean row sum 0.9934, rms
# deviation 1.20e-02, worst row 0.9506, and fp32 storage does not fix it (0.9954). So what
# the card computes is `c * softmax(x)` for a per-row `c`, and the vjp of THAT is
# `y * (g - sum(g*y)/sum(y))`. The plain rule `y * (g - sum(g*y))` is the vjp of a function
# the card did not evaluate.
#
# The cost is not in the rel_l2 of `dx`, which barely moves (2.0156e-02 against 2.0154e-02
# at the trunk shape, so an op-level audit is blind to this). It is that the plain rule
# leaves `dx` with a nonzero ROW SUM -- 4.62e-03 rms against the reference's 1.29e-16 --
# and the attention backward below it consumes exactly that: `dq_i = sum_j dx_ij k_j`
# equals `sum_j dx_ij (k_j - kbar)` only when the row sums vanish. When they do not, `dq`
# picks up `(row residual) * kbar`, a term the true gradient does not contain, worth 1.00x
# to 13.09x on `||dq||` as the common component of k grows (`perf/of3t_d116/amplify.json`).
#
# Off by default: it moves a gradient, so it is release-gated and `land-standing` owns the
# default. It is inside backward closures only, so no forward and no inference result can
# move whichever way the flag is set.
SOFTMAX_BW_RENORM = env_flag("TT_BIO_SOFTMAX_BW_RENORM", False)


def softmax_bw_inner(y, g, dim=-1, config=None):
    """`sum_j g_j y_j` for the softmax backward `dx = y * (g - inner)`, row-sum corrected.

    One helper for both callers -- `triangle_attention` below and
    `taped_ttnn._v_softmax` -- because the defect is the rule, not the site, and a repair
    applied to one of two identical expressions is the kind of half-fix that reads as fixed.
    """
    inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
    if not SOFTMAX_BW_RENORM:
        return inner
    return ttnn.divide(inner, ttnn.sum(y, dim=dim, keepdim=True,
                                       compute_kernel_config=config or precise_config()))


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

    __slots__ = ("value", "grad", "requires_grad", "node", "pinned", "evictable")

    def __init__(self, value, requires_grad: bool = False):
        self.value = value
        self.grad = None
        self.requires_grad = requires_grad
        self.node = None
        # Set by `_tape` the moment a closure is built that can read this value. It is the
        # whole of the lifetime rule: `free` refuses a pinned tensor and nothing else.
        self.pinned = False
        # Whether `free` may touch this value at all. Cleared in two cases. First, the few
        # ops whose backward reads their own OUTPUT -- relu, sigmoid, softmax, max --
        # because those closures hold the handle directly and deliberately do not hold the
        # `Tensor` (a closure that did would make the cycle out -> node -> fn -> out that
        # CPython cannot collect). Second, a tensor that SHARES STORAGE with another taped
        # tensor, which `_tape` detects.
        self.evictable = True

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
        # Resolved lazily: the slicing lives with the taped verb surface, and importing it
        # here would make `import tt_bio` pull the whole surface in for an inference run.
        from .taped_ttnn import _getitem
        return _getitem(self, index)

    def free(self) -> None:
        """Release the device buffer, unless a backward can still read it.

        This is what ``ttnn.deallocate`` becomes while a tape is open. Inference frees an
        activation the moment its consumer has read it; a backward reads it again, much
        later, and the crash that follows a premature free surfaces in an unrelated op
        several layers away. A leaf parameter is never freed either -- it is not an
        intermediate, and its gradient lands on it.
        """
        if not self.evictable:
            return
        if not self.pinned and not self.requires_grad:
            ttnn.deallocate(self.value)
        elif self.value.memory_config().buffer_type == ttnn.BufferType.L1:
            # EVICT rather than refuse. The tuned forward puts an activation in L1 and then
            # frees it the moment its consumer has read it, and the next kernel's circular
            # buffers are sized against the room that leaves. A tape that simply declines
            # the free keeps the room occupied and the next kernel cannot lay out: measured
            # on the shipped Transition at a [1,256,256,128] pair track, "Statically
            # allocated circular buffers in program 11 clash with L1 buffers ... L1 buffer
            # allocated at 692224 and static circular buffer region ends at 893440".
            #
            # The backward needs the VALUE; the forward's tuning needs the PLACE. Moving to
            # DRAM gives both, and it is the honest price of a gradient: one DRAM write per
            # L1-resident activation that inference does not pay.
            old = self.value
            self.value = ttnn.to_memory_config(old, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(old)

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
            if g is None:
                # On the tape and reachable from a root, but no gradient ever landed on it. A
                # multi-output segment makes this ordinary rather than exceptional: a
                # checkpointed `PairformerLayer` publishes both `s` and `z` as parents of the
                # next block, and an objective that seeds only the pair track leaves the single
                # track's node with nothing to propagate. Zero in, zero out -- running the
                # closure on a None is the crash, not the diagnosis.
                continue
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
    # A ttnn shape op can return a VIEW rather than a copy -- measured, `ttnn.reshape` of
    # (1,4,32,64) to (1,128,64) hands back the input's own buffer -- and the tape then
    # holds two `Tensor`s over one allocation. Freeing or evicting either kills both, and
    # the throw lands much later and somewhere else: in AttentionPairBias it surfaced as
    # "Buffer is not allocated" inside a sigmoid in the gate's backward, two ops
    # downstream of the reshape that caused it. Neither may be released.
    #
    # Checked on EVERY op, not only the differentiated ones: a view whose own gradient
    # nobody wants still shares storage with one that somebody does, and it is the view
    # that the shipped code deallocates.
    try:
        addr = out_value.buffer_address()
    except Exception:                                       # host tensor, or no buffer yet
        addr = None
    if addr is not None:
        for p in parents:
            try:
                shared = p.value.buffer_address() == addr
            except Exception:
                shared = False
            if shared:
                p.evictable = out.evictable = False
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


def _reduce_to(g, shape):
    """A broadcast operand's gradient: the output's, summed over the axes it was spread along.

    `_sum_leading` is the bias case -- reduce everything down to a trailing shape -- and it is
    wrong here, because the operands of a broadcasting binary op differ in a LEADING axis that
    is 1 on one side, not in rank alone. `ttnn` broadcasts an operand of shape (497, 128)
    against (1, 497, 128) happily and the backward then hands back a gradient of the output's
    shape, which `add_grad` refuses, correctly: a gradient that does not have its value's shape
    has been summed over the wrong thing or not at all. Same volume is a reshape; a genuinely
    stretched axis is a sum, at fp32 fidelity for `_sum_leading`'s reason.
    """
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
        return ttnn.reshape(g, ws)
    pad = [1] * (len(gs) - len(ws)) + ws
    out = g
    for ax in range(len(gs)):
        if pad[ax] == 1 and gs[ax] != 1:
            out = ttnn.sum(out, dim=ax, keepdim=True, compute_kernel_config=precise_config())
    return ttnn.reshape(out, ws)


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
            # `x.value`, not a captured handle: `Tensor.free` may have evicted this to
            # DRAM since the forward, and the captured handle would be freed storage.
            xv = x.value
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

    out = _tape(y, [x], make)
    out.evictable = False
    return out


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

    out = _tape(out_v, [x], make)
    out.evictable = False          # the closure above holds this handle directly
    return out


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid. Backward ``y * (1 - y)``, computed from the retained output."""
    y = ttnn.sigmoid(x.value)

    def make():
        def bw(g):
            x.add_grad(ttnn.multiply(g, ttnn.multiply(y, ttnn.rsub(y, 1.0))))
        return bw

    out = _tape(y, [x], make)
    out.evictable = False
    return out


def silu(x: Tensor) -> Tensor:
    """SiLU, ``x * sigmoid(x)``. Eight shipped linears fuse this activation into the packer.

    The backward needs the INPUT, not the output: ``y = x*s`` is not invertible, so unlike
    relu and sigmoid there is no way back from what was materialised. ``dy/dx = s*(1 + x*(1
    - s))``, and the sigmoid is retained rather than recomputed because it is the expensive
    half.
    """
    out_v = ttnn.multiply(x.value, ttnn.sigmoid(x.value))

    def make():
        def bw(g):
            # The sigmoid is RECOMPUTED, not retained, and the input is read through the
            # Tensor in case `free` evicted it to DRAM.
            #
            # Retaining it was the older choice, on the grounds that it is the expensive
            # half. It is, and it is still the wrong trade here, because what the retention
            # actually costs is L1: the shipped `Transition` asks for its swiglu operands in
            # L1 and sizes the next kernel's circular buffers against what that leaves, and
            # the tape composes this activation where production fuses it into the packer.
            # Measured at a [1,256,256,128] pair track with hidden 512, the retained sigmoid
            # plus the composed output is two L1 tensors production does not hold, and fc2
            # then throws "Statically allocated circular buffers in program 11 clash with L1
            # buffers ... allocated at 692224 and static circular buffer region ends at
            # 893440" on the FIRST row chunk. One sigmoid per backward buys the shape back.
            xv = x.value
            sig = ttnn.sigmoid(xv)
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
                       q_chunk: Optional[int] = None, config=None, value=None) -> Tensor:
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

    ``value`` hands in a forward that has already been computed, and it is what lets the
    shipped fused SDPA share this backward instead of getting a second copy of it. The
    backward below reads q, k, v and bias and recomputes the scores; it never reads the
    forward output, so supplying the output changes nothing about the gradient and saves
    computing the attention twice. ``tt_bio.taped_ttnn`` passes the fused kernel's result
    here, which is how the production forward and this backward end up in one node.
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
    for b0 in range(0, B, cB) if value is None else ():
        b1 = min(b0 + cB, B)
        row_blocks = []
        for i0 in range(0, n_q, cQ):
            i1 = min(i0 + cQ, n_q)
            p = _scores(q.value[b0:b1, :, i0:i1, :], b0, b1, i0, i1)
            row_blocks.append(ttnn.matmul(p, v.value[b0:b1], compute_kernel_config=cfg))
            ttnn.deallocate(p)
        out_blocks.append(row_blocks[0] if len(row_blocks) == 1
                          else ttnn.concat(row_blocks, dim=2))
    if value is not None:
        out_v = value
    else:
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
                    inner = softmax_bw_inner(p, dp, dim=-1, config=cfg)
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


# Every input a checkpointed segment pinned. The pin has to outlive the forward, because the
# recompute happens in the backward, so nothing in `checkpoint` itself can drop it -- and a pinned
# tensor is never released, so a run that does several forwards without a backward between them
# accumulates one whole segment input per block per forward. Measured: a finite-difference sweep
# at 384 aa (six extra forwards) left the next training step's backward OOM at 31.9 GB. The run
# owns the release, and `release_pins` is how it says so.
_CKPT_PINS: list = []


def release_pins() -> None:
    """Drop every pin a checkpointed segment took. Call after a backward, or after a forward
    whose tape is being discarded."""
    for t in _CKPT_PINS:
        t.pinned = False
    _CKPT_PINS.clear()


# Sentinel: "whichever registered parameters this segment turns out to read", resolved by the
# touch census around the untaped forward. A caller that names its own parameters still can.
_ALL_PARAMS = object()


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
    # PIN the inputs across the untaped forward. A shipped block deallocates the tensor it was
    # handed the moment it has read it -- that is what the tuned forward is -- and under
    # `no_grad` nothing on the tape objects, so the segment frees its own input and the
    # recompute later reads a dead buffer: "Buffer is not allocated", raised inside the
    # BACKWARD from a free issued in the forward. Pinning is exactly `free`'s existing
    # mechanism: a pinned tensor is evicted to DRAM rather than released, so the block's
    # tuning still gets its L1 place back and the recompute still gets its value. The pin
    # outlives the forward because the recompute happens in the backward.
    held = [t for t in inputs if isinstance(t, Tensor)]
    for t in held:
        t.pinned = True
        _CKPT_PINS.append(t)
    _TOUCHED.clear()
    with no_grad():
        produced = fn(*inputs)
    # Only the parameters this segment READ. Naming all of them would give every block a node,
    # including the ones that touch no trainable weight, and those recompute to an empty tape.
    touched = [t for k, t in _PARAMS.items() if k in _TOUCHED] if params is _ALL_PARAMS \
        else list(params)
    _TOUCHED.clear()
    parents = list(held) + touched

    def _recompute(k, g):
        """Re-run the segment on fresh nodes over the same input VALUES, seed output `k`."""
        from .taped_ttnn import recompute_scope
        inner = [Tensor(t.value, requires_grad=t.requires_grad) if isinstance(t, Tensor) else t
                 for t in inputs]
        with recompute_scope():
            y = fn(*inner)
        y = y[k] if k is not None else y
        if y.node is None:
            raise RuntimeError("checkpoint(fn): the recomputed segment built no tape; fn must "
                               "use taped ops and at least one input must require a gradient")
        y.backward(seed=g)
        for src, dup in zip(inputs, inner):
            if isinstance(src, Tensor) and isinstance(dup, Tensor) and dup.grad is not None:
                src.add_grad(dup.grad)
        # Drop the inner tape NOW, and collect, because refcounting will not. A node closure
        # that reads its own output makes the cycle out -> node -> fn -> out, which is
        # `hallgrad-tape-self-closure-leak` and is exactly what the ops whose backward reads
        # their output (relu, sigmoid, softmax, max) build. One leaked inner tape per block is
        # invisible; 96 of them -- two recomputes per block over 48 blocks -- is the difference
        # between a backward that peaks at 8.92 GiB and one that is refused 2.4 GB with 30.3 GiB
        # held. The collect is per segment, not per op, so it costs a few seconds over a trunk.
        del y, inner
        gc.collect()

    if not isinstance(produced, (tuple, list)):
        return _tape(produced.value if isinstance(produced, Tensor) else produced, parents,
                     lambda: (lambda g: _recompute(None, g)))

    # A segment with SEVERAL outputs, which is what a real block is: a `PairformerLayer`
    # returns the pair (s, z) and the single-output form cannot express it.
    #
    # One node per output, and each one recomputes the segment seeding ONLY itself. That costs
    # a recompute per output rather than per segment, and it is exact rather than approximate:
    # the derivative through a shared ancestor is the SUM over the output paths that reach it,
    # so two backward passes with one seed each add up to precisely what one pass with both
    # seeds would have produced. The alternative -- one node that waits for every output's
    # gradient before recomputing -- needs to know when the last one has arrived, and the
    # reverse-topological walk gives no such signal per tensor.
    outs = []
    for k, prod in enumerate(produced):
        if prod is None:
            outs.append(None)
            continue
        outs.append(_tape(prod.value if isinstance(prod, Tensor) else prod, parents,
                          (lambda k=k: (lambda g: _recompute(k, g)))))
    return tuple(outs)


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


# Wrappers built over raw ttnn handles the CALLER still owns, keyed by the handle's id.
# Cleared when the tape closes; the wrapper holds the handle, so the id cannot be reused
# while the entry is live.
_WRAPPED: dict = {}


# Weights a caller has declared TRAINABLE, keyed by the raw handle's id. Distinct from
# `_WRAPPED`, which is bookkeeping the tape does for itself and drops when the tape closes:
# this one is the user's parameter set and outlives a tape, because the optimizer holds it.
#
# It exists because the two seams did not meet. `tt_bio.train.lora.weights_for` censuses
# `ops.linear` sites, and the shipped pairformer routes none of its calls through `ops.linear`
# (LEDGER K3) -- it calls `ttnn.linear`, which `taped_ttnn` covers. But that surface
# short-circuits to the shipped op when no ARGUMENT is already a `Tensor`, and a weight the
# caller wants to train arrives as a raw handle among raw handles, so the short circuit fired
# first and the parameter was never seen. A model could therefore be fully differentiable and
# still have no trainable weight. Registering here is what makes a raw weight a leaf at all 331
# taped calls rather than at the four routed ones.
_PARAMS: dict = {}


def parameter(raw, requires_grad: bool = True):
    """Declare a device weight trainable. Idempotent per handle; returns the leaf.

    The leaf, not a copy: `_wrap` hands the same object to every call site that reads this
    weight, so a 48-block trunk sharing one tensor accumulates into one gradient.

    Pass the LEAF back, not its new value, after an optimizer step. `AdamW.step` replaces
    `t.value` with a fresh device tensor, so the registry is keyed on a handle that no longer
    exists and a bare `parameter(t.value)` would mint a SECOND leaf over the same weight --
    the tape would accumulate into the new one and the optimizer would keep stepping the old,
    which reads as a run whose second step has no gradients at all. Re-keying is what is
    wanted, and it is what passing the leaf does.
    """
    if isinstance(raw, Tensor):
        _PARAMS[id(raw.value)] = raw
        return raw
    t = _PARAMS.get(id(raw))
    if t is not None and t.value is raw:
        return t
    t = Tensor(raw, requires_grad=requires_grad)
    _PARAMS[id(raw)] = t
    return t


def forget_parameters() -> None:
    """Drop the trainable set. A run owns it; nothing here survives the run."""
    _PARAMS.clear()


# True only at the OUTERMOST taped call. A taped verb computes its value by calling the
# SHIPPED verb, and the shipped verb is sometimes itself a tt-bio function whose body calls
# `ttnn` -- `ops.linear`'s fallback is literally `ttnn.linear`, and inside `tt_bio.ops` that
# name is the shim. With a registered parameter among the operands, that re-entrant call would
# tape a SECOND time and hand the outer verb a `Tensor` where it expects a raw handle, so the
# outer `_tape` wraps a wrapper and the next op gets an `autograd.Tensor` as a pybind argument.
# Observed exactly once, as `ttnn.matmul(): incompatible function arguments ... invoked with
# (ttnn.Tensor, tt_bio.autograd.Tensor)` from the denoiser's atom decoder. Activations do not
# have this problem because the inner call sees them already unwrapped; a parameter is
# recognised by IDENTITY of a raw handle, which unwrapping cannot hide. So the lookup is
# switched off for the duration of a verb, and `_wrap` is deliberately NOT gated: the outer
# verb still resolves the parameter to its leaf, which is the whole point.
_PARAM_SCAN = True


class _no_param_scan:
    """Suppress parameter lookup for the duration of one taped verb's shipped call."""

    def __enter__(self):
        global _PARAM_SCAN
        self._prev = _PARAM_SCAN
        _PARAM_SCAN = False

    def __exit__(self, *exc):
        global _PARAM_SCAN
        _PARAM_SCAN = self._prev
        return False


# Which parameters a stretch of forward actually read. `checkpoint` needs it: a 48-block trunk
# where only the last block's weights are trainable would otherwise name every registered
# parameter as a parent of every block, so every block gets a node, and 47 of those nodes
# recompute a segment that touches no parameter and builds no tape.
_TOUCHED: set = set()


def _param(v):
    t = _PARAMS.get(id(v))
    if t is not None and t.value is v:
        _TOUCHED.add(id(v))
        return t
    return None


def _param_on_tape(v):
    return _param(v) if _PARAM_SCAN else None


def _wrap(t):
    """A raw ttnn tensor joins the tape as an untracked leaf; a `Tensor` passes through.

    The wrapper is REMEMBERED, and that is not bookkeeping for its own sake. A raw handle
    can reach a taped op as one operand among taped ones -- the shipped attention does it
    twice, `batched_matmul(probs, v)` at `tenstorrent.py:8209` and the gate multiply at
    `:8290`, where the pair track is taped and the single track is not. The tape wraps and
    pins the raw operand, but the pin is invisible to the caller, which still holds the raw
    handle and deallocates it three lines later. The buffer goes, and the throw arrives
    much later inside the backward: "Buffer is not allocated" in a sigmoid, in a module
    whose forward completed cleanly. `deallocate` consults this map so a raw handle the
    tape has wrapped is routed through `Tensor.free` and respects the pin.
    """
    if t is None or isinstance(t, Tensor):
        return t
    p = _param(t)
    if p is not None:
        return p
    w = _WRAPPED.get(id(t))
    if w is not None and w.value is t:
        # Idempotent per handle. Wrapping the same raw tensor twice would give the tape a
        # pinned parent and the caller's deallocate a DIFFERENT, unpinned wrapper over the
        # same buffer, and the unpinned one takes the release branch.
        return w
    w = Tensor(t)
    _WRAPPED[id(t)] = w
    return w


def wrapper_for(raw):
    """The `Tensor` the tape built over this raw handle, if it built one."""
    w = _WRAPPED.get(id(raw))
    return w if w is not None and w.value is raw else None


def forget_wrappers() -> None:
    """Drop the raw-handle map. Called when a tape closes; nothing survives it."""
    _WRAPPED.clear()


def _unwrap(t):
    return t.value if isinstance(t, Tensor) else t


def _walk(args, kwargs):
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, (list, tuple)):
            yield from v
        else:
            yield v


def _on_tape(args, kwargs):
    # A registered parameter counts, even though it arrives as a raw handle: it is the first
    # taped thing in a forward whose inputs are all data, and without it the short circuit
    # below reaches the shipped op and the whole downstream chain is never taped.
    return any(isinstance(v, Tensor) or _param_on_tape(v) is not None
               for v in _walk(args, kwargs))


def _differentiating(args, kwargs):
    if not _GRAD_ENABLED:
        return False
    for v in _walk(args, kwargs):
        if isinstance(v, Tensor):
            if v.requires_grad:
                return True
        else:
            p = _param_on_tape(v)
            if p is not None and p.requires_grad:
                return True
    return False


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
                # `_reduce_to` because a matmul normalises rank: an x of (1, N, N, c)
                # comes back as (N, N, c) and `add_grad` refuses a gradient that is not
                # its value's shape, correctly.
                x.add_grad(_reduce_to(ttnn.matmul(g, w.value, transpose_b=True,
                                                  compute_kernel_config=bwcfg),
                                      x.value.shape))
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
                w.add_grad(_reduce_to(
                    ttnn.matmul(_flat2d(x.value), _flat2d(g), transpose_a=True,
                                compute_kernel_config=bwcfg, dtype=ttnn.float32),
                    w.value.shape))
            if bias is not None and bias.requires_grad:
                bias.add_grad(_sum_leading(g, bias.value.shape))
        return bw

    out = _tape(out_v, parents, make)
    if act is None:
        return out
    activated = act(out)
    # The pre-activation is now read only by the activation's own backward, which reads it
    # through its `Tensor`. Production never materialises it at all -- `ttnn.linear` fuses
    # the activation into the packer -- so leaving it in the L1 the site asked for is the
    # tape holding a tensor the tuning did not budget. Hand its PLACE back; `free` evicts
    # to DRAM rather than releasing it, because the backward still needs the value.
    out.free()
    return activated


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
            # `x.value`, not a captured handle: `Tensor.free` may have evicted this to
            # DRAM since the forward, and the captured handle would be freed storage.
            xv = x.value
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
        with _no_param_scan():
            return Tensor(shipped(*ra, **rk))
    impl = _TAPED.get(name)
    if impl is None:
        raise NotImplementedError(
            f"tt_bio.ops.{name} has no taped implementation. Add one to "
            f"tt_bio.autograd._TAPED; declining here would silently drop the gradient.")
    with _no_param_scan():
        return impl(shipped, args, kwargs)


def _checkpoint_segment(fn, *inputs):
    """`ops.checkpoint_segment`'s implementation: checkpoint a block, or just run it.

    Run it plainly whenever a tape could not use the trade -- grad off, nothing on the tape and
    no registered parameter. Under a tape it is not an optimisation that buys depth, it is what
    makes the thing fit: at 384 aa the composed backward exhausted all 34.23 GB of the card
    without it, and `ptx-crop` measured the same trunk WITH per-block checkpointing at 7.762 GB.
    """
    if not _GRAD_ENABLED:
        return fn(*inputs)
    live = [t for t in inputs if isinstance(t, Tensor) and t.requires_grad]
    if not live and not _PARAMS:
        return fn(*inputs)
    return checkpoint(fn, *inputs, params=_ALL_PARAMS)


def install():
    """Route `tt_bio.ops` through the tape. Idempotent; returns the hook it replaced.

    This is the narrow seam -- two verbs. `tape()` is the whole shipped forward.
    """
    from . import ops
    # A recycling model asks `ops.recycle_region` whether a non-final cycle is differentiated.
    # Installed together with the verb hook because the two are the same opt-in.
    ops.set_recycle_hook(no_grad)
    ops.set_checkpoint_hook(_checkpoint_segment)
    return ops.set_grad_hook(_hook)


def uninstall() -> None:
    """Put the inference path back. Idempotent."""
    from . import ops
    ops.set_recycle_hook(None)
    ops.set_checkpoint_hook(None)
    ops.set_grad_hook(None)


def installed() -> bool:
    from . import ops
    return ops.grad_hook() is _hook


# The taped ttnn surface lives in `taped_ttnn.py`, not here: `state/ptx/DESIGN.md` §6 puts
# this file under `ptx-unify` and the surface under `ptx-fastpath`, and they are genuinely
# different things -- this module is the tape, that one is how the shipped modules reach it.
# Re-exported because `tt_bio.autograd.tape()` is the one entry point a caller should need.
def tape():
    """Make the shipped forward differentiable for the duration of the block.

        with tt_bio.autograd.tape():
            out = model(x)          # the production module, the production kernels
        out.backward()

    See `tt_bio.taped_ttnn.tape` for what it does and what it deliberately does not.
    """
    from .taped_ttnn import tape as _tape_cm
    return _tape_cm()

