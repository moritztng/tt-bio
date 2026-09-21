"""ttnn as tt-bio's own modules see it while a tape is open.

`tt_bio/ops.py` is the narrow seam: two verbs, dispatched at the call site, with the grad
hook `tt_bio.autograd` installs. This module is the wide one, and it exists because the
Pairformer chain calls 380 ttnn verbs across 56 names (`perf/ptx_fastpath/census.py`).
Routing all of them through `ops` by hand would mean editing the most heavily tuned file in
the repo in 380 places, and then keeping every future perf lever in step with the tape by
hand. It also puts the gradient in the call site rather than in the op, and the call site is
what goes stale: a lever landed next week in the trimul would take the gradient with it,
silently.

So `tape()` rebinds the module-global name `ttnn` inside tt-bio's own modules to the proxy
below, which is ttnn with the differentiable verbs taped and every other attribute ttnn's
own. One rebinding per module, restored on exit even if the forward raises, live only while
a tape is open. It is not a monkeypatch of ttnn's namespace: nothing outside tt-bio sees a
different ttnn, and `tt_bio.autograd` is excluded so its backward closures call the real
verbs and cannot re-enter the tape.

The mechanism is `dispatch.OpSurface`'s, one level out. A taped verb computes its value BY
CALLING the shipped verb on unwrapped operands, so there is exactly one forward and the
grad-off column is bit-identical rather than nearly equal. The tuned paths are reached the
same way they always were, because the tuning reads shapes and memory configs off the
tensor and `autograd.Tensor` answers those.

A verb with no entry that is handed a taped tensor raises. That is deliberate: an op the
tape cannot follow is loud, not a silently dropped gradient.
"""

from __future__ import annotations

import contextlib
import sys

import ttnn

from . import autograd as ag
from .autograd import Tensor, precise_config
from .autograd import (_axis, _differentiating, _flat2d, _on_tape, _raw,
                       _reduce_to, _sum_leading, _tape,
                       _taped_layer_norm, _taped_linear, _unwrap, _wrap)

__all__ = ["tape", "recompute_scope", "VERBS", "taped_ttnn"]

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
VERBS = _VERBS


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
    kw = dict(kwargs)
    # `ttnn.matmul` takes its operands positionally; `experimental.minimal_matmul` names
    # them, and the shipped QKV and gate projections call it that way. One entry covers
    # both rather than two entries drifting.
    a = _wrap(args[0] if args else kw.get("input_tensor"))
    b = _wrap(args[1] if len(args) > 1 else kw.get("weight_tensor"))
    bias = _wrap(kw.get("bias_tensor"))
    ta = bool(kw.get("transpose_a", False))
    tb = bool(kw.get("transpose_b", False))
    if kw.get("activation") is not None:
        raise NotImplementedError(
            f"tt_bio.autograd has no backward for the fused activation "
            f"{kw['activation']!r} on a matmul. `ops.linear` composes its activation on the "
            f"tape; do the same here rather than dropping it.")
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
                # The weight reduces over every token, so it is `_flat2d`'s DRAM-normalised
                # long-K path and fp32 out, for the reason documented there.
                # A 2-D right operand is a WEIGHT, and only then does the contraction
                # run over every leading axis. A batched matmul whose right operand is
                # another activation contracts over the batch as well, and `_flat2d`
                # would fold that away: measured as a (64, 32) gradient arriving at a
                # (1, 4, 64, 32) value in AttentionPairBias.
                if (not ta and not tb and len(a.value.shape) > 2
                        and len(b.value.shape) == 2):
                    b.add_grad(ttnn.matmul(_flat2d(a.value), _flat2d(g), transpose_a=True,
                                           compute_kernel_config=cfg, dtype=ttnn.float32))
                else:
                    b.add_grad(ttnn.matmul(a.value, g, transpose_a=not ta,
                                           compute_kernel_config=cfg) if not tb else
                               ttnn.matmul(g, a.value, transpose_a=True, transpose_b=ta,
                                           compute_kernel_config=cfg))
            if bias is not None and bias.requires_grad:
                bias.add_grad(_sum_leading(g, bias.value.shape))
        return bw

    return _tape(out_v, [p for p in (a, b, bias) if p is not None], make)


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
            # The same expression as `autograd.softmax` and `triangle_attention`; the helper
            # carries the TT_BIO_SOFTMAX_BW_RENORM branch all three used to inline.
            inner = ag.softmax_bw_inner(y, g, dim=dim)
            x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
        return bw

    out = _tape(y, [x], make)
    out.evictable = False          # the closure holds y, this tensor's own value
    return out


def _unary(fn, reads_output=False):
    """Register a unary eltwise verb whose backward is `fn(x_value, out_value) -> dy/dx`.

    An `output_tensor=` argument is dropped, which is the in-place case: the shipped verb
    writes its result over its input and the tape needs the input to still be there.
    """
    def impl(shipped, args, kwargs):
        x = _wrap(args[0])
        inplace = kwargs.get("output_tensor") is not None
        kw = {k: v for k, v in kwargs.items() if k != "output_tensor"}
        out_v = shipped(x.value, *[_unwrap(a) for a in args[1:]], **kw)
        def make():
            def bw(g):
                # Through the Tensor: `free` may have evicted the input to DRAM.
                x.add_grad(ttnn.multiply(g, fn(x.value, out_v)))
            return bw

        out = _tape(out_v, [x], make)
        if reads_output:
            out.evictable = False
        elif inplace:
            # `ttnn.silu(x, output_tensor=x)` is the unary form of the same signal, and
            # the same ordering rule applies: after `_tape`, so `free` can see that a
            # closure reads this and evict instead of releasing. Only safe when the rule
            # reads the INPUT; a rule that reads the output would be reading this tensor.
            x.free()
        return out
    return impl


_VERBS["silu"] = _unary(
    # sigma * (1 + x * (1 - sigma)). The output is not invertible, so the input is read.
    lambda xv, y: (lambda s: ttnn.multiply(
        s, ttnn.add(ttnn.multiply(xv, ttnn.rsub(s, 1.0)), 1.0)))(ttnn.sigmoid(xv)))
_VERBS["sigmoid"] = _unary(lambda xv, y: ttnn.multiply(y, ttnn.rsub(y, 1.0)),
                           reads_output=True)
_VERBS["relu"] = _unary(lambda xv, y: ttnn.gtz(xv))
_VERBS["exp"] = _unary(lambda xv, y: y, reads_output=True)


# A fused eltwise activation, and its derivative from whichever of (input, output) is
# cheaper. `ttnn.multiply(a, b, input_tensor_b_activations=[SIGMOID])` applies the unary to
# the operand BEFORE the binary op, and a tape that forwards the keyword to the shipped verb
# but ignores it in the backward gets a correct forward and a silently wrong gradient. That
# is exactly what happened: the shipped `TriangleAttention` gate is
# `ttnn.multiply_(o, g, input_tensor_b_activations=[SIGMOID])` at `tenstorrent.py:7532`, and
# the composed module's dL/dz came out 4.88x too large -- flat across a 50x eps sweep, so
# not noise. Every op-scale check in this directory passed while that was true, because none
# of them composes a gate. `perf/ptx_fastpath/modulecheck.py` is what found it.
_FUSED_UNARY = {}


def _register_fused_unary():
    u = getattr(ttnn, "UnaryOpType", None)
    if u is None:                                                  # pragma: no cover
        return
    if hasattr(u, "SIGMOID"):
        _FUSED_UNARY[u.SIGMOID] = (
            ttnn.sigmoid, lambda x, y: ttnn.multiply(y, ttnn.rsub(y, 1.0)))
    if hasattr(u, "RELU"):
        _FUSED_UNARY[u.RELU] = (ttnn.relu, lambda x, y: ttnn.gtz(x))
    if hasattr(u, "SILU"):
        _FUSED_UNARY[u.SILU] = (
            ttnn.silu,
            lambda x, y: (lambda sg: ttnn.multiply(
                sg, ttnn.add(ttnn.multiply(x, ttnn.rsub(sg, 1.0)), 1.0)))(ttnn.sigmoid(x)))


_register_fused_unary()


def _activation(kwargs, key):
    """The single fused unary on one operand, or None. Anything unmodelled raises.

    Declining loudly is the whole point. The forward would be right either way, because it
    comes from the shipped verb; it is the backward that would quietly differentiate a
    different function.
    """
    acts = kwargs.get(key) or ()
    acts = list(acts)
    if not acts:
        return None
    if len(acts) > 1:
        raise NotImplementedError(
            f"tt_bio.autograd tapes one fused activation per operand; {key} has {len(acts)}.")
    op = acts[0]
    if op not in _FUSED_UNARY:
        raise NotImplementedError(
            f"tt_bio.autograd has no backward for the fused activation {op!r} on {key}. "
            f"Add it to tt_bio.taped_ttnn._FUSED_UNARY -- forwarding it to the shipped verb "
            f"and ignoring it here gives a correct forward and a wrong gradient, which is "
            f"how the TriangleAttention gate read 4.88x high.")
    return _FUSED_UNARY[op]


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
        inplace = out_of_place is not None
        if inplace:
            shipped = out_of_place
        if not isinstance(raw_b, (Tensor, ttnn.Tensor)):
            _activation(kwargs, "input_tensor_a_activations")      # raises if unmodelled
            _activation(kwargs, "input_tensor_b_activations")
            if kwargs.get("input_tensor_a_activations") or kwargs.get(
                    "input_tensor_b_activations"):
                raise NotImplementedError(
                    "tt_bio.autograd does not tape a fused activation on a scalar operand")
            f = float(raw_b)
            out_v = shipped(a.value, raw_b, **kw)

            def make_s():
                def bw(g):
                    a.add_grad(scalar(g, f))
                return bw

            return _tape(out_v, [a], make_s)
        b = _wrap(raw_b)
        fa = _activation(kwargs, "input_tensor_a_activations")
        fb = _activation(kwargs, "input_tensor_b_activations")
        out_v = shipped(a.value, b.value, **kw)
        def make():
            def bw(g):
                # Read through the Tensors: either operand may have been evicted to DRAM
                # by a `deallocate` the tape declined between here and the backward.
                av, bv = a.value, b.value
                # The binary op sees the ACTIVATED operands, so the binary rule is
                # evaluated on those and the unary derivative is applied after. Recomputed
                # rather than retained: the fused forward never materialised them, and
                # holding two extra tensors is what the L1 budget cannot take.
                ea = fa[0](av) if fa else av
                eb = fb[0](bv) if fb else bv
                if a.requires_grad:
                    da = grad_a(g, ea, eb)
                    da = ttnn.multiply(da, fa[1](av, ea)) if fa else da
                    a.add_grad(_reduce_to(da, av.shape))
                if b.requires_grad:
                    db = grad_b(g, ea, eb)
                    db = ttnn.multiply(db, fb[1](bv, eb)) if fb else db
                    b.add_grad(_reduce_to(db, bv.shape))
            return bw

        out = _tape(out_v, [a, b], make)
        if inplace:
            # An in-place verb is a free of its destination plus a write, and the tuned
            # forward budgets the next kernel's L1 against that free. Taping it out of
            # place gives the result a new home and leaves the old one with no owner to
            # release it -- one stranded L1 tensor per residual, per chunk. Measured on
            # the shipped Transition at a [1,256,256,128] pair track: the row loop leaks
            # `x_1` on every chunk, because `multiply_` is what consumed it in inference
            # and nothing else deallocates it, and fc1 then cannot lay out its circular
            # buffers. So the verb's own signal is honoured: the destination's PLACE is
            # released, and `free` evicts rather than frees if a backward reads it.
            #
            # AFTER `_tape`, never before. `free` reads `pinned` to choose between
            # releasing and evicting, and `_tape` is what sets it, so releasing first
            # deallocates an operand the node is about to read. The throw lands far
            # away -- "TT_THROW @ ttnn/core/tensor/storage.cpp:60", in
            # AttentionPairBias's backward, from a free issued in the forward.
            a.free()
            # And the caller MUST take the return value. `ttnn.add_(a, b)` returns its
            # destination, so inference reads the same object whether or not it is rebound --
            # but under the tape the result has a NEW home and `a.free()` above has just
            # released the old one, so a call site that discards the return is left holding a
            # freed buffer and throws "Buffer is not allocated" on its next use. Five sites in
            # this tree discarded it and all five now rebind: `PairWeightedAveraging`'s head
            # accumulator, the attention output accumulator, `OuterProductMean`'s row
            # accumulator, and two MSA residuals in `openfold3_msa_embedder`. It is a
            # one-token change with no inference effect and there is no way to make the
            # discarding form work: the tape cannot write into a buffer whose size and
            # lifetime the backward still needs.
        return out
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
    #  must not be evaluated when  arrives as a keyword, which is how
    # the trimul calls it (tenstorrent.py:6659).
    n = int(kwargs["chunks"] if "chunks" in kwargs else args[1])
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


# --- attention ---------------------------------------------------------------------------

# One score block, in bytes, that the backward is allowed to hold while it recomputes.
# The object being bounded is concrete: at 800 aa with 8 heads the full [S, H, S, S] bf16
# score tensor is 8.19 GB on a 34.23 GB card, and the backward needs one while the whole
# forward tape is already resident. A leading-axis chunk of 1 is 10.24 MB.
SDPA_SCORE_BUDGET = 256 << 20


def _sdpa_chunking(B, H, n_q, n_k, itemsize, budget=None):
    """Pick (leading, query) chunk bounds so one recomputed score block fits `budget`.

    Gradients are invariant to both bounds -- `perf/hallgrad/gradcheck.py --cases
    triatt_chunked` checks that by differencing two chunkings rather than trusting one --
    so this is a memory decision and not a numeric one, and it is derived from the shape
    rather than tuned. The leading axis is spent first because chunking it costs nothing:
    each of the B attention problems is independent, so a leading chunk's softmax is
    already exact and complete and needs no running row statistics.
    """
    budget = SDPA_SCORE_BUDGET if budget is None else budget
    row = H * n_k * itemsize
    if row * n_q <= budget:
        per = max(1, min(B, budget // (row * n_q)))
        return (B if per >= B else per), None
    return 1, max(1, min(n_q, budget // row))


@_verb("transformer.scaled_dot_product_attention")
def _v_sdpa(shipped, args, kwargs):
    """The shipped fused SDPA, with the chunked-recompute backward behind it.

    The forward is the production kernel -- every rung of `_tri_att_sdpa`'s ladder, its
    program config, its L1 routing -- called here and handed to `autograd.triangle_attention`
    as its value. There is no second attention implementation: that function's backward
    reads q, k, v and bias and recomputes the scores per chunk, and never reads the forward
    output, so the production forward and the verified backward compose into one node.

    This entry is the single shared implementation `ptx-diffusion` consumes as well
    (`state/ptx/LEDGER.md` K17). It is registered on the verb rather than in a model file,
    so `AttentionPairBias` in the pairformer and the denoiser's attention sites reach the
    same code without either of them naming it.

    `is_causal` is refused rather than approximated. No tt-bio caller sets it, and a causal
    mask the backward did not apply would be a wrong gradient that the forward agrees with.

    THE MASK IS ADDED BEFORE THE SCALE, and getting that backwards is a wrong gradient the
    forward agrees with, which is the failure mode this row exists to prevent. Measured on
    a p300c: the fused kernel computes ``softmax((q @ k^T + mask) * scale) @ v``, not
    ``softmax((q @ k^T) * scale + mask) @ v``. Against the first the error sits flat at
    3.3e-02 for every mask magnitude tested (0, 0.05, 0.5, 2.0), which is the fused
    softmax's own known deficit; against the second it grows with the mask -- 3.3e-02,
    5.1e-02, 3.7e-01, 8.3e-01. Production already knows this and compensates:
    `tenstorrent.py:7205` sets `self.scale = head_dim ** 0.5`, pre-multiplies
    `bias_weight` by it, and passes `scale ** -1` to the kernel, so the pair bias arrives
    pre-scaled and the composite is standard attention.

    `autograd.triangle_attention` uses the other convention -- it scales the scores and
    then adds the bias -- and it is not being changed, because it is a correct op with its
    own verified gradients and another campaign owns that file. Instead the mask is scaled
    ON THE TAPE here, with `autograd.scale`, so the composite the tape differentiates is
    the composite the kernel computes and the chain rule back to the caller's unscaled
    mask is the tape's own rather than a factor written out by hand.
    """
    q, k, v = (_wrap(a) for a in args[:3])
    kw = dict(kwargs)
    bias = _wrap(kw.get("attn_mask") if "attn_mask" in kw else
                 (args[3] if len(args) > 3 else None))
    if kw.get("is_causal"):
        raise NotImplementedError(
            "tt_bio.autograd has no backward for a causal SDPA. No tt-bio caller sets "
            "is_causal, and applying the mask in the forward but not the backward would be "
            "a wrong gradient the forward agrees with.")
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)
    qs = [int(d) for d in q.value.shape]
    B, H, n_q, head_dim = qs
    n_k = int(k.value.shape[2])
    scale = kw.get("scale")
    if scale is None:
        scale = head_dim ** -0.5
    if bias is not None:
        bias = ag.scale(bias, scale)
    itemsize = 4 if q.value.dtype == ttnn.float32 else 2
    cB, cQ = _sdpa_chunking(B, H, n_q, n_k, itemsize)
    return ag.triangle_attention(q, k, v, bias, scale=scale, chunk=cB, q_chunk=cQ,
                                 value=out_v)


# --- attention head packing ------------------------------------------------------------
# Both layouts below are DERIVED FROM THE DEVICE with an index-valued tensor
# (`perf/ptx_fastpath/heads.py`), not read off a docstring. A head-split backward written
# from a guess about the packing is exactly how a wrong gradient ships: it is a pure
# rearrangement, so nothing about its magnitude looks wrong, and every downstream head
# would train on another head's signal.

@_verb("experimental.nlp_concat_heads")
def _v_concat_heads(shipped, args, kwargs):
    """``[B, H, L, dh] -> [B, 1, L, H*dh]``, measured equal to
    ``permute(0, 2, 1, 3).reshape(B, 1, L, H*dh)``. The backward is that inverted."""
    x = _wrap(args[0])
    B, H, L, dh = (int(d) for d in x.value.shape)
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            x.add_grad(ttnn.permute(ttnn.reshape(g, [B, L, H, dh]), [0, 2, 1, 3]))
        return bw

    return _tape(out_v, [x], make)


@_verb("experimental.nlp_create_qkv_heads")
def _v_create_qkv_heads(shipped, args, kwargs):
    """``[B, 1, L, 3*H*dh] -> three [B, H, L, dh]``, measured equal to
    ``reshape(B, 1, L, 3, H, dh)[:, 0, :, s].permute(0, 2, 1, 3)`` for s in 0, 1, 2.

    Each output gets its own node, so a consumer that differentiates only q -- which the
    tape cannot know in advance -- contributes only q. Each scatters into the full packed
    width with zeros in the other two slots and `add_grad` sums them, which costs two
    packed-width temporaries more than a single shared closure would. It is the price of
    not having to know the fan-out at forward time, and the packed tensor is the smallest
    thing in the block.
    """
    x = _wrap(args[0])
    B, _, L, wide = (int(d) for d in x.value.shape)
    H = int(kwargs.get("num_heads", 1))
    dh = wide // (3 * H)
    outs = shipped(x.value, *[_unwrap(a) for a in args[1:]],
                   **{k: _unwrap(v) for k, v in kwargs.items()})

    def slot(s):
        def make():
            def bw(g):
                # [B, H, L, dh] -> [B, L, 1, H*dh], then into slot s of the packed axis.
                rows = ttnn.reshape(ttnn.permute(g, [0, 2, 1, 3]), [B, L, 1, H * dh])
                # One zero tensor for both empty slots, then one concat. `ttnn.pad` would be
                # the single-allocation form and cannot be used: it refuses front padding
                # (`pad.cpp:278 front_padding_is_zero`), so slots 1 and 2 have no pad
                # expression. The packed width is 2,415,919,104 B at a 384-token pair track,
                # which is why this op is where the backward runs out of card.
                zero = ttnn.zeros([B, L, 1, H * dh], dtype=rows.dtype,
                                  layout=ttnn.TILE_LAYOUT, device=rows.device())
                parts = [rows if i == s else zero for i in range(3)]
                x.add_grad(ttnn.reshape(ttnn.concat(parts, dim=2), [B, 1, L, 3 * H * dh]))
            return bw
        return make

    return tuple(_tape(o, [x], slot(s)) for s, o in enumerate(outs))


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
    if not isinstance(t, Tensor):
        # A raw handle the tape has wrapped is still one the tape may be reading; one it
        # has never seen is nobody's activation and goes straight through.
        t = ag.wrapper_for(t) or t
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


# Verbs that must reach their tape entry even when no argument is a `Tensor`. There is
# exactly one, and the reason is the raw-handle map: a raw ttnn tensor can be an operand of
# a taped op -- the shipped attention does it at `tenstorrent.py:8209` and `:8290`, where
# the pair track is taped and the single track is not -- and the tape wraps and pins it.
# The caller then deallocates the RAW handle, whose argument list contains no `Tensor` at
# all, so an `_on_tape` short circuit hands it straight to ttnn and the pin never gets a
# say. The buffer goes and the throw arrives later, in the backward, in a different module.
_ALWAYS_DISPATCH = frozenset({"deallocate"})


def _taped_verb(qual, shipped):
    impl = _VERBS.get(qual)
    always = qual in _ALWAYS_DISPATCH

    def call(*args, **kwargs):
        if not always and not _on_tape(args, kwargs):
            return shipped(*args, **kwargs)
        if impl is None:
            raise NotImplementedError(
                f"ttnn.{qual} has no tape entry, and it was handed a taped tensor. Add one "
                f"to tt_bio.autograd._VERBS -- unwrapping here would drop the gradient of "
                f"everything upstream of this call, silently.")
        with ag._no_param_scan():
            out = impl(shipped, args, kwargs)
        return None if out is _FREED else out

    return call


_SHIM = _Ttnn(ttnn)
_SHIMMED: list = []
_NEVER_SHIM = (__name__, "tt_bio.autograd")


def taped_ttnn():
    """The proxy itself, for a caller outside `tt_bio` that wants to address it directly --
    a verification script, mostly. A shipped module never needs this; it writes `ttnn.`."""
    return _SHIM


def _swap(to_shim: bool) -> None:
    """Rebind the name `ttnn` in every tt-bio module that holds one.

    Every module, rather than a named list, because the chain reaches nine of them today
    and a tenth added next week would otherwise be the one place the tape stops. Two are
    excluded: this one, and `tt_bio.autograd`, whose backward closures must call the real
    verbs or they would tape their own gradients.
    """
    global _SHIMMED
    if to_shim:
        _SHIMMED = []
        for name, mod in list(sys.modules.items()):
            if not name.startswith("tt_bio") or mod is None or name in _NEVER_SHIM:
                continue
            if getattr(mod, "ttnn", None) is ttnn:
                mod.ttnn = _SHIM
                _SHIMMED.append(mod)
    else:
        for mod in _SHIMMED:
            mod.ttnn = ttnn
        _SHIMMED = []


@contextlib.contextmanager
def recompute_scope():
    """Make the shipped modules taped again for a recomputation inside a BACKWARD.

    `tape()` is a forward-time context: it swaps the shim in, and on the way out it forgets the
    raw-handle wrappers and puts the grad hook back. A checkpointed segment recomputes itself
    from inside `backward`, which the documented usage runs AFTER the tape block has closed --
    so the shipped module is looking at the real `ttnn` again and hands it an `autograd.Tensor`,
    which pybind refuses. This is the narrower thing that recompute needs: swap the shim in if
    it is not already in, put it back exactly as found, and touch neither the wrapper map nor
    the hook, because the backward that is running owns both.
    """
    if _SHIMMED:
        yield
        return
    from . import ops
    # The hook as well as the shim. `ops.linear` and `ops.layer_norm` tape through the hook,
    # not the shim, so a recompute without it would rebuild the segment with those sites
    # untaped and produce a partial gradient rather than an error. And `ops.taping()` -- which
    # is how nine fused kernels decide to decline, none of which has a backward -- is defined
    # as the hook being installed, so without it the recompute walks straight into
    # `generic_op has no tape entry`.
    prev = ops.grad_hook()
    if prev is None:
        ops.set_grad_hook(ag._hook)
    _swap(True)
    try:
        yield
    finally:
        _swap(False)
        if prev is None:
            ops.set_grad_hook(None)


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
    prev = ag.install()
    _swap(True)
    try:
        yield
    finally:
        _swap(False)
        ag.forget_wrappers()
        from . import ops
        ops.set_grad_hook(prev)


# --- padding and reductions ---------------------------------------------------------------

@_verb("pad")
def _v_pad(shipped, args, kwargs):
    """The backward is the slice this op is the inverse of, and the pad VALUE is a constant
    so it carries no gradient of its own. Every tt-bio caller passes per-dimension
    (before, after) pairs, which is the form read here; a bare-int form would land on a
    different ttnn overload and is refused rather than guessed."""
    x = _wrap(args[0])
    padding = kwargs.get("padding", args[1] if len(args) > 1 else None)
    pairs = [(int(a), int(b)) for a, b in padding]
    shape = [int(d) for d in x.value.shape]
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            starts = [b for b, _ in pairs]
            ends = [b + n for (b, _), n in zip(pairs, shape)]
            x.add_grad(ttnn.slice(g, starts, ends))
        return bw

    return _tape(out_v, [x], make)


@_verb("sum")
def _v_sum(shipped, args, kwargs):
    """Sum over one axis. The backward broadcasts the gradient back along it, which is one
    add against a zero tensor of the input's shape rather than a repeat."""
    x = _wrap(args[0])
    keepdim = bool(kwargs.get("keepdim", False))
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    shape = [int(d) for d in x.value.shape]
    if dim is None or isinstance(dim, (list, tuple)):
        raise NotImplementedError(
            "tt_bio.autograd tapes ttnn.sum over a single named axis only; an all-axis or "
            "multi-axis sum has a different broadcast and no tt-bio caller needs one.")
    ax = _axis(int(dim), len(shape))
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            if not keepdim:
                kept = list(shape)
                kept[ax] = 1
                g = ttnn.reshape(g, kept)
            x.add_grad(ttnn.add(ttnn.zeros(shape, dtype=g.dtype, layout=ttnn.TILE_LAYOUT,
                                           device=g.device()), g))
        return bw

    return _tape(out_v, [x], make)


@_verb("max")
def _v_max(shipped, args, kwargs):
    """Max over one axis, gradient routed to the maximal elements by an equality mask.

    Ties SPLIT: k equal maxima each receive the full gradient rather than 1/k of it, which
    differs from torch's `max(dim)`, and the reason to accept it here is that the only
    tt-bio caller is the numerical-stability shift in `_accurate_softmax`
    (`tenstorrent.py:3444`). There the gradient through the max provably contributes zero
    whatever the tie rule: softmax is invariant to a uniform row shift, so the perturbation
    this path sends into the softmax lies along the all-ones direction, and the softmax
    Jacobian annihilates it. A caller wanting max as a real selection should ask for one.
    """
    x = _wrap(args[0])
    keepdim = bool(kwargs.get("keepdim", False))
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    shape = [int(d) for d in x.value.shape]
    if dim is None or isinstance(dim, (list, tuple)):
        raise NotImplementedError("tt_bio.autograd tapes ttnn.max over a single named axis only")
    ax = _axis(int(dim), len(shape))
    ra, rk = _raw(args, kwargs)
    out_v = shipped(*ra, **rk)
    xv = x.value

    def make():
        def bw(g):
            m = out_v
            if not keepdim:
                kept = list(shape)
                kept[ax] = 1
                m = ttnn.reshape(m, kept)
                g = ttnn.reshape(g, kept)
            x.add_grad(ttnn.multiply(g, ttnn.eq(x.value, m)))
        return bw

    out = _tape(out_v, [x], make)
    out.evictable = False          # the closure holds out_v to build the equality mask
    return out
