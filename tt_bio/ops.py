"""The linear and layer-norm call every device module makes.

Eleven private wrappers around ``ttnn.linear`` and ``ttnn.layer_norm`` lived across
``protenix.py``, ``tenstorrent.py`` and the three OpenFold3 modules. They differed only in
which defaults they hard-coded and which of the two tuned L1 routings they knew about, so
a lever added to one -- the narrow-projection program config, the L1-resident norm -- was
absent from the other ten. There is one implementation of each call here now, and the
per-class wrappers keep only the defaults that class actually measured.

The second reason it is one function. Training needs the same forward inference runs, not a
copy of it, and a copy is what a differentiable second implementation becomes: the taped
twin under ``perf/ptxft`` had already drifted 2.56e-02 on the denoiser before anyone
looked. The twin was deleted in commit 81eaa6a6a, which is where that reading is on the
record. So the call site dispatches. With no hook installed -- every inference path,
always -- ``linear`` is the ``ttnn.linear`` call that was written here before, same
arguments, same kernel, same output bytes. ``tt_bio.autograd`` installs a hook when a user
opts into training, and then the same call site records a tape node.

The hook is injected rather than imported, and that is the whole of what keeps the tape off
the inference path: nothing in this module knows ``tt_bio.autograd`` exists, so importing
``tt_bio`` cannot reach it and grad-off costs one ``is None`` test per call.

Why a hook and not the tape's own ``_GRAD_ENABLED``: that flag prunes the tape NODE, it does
not gate the forward (``autograd.py:151`` -- the caller has already computed ``out_value``).
A differentiable layer norm on this hardware must be composite, five ops against production's
one kernel, because ``ttnn.layer_norm`` returns neither mean nor rstd and
``moreh_layer_norm_backward`` needs both and then refuses bfloat8_b. Substituting it would
move inference numerics whether or not a tape node survived. Dispatching cannot.
"""

from __future__ import annotations

import contextlib

import ttnn

from .dispatch import OpSurface

__all__ = ["linear", "layer_norm", "set_grad_hook", "grad_hook",
           "set_recycle_hook", "recycle_region", "taping",
           "set_checkpoint_hook", "checkpoint_segment",
           "set_host_softmax_hook", "host_softmax_hook"]


# The slot, and the decorator that uses it, are `tt_bio/dispatch.py`'s -- shared with
# `abodybuilder3_ops.py` rather than written twice. The slot itself is per surface: one
# global hook would mean installing a tape for one model taped every other.
_SURFACE = OpSurface("tt_bio.ops")
set_grad_hook = _SURFACE.set_grad_hook
grad_hook = _SURFACE.grad_hook
_dispatching = _SURFACE.dispatching


# A recycling stack differentiates its LAST cycle only -- AF3's own training structure, and
# the difference between a tape holding one cycle and a tape holding ten of them. Which cycle
# that is, is the model's business; whether "not differentiated" means anything at all is the
# tape's. So the model asks here and the answer is injected, exactly like the grad hook above
# and for the same reason: nothing in this module may know `tt_bio.autograd` exists, or
# importing `tt_bio` would reach the training stack (`tests/test_training_opt_in.py`
# ::test_no_inference_module_imports_training). With nothing installed this is
# `nullcontext`, so every inference path pays one `is None` test per recycling cycle.
_RECYCLE = None


def set_recycle_hook(fn):
    """Install the region a non-differentiated recycling cycle runs in. Returns the old one."""
    global _RECYCLE
    prev, _RECYCLE = _RECYCLE, fn
    return prev


def taping():
    """Is a tape open? Asked by the fused kernels that have no backward.

    A handful of shipped kernels are driven through `ttnn.generic_op` -- the F1 trimul tail,
    the head-major QKV projection, the fused QKV+SDPA fold. `generic_op` has no tape entry and
    `taped_ttnn` raises rather than unwrap, which is right: unwrapping would drop the gradient
    of everything upstream silently. Each of those kernels already has a decline path, because
    each returns None for a call its descriptor does not cover and the composed ops run
    unchanged. So while a tape is open they decline, and the training forward takes the
    composed path the tape can follow. It costs the fused kernel's win in training and nothing
    at all in inference, where `grad_hook()` is None.
    """
    return grad_hook() is not None


# The host float64 softmax a construction site may ask for, injected the same way and for the
# same reason as the hooks above: `tt_bio/autograd.py` owns the implementation and nothing on
# the inference path imports it, so with no tape open there is no function to reach.
_HOST_SOFTMAX = None


def set_host_softmax_hook(fn):
    """Install the host float64 softmax. Returns the old one."""
    global _HOST_SOFTMAX
    prev, _HOST_SOFTMAX = _HOST_SOFTMAX, fn
    return prev


def host_softmax_hook():
    """The host float64 softmax if a tape is open to want it, otherwise None.

    `tenstorrent.site_softmax` asks here, and None is the whole of what keeps the path off
    inference. The site selector reads an environment variable and the call sites it decides are
    shared with every model's inference, so gating on the selector alone left a training-only
    lever one `export` away from a user's fold. The path buys gradient fidelity and costs a host
    round trip per softmax, so there was never an inference reading to gain either.

    Two conditions rather than one. `taped_ttnn.tape()` restores the grad hook on the way out
    and leaves the other slots filled, so the slot on its own would stay live for the rest of
    the process and a fold after a training block could still reach the path.
    """
    return _HOST_SOFTMAX if grad_hook() is not None else None


_CHECKPOINT = None


def set_checkpoint_hook(fn):
    """Install the per-block checkpointing implementation. Returns the old one."""
    global _CHECKPOINT
    prev, _CHECKPOINT = _CHECKPOINT, fn
    return prev


def checkpoint_segment(fn, *inputs):
    """Run one block. Under a tape, as a checkpointed segment; otherwise just run it.

    A deep stack asks here instead of calling its block directly, so the memory/recompute
    trade is the tape's decision and not something the model file has to know about. With
    nothing installed this is `fn(*inputs)` and inference pays one `is None` test per block.
    """
    if _CHECKPOINT is None:
        return fn(*inputs)
    return _CHECKPOINT(fn, *inputs)


def recycle_region(cyc, last):
    """The context a recycling cycle runs in: inert during inference, `no_grad` under a tape."""
    if _RECYCLE is None or cyc == last:
        return contextlib.nullcontext()
    return _RECYCLE()


# `_narrow_proj_linear` and `_l1_layer_norm` are tuning that belongs beside the program
# configs it builds, so they stay in tenstorrent.py and are resolved on first use -- a
# module-level import here would be circular, since tenstorrent.py imports this module.
_NARROW_PROJ = None
_L1_NORM = None


@_dispatching
def linear(x, w, bias=None, *, activation=None, compute_kernel_config=None, dtype=None,
           core_grid=None, narrow_proj=False, **kw):
    """``x @ w (+ bias)``. ``w`` is (in, out), tt-bio's convention throughout.

    ``narrow_proj`` offers the call to the tuned narrow-output pair projection first, which
    is the path that wants an L1-resident operand to stay in L1. It declines any output wider
    than two tiles and any call carrying a bias or an activation, so it is the caller's
    assertion that this is a projection, not a claim about the shape.

    Every keyword defaults to ``None`` because that is ttnn's own default for each of them:
    passing ``None`` is indistinguishable from omitting the argument, which is what lets
    eleven differently-shaped call sites share one body.
    """
    if narrow_proj and bias is None and activation is None:
        global _NARROW_PROJ
        if _NARROW_PROJ is None:
            from .tenstorrent import _narrow_proj_linear
            _NARROW_PROJ = _narrow_proj_linear
        # An L1 operand only ever arrives from a caller that deliberately put it there, and
        # such a caller wants the result in L1 too, so the buffer type carries the decision
        # and no flag has to be threaded through.
        in_l1 = x.memory_config().buffer_type == ttnn.BufferType.L1
        out = _NARROW_PROJ(x, w, compute_kernel_config, dtype, l1_out=in_l1)
        if out is not None:
            return out
    return ttnn.linear(x, w, bias=bias, activation=activation,
                       compute_kernel_config=compute_kernel_config, dtype=dtype,
                       core_grid=core_grid, **kw)


@_dispatching
def layer_norm(x, weight=None, bias=None, *, epsilon=1e-5, compute_kernel_config=None,
               l1_headroom=None, **kw):
    """Layer norm over the last dim.

    ``epsilon`` defaults to production's 1e-5, not ttnn's 1e-12 and not the tape's 1e-6.
    Every routed site passed 1e-5 explicitly; the default is here so that a site that stops
    passing it lands on the shipped value rather than a hundred-fold smaller one.

    ``l1_headroom`` asks for an L1-resident result when that many copies of ``x`` fit across
    the grid's banks, for a norm whose consumers are narrow projections of the whole tensor:
    they are bound by reading it, so where the result lives is the lever. A refusal falls
    back to DRAM and changes nothing.
    """
    if l1_headroom is not None:
        global _L1_NORM
        if _L1_NORM is None:
            from .tenstorrent import _l1_layer_norm
            _L1_NORM = _l1_layer_norm
        return _L1_NORM(x, l1_headroom, weight=weight, bias=bias, epsilon=epsilon,
                        compute_kernel_config=compute_kernel_config, **kw)[0]
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=epsilon,
                           compute_kernel_config=compute_kernel_config, **kw)
