"""The linear and layer-norm call every device module makes.

Eleven private wrappers around ``ttnn.linear`` and ``ttnn.layer_norm`` lived across
``protenix.py``, ``tenstorrent.py`` and the three OpenFold3 modules. They differed only in
which defaults they hard-coded and which of the two tuned L1 routings they knew about, so
a lever added to one -- the narrow-projection program config, the L1-resident norm -- was
absent from the other ten. There is one implementation of each call here now, and the
per-class wrappers keep only the defaults that class actually measured.

The second reason it is one function. Training needs the same forward inference runs, not a
copy of it, and a copy is what a differentiable second implementation becomes: the taped
twin in ``perf/ptxft/tape_block.py`` had already drifted 2.56e-02 on the denoiser before
anyone looked. So the call site dispatches. With no hook installed -- every inference path,
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

import ttnn

__all__ = ["linear", "layer_norm", "set_grad_hook", "grad_hook"]


_GRAD_HOOK = None


def set_grad_hook(hook):
    """Install the differentiable implementation, or clear it with ``None``.

    Returns the previous hook so a caller can restore it. A hook exposes ``linear`` and
    ``layer_norm`` with the signatures below and returns ``None`` to decline a call, which
    falls through to production -- so a hook only has to handle the operands it tracks.
    """
    global _GRAD_HOOK
    prev, _GRAD_HOOK = _GRAD_HOOK, hook
    return prev


def grad_hook():
    return _GRAD_HOOK


# `_narrow_proj_linear` and `_l1_layer_norm` are tuning that belongs beside the program
# configs it builds, so they stay in tenstorrent.py and are resolved on first use -- a
# module-level import here would be circular, since tenstorrent.py imports this module.
_NARROW_PROJ = None
_L1_NORM = None


def linear(x, w, bias=None, *, activation=None, compute_kernel_config=None, dtype=None,
           core_grid=None, narrow_proj=False, **kw):
    """``x @ w (+ bias)``. ``w`` is (in, out), tt-bio's convention throughout.

    ``narrow_proj`` offers the call to the tuned narrow-output pair projection first, which
    is the path that wants an L1-resident operand to stay in L1. It declines any output wider
    than two tiles and any call carrying a bias or an activation, so it is the caller's
    assertion that this is a projection, not a claim about the shape.

    Every keyword defaults to ``None`` here because that is ttnn's own default for each of
    them: passing ``None`` is indistinguishable from omitting the argument, which is what
    lets eleven differently-shaped call sites share one body.
    """
    if _GRAD_HOOK is not None:
        out = _GRAD_HOOK.linear(x, w, bias, activation=activation,
                                compute_kernel_config=compute_kernel_config, dtype=dtype,
                                core_grid=core_grid, **kw)
        if out is not None:
            return out
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
    if _GRAD_HOOK is not None:
        out = _GRAD_HOOK.layer_norm(x, weight, bias, epsilon=epsilon,
                                    compute_kernel_config=compute_kernel_config, **kw)
        if out is not None:
            return out
    if l1_headroom is not None:
        global _L1_NORM
        if _L1_NORM is None:
            from .tenstorrent import _l1_layer_norm
            _L1_NORM = _l1_layer_norm
        return _L1_NORM(x, l1_headroom, weight=weight, bias=bias, epsilon=epsilon,
                        compute_kernel_config=compute_kernel_config, **kw)[0]
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=epsilon,
                           compute_kernel_config=compute_kernel_config, **kw)
