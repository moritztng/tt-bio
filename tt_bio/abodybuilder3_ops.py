"""The device ops the ABodyBuilder3 port is built from. Inference module: imports only ttnn.

One definition per op, and it is the production call. Nothing here knows the tape exists.

`tt_bio/ops.py` holds the same slot for the linear and layer norm every device module makes
(`train-a1-defork`); this file is the rest of what a geometry model needs -- the matmul with its
two transpose flags, the eltwise set, the reductions and the shape ops -- and it holds the slot the
same way. `_GRAD_HOOK` is `None` on every inference path, always, so an op is literally the
`ttnn` call it was written as plus one `is None` test. `tt_bio.train.abodybuilder3_grad.install()`
fills the slot and then the same call site records a tape node.

The dependency points from training TO inference and never the other way, which is what makes
`importing the model cannot reach the tape` structural rather than a claim. It also makes the
stronger property true by construction: with no hook installed there is no closure to record, so
the training forward IS the served forward rather than a second implementation that agrees with it.

Every op is fp32. Measured on qb1 card 3 (`scripts/abb3_port/precision_probe.py`): eltwise is
fp32-exact at 3.0e-07, a matmul on fp32 operands keeps ~11 mantissa bits at 1.25e-03 relative
whatever the kernel config says, and a reduction rounds like a matmul. That asymmetry decides which
algebraic form each op in the model takes, so it is recorded here beside the ops.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence

import ttnn

from .tenstorrent import CORE_GRID_MAIN, get_device

__all__ = ["set_grad_hook", "grad_hook", "kernel_config",
           "linear", "matmul", "add", "sub", "sub_square", "mul", "div", "scale", "shift", "sum_last", "sqrt_plus",
           "softplus", "clamp_min", "norm_from_sq", "relu", "sum_dim", "softmax", "layer_norm", "reshape", "permute", "transpose_last",
           "slice_dim", "concat"]

_GRAD_HOOK = None


def set_grad_hook(hook):
    """Install a differentiable implementation, or clear it with `None`. Returns the previous one.

    A hook is called as `hook(name, shipped, args, kwargs)` and returns `None` to decline, which
    falls through to production. Declining is how a hook handles only the operands it tracks, and
    it is also how a taped module runs the shipped op for a tensor that is on the tape but not
    being differentiated -- a frozen block, or anything under `no_grad`.
    """
    global _GRAD_HOOK
    prev, _GRAD_HOOK = _GRAD_HOOK, hook
    return prev


def grad_hook():
    return _GRAD_HOOK


def _dispatching(fn):
    """Offer the call to the hook first, then run `fn`, which is the production op.

    The decorated name is what the hook dispatches on, and `fn` is handed over with it so the hook
    can compute its forward by calling the shipped op rather than by re-implementing it. That is
    the whole mechanism by which there is one forward: a taped op's value comes from this function.
    """
    name = fn.__name__

    @functools.wraps(fn)
    def call(*args, **kwargs):
        hook = _GRAD_HOOK
        if hook is not None:
            out = hook(name, fn, args, kwargs)
            if out is not None:
                return out
        return fn(*args, **kwargs)

    call.shipped = fn
    return call


_KERNEL_CONFIG = None


def kernel_config():
    """HiFi4 with an fp32 accumulator: the repo's trunk config, and the best this card offers.

    `precision_probe.py` measured what best means here -- 1.25e-03 relative on a matmul against
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


# ------------------------------------------------------------------------------ matmul family

@_dispatching
def linear(x, w, bias=None, *, activation: Optional[str] = None, dtype=ttnn.float32,
           core_grid=CORE_GRID_MAIN):
    """`x @ w (+ bias)`, with `w` in `(in, out)` layout -- tt-bio's convention throughout."""
    return ttnn.linear(x, w, bias=bias, activation=activation, dtype=dtype, core_grid=core_grid,
                       compute_kernel_config=kernel_config())


@_dispatching
def matmul(a, b, *, transpose_a: bool = False, transpose_b: bool = False):
    """`op(a) @ op(b)` where op is transpose-or-not on the last two axes.

    Both flags are free: the gradcheck measured 1.28e-03 and 1.54e-03 against a plain matmul's
    1.37e-03, because the transpose happens inside the datapath that rounds anyway. That is what
    makes a transposed operand ORIENTATION free as well, which the port relies on -- a standalone
    reorientation of the last two axes rounds fp32 at 7.5e-04 whichever call it goes through
    (`scripts/abb3_port/relayout_probe.py`), so the projections that need residues on the width
    are produced by `matmul(w, x, transpose_a=True, transpose_b=True)` instead of being moved after.
    """
    return ttnn.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b,
                       compute_kernel_config=kernel_config())


# ------------------------------------------------------------------------------- eltwise, fp32

@_dispatching
def add(a, b):
    """Elementwise sum, broadcasting allowed on either side."""
    return ttnn.add(a, b)


@_dispatching
def sub(a, b):
    """Elementwise difference, broadcasting allowed on either side.

    Two-sided broadcast is the shape Alg. 22's point term is built from: `[*, N, 1]` against
    `[*, 1, N]` gives every pair of residues in one program, and it is fp32-exact.

    The alternative -- `|q-k|^2 = |q|^2 + |k|^2 - 2 q.k`, one matmul instead of this subtract and
    a square -- was measured and refused. The operands are global coordinates in Angstrom, so the
    terms reach 2e4 A^2 while the distances that matter are ~1, and at ~11 mantissa bits the
    identity lands 14.8 A^2 out on pairs under 100 A^2: a 0.70 error on a softmax logit. This form
    lands 2.7e-05 A^2. Centring the points first makes it worse, which is the tell that the error
    is set by the magnitude the matmul rounds rather than by the cancellation in the subtraction.
    """
    return ttnn.subtract(a, b)


@_dispatching
def sub_square(a, b):
    """`(a - b) ** 2`, with the square fused into the subtract's packer.

    Alg. 22's point term is three of these per block on a 50 MB tensor at micro-batch 4 and 256
    tokens, so the fusion deletes one full-size write and one full-size read per coordinate, and its
    backward recomputes the difference instead of retaining it -- which is the bigger win, because
    the retained difference was 150 MB per block held for the whole backward pass.

    `ttnn.subtract(activations=[SQUARE])` is exact here: measured 1.40e-07 against float64, the same
    as the unfused pair, because both halves are eltwise.
    """
    return ttnn.subtract(a, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)])


@_dispatching
def mul(a, b):
    """Elementwise product, broadcasting allowed on either side.

    The IPA head weights against a `[B, H, N, N]` logit tensor is the broadcast in this model
    where the small side is a parameter, so it is the one whose backward has to reduce.
    """
    return ttnn.multiply(a, b)


@_dispatching
def scale(x, factor: float):
    """Multiply by a python scalar."""
    return ttnn.multiply(x, float(factor))


@_dispatching
def shift(x, offset: float):
    """Add a python scalar."""
    return ttnn.add(x, float(offset))


@_dispatching
def sqrt_plus(x, eps: float):
    """`sqrt(x + eps)`.

    The epsilon is an argument and not a default because it is the difference between a gradient
    and a NaN: Alg. 22's point norm is exactly zero wherever a value point collapses, and
    upstream's own 1e-7 (`params.yaml` `epsilon`) is what keeps the derivative finite there.
    """
    return ttnn.sqrt(ttnn.add(x, float(eps)))


@_dispatching
def div(a, b):
    """`a / b`, broadcasting allowed. Used to normalise a quaternion and a sin/cos pair."""
    return ttnn.divide(a, b)


@_dispatching
def clamp_min(x, value: float):
    """`max(x, value)`, with `clamp`'s gradient and not `sqrt(x + eps)`'s.

    The angle resnet normalises by `sqrt(clamp(sum(s^2), min=eps))` (`structure_module.py:167`) and
    the difference from `sqrt(sum + eps)` is invisible in the value and decisive in the gradient.
    At the released initialisation `linear_out` is zero, so at step 0 every `sum(s^2)` is exactly
    zero: `clamp` passes no gradient there, `sqrt(x + eps)` passes `1 / (2 sqrt(1e-7))` = 1581 times
    whatever arrives. One of those trains and the other one detonates on the first step.
    """
    return ttnn.maximum(x, float(value))


@_dispatching
def norm_from_sq(x, eps: float):
    """`sqrt(x + eps)`, forced to zero wherever `x` is zero: the 2-norm's own subgradient.

    The pairwise distance feature map has an exactly zero diagonal that is never masked away, and
    at block 0 every translation is zero so the WHOLE map is zero. `torch.linalg.vector_norm`
    defines the 2-norm backward as zero at the origin; a bare `sqrt` has an infinite derivative
    there. Gating on `x > 0` reproduces both the value and the derivative, and `eps` only keeps the
    forward finite in between.
    """
    # `relu` before the sqrt is not redundant with the gate after it: a negative input would make
    # the sqrt NaN, and NaN times zero is NaN, so the gate alone does not protect the op. The real
    # input is a sum of squares and cannot be negative, which is exactly why an unprotected version
    # would survive every test until something upstream changed.
    return ttnn.multiply(ttnn.sqrt(ttnn.add(ttnn.relu(x), float(eps))), ttnn.gtz(x))


@_dispatching
def softplus(x):
    """`log(1 + exp(x))`.

    Measured at 3.37e-03 relative, which is why the IPA keeps its 12 head weights as HOST
    parameters and uploads the softplus of them: they are 12 scalars per block, and a round trip
    is exact where this kernel is not.
    """
    return ttnn.softplus(x, beta=1.0, threshold=20.0)


@_dispatching
def relu(x):
    """ReLU."""
    return ttnn.relu(x)


# -------------------------------------------------------------------------------- reductions

@_dispatching
def sum_last(x, *, keepdim: bool = True):
    """Sum over the last axis.

    Reductions round like the matmul and not like eltwise: 1.2e-02 on 96 fp32 terms. That is why
    the point term accumulates its 12 channels with `add` rather than laying them along the last
    axis and calling this.
    """
    out = ttnn.sum(x, dim=-1, keepdim=True)
    if not keepdim:
        out = ttnn.reshape(out, [int(d) for d in x.shape][:-1])
    return out


@_dispatching
def sum_dim(x, dim: int, *, keepdim: bool = True):
    """Sum over one axis.

    The point term reduces the points inside each head with this, over an axis it has already
    grouped by a free reshape. A reduction rounds like a matmul at ~1e-3 relative, which is
    harmless here and only here: the terms are the squared coordinate differences of one point
    pair, all positive and of the same magnitude, so the error scales with the answer. That is the
    property `|q|^2 + |k|^2 - 2 q.k` does not have, and the reason one is refused and this is not.
    """
    out = ttnn.sum(x, dim=dim, keepdim=True)
    if not keepdim:
        shape = [int(d) for d in x.shape]
        out = ttnn.reshape(out, shape[:dim] + shape[dim + 1:])
    return out


@_dispatching
def softmax(x, dim: int = -1):
    """Softmax over `dim`."""
    return ttnn.softmax(x, dim=dim, compute_kernel_config=kernel_config())


@_dispatching
def layer_norm(x, gamma, beta, *, eps: float = 1e-5):
    """Layer norm over the last axis, on the production kernel.

    This is the op a differentiable second implementation did not need to fork. `ttnn.layer_norm`
    returns neither mean nor rstd, and `moreh_layer_norm_backward` needs both and then refuses
    bfloat8_b, which is what ruled moreh out -- but both are two reductions away from the input the
    backward retains anyway, so the tape recomputes them and this forward stays the shipped one.

    `eps` defaults to production's 1e-5, which is also upstream's (`primitives.LayerNorm`), and not
    ttnn's 1e-12.
    """
    return ttnn.layer_norm(x, weight=gamma, bias=beta, epsilon=eps,
                           compute_kernel_config=kernel_config())


# --------------------------------------------------------------------------------- shape ops

@_dispatching
def reshape(x, shape: Sequence[int]):
    """Reshape. Bit-exact, including the cases the port depends on: adding a trailing or leading
    1 axis (`[B, C, N] -> [B, C, N, 1]` or `[B, C, 1, N]`) measured at fp32 round-trip error."""
    return ttnn.reshape(x, [int(d) for d in shape])


@_dispatching
def permute(x, dims: Sequence[int]):
    """Permute. Exact for leading axes; see `transpose_last` for the axes it is not exact on."""
    return ttnn.permute(x, [int(d) for d in dims])


@_dispatching
def transpose_last(x):
    """Swap the last two axes. Measured, and the answer is not the one the name suggests.

    Moving the last two axes rounds fp32 at 4.95e-04 whichever call it goes through --
    `ttnn.transpose`, `ttnn.permute`, or a ROW_MAJOR round trip, which also costs 1.5 s -- while a
    permute of the LEADING axes is exact at 5.3e-08. The last two axes are the tiled ones, so this
    is the tile transpose and it is arithmetic, not data movement.

    Nothing on the port's precision-critical path calls it: the logits take k^T inside
    `matmul(transpose_b=True)`, and the reorientations the IPA needs are leading-axis permutes and
    reshapes. Kept, measured and documented so that stays a decision rather than an accident.
    """
    rank = len(x.shape)
    dims = list(range(rank))
    dims[-2], dims[-1] = dims[-1], dims[-2]
    return ttnn.permute(x, dims)


@_dispatching
def slice_dim(x, dim: int, start: int, end: int):
    """Slice one axis.

    The port slices the head axis and never the tiled last two, which is why this is cheap: the
    coordinate index of a point is folded into the head axis exactly so that the rotation, which
    mixes coordinates, mixes slices of an untiled axis instead of sub-tile channel ranges.
    """
    src = [int(d) for d in x.shape]
    dim = dim % len(src)
    starts = [0] * len(src)
    ends = list(src)
    starts[dim], ends[dim] = int(start), int(end)
    return ttnn.slice(x, starts, ends)


@_dispatching
def concat(xs: Sequence, dim: int = -1):
    """Concatenate along `dim`."""
    return ttnn.concat(list(xs), dim=dim)
