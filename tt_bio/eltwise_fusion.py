"""The three ttnn eltwise/norm fusions that ship at our pinned ``ttnn==0.68.0``.

Each one collapses a two-op chain we write by hand into a single device op:

===================  =======================================  ==================
helper               chain it replaces                         ttnn op
===================  =======================================  ==================
``scale_add``        ``add(multiply(x, scale), bias)``         ``addalpha``
``mask_add``         ``add(x, multiply(y, mask))``             ``addcmul``
``norm_residual``    ``layer_norm(add(x, residual))``          ``layer_norm(residual_input_tensor=)``
===================  =======================================  ==================

One mechanism in one place: eleven call sites across six models call these three
helpers, so there is no per-model fusion code to keep in sync.

**The fused ops are MORE accurate than the chains they replace, not less.** Both arms
read the same inputs, but the chain packs the intermediate to the tensor dtype and
reloads it, rounding twice, while the fused op keeps the product in an fp32 SFPU
register and packs once. Screened at every real site shape on Blackhole
(``scripts/profiling/eltwise_fusion_screen.py``, artifacts/eltwise_screen.json): the
fused arm is closer to a float64 reference at every site, and its disagreement with
the current chain is exactly one bf16 ULP (6.25e-2 on logits reaching ~11.6, rel
~5e-3). Where the scale is a power of two the fused arm is bit-identical, because
then the chain's extra rounding is a no-op -- so a bit-exactness check on a
power-of-two scale proves nothing about the other sites.

Each fusion has its own gate because they are independent ops at independent sites
with independently measured fold-level effects. The environment is read once at
import, but the helpers read the module global at call time, so an A/B driver can
hold both arms in one process (one model load, one device, one program cache) by
assigning the module attribute, the same way the other lever A/Bs do.
"""

from __future__ import annotations

import ttnn

from tt_bio.envflags import env_flag

#: ``addalpha``: attention's scale-then-bias. Highest call count of the three.
FUSE_SCALE_ADD = env_flag("TT_BIO_FUSE_SCALE_ADD", True)
#: ``addcmul``: the gated-residual write-back (multiply by a mask, add the residual).
FUSE_MASK_ADD = env_flag("TT_BIO_FUSE_MASK_ADD", True)
#: ``layer_norm(residual_input_tensor=)``: an add whose only consumer is a norm.
FUSE_NORM_RESIDUAL = env_flag("TT_BIO_FUSE_NORM_RESIDUAL", True)


def scale_add(x, scale: float, bias, **kwargs):
    """``x * scale + bias`` -- one ``ttnn.addalpha`` instead of a multiply then an add.

    Argument order follows the call sites (scale the scores, then add the bias), not
    ``addalpha``'s own (bias, x, alpha) order.
    """
    if FUSE_SCALE_ADD:
        return ttnn.addalpha(bias, x, scale, **kwargs)
    return ttnn.add(ttnn.multiply(x, scale), bias, **kwargs)


def mask_add(x, y, mask, **kwargs):
    """``x + y * mask`` -- one ``ttnn.addcmul`` instead of a multiply then an add.

    ``mask`` broadcasts, so the usual column form ``[..., N, 1]`` is fine.
    """
    if FUSE_MASK_ADD:
        return ttnn.addcmul(x, y, mask, value=1.0, **kwargs)
    return ttnn.add(x, ttnn.multiply(y, mask), **kwargs)


def norm_residual(x, residual, **kwargs):
    """``layer_norm(x + residual)`` -- the add folded into the norm's own kernel.

    Only valid where the add's result feeds nothing but the norm; otherwise the caller
    still needs the sum as a tensor and there is nothing to fuse.
    """
    if FUSE_NORM_RESIDUAL:
        return ttnn.layer_norm(x, residual_input_tensor=residual, **kwargs)
    return ttnn.layer_norm(ttnn.add(x, residual), **kwargs)
