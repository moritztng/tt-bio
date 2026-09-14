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

#: ``addalpha``: attention's scale-then-bias, 1503 calls per 298 aa fold in both
#: openfold3 and protenix-v2 -- the highest count of the three. **Default OFF.** It is
#: bit-exact wherever the site runs fp32 (protenix's DiT, which is 1200 of those calls),
#: but at bf16 in openfold3 it moves the 298 aa structure 1.475 A all-atom against a
#: 0.000 A A/A floor. That is 4.0x inside openfold3's own 5.86 A seed floor on the same
#: fixture, and 4.2x OVER the 0.35 A absolute 298 aa bar -- and the bar is the criterion,
#: so it stays off until an accuracy reading with teeth (CA-lDDT against an experimental
#: structure, not RMSD against the seed floor) says otherwise. The deviation is one bf16
#: ULP per call and it is in the MORE accurate direction; what moves the structure is
#: that a diffusion trajectory amplifies any perturbation at all.
FUSE_SCALE_ADD = env_flag("TT_BIO_FUSE_SCALE_ADD", False)
#: ``addcmul``: the gated-residual write-back (multiply by a mask, add the residual).
#: Bit-exact at fold level (openfold3 cdk2x2_298, 1504 calls, byte-identical digest), so
#: it is on: it deletes 1504 dispatches and cannot move the structure.
FUSE_MASK_ADD = env_flag("TT_BIO_FUSE_MASK_ADD", True)
#: ``layer_norm(residual_input_tensor=)``: an add whose only consumer is a norm. Bit-exact
#: at fold level and the largest per-op win of the three (1.880x on a [1,512,512,128]
#: norm), but its one site (protenix.py:1637, the confidence head's pde branch) does not
#: execute in the 298 aa protocol, so the fold-level win is unpriced rather than measured.
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
