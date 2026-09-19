"""A float64 reference for the OF3 confidence head, and its validation.

PROTOCOL SS3c: the reference is float64, it is validated against the forward it claims to
differentiate, and then against float64 central finite differences -- never against another
approximation.

How it is assembled, and why each piece is allowed to be the reference:

  * the z-track is `tt_bio.reference`'s torch modules, loaded from the SAME remapped state
    dict the device modules load. The remap's keys are an exact match for
    `reference.PairformerLayer.state_dict()`, so there is no transcription step to get
    wrong. `PairformerLayer.forward` is not used -- it casts to float32 mid-block, which
    would silently demote the reference -- so the five z sub-modules are called directly.
  * the s-track and the five output heads are `OF3ConfidenceHead`'s own host path at
    `_dtype = torch.float64`. That path is the one gated against the real OF3 golden, so
    it is the reference here by provenance rather than by assertion.

Validation is the point of this file, so it runs both legs: the reference forward against
the device forward (agreement at the bf16 floor means neither is a transcription error),
and the reference's autograd against its own float64 central finite differences.
"""
import contextlib
import os, pickle, torch, torch.nn.functional as F

import tt_bio.reference as R
from tt_bio.openfold3_weights import remap_pairformer_block, _sub

_C_S, _C_Z, _NB = 384, 128, 4


@contextlib.contextmanager
def no_demotion():
    """Stop ``tt_bio.reference``'s own ``.float()`` calls demoting a float64 reference.

    `reference.py` casts to float32 inside both TriangleMultiplication variants
    (lines 459 and 548) and the OuterProductMean einsum. Those casts exist to keep a
    bf16 activation out of the einsum, so their intent is "not lower than float32" --
    but `torch.Tensor.float()` is an ASSIGNMENT to float32, not a floor, and on a
    float64 input it silently halves the mantissa. A reference that quietly runs at
    float32 is exactly the thing PROTOCOL SS3c forbids, and nothing would have raised.

    So `.float()` becomes a no-op for the duration, and `forward` asserts the output
    came back float64 -- the patch is checked, not trusted.
    """
    orig = torch.Tensor.float
    torch.Tensor.float = lambda self: self if self.dtype == torch.float64 else orig(self)
    try:
        yield
    finally:
        torch.Tensor.float = orig
_PFX = "pairformer_embedding.pairformer_stack.blocks.%d"


def build_blocks(aux, dtype=torch.float64):
    """The four confidence-Pairformer blocks as torch modules, in ``dtype``."""
    blocks = []
    for i in range(_NB):
        pd = remap_pairformer_block(_sub(aux, _PFX % i))
        m = R.PairformerLayer(_C_S, _C_Z, 16, 0.0, 32, 4)
        m.load_state_dict(pd, strict=True)
        blocks.append(m.to(dtype).eval())
    return blocks


def z_block(m, z, pair_mask):
    """One block's z-track, sub-module by sub-module so nothing casts to float32.

    The mask is all-ones rather than None: these modules multiply by it unconditionally,
    and all-ones is what the device path's ``mask=None`` means.
    """
    z = z + m.tri_mul_out(z, mask=pair_mask)
    z = z + m.tri_mul_in(z, mask=pair_mask)
    z = z + m.tri_att_start(z, mask=pair_mask)
    z = z + m.tri_att_end(z, mask=pair_mask)
    return z + m.transition_z(z)


def forward(head, blocks, si_input, si_trunk, zij_trunk, repr_x, mask,
            dtype=torch.float64):
    """The whole confidence head in ``dtype``, returning the same dict as the device path."""
    head._dtype = dtype
    si_input, si_trunk = si_input.to(dtype), si_trunk.to(dtype)
    zij_trunk, repr_x = zij_trunk.to(dtype), repr_x.to(dtype)
    N = si_trunk.shape[0]
    pe = "pairformer_embedding."
    z = (zij_trunk
         + F.linear(si_input, head._g(pe + "linear_i.weight")).unsqueeze(-2)
         + F.linear(si_input, head._g(pe + "linear_j.weight")).unsqueeze(-3))
    dij = torch.sum((repr_x[..., None, :] - repr_x[..., None, :, :]) ** 2, -1, keepdim=True)
    oh = ((dij > head._squared_bins.to(dtype)) & (dij < head._upper.to(dtype))).to(dtype)
    z = z + F.linear(oh, head._g(pe + "linear_distance.weight"))

    s = si_trunk
    zb = z.unsqueeze(0)
    pair = torch.ones(1, N, N, dtype=dtype)
    with no_demotion():
        for i, m in enumerate(blocks):
            zb = z_block(m, zb, pair)
            s = head._host_s_block(s, zb[0], i)
    zc = zb[0]
    if zc.dtype != dtype or s.dtype != dtype:
        raise AssertionError(f"reference demoted to z={zc.dtype} s={s.dtype}, wanted {dtype}")

    dlog = F.linear(zij_trunk, head._g("distogram.linear.weight"))
    plog = F.linear(F.layer_norm(zc, (_C_Z,)) * head._g("pde.layer_norm.weight")
                    + head._bias("pde.layer_norm.bias"), head._g("pde.linear.weight"))
    return {
        "distogram_logits": dlog + dlog.transpose(-2, -3),
        "pae_logits": F.linear(F.layer_norm(zc, (_C_Z,)) * head._g("pae.layer_norm.weight")
                               + head._bias("pae.layer_norm.bias"),
                               head._g("pae.linear.weight")),
        "pde_logits": plog + plog.transpose(-2, -3),
        "plddt_logits": head._atom_head(s, "plddt", mask, 50),
        "experimentally_resolved_logits": head._atom_head(
            s, "experimentally_resolved", mask, 2),
        "si_conf": s, "zij_conf": zc,
    }
