"""The pair-mask skip in the host glue: skipping ``x * mask`` must give the same bits as doing it."""
import torch

from tt_bio._vendor.rfd4_prep._cif.nn_modules_adaptive_ln_scale import AdaptiveLayerNorm
from tt_bio._vendor.rfd4_prep._cif.nn_modules_norm import is_identity_mask


def test_skip_only_when_the_multiply_is_the_identity():
    x = torch.randn(2, 8, 8, 16)
    ones = torch.ones(2, 8, 8)
    assert is_identity_mask(ones, x)
    assert torch.equal(x * ones[..., None], x)
    holed = ones.clone()
    holed[1, 3, 5] = 0
    assert not is_identity_mask(holed, x)
    assert not is_identity_mask(torch.full((2, 8, 8), 0.5), x)
    assert not is_identity_mask(ones, x.bfloat16())  # the product would promote to fp32
    assert is_identity_mask(ones.bool(), x)


def test_adaln_bits_unchanged_by_the_skip():
    torch.manual_seed(0)
    ada = AdaptiveLayerNorm(dim=16, dim_cond=8, preserve_dtype=True).eval()
    x, cond = torch.randn(2, 8, 8, 16), torch.randn(2, 8)
    ones = torch.ones(2, 8, 8)
    with torch.no_grad():
        skipped = ada(x, cond, ones)
        normed = ada.norm(x)
        nc = ada.norm_cond(cond)
        ref = (normed * ada.to_gamma(nc)[:, None, None] + ada.to_beta(nc)[:, None, None]) * ones[..., None]
    assert torch.equal(skipped, ref)
