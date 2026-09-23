"""OpenFold3's template distogram, binned by index, is upstream's broadcast compare bit for bit.

The vendored `create_template_distogram` bins each squared pair distance with a direct index
into the sorted edges instead of comparing every pair against all 39 of them. The reference below
is the upstream expression verbatim. Opens no device.
"""

import numpy as np
import torch

from tt_bio._vendor.openfold3.core.data.primitives.featurization.template import (
    create_template_distogram,
)


def _upstream(coords, mask, pair_mask, min_bin=3.25, max_bin=50.75, n_bins=39, inf_value=1e8):
    distogram = np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2,
                       axis=-1, keepdims=True)
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)], axis=-1)
    binned = torch.tensor(((distogram > lower) * (distogram < upper)).astype(distogram.dtype),
                          dtype=torch.float)
    return binned * (mask[..., None] * mask[..., None, :])[..., None] * pair_mask


def _case(n_templates, n_tokens, dtype, seed):
    rng = np.random.default_rng(seed)
    coords = (rng.standard_normal((n_templates, n_tokens, 3)) * 25.0).astype(dtype)
    coords[:, rng.random(n_tokens) < 0.1] = np.nan              # unresolved pseudo-beta atoms
    # Distances exactly on a bin edge and past the last one: token 1 sits 3.25 A (edge 0) from
    # token 0 along x, token 2 exactly on the last edge, token 3 far past it.
    coords[:, :4] = 0.0
    coords[:, 1, 0], coords[:, 2, 0], coords[:, 3, 0] = 3.25, 50.75, 1e4
    mask = torch.tensor(~np.isnan(coords).any(axis=-1), dtype=torch.float)
    asym = torch.tensor(rng.integers(1, 4, n_tokens))
    pair_mask = (asym[..., None] == asym[..., None, :])[..., None, :, :, None]
    return coords, mask, pair_mask


def test_indexed_binning_is_the_broadcast_compare():
    for n_templates, n_tokens, dtype, seed in ((1, 37, np.float64, 0), (4, 160, np.float64, 1),
                                               (2, 96, np.float32, 2)):
        coords, mask, pair_mask = _case(n_templates, n_tokens, dtype, seed)
        want = _upstream(coords, mask, pair_mask)
        got = create_template_distogram(coords, mask, pair_mask)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want), (n_templates, n_tokens, dtype)
        # the edge cases really are exercised: the 3.25 A pair is in no bin (strict compare)
        assert float(got[0, 0, 1].sum()) == 0.0
