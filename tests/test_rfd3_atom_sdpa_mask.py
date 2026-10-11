"""Host-only invariants of the RFD3_ATOM_SDPA mask. No device.

ttnn's SDPA computes softmax((q k^T + mask) * scale) v: the mask goes in before the scale. The atom
attention wants softmax(q k^T * scale + bias) v. So the model pre-scales the compact neighbour bias
by sqrt(head_dim) and leaves the kernel's bf16(-1e4) fill alone. These tests pin that the two forms
agree and that the unscaled fill still gives every masked key a weight of exactly 0.
"""
import torch

FILL = -9984.0       # bf16(-1e4), the sparse-bias kernel's fill
L, K, H, D = 193, 32, 4, 32


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    idx = torch.stack([torch.randperm(L, generator=g)[:K].sort().values for _ in range(L)])
    pb = torch.randn(1, H, L, K, generator=g) * 3.0
    return q, k, idx.expand(1, H, L, K), pb


def _spread(idx, pb):
    return torch.full((1, H, L, L), FILL).scatter_(3, idx, pb)


def test_prescaled_mask_matches_the_atom_attention():
    q, k, idx, pb = _fixture()
    scale = D ** -0.5
    want = torch.softmax(q @ k.transpose(-1, -2) * scale + _spread(idx, pb), -1)
    got = torch.softmax((q @ k.transpose(-1, -2) + _spread(idx, pb * D ** 0.5)) * scale, -1)
    assert (want - got).abs().max().item() < 1e-6


def test_unscaled_fill_is_still_an_exact_zero():
    q, k, idx, pb = _fixture()
    scale = D ** -0.5
    w = torch.softmax((q @ k.transpose(-1, -2) + _spread(idx, pb * D ** 0.5)) * scale, -1)
    masked = torch.ones_like(w, dtype=torch.bool).scatter_(3, idx, False)
    assert w[masked].max().item() == 0.0


def test_a_padded_query_row_of_fill_is_uniform_not_nan():
    """The mask is allocated out to the tile-padded query length and those rows are all fill. Their
    softmax must be finite (they are sliced off), which max-subtraction guarantees."""
    row = torch.full((1, L), FILL) * D ** -0.5
    w = torch.softmax(row, -1)
    assert torch.isfinite(w).all() and abs(w.sum().item() - 1.0) < 1e-5
