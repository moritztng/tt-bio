"""The joined OPM (`AF2MaskedOuterProductMean._sum_rows_joined`) is the per-row contraction,
card-free: its block-diagonal proj_o, the view and the permutes, replayed in float64 torch."""
import pytest
import torch

from tt_bio.af2 import AF2MaskedOuterProductMean, rows_joined_blocks


@pytest.mark.parametrize("S", [2, 3, AF2MaskedOuterProductMean.ROWS_JOINED_MAX])
def test_joined_layout_is_the_row_sum(S):
    I, J, C, D, c_z = 40, 24, 32, 32, 128
    g = torch.Generator().manual_seed(S)
    a, b = torch.randn(S, I, C, generator=g, dtype=torch.float64), torch.randn(S, J, D, generator=g, dtype=torch.float64)
    o = torch.randn(C * D, c_z, generator=g, dtype=torch.float64)
    w = rows_joined_blocks(o, S, C, D)
    T = w @ a.permute(0, 2, 1).reshape(S * C, I)
    A = T.reshape(c_z, S * D, I).permute(2, 0, 1)
    out = (A @ b.permute(0, 2, 1).reshape(S * D, J)).permute(0, 2, 1)
    ref = torch.einsum("sic,sjd,cdk->ijk", a, b, o.reshape(C, D, c_z))
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-9)


def test_off_diagonal_blocks_are_exact_zeros():
    S, C, D, c_z = 3, 32, 32, 128
    w = rows_joined_blocks(torch.ones(C * D, c_z), S, C, D).reshape(c_z, S, D, S, C)
    for s in range(S):
        for t in range(S):
            assert bool((w[:, s, :, t, :] != 0).any()) == (s == t)
