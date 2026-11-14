"""RFD3's token-initializer pair mean-pool: the index_add form (RFD3_POOL_SCATTER) against the one-hot einsum and a
float64 pool. Host only."""
import pytest
import torch

M = pytest.importorskip("tt_bio.rfd3.model")


def test_scatter_pool_matches_einsum_and_float64(monkeypatch):
    g = torch.Generator().manual_seed(0)
    I = 40
    tok = torch.repeat_interleave(torch.arange(I), torch.randint(1, 12, (I,), generator=g))
    tok[5:9] = 7  # a token whose atoms are not contiguous
    P = torch.randn(len(tok), len(tok), 16, generator=g)
    monkeypatch.setattr(M, "_POOL_SCATTER", False)
    ref = M._scatter_mean_pool(P, tok, I)
    monkeypatch.setattr(M, "_POOL_SCATTER", True)
    new = M._scatter_mean_pool(P, tok, I)
    exact = M._scatter_mean_pool(P.double(), tok, I)
    assert new.dtype == ref.dtype and new.shape == ref.shape == (I, I, 16)
    assert torch.allclose(new, ref, rtol=0, atol=1e-6)
    assert (new.double() - exact).abs().max() <= (ref.double() - exact).abs().max() * 4 + 1e-7
