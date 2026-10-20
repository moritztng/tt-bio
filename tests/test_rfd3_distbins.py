"""rfd3.distbins equals the torch distogram bin chain it replaces, near-edge pairs included."""
import pytest
import torch

from tt_bio.rfd3 import distbins


def _ref(R, edges):
    return torch.bucketize(torch.linalg.norm(R.unsqueeze(-2) - R.unsqueeze(-3), dim=-1), edges).to(torch.int32)


@pytest.mark.parametrize("B,N", [(1, 685), (2, 250), (1, 33)])
@pytest.mark.parametrize("sigma", [0.0, 0.02, 0.3, 1.0, 10.0])
def test_bins_equal_torch(B, N, sigma):
    g = torch.Generator().manual_seed(N)
    ca = torch.cumsum(torch.randn(B, N, 3, generator=g) * 2.2, 1) / 16.0
    R = (ca + torch.randn(B, N, 3, generator=g) * sigma).float()
    edges = torch.linspace(1.0 / 16, 30.0 / 16, 64)
    assert torch.equal(distbins.bins(R, edges), _ref(R, edges))


def test_bins_on_edges():
    """Pairs whose distance is exactly an edge (and one ulp either side) go to torch's bin."""
    edges = torch.linspace(1.0 / 16, 30.0 / 16, 64)
    d = torch.cat([edges, torch.nextafter(edges, torch.full_like(edges, 9.0)),
                   torch.nextafter(edges, torch.zeros_like(edges)), torch.tensor([0.0, 5.0])])
    R = torch.zeros(1, len(d) + 1, 3)
    R[0, 1:, 0] = d
    assert torch.equal(distbins.bins(R, edges), _ref(R, edges))


def test_model_uses_it_by_default():
    from tt_bio.rfd3 import model as M
    R = torch.randn(1, 40, 3)
    assert M._BINS_NB is True or M.env_flag("RFD3_BINS_NB", True) is False
    assert torch.equal(M._scaled_distogram_bins(R), _ref(R, torch.linspace(1.0 / 16, 30.0 / 16, 64)))


def test_bins_leaves_numbas_thread_pool_unlaunched():
    """A launched numba OpenMP pool slows every later torch CPU op in the process ~20x (see distbins' docstring)."""
    from numba.np.ufunc import parallel
    R = torch.randn(2, 64, 3)
    distbins.bins(R, torch.linspace(1.0 / 16, 30.0 / 16, 64))
    assert not parallel._is_initialized
