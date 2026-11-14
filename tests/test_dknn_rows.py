"""tt_bio.dknn on the card: a row range's result equals the same rows of the full launch, and both equal the
float64 k-d tree's index (_neighbours_kdtree) row for row."""
import numpy as np
import pytest
import torch
import ttnn

from tt_bio import dknn
from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


@pytest.mark.parametrize("rows", [(0, 2400), (1700, 2400), (333, 1001)])
def test_row_range_equals_full_and_kdtree(rows):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    L, K, NT = 2400, 128, 240
    rng = np.random.default_rng(5)
    tok = np.repeat(np.arange(NT), L // NT)
    x = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3))
    x = torch.from_numpy(x.astype(np.float32))
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    xd = x.double().numpy()
    full = ttnn.to_torch(dknn.neighbours(dev, xd, mask, K)).numpy()
    r0, r1 = rows
    part = ttnn.to_torch(dknn.neighbours(dev, xd, mask, K, rows)).numpy()
    assert part.shape[0] == r1 - r0
    assert np.array_equal(part, full[r0:r1])
    ref = M._neighbours_kdtree(mask, seq, x.unsqueeze(0), K)
    assert torch.equal(torch.from_numpy(full[:, :K]).long().unsqueeze(0), ref)


@pytest.mark.parametrize("sigma", [0.5, 400.0])
def test_static_candidates_equal_full(sigma):
    """Candidate mode (dknn.Static: static rows against their static candidates and the movers) gives the full
    launch's rows, including the first call that only learns and a call after the movers moved again."""
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    L, K, NT, NM = 2400, 128, 240, 560
    rng = np.random.default_rng(7)
    tok = np.repeat(np.arange(NT), L // NT)
    x0 = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3))
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    st = dknn.Static()
    for step in range(3):
        x = x0.copy()
        x[L - NM:] += rng.normal(size=(NM, 3)) * sigma / (1 + step)
        x = x.astype(np.float32)
        got = ttnn.to_torch(dknn.neighbours(dev, x, mask, K, static=st)).numpy()
        full = ttnn.to_torch(dknn.neighbours(dev, x, mask, K)).numpy()
        assert np.array_equal(got, full), step
    assert st.ids is not None and len(st.ids) == L - NM


@pytest.mark.parametrize("async_read", [True, False])
def test_pending_read_later_equals_blocking(monkeypatch, async_read):
    """Pending's readback (dknn.read_later) returns the blocking read's index, with card work queued after the launch
    and before .result()."""
    ensure_p300_mesh_descriptor()
    monkeypatch.setattr(dknn, "ASYNC_READ", async_read)
    dev = M.get_device()
    L, K, NT = 2400, 128, 240
    rng = np.random.default_rng(9)
    tok = np.repeat(np.arange(NT), L // NT)
    x = torch.from_numpy((np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3))).astype(np.float32))
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    p = dknn.Pending(dev, x.unsqueeze(0), mask, K)
    a = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for _ in range(20):
        a = ttnn.matmul(a, a)
    got = p.result()
    ttnn.deallocate(a)
    assert torch.equal(got, M._neighbours_kdtree(mask, seq, x.unsqueeze(0), K))
