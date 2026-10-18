"""tt_bio.dknn's contract, host only: its fixed point, packed mask and join, with the kernel's per-row rule run in
numpy, reproduce _neighbours_kdtree (the float64 k-d tree the model uses today) row for row."""
import numpy as np
import pytest
import torch

dknn = pytest.importorskip("tt_bio.dknn")
M = pytest.importorskip("tt_bio.rfd3.model")


def _kernel_rows(q, mw, L, K, WO):
    """dknn.cpp per row: k-th uint64 key (quickselect on the atoms under a 1-in-8 sample's bound, or on every atom if
    fewer than K are under it), outside-mask atoms within it, the n_fill nearest (ties to the lower index)."""
    out = np.full((L, WO), L, np.int32)
    j = np.arange(L)
    ts, sampled = 2 * K // 8 - 1, L >= 4 * K
    for r in range(L):
        d = ((q[:, :3].astype(np.int64) - q[r, :3]) ** 2).sum(1).astype(np.uint64)
        under = d <= np.sort(d[::8])[ts] if sampled else np.ones(L, bool)
        if under.sum() < K:
            under[:] = True
        dk = np.sort(d[under])[K - 1]
        assert dk == np.sort(d)[K - 1]
        bit = (mw[r, 1 + (j >> 5)] >> (j & 31).astype(np.uint32)) & 1
        c = np.where((d <= dk) & (bit == 0))[0]
        if len(c) > mw[r, 0]:
            c = c[np.lexsort((c, d[c]))][: mw[r, 0]]
        out[r, : len(c)] = np.sort(c)
    return out


@pytest.mark.parametrize("sigma", [0.0, 5.0, 160.0])
def test_dknn_rule_equals_kdtree(sigma):
    L, K, NT = 700, 64, 80
    rng = np.random.default_rng(3)
    tok = np.repeat(np.arange(NT), np.diff(np.linspace(0, L, NT + 1).astype(int)))
    x = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3)) * (1.0 + sigma)
    x = torch.from_numpy(x.astype(np.float32))
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    n_fill = (K - (seq < L).sum(1)).numpy()
    mw = dknn.mask_words(mask.numpy(), n_fill).view(np.uint32)
    assert (mw[:, 0] == n_fill).all()
    q = dknn.fixed_point(x.numpy())
    assert np.abs(q).max() <= 2 ** 30
    out = _kernel_rows(q, mw, L, K, dknn._align(int(n_fill.max())))
    got = dknn.join(seq.numpy(), out, L, K)
    ref = M._neighbours_kdtree(mask, seq, x.unsqueeze(0), K)
    assert torch.equal(got, ref)



def test_fits_l1_bound():
    """The card index only takes designs whose two RISC-Vs' buffers fit one core; bigger ones keep the k-d tree."""
    assert dknn.fits(6051, 128)
    assert dknn.fits(16000, 128)
    assert not dknn.fits(20000, 128)
    big = max(L for L in range(1000, 30000, 8) if dknn.fits(L, 128))
    MW, WO = dknn._align(1 + (big + 31) // 32), dknn._align(128)
    assert 2 * sum(dknn._cb_sizes(big, MW, WO)) + dknn.PROGRAM_RESERVE <= dknn.L1_PER_CORE
