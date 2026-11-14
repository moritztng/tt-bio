"""ESMFold2's windowed atom attention (TT_BIO_ESMFOLD2_SWA_WINDOW) is the dense band attention.

Host-only: the per-block bias `_window_bias` builds is replayed in float64 the way the
`kv_window` SDPA reads it (block i of 32 queries against frame rows [32 i, 32 i + 160) of a K/V
with 64 zero rows on each side) and compared with softmax over the dense [N, N] band."""
import pytest
import torch

from tt_bio.esmfold2 import _window_bias, band_mask


def _windowed(q, k, v, win, N, lead=64, nq=32):
    H, _, dh = q.shape
    nb, _, W = win.shape
    NP = nb * nq
    qp = torch.zeros(H, NP, dh, dtype=q.dtype); qp[:, :N] = q
    kp = torch.zeros(H, lead + NP + lead, dh, dtype=q.dtype); kp[:, lead:lead + N] = k
    vp = torch.zeros_like(kp); vp[:, lead:lead + N] = v
    bias = win.clamp(min=-1e4).to(q.dtype)
    out = torch.empty(H, NP, dh, dtype=q.dtype)
    for i in range(nb):
        s = qp[:, nq * i:nq * i + nq] @ kp[:, nq * i:nq * i + W].transpose(1, 2) * dh ** -0.5 + bias[i]
        out[:, nq * i:nq * i + nq] = torch.softmax(s, -1) @ vp[:, nq * i:nq * i + W]
    return out[:, :N]


@pytest.mark.parametrize("N,tail", [(4160, 0), (4100, 37), (96, 0), (1000, 5)])
def test_window_equals_dense_band(N, tail):
    mask = torch.ones(N)
    if tail:
        mask[-tail:] = 0
    dense = band_mask(mask, 64)[0, 0]
    win = _window_bias(mask, 64)
    assert win is not None and win.shape[1:] == (32, 160)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(4, N, 32, generator=g, dtype=torch.float64) for _ in range(3))
    ref = torch.softmax(q @ k.transpose(1, 2) * 32 ** -0.5 + dense.double(), -1) @ v
    got = _windowed(q, k, v, win, N)
    valid = mask > 0.5
    assert (got[:, valid] - ref[:, valid]).abs().max() < 1e-12


def test_padding_inside_the_atom_list_falls_back():
    mask = torch.ones(500)
    mask[100:300] = 0
    assert _window_bias(mask, 64) is None


def _window_from_dense(dense, half_window, nq=32):
    """The window as it used to be built: cut from the dense band, None when a row's keys spill."""
    N = dense.shape[-1]
    lead = -(-half_window // nq) * nq
    W, NP = 2 * lead + nq, -(-N // nq) * nq
    nb = NP // nq
    full = torch.full((NP, lead + NP + lead), float("-inf"))
    full[:N, lead:lead + N] = dense
    cols = torch.arange(nb)[:, None] * nq + torch.arange(W)[None, :]
    win = full.view(nb, nq, -1).gather(2, cols[:, None, :].expand(nb, nq, W))
    if not torch.equal((win.reshape(NP, W)[:N] == 0).sum(1), (dense == 0).sum(1)):
        return None
    return win


@pytest.mark.parametrize("N,holes", [(4160, []), (4100, [(4063, 4100)]), (96, []), (1000, [(995, 1000)]),
                                     (500, [(100, 300)]), (777, [(0, 20), (700, 777)]), (640, [(300, 330)]),
                                     (33, [(5, 6)]), (300, [(0, 300)])])
def test_window_is_bit_identical_to_the_cut_from_the_dense_band(N, holes):
    mask = torch.ones(N)
    for a, b in holes:
        mask[a:b] = 0
    ref = _window_from_dense(band_mask(mask, 64)[0, 0], 64)
    got = _window_bias(mask, 64)
    assert (ref is None) == (got is None)
    if ref is not None:
        assert got.dtype == ref.dtype and torch.equal(got, ref)
