"""The atom transformer's superset key window computes the same attention as the 128-key window.

`AtomTransformer._attention_superset` attends each 32-atom query block over the five whole tiles
around it (160 keys) instead of its 128-key window, which starts 48 atoms to the left and so does
not sit on a tile. `_superset_bias` gives the 16 slack keys on each side, and every key the window
mask rules out, a -1e9 bias. This test runs that arithmetic in float64 torch against the windowed
definition (padded row i*32 + j for key j of block i), with the bias built by the real
`_superset_bias` and K/V laid out exactly as `_attention_superset` lays them out on the device.

Opens no device: ttnn's host conversions are replaced by identity functions.
"""
import types

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from tt_bio import protenix as PX                                           # noqa: E402

H, DH, NQ, NK, PAD_LEFT = 4, 32, 32, 128, 48


def _at(sdpa=False):
    at = types.SimpleNamespace(N_HEADS=H, HEAD_DIM=DH, N_QUERIES=NQ, N_KEYS=NK, PAD_LEFT=PAD_LEFT,
                               _sdpa=sdpa, device=None)
    at._superset = lambda: PX.AtomTransformer._superset(at)
    return at


@pytest.fixture
def host_ttnn(monkeypatch):
    monkeypatch.setattr(PX.ttnn, "to_torch", lambda t: t)
    monkeypatch.setattr(PX.ttnn, "from_torch", lambda t, **kw: t)


def _windowed(q, k, v, z, mask, N, NP):
    """The definition: (M, N, H*dh) inputs, z (nb, H, nq, nk), mask (nb, nq, nk)."""
    M, nb = q.shape[0], NP // NQ
    sp = lambda x: x.reshape(M, -1, H, DH).permute(0, 2, 1, 3)               # (M, H, L, dh)
    qb = sp(torch.nn.functional.pad(q, (0, 0, 0, NP - N))).reshape(M, H, nb, NQ, DH)
    kp = sp(torch.nn.functional.pad(k, (0, 0, PAD_LEFT, NP + NK - PAD_LEFT - N)))
    vp = sp(torch.nn.functional.pad(v, (0, 0, PAD_LEFT, NP + NK - PAD_LEFT - N)))
    idx = torch.arange(nb)[:, None] * NQ + torch.arange(NK)[None, :]         # (nb, nk)
    kb, vb = kp[:, :, idx], vp[:, :, idx]                                    # (M, H, nb, nk, dh)
    bias = z.permute(1, 0, 2, 3) + torch.where(mask < 0.5, -1e9, 0.0)[None]  # (H, nb, nq, nk)
    sc = qb @ kb.transpose(-1, -2) * DH ** -0.5 + bias
    o = torch.softmax(sc, -1) @ vb                                           # (M, H, nb, nq, dh)
    return o.reshape(M, H, NP, DH).permute(0, 2, 1, 3).reshape(M, NP, H * DH)[:, :N]


def _superset(q, k, v, zs, N, NP, lead, W):
    """`_attention_superset`'s layout and arithmetic, op for op, in torch."""
    M, nb, S = q.shape[0], NP // NQ, W // NQ
    nbk = nb + S - 1

    def heads(x, front, rows):
        x = torch.nn.functional.pad(x, (0, 0, front, rows - front - N))
        return x.reshape(M, rows, H, DH).permute(0, 2, 1, 3)

    def windows(x):
        x = heads(x, lead, nbk * NQ).reshape(M * H, nbk, NQ, DH)
        x = torch.cat([x[:, j:j + nb] for j in range(S)], dim=2)
        return x.reshape(M, H * nb, W, DH)

    qs = heads(q, 0, NP).reshape(M, H * nb, NQ, DH)
    ks, vs = windows(k), windows(v)
    o = torch.softmax(qs @ ks.transpose(-1, -2) * DH ** -0.5 + zs, -1) @ vs
    return o.reshape(M, H, NP, DH).permute(0, 2, 1, 3).reshape(M, NP, H * DH)[:, :N]


@pytest.mark.parametrize("N,M", [(5919, 5), (100, 1), (33, 2), (64, 3)])
def test_superset_equals_the_window(host_ttnn, N, M):
    torch.manual_seed(N)
    NP = -(-N // NQ) * NQ
    nb = NP // NQ
    q, k, v = (torch.randn(M, N, H * DH, dtype=torch.float64) for _ in range(3))
    z = torch.randn(nb, H, NQ, NK).double()       # fp32 values, as the device bias holds
    # Window validity as Protenix builds it: a key outside [0, N) is invalid, plus random holes.
    key = torch.arange(nb)[:, None, None] * NQ - PAD_LEFT + torch.arange(NK)[None, None, :]
    mask = ((key >= 0) & (key < N)).expand(nb, NQ, NK).double()
    mask = mask * (torch.rand(nb, NQ, NK) > 0.05).double()
    mask[..., NK // 2] = 1.0                       # every row keeps at least one valid key
    at = _at()
    lead, W = at._superset()
    assert (lead, W) == (64, 160)
    zs = PX.AtomTransformer._superset_bias(at, z, mask, nb)
    assert tuple(zs.shape) == (1, H * nb, NQ, W)
    ref = _windowed(q, k, v, z, mask, N, NP)
    got = _superset(q, k, v, zs.double(), N, NP, lead, W)
    assert (got - ref).abs().max().item() < 1e-12


def test_valid_entries_keep_the_exact_bias(host_ttnn):
    nb = 3
    z = torch.randn(nb, H, NQ, NK)
    mask = torch.ones(nb, NQ, NK)
    zs = PX.AtomTransformer._superset_bias(_at(), z, mask, nb).reshape(H, nb, NQ, 160)
    assert torch.equal(zs[..., 16:144], z.permute(1, 0, 2, 3))
    assert (zs[..., :16] == -1e9).all() and (zs[..., 144:] == -1e9).all()


def test_the_sdpa_bias_is_prescaled_and_clamped(host_ttnn):
    """The fused kernel adds the mask before it applies the scale."""
    nb = 2
    z = torch.randn(nb, H, NQ, NK)
    zs = PX.AtomTransformer._superset_bias(_at(sdpa=True), z, torch.ones(nb, NQ, NK), nb)
    zs = zs.reshape(H, nb, NQ, 160)
    assert torch.allclose(zs[..., 16:144], z.permute(1, 0, 2, 3) * DH ** 0.5)
    assert (zs[..., :16] == -1e4 * DH ** 0.5).all()


def test_a_merged_cond_carries_one_bias_per_member(host_ttnn):
    nb, B = 4, 3
    z = torch.randn(B * nb, H, NQ, NK)
    zs = PX.AtomTransformer._superset_bias(_at(), z, torch.ones(nb, NQ, NK), nb)
    assert tuple(zs.shape) == (B, H * nb, NQ, 160)
    assert torch.equal(zs.reshape(B, H, nb, NQ, 160)[1, :, :, :, 16:144],
                       z.reshape(B, nb, H, NQ, NK)[1].permute(1, 0, 2, 3))
