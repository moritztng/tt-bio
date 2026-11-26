"""The whole-trunk float64 reference under upstream `recycle grad: last` can be checked by central differences
only with the earlier cycles frozen at the base point (`rfd4_reference.trunk(state=...)`): `grad_last` alone
detaches them from the gradient, but a perturbed weight still moves them, so the difference disagrees."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rfd4_reference as REF  # noqa: E402
from tt_bio.train import checks as CK  # noqa: E402

N, DIM, DP, DC, H, LAYERS = 8, 32, 16, 16, 2, 3


def _setup():
    w = REF.trunk_weights(LAYERS, DIM, DP, DC, H, tri_c=16, seed=3)
    x, z, _c, mask = REF.block_inputs(1, N, DIM, DP, DC, lengths=[6], seed=1)
    g = torch.Generator().manual_seed(2)
    conds = [torch.randn(1, N, DC, generator=g, dtype=torch.float64) for _ in range(2)]
    pair = (mask[:, :, None] * mask[:, None, :])[..., None]
    return w, x * mask[..., None], z * pair, conds, mask


def _fd(loss, w):
    ps = [v.requires_grad_(True) for v in w.values()]
    return CK.fd_check(loss, ps, n_probe=2, seed=0)[0]


def test_last_cycle_from_a_frozen_state_differences_clean():
    w, x, z, conds, mask = _setup()
    frozen = REF.trunk(w, x, z, conds[:1], mask, LAYERS, n_heads=H)
    loss = lambda: REF.trunk(w, x, z, conds[1:], mask, LAYERS, n_heads=H, state=frozen)[0].sum()  # noqa: E731
    assert _fd(loss, w) < CK.FD_BAR


def test_grad_last_alone_does_not_difference_clean():
    w, x, z, conds, mask = _setup()
    loss = lambda: REF.trunk(w, x, z, conds, mask, LAYERS, n_heads=H, grad_last=True)[0].sum()  # noqa: E731
    assert _fd(loss, w) > CK.FD_BAR


def test_frozen_state_is_the_grad_last_forward_and_gradient():
    w, x, z, conds, mask = _setup()
    ps = {k: v.clone().requires_grad_(True) for k, v in w.items()}
    a = REF.trunk(ps, x, z, conds, mask, LAYERS, n_heads=H, grad_last=True)[0]
    ga = torch.autograd.grad(a.sum(), list(ps.values()), allow_unused=True)
    frozen = REF.trunk(w, x, z, conds[:1], mask, LAYERS, n_heads=H)
    b = REF.trunk(ps, x, z, conds[1:], mask, LAYERS, n_heads=H, state=frozen)[0]
    gb = torch.autograd.grad(b.sum(), list(ps.values()), allow_unused=True)
    assert torch.equal(a, b)
    for u, v in zip(ga, gb):
        assert (u is None and v is None) or torch.equal(u, v)
