"""The pose-update block's two card-free claims: Davenport's K assembly and the tie-free pick.

:class:`tt_bio.diffdock.device_loop.PoseUpdate` builds K from nine per-pose sums rather than a
batched `[3, 3]` transpose, and :func:`device_loop.proper_rotation` picks the eigenvector column
by the *first* index attaining the largest diagonal entry instead of `argmax`. Both are claims
about index algebra, not about ttnn, so they are graded here against
:mod:`tt_bio.diffdock.kabsch` itself without a card.
"""

import torch

from tt_bio.diffdock.kabsch import _davenport_k


def _k16(h):
    """The 16 entries `device_loop.PoseUpdate.align` assembles, in its own order."""
    xx, xy, xz = h[..., 0, 0], h[..., 0, 1], h[..., 0, 2]
    yx, yy, yz = h[..., 1, 0], h[..., 1, 1], h[..., 1, 2]
    zx, zy, zz = h[..., 2, 0], h[..., 2, 1], h[..., 2, 2]
    return torch.stack([xx + yy + zz, yz - zy, zx - xz, xy - yx,
                        yz - zy, xx - yy - zz, xy + yx, zx + xz,
                        zx - xz, xy + yx, yy - xx - zz, yz + zy,
                        xy - yx, zx + xz, yz + zy, zz - xx - yy], dim=-1)


def test_k16_is_davenports_k_row_major():
    h = torch.randn(7, 3, 3, dtype=torch.float64)
    assert torch.equal(_k16(h), _davenport_k(h).reshape(7, 16))


def test_k16_covers_the_reflection_case():
    h = torch.randn(4, 3, 3, dtype=torch.float64)
    h[:, :, 2] *= -torch.sign(torch.linalg.det(h))[:, None]   # det(H) < 0: upstream's branch
    assert (torch.linalg.det(h) < 0).all()
    assert torch.equal(_k16(h), _davenport_k(h).reshape(4, 16))


def _first_argmax(diag):
    """The pick `proper_rotation` builds from `eq` and a running `taken` column."""
    best = diag.max(dim=-1, keepdim=True).values
    taken, picks = None, []
    for j in range(diag.shape[-1]):
        hit = (diag[:, j:j + 1] == best).to(diag.dtype)
        pick = hit if taken is None else hit * (1 - taken)
        taken = pick if taken is None else taken + pick
        picks.append(pick)
    return torch.cat(picks, dim=-1)


def test_pick_is_one_hot_on_the_first_maximum():
    diag = torch.tensor([[1.0, 3.0, 3.0, 2.0],      # a tie: the first of the two wins
                         [5.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0],      # all equal: column 0
                         [1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    pick = _first_argmax(diag)
    assert torch.equal(pick.sum(-1), torch.ones(4, dtype=torch.float64))
    assert torch.equal(pick.argmax(-1), torch.tensor([1, 0, 0, 3]))


def test_pick_matches_argmax_where_the_maximum_is_unique():
    diag = torch.rand(64, 4, dtype=torch.float64)
    assert torch.equal(_first_argmax(diag).argmax(-1), diag.argmax(-1))
