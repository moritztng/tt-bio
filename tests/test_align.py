"""tt_bio.align reproduces, bit for bit, the two Kabsch implementations it replaced.

``openfold3_fold.kabsch_rmsd`` and ``pxdesign/write._kabsch`` were the same maths in different
notation, but not the same sequence of matmuls: OF3 applied ``(vtᵀ S uᵀ)ᵀ`` and PXDesign built
``u S vt``. Mathematically equal is not bit-equal in floating point, and both feed shipped
numbers (an OF3 gate RMSD, every coordinate in a PXDesign mmCIF), so the equality is pinned
here rather than asserted in a commit message.

`kabsch-inverse-rotation-swap-phantom-rmsd` is why this module exists at all: swapping the
rotation for its inverse leaves a plausible RMSD, so the convention has to live in one place.
"""
from __future__ import annotations

import pytest
import torch

from tt_bio import align


def _of3_pre_extraction(pred_ca, gt_ca):
    """openfold3_fold.kabsch_rmsd exactly as it stood at db56e207."""
    p = pred_ca.double() - pred_ca.double().mean(0)
    g = gt_ca.double() - gt_ca.double().mean(0)
    u, _, vt = torch.linalg.svd(p.t() @ g)
    d = torch.sign(torch.det(vt.t() @ u.t()))
    s = torch.eye(3, dtype=torch.float64)
    s[2, 2] = d
    p_aligned = p @ (vt.t() @ s @ u.t()).t()
    return float(torch.sqrt(((p_aligned - g) ** 2).sum(-1).mean()))


def _px_pre_extraction(a, b):
    """pxdesign/write._kabsch exactly as it stood at db56e207."""
    ca, cb = a.double().mean(0), b.double().mean(0)
    u, _, vt = torch.linalg.svd((a.double() - ca).T @ (b.double() - cb))
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return r, ca, cb


def _clouds(seed: int, n: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, generator=g, dtype=torch.float64) * 10,
            torch.randn(n, 3, generator=g, dtype=torch.float64) * 10)


@pytest.mark.parametrize("n", [4, 7, 32, 61, 196, 298])
def test_rmsd_is_bit_identical_to_the_of3_implementation(n):
    for seed in range(12):
        a, b = _clouds(seed, n)
        assert align.rmsd(a, b) == _of3_pre_extraction(a, b), (n, seed)


@pytest.mark.parametrize("n", [4, 7, 32, 61, 196, 298])
def test_rigid_transform_is_bit_identical_to_the_pxdesign_implementation(n):
    for seed in range(12):
        a, b = _clouds(seed, n)
        r, ma, mb = align.rigid_transform(a, b)
        r0, ma0, mb0 = _px_pre_extraction(a, b)
        assert torch.equal(r, r0) and torch.equal(ma, ma0) and torch.equal(mb, mb0), (n, seed)


def test_the_two_implementations_agreed_with_each_other():
    """The premise of the extraction, measured rather than read off the source: OF3's
    association order and PXDesign's give the same bits, so one module can serve both."""
    for seed in range(50):
        a, b = _clouds(seed, 4 + seed)
        p, g = a - a.mean(0), b - b.mean(0)
        r_px, _, _ = _px_pre_extraction(a, b)
        v_px = float(torch.sqrt(((p @ r_px - g) ** 2).sum(-1).mean()))
        assert v_px == _of3_pre_extraction(a, b), seed


def test_a_rigidly_moved_copy_scores_zero():
    a, _ = _clouds(3, 40)
    theta = torch.tensor(0.7, dtype=torch.float64)
    rot = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                        [torch.sin(theta), torch.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]], dtype=torch.float64)
    b = a @ rot + torch.tensor([3.0, -1.0, 8.0], dtype=torch.float64)
    assert align.rmsd(a, b) < 1e-12


def test_a_mirrored_copy_does_not_align_onto_its_reflection():
    """The determinant correction is the whole reason this is not just an SVD."""
    a, _ = _clouds(5, 40)
    mirrored = a * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    assert align.rmsd(a, mirrored) > 1.0
