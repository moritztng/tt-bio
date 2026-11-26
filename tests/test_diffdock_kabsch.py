"""The SVD-free Kabsch gives upstream's rotation, on the inputs the diffusion loop has.

Graded against a float64 SVD, not against another approximation, and on the quantity that
matters: where the aligned coordinates land, in Angstrom. A rotation that is 1e-6 off in its
matrix entries is not interesting; a coordinate that moves is.
"""

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.kabsch import apply_rigid, kabsch_batch, polar_rotation  # noqa: E402


def svd_reference(a, b):
    """``rigid_transform_Kabsch_3D_torch_batch`` verbatim, in float64."""
    a, b = a.double().permute(0, 2, 1), b.double().permute(0, 2, 1)
    ca, cb = a.mean(dim=2, keepdim=True), b.mean(dim=2, keepdim=True)
    h = (a - ca) @ (b - cb).transpose(1, 2)
    u, _, vt = torch.linalg.svd(h)
    r = vt.transpose(1, 2) @ u.transpose(1, 2)
    flip = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    rm = (vt.transpose(1, 2) @ flip) @ u.transpose(1, 2)
    r = torch.where(torch.linalg.det(r)[:, None, None] < 0, rm, r)
    return r, (-r @ ca) + cb


def torsion_like_pair(batch=10, n=38, angle=0.25, seed=0):
    """Two conformers a torsion update apart: what the loop actually aligns."""
    torch.manual_seed(seed)
    base = torch.randn(batch, n, 3) * 2.0
    axis = torch.nn.functional.normalize(torch.randn(batch, 3), dim=-1)
    moved = base.clone()
    half = n // 2
    k = torch.zeros(batch, 3, 3)
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3).expand(batch, 3, 3)
    rot = eye + torch.sin(torch.tensor(angle)) * k + (1 - torch.cos(torch.tensor(angle))) * (k @ k)
    moved[:, half:] = (base[:, half:] - base[:, half:half + 1]) @ rot.transpose(1, 2) \
        + base[:, half:half + 1]
    return moved, base


def test_polar_rotation_is_orthogonal():
    torch.manual_seed(1)
    m = torch.randn(16, 3, 3)
    q = polar_rotation(m)
    eye = torch.eye(3).expand_as(q)
    assert (q @ q.transpose(-1, -2) - eye).abs().max() < 1e-6


@pytest.mark.parametrize("angle", [0.02, 0.25, 1.1])
def test_aligned_coordinates_match_the_float64_svd(angle):
    a, b = torsion_like_pair(angle=angle)
    r, t, det = kabsch_batch(a, b)
    r_ref, t_ref = svd_reference(a, b)
    assert (det > 0).all(), "the reflection branch cannot arise on a torsion-perturbed copy"
    moved = apply_rigid(a, r, t)
    moved_ref = apply_rigid(a.double(), r_ref, t_ref)
    worst_a = (moved.double() - moved_ref).norm(dim=-1).max()
    assert worst_a < 1e-4, f"{worst_a:.2e} A at angle {angle}"


def test_a_far_rotation_still_converges():
    """Not the loop's regime, but the iteration must not quietly stop converging."""
    torch.manual_seed(2)
    a = torch.randn(8, 38, 3) * 2.0
    axis = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1) * 2.0
    rot = torch.matrix_exp(torch.stack([
        torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        for v in axis]))
    b = a @ rot.transpose(1, 2) + torch.randn(8, 1, 3)
    r, t, det = kabsch_batch(a, b)
    r_ref, t_ref = svd_reference(a, b)
    worst = (apply_rigid(a, r, t).double() - apply_rigid(a.double(), r_ref, t_ref)).norm(dim=-1).max()
    assert worst < 1e-4, f"{worst:.2e} A"


def test_four_newton_steps_reach_the_float32_floor():
    """The step count is measured, not guessed, and the margin is visible.

    If the iteration ever stops converging -- a scaling bug, a sign slip in the adjugate --
    this is where it shows, because two steps are deliberately still 2e-2 A out.
    """
    a, b = torsion_like_pair(angle=0.25)
    r_ref, t_ref = svd_reference(a, b)
    reference = apply_rigid(a.double(), r_ref, t_ref)

    def worst(steps):
        r, t, _ = kabsch_batch(a, b, steps=steps)
        return (apply_rigid(a, r, t).double() - reference).norm(dim=-1).max().item()

    assert worst(2) > 1e-3
    assert worst(3) < 1e-4
    assert worst(4) < 2e-6
    assert worst(8) < 2e-6
