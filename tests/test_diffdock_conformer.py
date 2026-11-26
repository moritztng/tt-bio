"""`tt_bio.diffdock.conformer` against upstream's own pose update.

The reference implementations below are DiffDock's `utils/geometry.py` and `utils/torsion.py`
copied verbatim (MIT, gcorso/DiffDock) so the comparison does not depend on the upstream
checkout being installed. They keep upstream's masked assignments and its SVD; the port
replaces both, and these tests are what says the replacement did not change the answer.
"""

from __future__ import annotations

import math

import pytest
import torch

from tt_bio.diffdock.conformer import (apply_torsions, axis_angle_to_matrix, modify_conformer,
                                       quaternion_to_matrix)

# The torsion realignment goes through the SVD-free Kabsch (Horn's quaternion by repeated
# squaring), whose measured float32 floor against upstream's SVD is ~1e-06 on the rotation
# matrix; everything else in the update is term-for-term identical to upstream.
POSE_TOL = 2e-4
EXACT_TOL = 1e-6


# --------------------------------------------------------------------------- upstream, verbatim

def ref_axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)


def ref_quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def ref_axis_angle_to_matrix(axis_angle):
    return ref_quaternion_to_matrix(ref_axis_angle_to_quaternion(axis_angle))


def ref_modify_conformer_torsion_angles_batch(pos, edge_index, mask_rotate, torsion_updates):
    pos = pos + 0
    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]
        rot_vec = pos[:, u] - pos[:, v]
        rot_mat = ref_axis_angle_to_matrix(
            rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True)
            * torsion_updates[:, idx_edge:idx_edge + 1])
        pos[:, mask_rotate[idx_edge]] = (
            torch.bmm(pos[:, mask_rotate[idx_edge]] - pos[:, v:v + 1],
                      torch.transpose(rot_mat, 1, 2)) + pos[:, v:v + 1])
    return pos


def ref_kabsch_batch(A, B):
    A, B = A.permute(0, 2, 1), B.permute(0, 2, 1)
    centroid_A, centroid_B = A.mean(dim=2, keepdim=True), B.mean(dim=2, keepdim=True)
    H = torch.bmm(A - centroid_A, (B - centroid_B).transpose(1, 2))
    U, _, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    # The reflection branch, which upstream has and an early draft of this reference did not:
    # without it the "upstream" answer is an improper matrix and the port looks wrong where it
    # is right. It fires on a badly distorted conformer, which a torsion update can produce.
    SS = torch.diag(torch.tensor([1., 1., -1.], dtype=A.dtype, device=A.device))
    Rm = torch.bmm(Vt.transpose(1, 2) @ SS, U.transpose(1, 2))
    R = torch.where(torch.linalg.det(R)[:, None, None] < 0, Rm, R)
    t = -torch.bmm(R, centroid_A) + centroid_B
    return R, t


def ref_modify_conformer_batch(pos, tr_update, rot_update, torsion_updates, bonds, mask_rotate):
    lig_center = torch.mean(pos, dim=1, keepdim=True)
    rot_mat = ref_axis_angle_to_matrix(rot_update)
    rigid = torch.bmm(pos - lig_center, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + lig_center
    if torsion_updates is None:
        return rigid
    flexible = ref_modify_conformer_torsion_angles_batch(
        rigid, torch.tensor(bonds), mask_rotate, torsion_updates)
    R, t = ref_kabsch_batch(flexible, rigid)
    return torch.bmm(flexible, R.transpose(1, 2)) + t.transpose(1, 2)


# --------------------------------------------------------------------------- a toy ligand

def chain_ligand(n_atoms=18, n_poses=4, seed=0):
    """A chain with every bond rotatable: bond k is (k, k+1), rotating atoms k+1 and beyond.

    The tree structure is what makes the mask legal (upstream asserts `v` is on the moving
    side and `u` is not), and a chain is the worst case for the sequential torsion loop --
    every bond's axis has been moved by every earlier bond.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.cumsum(torch.randn(n_poses, n_atoms, 3, generator=g), dim=1) * 1.5
    bonds = [(k, k + 1) for k in range(n_atoms - 1)]
    mask = torch.zeros(len(bonds), n_atoms, dtype=torch.bool)
    for k in range(len(bonds)):
        mask[k, k + 1:] = True
    return pos, bonds, mask, g


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-7, 9.9e-7, 1.1e-6, 1e-3, 0.5, 2.0, math.pi])
def test_axis_angle_matches_upstream_on_both_sides_of_its_epsilon(angle):
    axis = torch.nn.functional.normalize(torch.tensor([[0.3, -0.7, 0.4]]), dim=-1)
    v = axis * angle
    got, want = axis_angle_to_matrix(v), ref_axis_angle_to_matrix(v)
    assert torch.isfinite(got).all()
    assert (got - want).abs().max().item() < EXACT_TOL


def test_axis_angle_matches_upstream_on_a_random_batch():
    g = torch.Generator().manual_seed(7)
    v = torch.randn(512, 3, generator=g) * 1.4
    assert (axis_angle_to_matrix(v) - ref_axis_angle_to_matrix(v)).abs().max().item() < EXACT_TOL


def test_a_zero_rotation_is_the_identity_not_a_nan():
    """The branch the port removed read uninitialised memory; `where` must not reintroduce it."""
    m = axis_angle_to_matrix(torch.zeros(3, 3))
    assert torch.isfinite(m).all()
    assert (m - torch.eye(3)).abs().max().item() == 0.0


def test_quaternion_to_matrix_matches_upstream_unnormalised():
    g = torch.Generator().manual_seed(3)
    q = torch.randn(64, 4, generator=g) * 2.0
    assert (quaternion_to_matrix(q) - ref_quaternion_to_matrix(q)).abs().max().item() < EXACT_TOL


def test_apply_torsions_matches_upstreams_masked_write():
    pos, bonds, mask, g = chain_ligand()
    updates = torch.randn(pos.shape[0], len(bonds), generator=g)
    got = apply_torsions(pos, bonds, mask, updates)
    want = ref_modify_conformer_torsion_angles_batch(pos, torch.tensor(bonds), mask, updates)
    assert (got - want).abs().max().item() < EXACT_TOL


def test_the_full_pose_update_matches_upstream():
    """A realistic step: small torsion updates, so the realignment stays a proper rotation."""
    pos, bonds, mask, g = chain_ligand(n_atoms=24, n_poses=10, seed=11)
    tr = torch.randn(pos.shape[0], 3, generator=g) * 2.0
    rot = torch.randn(pos.shape[0], 3, generator=g) * 0.4
    tor = torch.randn(pos.shape[0], len(bonds), generator=g) * 0.08
    got, det = modify_conformer(pos, tr, rot, tor, bonds, mask)
    want = ref_modify_conformer_batch(pos, tr, rot, tor, bonds, mask)
    assert (det > 0).all(), "a reflection would mean the polar factor is the wrong branch"
    assert (got - want).abs().max().item() < POSE_TOL


def test_a_reflecting_pose_still_matches_upstream():
    """The case the polar factor cannot reach, and the reason `modify_conformer` uses Horn's.

    A 24-atom free chain with every bond rotated ~0.3 rad puts poses at `det(H) < 0`, where
    upstream switches to `V diag(1,1,-1) U^T`. The polar factor returns the improper matrix
    there and is 1.9 out on the matrix entries -- radian-scale, not rounding. Horn's quaternion
    gets both cases from one expression, so this test is a parity test and not a caveat.
    """
    pos, bonds, mask, g = chain_ligand(n_atoms=24, n_poses=10, seed=11)
    tr = torch.randn(pos.shape[0], 3, generator=g) * 2.0
    rot = torch.randn(pos.shape[0], 3, generator=g) * 0.4
    tor = torch.randn(pos.shape[0], len(bonds), generator=g) * 0.3
    got, det = modify_conformer(pos, tr, rot, tor, bonds, mask)
    want = ref_modify_conformer_batch(pos, tr, rot, tor, bonds, mask)
    assert (det < 0).any(), "fixture should contain the reflection case"
    assert (got - want).abs().max().item() < POSE_TOL


def test_the_rigid_only_path_preserves_every_interatomic_distance():
    pos, bonds, mask, g = chain_ligand(n_atoms=16, n_poses=3, seed=5)
    tr = torch.randn(3, 3, generator=g) * 3.0
    rot = torch.randn(3, 3, generator=g)
    got, det = modify_conformer(pos, tr, rot, None, bonds, mask)
    assert det is None
    d0 = torch.cdist(pos, pos)
    assert (torch.cdist(got, got) - d0).abs().max().item() < 1e-4
    want = ref_modify_conformer_batch(pos, tr, rot, None, bonds, mask)
    assert (got - want).abs().max().item() < EXACT_TOL


def test_pad_atoms_and_pad_bonds_change_nothing():
    """A padded ligand must give the real atoms the same coordinates as the unpadded one."""
    pos, bonds, mask, g = chain_ligand(n_atoms=20, n_poses=4, seed=2)
    n_real, n_bonds = pos.shape[1], len(bonds)
    tr = torch.randn(4, 3, generator=g)
    rot = torch.randn(4, 3, generator=g) * 0.3
    tor = torch.randn(4, n_bonds, generator=g) * 0.4

    pad_atoms, pad_bonds = 12, 12          # to the next multiple of 32 in both axes
    # Pad atoms sit at the origin, pad bonds point at the pad atom from the pad atom, and the
    # ligand centre is taken over the real atoms only -- so the padded run must be told which
    # atoms are real. Here the rigid centring is the only place that matters, so compare the
    # torsion stage, which is what the padding has to survive.
    ppos = torch.cat([pos, torch.zeros(4, pad_atoms, 3)], dim=1)
    pmask = torch.zeros(n_bonds + pad_bonds, n_real + pad_atoms, dtype=torch.bool)
    pmask[:n_bonds, :n_real] = mask
    pbonds = bonds + [(n_real + pad_atoms - 1, n_real + pad_atoms - 1)] * pad_bonds
    ptor = torch.cat([tor, torch.zeros(4, pad_bonds)], dim=1)

    got = apply_torsions(ppos, pbonds, pmask, ptor)
    want = apply_torsions(pos, bonds, mask, tor)
    assert torch.isfinite(got).all(), "a zero-length pad bond must not make a NaN axis"
    assert (got[:, :n_real] - want).abs().max().item() < EXACT_TOL
    assert (got[:, n_real:]).abs().max().item() == 0.0
