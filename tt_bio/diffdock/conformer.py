"""The pose update at the end of a diffusion step, in one fixed shape with no host branch.

Upstream's `utils/diffusion_utils.py::modify_conformer_batch` is the last thing a step does:
it takes the score network's translation, rotation and torsion updates and moves the ligand.
Three things in it are shaped by the host rather than by the data, and each one would end a
traced loop:

* **`pos[:, mask_rotate[idx_edge]] = ...`** -- a boolean-mask assignment, so the number of
  rotated atoms is read off the mask at run time and the write is a scatter of a
  data-dependent width. Here the same arithmetic is applied to *every* atom and the mask
  chooses: `where(mask, rotated, pos)`. The rotated value for an unselected atom is computed
  and discarded, which on a 60-atom ligand is free, and the shape never moves.
* **`sin_half_angles_over_angles[~small_angles] = ...`** inside `axis_angle_to_quaternion` --
  two masked assignments into an `empty_like`, which is both a branch and a read of
  uninitialised memory for any angle that hits neither side. Same two expressions, selected
  by `where`, with the division clamped so the small-angle branch cannot produce a NaN that
  the `where` would then propagate (`0 * NaN` is `NaN`, so selecting is not enough -- the
  *unselected* expression has to be finite too).
* **The Kabsch realignment** -- an SVD per pose per step, plus a `where` on the determinant
  for the reflection case. :mod:`tt_bio.diffdock.kabsch` does it as Horn's quaternion, which
  covers both cases in one branch-free expression; see that module for the grade.

The torsion loop itself stays sequential: bond `k`'s rotation axis is read from coordinates
that bonds `0..k-1` have already moved, so the updates do not commute and cannot be fused.
That is fine for a trace -- the bond count is fixed per complex, so the loop unrolls into a
fixed instruction sequence. It is not fine for a `for` loop over a *tensor* of bond indices:
:func:`apply_torsions` takes the pairs as Python ints precisely so the unrolling happens at
capture time. The topology is static per complex, so that is a property of the port, not an
assumption about the data.

Padding: a pad bond row (both ends the pad atom) gives a zero-length rotation axis, which
`unit` sends to zero, which gives the identity quaternion -- and its mask row is all zero, so
it writes nothing either way. A pad atom's mask column is zero in every row, so it never
moves.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .harmonics import unit
from .kabsch import apply_rigid, kabsch_proper

_SMALL_ANGLE = 1e-6      # upstream's eps, kept so the two agree term for term


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """``[..., 3] -> [..., 3, 3]``, upstream's rotation convention, branch-free.

    Magnitude is the angle in radians, direction is the axis, anticlockwise. Identical to
    `utils/geometry.py::axis_angle_to_matrix` (axis-angle -> quaternion -> matrix) except
    that the small-angle case is selected rather than scattered.
    """
    angles = axis_angle.norm(dim=-1, keepdim=True)
    half = 0.5 * angles
    # Clamp only the divisor: for angles above eps this changes nothing, and below eps the
    # quotient is discarded by the `where` -- but it still has to be finite to be discarded.
    safe = angles.clamp_min(_SMALL_ANGLE)
    sin_over = torch.where(
        angles.abs() < _SMALL_ANGLE,
        0.5 - (angles * angles) / 48,      # sin(x/2)/x to second order
        torch.sin(half) / safe,
    )
    q = torch.cat([torch.cos(half), axis_angle * sin_over], dim=-1)
    return quaternion_to_matrix(q)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """``[..., 4] -> [..., 3, 3]``, real part first. Upstream's formula verbatim.

    The ``2 / |q|^2`` factor normalises as it goes, so an unnormalised quaternion is fine.
    """
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def rotatable_bond_pairs(edge_index: Tensor, edge_mask: Tensor) -> list[tuple[int, int]]:
    """Upstream's ``edge_index.T[edge_mask]`` as the ``(u, v)`` list :func:`apply_torsions` wants.

    *edge_index* is one pose's ligand bond graph ``[2, M]``, *edge_mask* the boolean
    rotatable-bond selector ``[M]``. Python ints, because the torsion loop unrolls at capture.
    """
    pairs = edge_index[:, edge_mask.bool()].t().tolist()
    return [(int(u), int(v)) for u, v in pairs]


def apply_torsions(pos: Tensor, bonds, mask_rotate: Tensor, updates: Tensor) -> Tensor:
    """``modify_conformer_torsion_angles_batch`` with a select instead of a masked write.

    *pos*: ``[B, N, 3]``. *bonds*: the ``(u, v)`` pairs from :func:`rotatable_bond_pairs`,
    ``v`` on the side that rotates. *mask_rotate*: ``[R, N]``, 1 where that bond's rotation
    moves the atom. *updates*: ``[B, R]`` angles in radians.
    """
    if mask_rotate.shape[0] != len(bonds) or updates.shape[-1] != len(bonds):
        raise ValueError(f"{len(bonds)} bonds, mask {tuple(mask_rotate.shape)}, "
                         f"updates {tuple(updates.shape)}")
    if mask_rotate.shape[1] != pos.shape[1]:
        raise ValueError(f"mask has {mask_rotate.shape[1]} atoms, positions have {pos.shape[1]}")
    sel = mask_rotate.to(torch.bool)[:, None, :, None]          # [R, 1, N, 1]
    for idx, (u, v) in enumerate(bonds):
        pivot = pos[:, v:v + 1]                                 # [B, 1, 3]
        axis = unit(pos[:, u] - pos[:, v])                      # zero-safe
        rot = axis_angle_to_matrix(axis * updates[:, idx:idx + 1])
        rotated = torch.bmm(pos - pivot, rot.transpose(1, 2)) + pivot
        pos = torch.where(sel[idx], rotated, pos)
    return pos


def modify_conformer(pos: Tensor, tr_update: Tensor, rot_update: Tensor,
                     torsion_updates: Tensor | None, bonds, mask_rotate: Tensor | None):
    """Upstream's ``modify_conformer_batch``: rigid move, torsions, realign. Returns ``(pos, det)``.

    *pos*: ``[B, N, 3]``, one pose per row. *tr_update*: ``[B, 3]``. *rot_update*: ``[B, 3]``
    axis-angle. *torsion_updates*: ``[B, R]`` or ``None`` for a rigid ligand. *det* is the
    correlation determinant out of :func:`~tt_bio.diffdock.kabsch.kabsch_batch`, ``None`` when
    there are no torsions; accumulate it on device and read it once after the loop.
    """
    centre = pos.mean(dim=1, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update)
    rigid = torch.bmm(pos - centre, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + centre
    if torsion_updates is None or not len(bonds):
        return rigid, None
    flexible = apply_torsions(rigid, bonds, mask_rotate, torsion_updates)
    r, t, det = kabsch_proper(flexible, rigid)
    return apply_rigid(flexible, r, t), det
