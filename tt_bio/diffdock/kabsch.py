"""The in-loop Kabsch for a traced step: Horn's quaternion, no SVD and no branch.

`tt_bio/diffdock/sampling.py::modify_conformer` -- the canonical step, graded bit-identical
against upstream -- ends with upstream's Kabsch: an SVD per pose, then a `where` on the
determinant that swaps in `V diag(1,1,-1) U^T` when the best orthogonal map is a reflection.
Both halves are host-shaped. A traced loop cannot take the SVD, and a branch-free
replacement has to cover the reflection case, not just flag it: when it fires, the obvious
SVD-free route (the orthogonal polar factor by Newton iteration) returns the improper matrix
and is 1.9 out on the rotation entries. That case is reachable inside the loop -- a torsion
update at an early step is large enough -- which is D-dfd-trace-5.

Horn's formulation avoids it by construction. The optimal rotation is the unit quaternion
maximising `q^T K q` for Davenport's symmetric 4x4 `K`; every unit quaternion is a *proper*
rotation, so there is no reflection branch to take. The dominant eigenvector comes out of ten
4x4 squarings (`A^1024`, rescaled each time) with the column picked by a device argmax.
Graded against sampling's SVD Kabsch, reflection case included: 9.8e-07 in float32.

This is for the traced step only. Scoring and output alignments stay in `tt_bio/align.py`,
host fp64; the eager sampler keeps its own SVD Kabsch, which is upstream's arithmetic.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Squarings for :func:`proper_rotation`: `A^(2^SQUARINGS)`. Measured against the SVD Kabsch,
# reflection branch included, on torsion-perturbed and mirrored clouds: 6 squarings land
# 1.9e-02 out, 8 at 5.5e-05, and 10 reach 9.8e-07 in float32, which is the fp32 floor.
SQUARINGS = 10


def _inverse_transpose_and_det(m: Tensor):
    """``(inv(m)^T, det(m))`` for ``[..., 3, 3]``, from the adjugate: rows of cross products.

    For rows ``r0, r1, r2``, ``inv(m)`` has ``r1 x r2``, ``r2 x r0``, ``r0 x r1`` as its
    *columns*, so the transpose has them as rows and no transpose is actually taken. The
    determinant falls out of the same three cross products. Written this way rather than with
    ``torch.linalg.det`` and ``torch.linalg.inv`` because both of those are host LAPACK calls
    with no ttnn equivalent, and this is cross products and one division.
    """
    r0, r1, r2 = m[..., 0, :], m[..., 1, :], m[..., 2, :]
    c0 = torch.cross(r1, r2, dim=-1)
    adj_t = torch.stack([c0, torch.cross(r2, r0, dim=-1), torch.cross(r0, r1, dim=-1)], dim=-2)
    det = (r0 * c0).sum(-1)
    return adj_t / _nonzero(det)[..., None, None], det


def _nonzero(x: Tensor) -> Tensor:
    """Keep a division finite without changing any value that was not already degenerate."""
    tiny = torch.finfo(x.dtype).tiny
    return torch.where(x.abs() < tiny, torch.full_like(x, tiny), x)


def _davenport_k(h: Tensor) -> Tensor:
    """Davenport's symmetric 4x4 ``K`` for a correlation matrix ``H`` (``[..., 3, 3]``).

    The quaternion maximising ``q^T K q`` is the *proper* rotation that best superposes the
    two clouds -- Horn's solution. Unlike the polar factor it cannot return a reflection,
    because every unit quaternion is a rotation by construction.
    """
    xx, xy, xz = h[..., 0, 0], h[..., 0, 1], h[..., 0, 2]
    yx, yy, yz = h[..., 1, 0], h[..., 1, 1], h[..., 1, 2]
    zx, zy, zz = h[..., 2, 0], h[..., 2, 1], h[..., 2, 2]
    rows = [
        torch.stack([xx + yy + zz, yz - zy, zx - xz, xy - yx], dim=-1),
        torch.stack([yz - zy, xx - yy - zz, xy + yx, zx + xz], dim=-1),
        torch.stack([zx - xz, xy + yx, yy - xx - zz, yz + zy], dim=-1),
        torch.stack([xy - yx, zx + xz, yz + zy, zz - xx - yy], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def proper_rotation(h: Tensor, squarings: int = SQUARINGS) -> Tensor:
    """The best *proper* rotation for correlation matrix ``H``, by repeated squaring.

    ``K + cI`` with ``c`` a Gershgorin bound is positive semi-definite, so its dominant
    eigenvector is ``K``'s -- the optimal quaternion. Rather than iterate the matrix-vector
    product a few hundred times, square the matrix *squarings* times (``A^256`` in eight
    4x4 matmuls, rescaled each time so nothing overflows); the result is ``lambda v v^T`` to
    working precision, so any non-degenerate column of it is the eigenvector up to scale. The
    column is picked by the largest diagonal entry, which is ``lambda v_i^2`` -- a device
    argmax and a gather, not a host branch, and it cannot pick a column that has cancelled.

    This is the branch upstream takes with ``V diag(1,1,-1) U^T`` when ``det(H) < 0``, and the
    one :func:`polar_rotation` cannot reach. Returns ``R`` in upstream's convention.
    """
    k = _davenport_k(h)
    shift = k.abs().sum(dim=-1).amax(dim=-1)[..., None, None]     # Gershgorin
    a = k + shift * torch.eye(4, dtype=h.dtype, device=h.device)
    for _ in range(squarings):
        a = a @ a
        a = a / a.abs().amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(
            torch.finfo(h.dtype).tiny)
    diag = a.diagonal(dim1=-2, dim2=-1)                           # lambda * v_i^2, non-negative
    pick = diag.argmax(dim=-1)[..., None, None].expand(a.shape[:-2] + (4, 1))
    q = a.gather(-1, pick)[..., 0]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(h.dtype).tiny)
    return quaternion_to_matrix(q)


def kabsch_horn(a: Tensor, b: Tensor, squarings: int = SQUARINGS):
    """Sampling's ``kabsch_batch`` contract -- `b ~ a @ R.T + t.T`, both ``[B, N, 3]`` -- plus ``det(H)``.

    ``det`` is returned as a diagnostic only: a negative entry is a step where upstream took
    its reflection branch, which this matches without branching.
    """
    centroid_a, centroid_b = a.mean(dim=1, keepdim=True), b.mean(dim=1, keepdim=True)
    h = (a - centroid_a).transpose(1, 2) @ (b - centroid_b)
    r = proper_rotation(h, squarings)
    t = (-r @ centroid_a.transpose(1, 2)) + centroid_b.transpose(1, 2)
    return r, t, _inverse_transpose_and_det(h)[1]


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """``[..., 4] -> [..., 3, 3]``, real part first; the ``2/|q|^2`` factor normalises as it goes.

    The same formula sampling's ``axis_angle_to_matrix`` inlines; split out here because Horn's
    method produces a quaternion directly.
    """
    r, i, j, k = torch.unbind(q, -1)
    two = 2.0 / (q * q).sum(-1)
    out = torch.stack([
        1 - two * (j * j + k * k), two * (i * j - k * r), two * (i * k + j * r),
        two * (i * j + k * r), 1 - two * (i * i + k * k), two * (j * k - i * r),
        two * (i * k - j * r), two * (j * k + i * r), 1 - two * (i * i + j * j),
    ], dim=-1)
    return out.reshape(q.shape[:-1] + (3, 3))
