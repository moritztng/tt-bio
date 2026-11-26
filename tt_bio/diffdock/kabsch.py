"""Batched Kabsch alignment without an SVD, for the inside of the diffusion loop.

The torsion update in `modify_conformer_batch` ends with a Kabsch superposition of the
torsion-rotated conformer back onto the rigid one, once per pose per step. tt-bio's standing
position is that Kabsch belongs on the host in float64 (`tt_bio/align.py`, skill
`ttnn-host-kabsch`), and for scoring and output that is right: it moves every coordinate we
write. This one is different — it is inside the loop, it feeds the next step, and a host SVD
there is a device sync on every one of the 19 steps, which is what a traced loop exists to
avoid.

So compute the same rotation a different way. Kabsch's rotation is the orthogonal polar
factor of the correlation matrix: for `H = Am Bm^T`, upstream takes `H = U S V^T` and returns
`R = V U^T`, which is exactly the polar factor of `H^T`. The polar factor is reachable by Newton's
iteration `X <- 1/2 (X + X^-T)`, which on a 3x3 is three cross products and a division: no
SVD, no eigenvalues, no data-dependent branch, the same shape every iteration. With Higham's
determinant scaling it converges quadratically from any non-singular start, so it holds up on
a far-from-identity rotation as well as on the near-identity ones the loop produces.

(The inverse-free Newton-Schulz iteration `X <- 1/2 X (3I - X^T X)` is the other candidate and
needs no division, but its rate is set by the smallest singular value: it is still visibly
unconverged after 12 steps on an ill-conditioned matrix, where this one is done in six. A 3x3
inverse is cheap enough that there is no reason to take that trade.)

**A reflection needs the other routine in this module.** When `det(H) < 0` the closest
orthogonal matrix is not a rotation, and upstream switches to `V diag(1,1,-1) U^T`; that
correction flips the smallest singular direction, so the polar factor cannot take it and
returns the improper matrix instead. It is reachable -- a 24-atom free chain with every bond
rotated ~0.3 rad gets there -- and the error when it happens is radian-scale (1.9 on the
matrix entries), not a rounding, so flagging it is not enough. :func:`kabsch_proper` solves
the same problem through Horn's quaternion instead: the optimal quaternion is the dominant
eigenvector of Davenport's symmetric 4x4, every unit quaternion is a proper rotation by
construction, and the eigenvector comes out of ten 4x4 matmuls. That is the one
:func:`tt_bio.diffdock.conformer.modify_conformer` calls; :func:`kabsch_batch` stays as the
cheaper routine for the reflection-free case and because it is what the polar argument above
is about. Neither replaces `tt_bio.align` for scoring alignments: those stay host fp64.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Measured against a float64 SVD on torsion-perturbed and randomly rotated clouds
# (tests/test_diffdock_kabsch.py): 2 steps land 2.2e-02 A off, 3 steps 3.3e-05 A, and 4 steps
# reach 1.6e-06 A, which is the fp32 floor and does not improve after. Eight is margin on a
# 3x3, where a step costs three cross products.
NEWTON_STEPS = 8

# Squarings for :func:`proper_rotation`: `A^(2^SQUARINGS)`. Measured against upstream's full
# Kabsch, reflection branch included, on torsion-perturbed and mirrored clouds: 6 squarings
# land 1.9e-02 out, 8 at 5.5e-05, and 10 reach 9.8e-07 in float32, which is the fp32 floor.
# Ten 4x4 matmuls per pose per step is nothing next to one conv layer.
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


def polar_rotation(m: Tensor, steps: int = NEWTON_STEPS) -> Tensor:
    """The orthogonal polar factor of ``[..., 3, 3]``, by scaled Newton iteration.

    ``m = Q P`` with ``Q`` orthogonal and ``P`` symmetric positive semi-definite; this returns
    ``Q``, the closest orthogonal matrix to ``m`` in the Frobenius norm. A fixed step count,
    so it traces; *steps* is not tuned per input.
    """
    x = m
    for _ in range(steps):
        inv_t, det = _inverse_transpose_and_det(x)
        gamma = _nonzero(det).abs().pow(-1.0 / 3.0)[..., None, None]
        x = 0.5 * (gamma * x + inv_t / gamma)
    return x


def kabsch_batch(a: Tensor, b: Tensor, steps: int = NEWTON_STEPS):
    """Upstream's ``rigid_transform_Kabsch_3D_torch_batch`` without the SVD.

    *a*, *b*: ``[B, N, 3]``. Returns ``(R, t, det)`` with ``R`` ``[B, 3, 3]`` and ``t``
    ``[B, 3, 1]``, in upstream's convention, so the caller applies them the same way:
    ``a @ R.transpose(1, 2) + t.transpose(1, 2)``. *det* is the determinant of the correlation
    matrix; a negative entry means the reflection branch this does not implement.
    """
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError(f"expected two [B, N, 3] clouds, got {tuple(a.shape)} and "
                         f"{tuple(b.shape)}")
    centroid_a, centroid_b = a.mean(dim=1, keepdim=True), b.mean(dim=1, keepdim=True)
    h = (a - centroid_a).transpose(1, 2) @ (b - centroid_b)
    r = polar_rotation(h.transpose(1, 2), steps)
    t = (-r @ centroid_a.transpose(1, 2)) + centroid_b.transpose(1, 2)
    return r, t, _inverse_transpose_and_det(h)[1]


def apply_rigid(pos: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """``pos @ R^T + t^T``, upstream's application order."""
    return pos @ r.transpose(1, 2) + t.transpose(1, 2)


# --------------------------------------------------------------------------- proper rotation

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
    from .conformer import quaternion_to_matrix      # local: conformer imports kabsch
    return quaternion_to_matrix(q)


def kabsch_proper(a: Tensor, b: Tensor, squarings: int = SQUARINGS):
    """:func:`kabsch_batch`'s signature, but always a proper rotation (handles reflections)."""
    centroid_a, centroid_b = a.mean(dim=1, keepdim=True), b.mean(dim=1, keepdim=True)
    h = (a - centroid_a).transpose(1, 2) @ (b - centroid_b)
    r = proper_rotation(h, squarings)
    t = (-r @ centroid_a.transpose(1, 2)) + centroid_b.transpose(1, 2)
    return r, t, _inverse_transpose_and_det(h)[1]
