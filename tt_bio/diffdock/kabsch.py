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

**The one case this does not cover is a reflection.** When `det(H) < 0` the closest orthogonal
matrix is not a rotation, and upstream switches to `V diag(1,1,-1) U^T`; that correction
flips the smallest singular direction, which needs the singular vectors, so the polar factor
cannot take it -- it returns the improper orthogonal matrix instead.

It takes a badly distorted conformer to get there, well outside what a torsion update to a
real ligand produces, but it is reachable: a 24-atom free chain with every bond rotated by
~0.3 rad puts 1 pose in 10 on the wrong side (`tests/test_diffdock_conformer.py::
test_a_reflecting_pose_is_flagged_by_the_determinant`). So this is a *detected* limitation
rather than an impossible one: :func:`kabsch_batch` returns the determinant beside the
transform, and the traced loop accumulates it on device and reads it once at the end, the same
pattern the NaN guard uses. Do not reuse this function for aligning *different* structures;
use `tt_bio.align` for those.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Measured against a float64 SVD on torsion-perturbed and randomly rotated clouds
# (tests/test_diffdock_kabsch.py): 2 steps land 2.2e-02 A off, 3 steps 3.3e-05 A, and 4 steps
# reach 1.6e-06 A, which is the fp32 floor and does not improve after. Eight is margin on a
# 3x3, where a step costs three cross products.
NEWTON_STEPS = 8


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
