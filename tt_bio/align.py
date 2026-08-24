"""Rigid (Kabsch) superposition, host fp64. One implementation for the shipped callers.

Two shipped sites computed the same Kabsch in different notation: ``openfold3_fold.kabsch_rmsd``
returned the scalar, ``pxdesign/write._kabsch`` returned the transform. Same correlation matrix,
same determinant correction, same convention, applied the same way -- verified bit-identical over
400 random point clouds, including the two association orders they used
(``(vtᵀ S uᵀ)ᵀ`` against ``u S vt``), which is why the extraction can be exact rather than close.

Worth one module because ``kabsch-inverse-rotation-swap-phantom-rmsd`` is a real shipped bug of
exactly this shape: swap the rotation for its inverse and the RMSD is still a plausible number.

``rfd3/featurize._kabsch_align`` deliberately does NOT come here. It is a bit-faithful port of
``rfd3.inference.symmetry.frames._align``: float32 on purpose and translations discarded on
purpose, so moving it would change RFD3's numbers to match a convention it does not share. The
~20 copies under ``scripts/`` and ``perf/`` stay too, for the reason every unify sweep gives
about archived probes: their recorded numbers came from that exact code.

Alignment is a scoring and output step, not a device op. Keep it in host fp64: a device-side
approximation moves every coordinate we write (see the ``ttnn-host-kabsch`` skill).
"""

from __future__ import annotations

import torch


def rigid_transform(a: torch.Tensor, b: torch.Tensor):
    """The rigid map taking ``a`` onto ``b``: ``x -> (x - mean_a) @ R + mean_b``.

    ``a``, ``b``: ``[N, 3]``. Returns ``(R, mean_a, mean_b)``, all float64. The
    determinant correction keeps ``R`` a rotation, so a mirrored copy scores as
    mirrored instead of aligning onto its reflection.
    """
    a, b = a.double(), b.double()
    mean_a, mean_b = a.mean(0), b.mean(0)
    u, _, vt = torch.linalg.svd((a - mean_a).T @ (b - mean_b))
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return r, mean_a, mean_b


def rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """Optimal-superposition RMSD between two ``[N, 3]`` point clouds."""
    r, mean_a, mean_b = rigid_transform(a, b)
    aligned = (a.double() - mean_a) @ r
    return float(torch.sqrt(((aligned - (b.double() - mean_b)) ** 2).sum(-1).mean()))
