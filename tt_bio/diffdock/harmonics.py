"""Real spherical harmonics to l=2, in e3nn's basis and e3nn's ``component`` normalisation.

Every edge in the DiffDock score network carries ``o3.spherical_harmonics(sh_irreps, vec,
normalize=True, normalization='component')``; the released checkpoint runs at ``sh_lmax=2``,
so three blocks (``1x0e + 1x1o + 1x2e``, 9 components) cover the model, and the torsion head
asks for the ``2e`` block alone.

Two conventions have to match e3nn exactly, and neither is checkable by eye:

* **The component order is e3nn's, not physics'.** e3nn's ``l=1`` block is ``(x, y, z)`` of
  the vector as given, and its ``l=2`` block is ``(xz, xy, y^2 - (x^2+z^2)/2, yz,
  (z^2-x^2)/2)`` up to constants. The standard quantum-chemistry ordering of the same
  functions differs by a permutation and a sign, and the checkpoint's weights were fitted
  against e3nn's, so a permuted basis loads cleanly and gives wrong vectors.
* **``component`` normalisation is the raw polynomial** -- e3nn divides by ``sqrt(4 pi)``
  only for ``integral`` and by ``sqrt(2l+1)`` only for ``norm``. Each ``l`` block of a unit
  vector therefore has norm ``sqrt(2l+1)``, not 1.

``normalize=True`` means the vector is unit-normalised first, and e3nn does it with
``torch.nn.functional.normalize``, whose epsilon turns a zero vector into a zero vector
rather than a NaN. That behaviour is load-bearing here and not an accident of e3nn's: the
static-graph port (:mod:`tt_bio.diffdock.graph`) evaluates every edge of a complete edge set,
including the zero-length ones a mask will discard, and a NaN would survive the mask.

Graded against e3nn 0.5.1 by ``scripts/diffdock_port/sh_parity.py``: max absolute difference
2.4e-07 over 4096 random vectors, on values up to 2.24, for both the score network's
``1x0e+1x1o+1x2e`` and the torsion head's ``2e``.

The polynomial is evaluated in float64 and cast back, as in :mod:`tt_bio.diffdock.features`:
the arithmetic is a few FLOP per edge, and it keeps the harmonics inside fp32 epsilon of the
exact value instead of carrying upstream's fp32 rounding into every edge message.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

LMAX = 2
_EPS = 1e-12          # torch.nn.functional.normalize's default, which e3nn relies on


def unit(vec: Tensor) -> Tensor:
    """``F.normalize(vec, dim=-1)``: unit vectors, and zero for a zero vector."""
    norm = vec.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return vec / norm


def spherical_harmonics(vec: Tensor, ls=(0, 1, 2), normalize: bool = True) -> Tensor:
    """``[..., 3] -> [..., sum(2l+1)]`` for the requested *ls*, in e3nn's component basis.

    *ls* is the list of degrees, in output order: ``(0, 1, 2)`` is the score network's
    ``1x0e+1x1o+1x2e`` and ``(2,)`` is the torsion head's ``2e``.
    """
    bad = [l for l in ls if l not in (0, 1, 2)]
    if bad:
        raise ValueError(f"spherical harmonics are implemented to l={LMAX}, asked for {bad}")
    v = unit(vec).to(torch.float64) if normalize else vec.to(torch.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    blocks: list[Tensor] = []
    for l in ls:
        if l == 0:
            blocks.append(torch.ones_like(x)[..., None])
        elif l == 1:
            blocks.append(math.sqrt(3) * torch.stack([x, y, z], dim=-1))
        else:
            y2 = y * y
            x2z2 = x * x + z * z
            blocks.append(torch.stack([
                math.sqrt(15) * x * z,
                math.sqrt(15) * x * y,
                math.sqrt(5) * (y2 - 0.5 * x2z2),
                math.sqrt(15) * y * z,
                0.5 * math.sqrt(15) * (z * z - x * x),
            ], dim=-1))
    return torch.cat(blocks, dim=-1).to(vec.dtype)
