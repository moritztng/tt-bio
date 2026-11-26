"""Edge spherical harmonics, as e3nn 0.5 computes them for DiffDock.

Every convolution graph in DiffDock turns an edge vector into ``o3.spherical_harmonics(
sh_irreps, edge_vec, normalize=True, normalization='component')`` -- ``1x0e + 1x1o + 1x2e``
for the conv layers and ``2e`` alone for the torsion bonds. This is that function for
``l <= 2``, written out, so tt-bio does not import e3nn to compute nine polynomials.

Conventions that matter, all e3nn's:

* The input is read as ``(x, y, z)`` and the ``l = 1`` block is ``sqrt(3) * (x, y, z)``. (The
  ``(y, z, x)`` ordering e3nn documents belongs to its Wigner-D matrices, i.e. to how an
  ``l = 1`` irrep *rotates*, not to this function's input.)
* ``normalize=True`` goes through ``torch.nn.functional.normalize``, so a zero-length edge
  yields zeros for ``l >= 1`` (and 1 for ``l = 0``), not NaN. That case is real: the centre
  convolution's edge from a one-atom ligand's centre of mass to its only atom has length 0.
* ``component`` normalisation: each degree-``l`` block has squared norm ``2l + 1`` on the unit
  sphere.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_S3, _S5, _S15 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(15.0)
_EPS = 1e-12          # torch.nn.functional.normalize's default, which e3nn relies on


def unit(vec: Tensor) -> Tensor:
    """``F.normalize(vec, dim=-1)`` written out: unit vectors, and zero for a zero vector.

    Spelled out rather than called because the same epsilon has to hold for the rotation axes
    in :mod:`tt_bio.diffdock.conformer`, where a padded bond gives a zero-length axis and a
    NaN there would survive every mask downstream (`0 * NaN` is `NaN`).
    """
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def spherical_harmonics(ls, vec: Tensor) -> Tensor:
    """``[..., 3] -> [..., sum(2l+1)]`` for the degrees in *ls* (each 0, 1 or 2), in order.

    *ls* may be an int, a list of ints, or an irreps string such as ``"1x0e + 1x1o + 1x2e"``
    or ``"2e"``; parity labels are accepted and ignored, as e3nn checks them against
    ``(-1)^l`` and they carry no other information here.
    """
    degrees = _parse_ls(ls)
    if max(degrees) > 2:
        raise ValueError(f"DiffDock never needs l > 2 harmonics; got {degrees}")
    v = unit(vec)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    blocks = []
    for l in degrees:
        if l == 0:
            blocks.append(torch.ones_like(x).unsqueeze(-1))
        elif l == 1:
            blocks.append(_S3 * torch.stack([x, y, z], dim=-1))
        else:
            blocks.append(torch.stack([
                _S15 * x * z,
                _S15 * x * y,
                _S5 * (y * y - 0.5 * (x * x + z * z)),
                _S15 * y * z,
                0.5 * _S15 * (z * z - x * x),
            ], dim=-1))
    return torch.cat(blocks, dim=-1)


def _parse_ls(ls) -> list[int]:
    if isinstance(ls, int):
        return [ls]
    if isinstance(ls, str):
        out = []
        for term in ls.replace(" ", "").split("+"):
            mul, _, ir = term.rpartition("x")
            if mul not in ("", "1"):
                raise ValueError(f"harmonics come with multiplicity 1, got {term!r}")
            out.append(int(ir[:-1]))
        return out
    return [int(l) for l in ls]
