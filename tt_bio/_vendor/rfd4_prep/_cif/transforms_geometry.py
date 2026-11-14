# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
from collections.abc import Sequence

import numpy as np
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.ml.conditions import C_CRD
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray

from .utils_common import true_with_prob


def shift_conditioning_annotations(
    atom_array: AtomArray,
    shift: np.ndarray,
    annotation_names: Sequence[str] = (),
) -> None:
    """Shift coordinate-valued conditioning annotations in place to match a coordinate re-centering.

    Whenever a structure's coordinates are re-centered by subtracting ``shift``, any pre-existing
    coordinate-valued conditioning (e.g. :py:class:`~rfproteina.conditions.ChainCentroid`) must be
    shifted by the same amount, or it will silently refer to the wrong location once the structure
    has moved to its new frame.

    Args:
        atom_array: Atom array to update in place.
        shift: The offset (Å, shape ``(3,)``) being subtracted from ``atom_array.coord``.
        annotation_names: Names of the ``(n_atoms, 3)`` coordinate-valued annotations to shift.
            Each is a no-op if absent from ``atom_array``. Defaults to no annotations.
    """
    categories = atom_array.get_annotation_categories()
    for name in annotation_names:
        if name not in categories:
            continue
        values = atom_array.get_annotation(name)
        finite = np.isfinite(values).all(axis=-1)
        if not finite.any():
            continue
        values = values.copy()
        values[finite] -= shift
        atom_array.set_annotation(name, values)


class RandomRotationTransform(Transform):
    """Apply a uniform SO(3) rotation and center coordinates with probability ``p_apply``."""

    def __init__(self, p_apply: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.p_apply = p_apply

    def forward(self, data: dict) -> dict:
        """Apply random rotation to coordinates."""
        if not true_with_prob(self.p_apply):
            # Skip rotation with probability 1 - p_apply
            return data

        aa = data["atom_array"]

        # Skip if no atoms (can happen after filtering)
        if aa.array_length() == 0:
            return data

        # Center coordinates at origin before rotation
        # This ensures rotation is around the center of mass
        coords = aa.coord  # [n_atoms, 3]

        # Additional safety check for valid coordinates
        if len(coords) == 0 or np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            # Skip rotation if coordinates are invalid
            return data

        com = coords.mean(axis=0, keepdims=True)  # [1, 3]
        centered_coords = coords - com

        # Correct QR's column signs to sample a uniform orthogonal matrix.
        rand_mat = np.random.randn(3, 3)
        q, r = np.linalg.qr(rand_mat)
        q *= np.sign(np.diag(r))

        # Choose the third axis to preserve handedness (SO(3), not a reflection).
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1

        # Apply rotation to centered coordinates
        # coords shape: [n_atoms, 3]
        rotated_coords = centered_coords @ q.T  # Matrix multiplication for rotation

        # Keep coordinates centered; don't add COM back
        # (Later in training, coordinates will be re-centered based on conditioned atoms)
        aa.coord = rotated_coords

        data["atom_array"] = aa
        return data


class CenterOnCoordinateConditionedAtoms(Transform):
    """Center entire structure on the centroid of atoms with coordinate conditioning.

    Centers coordinates by subtracting the centroid of all atoms that are
    coordinate-conditioned (i.e., have known positions). If no atoms are
    coordinate-conditioned, centers on all atoms.

    Args:
        coord_annotation: Annotation name for coordinates to center.
            Defaults to ``"coord_to_be_noised"``.
        conditioning_annotations: Names of coordinate-valued conditioning annotations to re-center
            alongside the coordinates (see :py:func:`shift_conditioning_annotations`). Defaults to
            no annotations; configured via the pipeline (e.g. C_CRD and chain-centroid conditioning).
    """

    def __init__(
        self,
        coord_annotation: str = "coord_to_be_noised",
        conditioning_annotations: Sequence[str] = (),
    ):
        super().__init__()
        self.coord_annotation = coord_annotation
        self.conditioning_annotations = conditioning_annotations

    def forward(self, data: dict) -> dict:
        """Center entire structure on the centroid of the coordinate-conditioned atoms."""
        aa = as_atom_array_plus(data["atom_array"])
        coords = aa.get_annotation(self.coord_annotation)

        # Get coordinate-conditioned mask
        coord_mask = C_CRD.mask(aa, default="generate")

        # If no coordinate-conditioned atoms, use all atoms
        if not coord_mask.any():
            coord_mask = np.ones(aa.array_length(), dtype=bool)

        # Compute centroid of coordinate-conditioned atoms
        # Use nanmean to ignore NaN coordinates from unresolved atoms
        centroid = np.nanmean(coords[coord_mask], axis=0)

        # Center coordinates
        coords = coords - centroid

        # Update annotation
        aa.set_annotation(self.coord_annotation, coords)

        # Any pre-existing coordinate-valued conditioning lives in the same frame as `coords` and
        # must be re-centered by the same amount, or it will point to the wrong location.
        shift_conditioning_annotations(aa, centroid, self.conditioning_annotations)

        data["atom_array"] = aa

        return data
