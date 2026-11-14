# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Internal-coordinate group assignment and pipeline target materialization."""

from typing import ClassVar

from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.ml.conditions import C_CRD
from tt_bio._vendor.atomworks.ml.transforms.base import Transform

from .conditions import C_CRDGRP
from .transforms_utils_conditioning_utils import convert_to_internal_coords, materialize_internal_coords


class ConvertAbsoluteToInternalCoords(Transform):
    """Assign all absolute coordinate conditioning to one internal group for later materialization."""

    # Prohibited prior transforms: this conversion must run before each of these.
    incompatible_previous_transforms: ClassVar[list[str]] = [
        "UnindexFlaggedTokens",
        "CenterOnCoordinateConditionedAtoms",
    ]

    def forward(self, data: dict) -> dict:
        atom_array = as_atom_array_plus(data["atom_array"])

        coord_mask = C_CRD.mask(atom_array, default="generate")
        if coord_mask.any():
            atom_array = convert_to_internal_coords(atom_array, coord_mask)

        # Invariant: no absolute coordinate conditioning survives this transform.
        assert not C_CRD.mask(
            atom_array, default="generate"
        ).any(), "ConvertAbsoluteToInternalCoords: C_CRD mask must be empty after conversion."

        data["atom_array"] = atom_array
        return data


class MaterializeInternalCoords(Transform):
    """Validate coordinate conditioning and fill unset internal groups before unindexing.

    Complete saved targets are preserved; new targets use each group's live coordinates.
    """

    incompatible_previous_transforms: ClassVar[list[str]] = ["UnindexFlaggedTokens"]

    def forward(self, data: dict) -> dict:
        arr = data["atom_array"]
        if not C_CRDGRP.has_annotation(arr):
            C_CRDGRP.set_annotation(arr, C_CRDGRP.default_annotation(arr))
        materialize_internal_coords(arr)
        return data
