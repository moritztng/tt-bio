# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Transforms for atom name annotation and featurization."""

from typing import ClassVar

import numpy as np
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.conditions import C_SEQ
from tt_bio._vendor.atomworks.ml.transforms._checks import check_atom_array_annotation
from tt_bio._vendor.atomworks.ml.transforms.af3_reference_molecule import _encode_atom_names_like_af3
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray


def annotate_atom_name_visibility(atom_array: AtomArray, annotation_name: str = "should_show_atom_name") -> AtomArray:
    """Annotate which atoms should have their names shown to the model.

    Show atom names for:
    1. Sequence-conditioned standard residues (all atoms)
    2. Non-sequence-conditioned standard residues (backbone only)

    Atomized residues NEVER have atom names shown to maximize transfer learning with small molecules.
    """
    atom_array = atom_array.copy()

    ensure_annotations(
        atom_array,
        "is_protein_backbone",
        "is_nucleic_acid_backbone",
        "is_standard_aa",
        "is_standard_rna",
        "is_standard_dna",
    )

    seq_mask = C_SEQ.mask(atom_array, default="raise")
    is_standard = atom_array.is_standard_aa | atom_array.is_standard_rna | atom_array.is_standard_dna
    is_backbone = atom_array.is_protein_backbone | atom_array.is_nucleic_acid_backbone
    is_atomized = atom_array.atomize

    should_show = (seq_mask & is_standard) | (~seq_mask & is_backbone)
    should_show &= ~is_atomized

    atom_array.set_annotation(annotation_name, should_show)
    return atom_array


class AnnotateAtomNameVisibility(Transform):
    """Annotate which atoms should have their names shown to the model.

    Show atom names for:
    - Sequence-conditioned standard residues: all atoms
    - Sequence-conditioned non-standard residues: backbone atoms only

    Atomized residues NEVER have atom names shown to maximize transfer learning with small molecules.
    """

    requires_previous_transforms: ClassVar[list[str]] = ["AtomizeByCCDName"]

    def __init__(self, annotation_name: str = "should_show_atom_name"):
        super().__init__()
        self.annotation_name = annotation_name

    def forward(self, data: dict) -> dict:
        """Add atom name visibility annotation to atom array."""
        data["atom_array"] = annotate_atom_name_visibility(data["atom_array"], annotation_name=self.annotation_name)
        return data


def featurize_atom_names_like_af3(
    atom_array: AtomArray, visibility_annotation: str = "should_show_atom_name"
) -> np.ndarray:
    """Encode atom names using AF3 encoding scheme.

    Encodes atom names as 4 uint8 values (one per character).
    Masked atoms (indicated by the visibility annotation) get zeros.

    Encoding: ``ord(c) - 32`` → [0, 64] range:
    - Space ' ' (ord 32) → 0
    - 'C' (ord 67) → 35, 'A' (ord 65) → 33
    - Example: "CA  " → [35, 33, 0, 0]

    Args:
        atom_array: Input structure with visibility annotation.
        visibility_annotation: Name of boolean visibility annotation.
            Defaults to ``"should_show_atom_name"``.

    Returns:
        Array of shape ``[n_atoms, 4]`` with encoded atom names.
    """
    n_atoms = atom_array.array_length()
    atom_name_encoded = np.zeros((n_atoms, 4), dtype=np.uint8)

    should_show = getattr(atom_array, visibility_annotation)

    if np.any(should_show):
        atom_names_to_encode = atom_array.atom_name[should_show]
        encoded_values = _encode_atom_names_like_af3(atom_names_to_encode)
        atom_name_encoded[should_show] = encoded_values

    return atom_name_encoded


class FeaturizeAtomNamesLikeAF3(Transform):
    """Featurize atom names using AF3 encoding in an annotation-aware manner.

    Encodes canonical AA/RNA/DNA atom names as 4 uint8 values (one per character).
    Uses the visibility annotation to determine which atoms to encode.

    Output: ``atom_name_af3_encoded`` feature added to ``feats`` dictionary ``[n_atoms, 4]``.

    Pipeline flow (downstream of this transform):

    1. Transform: strings → ``[n_atoms, 4]`` uint8
    2. Collate: pad → ``[B, max_atoms, 4]``
    3. Model: 4 embeddings → concat → Linear → mean pool to tokens

    See Also:
        :py:class:`AnnotateAtomNameVisibility`
    """

    requires_previous_transforms: ClassVar[list[str]] = ["AnnotateAtomNameVisibility"]

    def __init__(self, visibility_annotation: str = "should_show_atom_name"):
        """Initialize the transform.

        Args:
            visibility_annotation: Name of boolean visibility annotation.
                Defaults to ``"should_show_atom_name"``.
        """
        super().__init__()
        self.visibility_annotation = visibility_annotation

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, [self.visibility_annotation])

    def forward(self, data: dict) -> dict:
        """Encode atom names and add to features dict."""
        encoded = featurize_atom_names_like_af3(data["atom_array"], visibility_annotation=self.visibility_annotation)

        if "feats" not in data:
            data["feats"] = {}

        data["feats"]["atom_name_af3_encoded"] = encoded
        return data
