# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Project-specific StandardAnnotations.

Registering these as StandardAnnotations (rather than plain extra_fields) gives:
- Automatic saving to CIF files (no need to list in ``extra_fields``).
- Easy loading from CIF files through ``parse``
- Default generation via
  :py:meth:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase.default_annotation`.
"""

import numpy as np
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level, StandardAnnotationBase
from biotite.structure import AtomArray


class SrcCoordinateConditionedRegion(StandardAnnotationBase):
    """Per-atom source identifier for the coordinate-conditioned region.

    Contig-style string identifier naming the source component that provided the coordinate
    conditioning for each atom. ``"."`` means the atom was not coordinate-conditioned.
    """

    name = "src-coordinate-conditioned-region"
    aliases = ("SA_CRDSRC",)
    n_body = 1
    level = Level.ATOM
    dtype = np.dtype("<U10")

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != "."

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``"."`` (unconditioned) for all atoms."""
        return np.full(atom_array.array_length(), ".", dtype="<U10")


class SrcComponent(StandardAnnotationBase):
    """Per-atom source component identifier.

    Contig-style string identifier naming the source component each atom
    came from. ``"."`` means the component is unassigned.
    """

    name = "src-component"
    aliases = ("SA_SRC",)
    n_body = 1
    level = Level.ATOM
    dtype = np.dtype("<U10")

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != "."

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``"."`` (unassigned) for all atoms."""
        return np.full(atom_array.array_length(), ".", dtype="<U10")


class SrcIndexGroup(StandardAnnotationBase):
    """Per-atom source index group identifier.

    Integer recording the original *unindexed* ``IndexGroup`` a mapped motif token came from,
    before insertion re-derives ``C_IDXGRP`` into the indexed frame. ``-1`` means the token was
    not an unindexed insertion.
    """

    name = "src-index-group"
    aliases = ("SA_IDXGRPSRC",)
    n_body = 1
    level = Level.ATOM
    dtype = np.int32

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != -1

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``-1`` (not mapped) for all atoms."""
        return np.full(atom_array.array_length(), -1, dtype=np.int32)


class IsUnindexedInsertion(StandardAnnotationBase):
    """Per-atom mask flagging atoms that map to unindexed in-context conditioning regions.

    Set by :py:func:`~rfproteina.transforms.unindexing.annotate_unindexed_insertion`.
    ``True`` means the atom was identified as the insertion target for an unindexed token.
    """

    name = "is-unindexed-insertion"
    aliases = ("SA_UII",)
    n_body = 1
    level = Level.ATOM
    dtype = bool

    @classmethod
    def get_full_name(cls) -> str:
        return "is_unindexed_insertion"

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: all-False (no unindexed insertion)."""
        return np.zeros(atom_array.array_length(), dtype=bool)


class CoordsWithInsertion(StandardAnnotationBase):
    """Per-atom reference coordinates for coordinate-conditioned atoms.

    Defined as:
    - Absolute coords from ``C_CRD`` plus the per-group superposition of the ``C_INTCRD``
      internal-coord reference onto the generated structure
    - NaN where an atom is not coordinate-conditioned

    Useful for computing motif-RMSD without needing to run any superposition; registered
    as a standard annotation so it is automatically saved to CIF and survives re-parse.
    """

    name = "coords-with-insertion"
    aliases = ("SA_CWI",)
    n_body = 1
    level = Level.ATOM
    dtype = np.float32
    value_shape = (3,)

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return ~np.isnan(annotation).any(axis=-1)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: NaN coordinates for all atoms."""
        return np.full((atom_array.array_length(), 3), np.nan, dtype=np.float32)


# --- Convenience aliases ---
SA_CRDSRC = SrcCoordinateConditionedRegion
SA_SRC = SrcComponent
SA_IDXGRPSRC = SrcIndexGroup
SA_UII = IsUnindexedInsertion
SA_CWI = CoordsWithInsertion
