# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
from enum import IntEnum
from typing import ClassVar, Literal

import numpy as np
from tt_bio._vendor.atomworks.constants import STANDARD_POLYMER_RESIDUES
from tt_bio._vendor.atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
)
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AnnotationList2D
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_IDX, CONDITIONS, ConditionBase
from tt_bio._vendor.atomworks.ml.transforms.dssp import SSEnum
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Int, Shaped

from .encoding import UNIFIED_ATOM37_ENCODING

Array = np.ndarray


# TODO: Generalize to give both hotspots and antihotspots!
class Hotspot(ConditionBase):
    """Hotspot annotation for protein design."""

    name = "hotspot"
    n_body = 1
    level = Level.ATOM
    dtype = bool
    aliases = ("C_HOT", "mask_hotspot_1_atom")
    default_value = False


class RasaEnum(IntEnum):
    UNSPECIFIED = 0
    BURIED = 1
    INTERMEDIATE = 2
    EXPOSED = 3


class Rasa(ConditionBase):
    """Conditioning that provides binned RASA values to the model. Zero denotes no conditioning."""

    name = "rasa-binned"
    n_body = 1
    level = Level.ATOM
    dtype = RasaEnum
    aliases = ("C_RAS",)
    default_value = RasaEnum.UNSPECIFIED

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != RasaEnum.UNSPECIFIED.value


class HBondEnum(IntEnum):
    NONE = 0
    DONOR = 1
    ACCEPTOR = 2
    BOTH = 3  # DONOR | ACCEPTOR -- for atoms that donate in one H-bond and accept in another


class HydrogenBond(ConditionBase):
    """Conditioning that provides hydrogen bond information (donor/acceptor) to the model."""

    name = "hbond"
    n_body = 1
    level = Level.ATOM
    dtype = HBondEnum
    aliases = ("C_HB",)
    default_value = HBondEnum.NONE

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != HBondEnum.NONE.value


class PiStacking(ConditionBase):
    """Conditioning for pi-stacking interactions."""

    name = "pistack"
    n_body = 1
    level = Level.ATOM
    dtype = bool
    aliases = ("C_PISTACK", "mask_pistack_1_atom")
    default_value = False


class PiCationEnum(IntEnum):
    NONE = 0
    AROMATIC = 1
    CATION = 2
    BOTH = 3


class PiCation(ConditionBase):
    """Conditioning for pi-cation interactions (aromatic ring / cation role)."""

    name = "pication"
    n_body = 1
    level = Level.ATOM
    dtype = PiCationEnum
    aliases = ("C_PICATION",)
    default_value = PiCationEnum.NONE

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != PiCationEnum.NONE.value


class SaltBridgeEnum(IntEnum):
    NONE = 0
    POSITIVE = 1
    NEGATIVE = 2


class SaltBridge(ConditionBase):
    """Conditioning for salt bridge interactions."""

    name = "saltbridge"
    n_body = 1
    level = Level.ATOM
    dtype = SaltBridgeEnum
    aliases = ("C_SALT",)
    default_value = SaltBridgeEnum.NONE

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != SaltBridgeEnum.NONE.value


class Hydrophobic(ConditionBase):
    """Conditioning for hydrophobic contacts."""

    name = "hydrophobic"
    n_body = 1
    level = Level.ATOM
    dtype = bool
    aliases = ("C_HPHOB", "mask_hydrophobic_1_atom")
    default_value = False


class MetalCoordEnum(IntEnum):
    NONE = 0
    METAL = 1
    COORDINATING = 2


class MetalCoordination(ConditionBase):
    """Conditioning for metal coordination interactions."""

    name = "metal-coord"
    n_body = 1
    level = Level.ATOM
    dtype = MetalCoordEnum
    aliases = ("C_MCOORD",)
    default_value = MetalCoordEnum.NONE

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != MetalCoordEnum.NONE.value


class SecondaryStructure(ConditionBase):
    """Conditioning that provides secondary structure information to the model."""

    name = "secondary-structure"
    n_body = 1
    level = Level.RESIDUE
    dtype = SSEnum
    aliases = ("C_SEC",)
    default_value = SSEnum.NONE

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != SSEnum.NONE.value


class IndexGroup(ConditionBase):
    """Indicates atoms within the same equivalence class of known index offsets.

    Does not differentiate between globally indexed vs unindexed groups (that is annotated in
    ``C_IDX.mask``). While the default annotation is monotonically increasing by one, this should
    not be expected to always hold.

    Groups by molecule to ensure that all tokens of one molecule share a group even if they
    are not contiguous in the atom array.
    """

    name = "index-group"
    n_body = 1
    level = Level.TOKEN
    dtype = int
    aliases = ("C_IDXGRP",)

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return np.ones(annotation.shape[0], dtype=bool)  # Every atom has an index group

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Int[Array, "n_atoms"]:
        # Add molecule annotations if not already present
        if "molecule_iid" not in atom_array.get_annotation_categories():
            ensure_annotations(atom_array, "is_polymer")
            atom_array = add_id_and_entity_annotations(atom_array)
            if "transformation_id" not in atom_array.get_annotation_categories():
                atom_array.set_annotation("transformation_id", np.full(atom_array.array_length(), "1", dtype=str))
            atom_array = add_molecule_iid_annotation(atom_array)

        if atom_array.array_length() == 0:
            return np.zeros(0, dtype=int)

        molecule_annot = atom_array.molecule_iid
        is_globally_indexed = C_IDX.mask(atom_array, default="generate")

        all_index_groups = np.empty(atom_array.array_length(), dtype=int)

        # Indexed atoms use one group per molecule, including non-adjacent parts of that molecule.
        _, molecule_iids = np.unique(molecule_annot, return_inverse=True)
        all_index_groups[is_globally_indexed] = molecule_iids[is_globally_indexed]
        n_molecules = int(molecule_iids.max()) + 1

        # Unindexed atoms use one group per contiguous same-molecule run, after the molecule IDs.
        if not is_globally_indexed.all():
            run_starts = np.concatenate(
                (
                    [True],
                    (molecule_annot[1:] != molecule_annot[:-1]) | (is_globally_indexed[1:] != is_globally_indexed[:-1]),
                )
            )
            run_ids = np.cumsum(run_starts) - 1
            _, unindexed_ids = np.unique(run_ids[~is_globally_indexed], return_inverse=True)
            all_index_groups[~is_globally_indexed] = unindexed_ids + n_molecules

        return all_index_groups


class CathLabel(ConditionBase):
    """Conditioning that provides cath labels to the model. 'x.x.x.x' is null conditioning."""

    name = "cath-label"
    n_body = 1
    level = Level.RESIDUE  # In practice this is often chain-level
    dtype = str
    aliases = ("C_CATH",)
    NULL_VALUE = "x.x.x.x"
    default_value = NULL_VALUE

    @classmethod
    def mask_from_annotation(cls, annotation: Shaped[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != cls.NULL_VALUE

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Shaped[Array, "n_atoms"]:
        # Max size of 23 is more than enough (allows for 5-digit ids at each level of the cath hierarchy)
        return np.full(atom_array.array_length(), cls.NULL_VALUE, dtype=np.dtype("U23"))


class InternalCoordinate(ConditionBase):
    """Materialized internal-coordinate payload for atoms in non-reserved coordinate groups.

    ``C_CRDGRP`` is the source of truth for membership. These coordinates provide
    group-local structure without fixing the groups' relative positions or orientations.
    """

    # NOTE: CamelCase is used since underscores are forbidden and dashes break the query syntax
    name = "InternalCoordinate"
    n_body = 1
    level = Level.ATOM
    dtype = float
    aliases = ("C_INTCRD",)
    default_value = np.nan
    value_shape = (3,)

    @classmethod
    def mask_from_annotation(cls, annotation: Float[Array, "n_atoms 3"]) -> Bool[Array, "n_atoms"]:
        """Report payload presence for the Condition API; do not use as group membership."""
        return np.isfinite(annotation).any(axis=1)


class ChainCentroid(ConditionBase):
    """Per-chain centroid xyz (Å, ``(n_atoms, 3)``), broadcast per atom; NaN where unincluded."""

    # NOTE: CamelCase is used since underscores are forbidden and dashes break the query syntax
    name = "ChainCentroid"
    n_body = 1
    level = Level.ATOM
    dtype = float
    aliases = ("C_CENTROID",)
    default_value = np.nan
    value_shape = (3,)

    @classmethod
    def mask_from_annotation(cls, annotation: Float[Array, "n_atoms 3"]) -> Bool[Array, "n_atoms"]:
        return np.isfinite(annotation).any(axis=1)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Float[Array, "n_atoms 3"]:
        return np.full((atom_array.array_length(), 3), fill_value=np.nan, dtype=np.float32)


class CoordinateGroup(ConditionBase):
    """Indicates atoms within the same equivalence class of known relative coordinates."""

    # NOTE: CamelCase is used since underscores are forbidden and dashes break the query syntax
    name = "CoordinateGroup"
    n_body = 1
    level = Level.ATOM
    dtype = int
    aliases = ("C_CRDGRP",)
    KNOWN_ABSOLUTE_COORDS = 0  # Standard absolute coordinate conditioning
    UNKNOWN_COORDS = -1  # No form of relative or absolute coordinate conditioning
    MAX_GROUP_ID = int(np.iinfo(np.int16).max)  # Model batches store group IDs as int16.

    @classmethod
    def set_annotation(
        cls,
        atom_array: AtomArray,
        array: Int[Array, "n_atoms"],
        *,
        on_internal_coord_mismatch: Literal["raise", "clear"] = "raise",
    ) -> None:
        """Set membership, preserving saved targets for unchanged whole atom sets.

        Args:
            atom_array: Structure to modify in place.
            array: New group IDs; pass a copy instead of mutating the existing annotation.
                IDs must be in [-1, 32767]. Group 0 must match absolute conditioning;
                modify ``C_CRD`` first.
            on_internal_coord_mismatch: Raise before writing if membership changes invalidate
                saved targets, or clear targets throughout affected groups with ``"clear"``.
                Whole-group renaming preserves targets under either policy.
        """
        if on_internal_coord_mismatch not in ("raise", "clear"):
            raise ValueError("on_internal_coord_mismatch must be 'raise' or 'clear'.")
        groups = np.asarray(array)
        if groups.shape != (len(atom_array),) or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("Coordinate groups must be a one-dimensional integer array with one ID per atom.")
        if np.any(groups < cls.UNKNOWN_COORDS):
            raise ValueError("Coordinate groups must be -1 (unknown), 0 (absolute), or positive (internal).")
        if np.any(groups > cls.MAX_GROUP_ID):
            raise ValueError(f"Coordinate group IDs must not exceed {cls.MAX_GROUP_ID} (the int16 model limit).")
        absolute = C_CRD.annotation(atom_array, default="generate")
        absolute_present = np.isfinite(absolute).all(axis=1)
        if not np.all(absolute_present | np.isnan(absolute).all(axis=1)):
            raise ValueError("Absolute coordinate targets must be entirely finite or entirely unset (NaN) per atom.")
        if not np.array_equal(absolute_present, groups == cls.KNOWN_ABSOLUTE_COORDS):
            raise ValueError("Coordinate group 0 must match absolute coordinate conditioning; modify C_CRD first.")

        old = cls.annotation(atom_array, default="generate")
        payload = None
        if InternalCoordinate.has_annotation(atom_array):
            preserved = np.zeros(len(old), dtype=bool)
            for gid in np.unique(groups[groups > cls.KNOWN_ABSOLUTE_COORDS]):
                members = groups == gid
                previous = np.unique(old[members])
                if (
                    len(previous) == 1
                    and previous[0] > cls.KNOWN_ABSOLUTE_COORDS
                    and np.array_equal(members, old == previous[0])
                ):
                    preserved |= members
            payload = InternalCoordinate.annotation(atom_array).copy()
            if not np.isnan(payload[~preserved]).all() and on_internal_coord_mismatch == "raise":
                raise ValueError(
                    "Coordinate group membership changes invalidate saved C_INTCRD targets; "
                    "use on_internal_coord_mismatch='clear' to discard affected targets."
                )
            payload[~preserved] = np.nan
        super().set_annotation(atom_array, groups)
        if payload is not None:
            InternalCoordinate.set_annotation(atom_array, payload)

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return np.ones(annotation.shape[0], dtype=bool)  # Every atom has a coordinate group

    @classmethod
    def is_internal(cls, atom_array: AtomArray) -> Bool[Array, "n_atoms"]:
        """Return atoms assigned to a non-reserved internal-coordinate group."""
        groups = cls.annotation(atom_array, default="generate")
        return (groups != cls.UNKNOWN_COORDS) & (groups != cls.KNOWN_ABSOLUTE_COORDS)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Int[Array, "n_atoms"]:
        # Coordinate groups are the source of truth for internal-coordinate membership;
        # defaults can therefore infer only absolute conditioning from C_CRD.
        all_coordinate_groups = np.full(atom_array.array_length(), cls.UNKNOWN_COORDS, dtype=int)
        if C_CRD.has_annotation(atom_array):
            has_known_absolute_coords = C_CRD.mask(atom_array, default="raise")
            all_coordinate_groups[has_known_absolute_coords] = cls.KNOWN_ABSOLUTE_COORDS
        return all_coordinate_groups


class BasePairEnum(IntEnum):
    """
    the Enum to hold int info corresponding to BasePair annotations
    """

    UNSPECIFIED = 0  # mask
    PAIR = 1  # base paired positions
    LOOP = 2  # explicitl-unpaired


class BasePair(ConditionBase):
    """NA secondary-structure pair condition (residue semantics, 2-body sparse storage)."""

    name = "basepair"
    n_body = 2
    level = Level.RESIDUE
    is_symmetric = True
    dtype = BasePairEnum
    aliases = ("C_BP",)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        # Sparse default: empty/unspecified
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=np.array([], dtype=int),
            values=np.empty((0,), dtype=cls.storage_dtype()),
        )

    @classmethod
    def mask_from_annotation(cls, annotation: AnnotationList2D) -> AnnotationList2D:
        return AnnotationList2D(
            n_atoms=annotation.n_atoms,
            pairs=annotation.pairs,
            values=np.ones(len(annotation.values), dtype=bool),
        )


class NucleicSSState(ConditionBase):
    """Per-residue NA secondary-structure state encoded with ``BasePairEnum``."""

    name = "na-ss-state"
    n_body = 1
    level = Level.RESIDUE
    dtype = BasePairEnum
    aliases = ("C_NA_SS_STATE",)
    default_value = BasePairEnum.UNSPECIFIED

    @classmethod
    def mask_from_annotation(cls, annotation: Int[Array, "n_atoms"]) -> Bool[Array, "n_atoms"]:
        return annotation != BasePairEnum.UNSPECIFIED.value


class SequenceProbability(ConditionBase):
    """Base class for an explicitly specified unified sequence-token probability.

    NaN leaves an identity unspecified; zero explicitly excludes it. Presence is
    derived from the stored value, including invalid infinities for validation.
    """

    token_name: ClassVar[str]
    token_index: ClassVar[int]
    alias_suffix: ClassVar[str]
    default_value = np.nan

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return ~np.isnan(annotation)


def _make_sequence_probability_condition(token_name: str, token_index: int) -> type[SequenceProbability]:
    """Create and register the probability Condition for one unified sequence token."""

    return type(
        f"{token_name.title()}SequenceProbability",
        (SequenceProbability,),
        {
            "__module__": __name__,
            "__doc__": f"Explicit probability conditioning for unified sequence token {token_name!r}.",
            "name": f"sequence-probability-{token_name}",
            "token_name": token_name,
            "token_index": token_index,
            "alias_suffix": token_name,
            "n_body": 1,
            "level": Level.RESIDUE,
            "dtype": np.float32,
            "aliases": (f"C_SEQ_PROB_{token_name}",),
        },
    )


# Use the unified encoding as the source of truth for channel order, excluding
# unknown, mask, atomized, and gap tokens from the conditioning vocabulary.
SEQUENCE_PROBABILITY_CONDITIONS: tuple[type[SequenceProbability], ...] = tuple(
    _make_sequence_probability_condition(str(token_name), token_index)
    for token_index, token_name in enumerate(UNIFIED_ATOM37_ENCODING.tokens)
    if token_name in STANDARD_POLYMER_RESIDUES
)
SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN = {
    condition.token_name: condition for condition in SEQUENCE_PROBABILITY_CONDITIONS
}
SEQUENCE_PROBABILITY_CHANNEL_BY_TOKEN = {
    condition.token_name: channel for channel, condition in enumerate(SEQUENCE_PROBABILITY_CONDITIONS)
}

# User-facing condition-spec aliases, e.g. ``C_SEQ_PROB_CYS: {0: ...}``.
for _condition in SEQUENCE_PROBABILITY_CONDITIONS:
    globals()[f"C_SEQ_PROB_{_condition.alias_suffix}"] = _condition


# Aliases for convenience
C_HOT = CONDITIONS.hotspot
C_RAS = CONDITIONS.rasa_binned
C_SEC = CONDITIONS.secondary_structure
C_HB = CONDITIONS.hbond
C_PISTACK = CONDITIONS.pistack
C_PICATION = CONDITIONS.pication
C_SALT = CONDITIONS.saltbridge
C_HPHOB = CONDITIONS.hydrophobic
C_MCOORD = CONDITIONS.metal_coord
C_IDXGRP = CONDITIONS.index_group
C_CATH = CONDITIONS.cath_label
C_INTCRD = CONDITIONS.InternalCoordinate
C_CRDGRP = CONDITIONS.CoordinateGroup
C_CENTROID = CONDITIONS.ChainCentroid
C_BP = CONDITIONS.basepair
C_NA_SS_STATE = CONDITIONS.na_ss_state
