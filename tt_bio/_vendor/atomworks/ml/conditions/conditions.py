"""Concrete implementations of common molecular design conditions."""

import numpy as np
from biotite.structure import AtomArray
from sympy.sets.sets import true

from tt_bio._vendor.atomworks.constants import MASKED
from tt_bio._vendor.atomworks.io.utils.atom_array import apply_and_spread
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AnnotationList2D
from tt_bio._vendor.atomworks.ml.conditions.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.conditions.base import ConditionBase, Level


class Sequence(ConditionBase):
    name = "sequence"
    n_body = 1
    level = Level.RESIDUE
    dtype = str

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return cls.annotation(atom_array, default="raise") != MASKED
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        seq = np.full(atom_array.array_length(), fill_value=MASKED)
        if cls.mask_name in atom_array.get_annotation_categories():
            # ... if mask exists, use it to get the sequence
            seq_mask = cls.mask(atom_array, default="raise")
            seq[seq_mask] = atom_array.res_name[seq_mask]
        return seq


class Coordinate(ConditionBase):
    name = "coordinate"
    n_body = 1
    level = Level.ATOM
    dtype = float

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return np.isfinite(cls.annotation(atom_array, default="raise")).any(axis=1)
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        coords = np.full((atom_array.array_length(), 3), fill_value=np.nan)
        if cls.mask_name in atom_array.get_annotation_categories():
            # ... if mask exists, use it to get the coordinates
            mask = cls.mask(atom_array, default="raise")
            coords[mask] = atom_array.coord[mask]
        return coords


class Index(ConditionBase):
    name = "index"
    n_body = 1
    level = Level.RESIDUE
    dtype = int
    MASKED_INDEX = -1

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return cls.annotation(atom_array, default="raise") != cls.MASKED_INDEX
        else:
            # ... otherwise, default to a full mask (all indices leaked)
            return np.ones(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "within_chain_res_idx")
        indices = atom_array.get_annotation("within_chain_res_idx").copy()
        if cls.mask_name in atom_array.get_annotation_categories():
            # ... if mask exists, use it to get the indices
            mask = cls.mask(atom_array, default="raise")
            indices[~mask] = cls.MASKED_INDEX
        return indices


class Distance(ConditionBase):
    name = "distance"
    n_body = 2
    level = Level.ATOM
    is_symmetric = True
    dtype = float

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotation = cls.annotation(atom_array, default="generate")
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=annotation.pairs,
            values=annotation.values > 0,
        )

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=np.array([], dtype=int),
            values=np.array([], dtype=cls.dtype),
        )


class NTerminus(ConditionBase):
    name = "n-terminus"
    n_body = 1
    level = Level.RESIDUE
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for N-terminus atoms
        is_n_terminus = atom_array.is_polymer & (atom_array.within_chain_res_idx == 0) & atom_array.is_chain_start

        # ... spread to full residue
        residue_segments = np.concatenate([np.where(atom_array.is_res_start)[0], [atom_array.array_length()]])
        is_n_terminus = apply_and_spread(residue_segments, is_n_terminus, np.any)

        return is_n_terminus


class CTerminus(ConditionBase):
    name = "c-terminus"
    n_body = 1
    level = Level.RESIDUE
    is_mask = true
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for C-terminus atoms
        annotations = atom_array.get_annotation_categories()
        chain_ids = atom_array.chain_iid if "chain_iid" in annotations else atom_array.chain_id
        # ... find max within_chain_res_idx for each chain
        is_max_within_chain_res_idx = np.zeros(atom_array.array_length(), dtype=bool)
        for chain_id in np.unique(chain_ids):
            is_this_chain = chain_ids == chain_id
            max_chain_idx = np.max(atom_array.get_annotation("within_chain_res_idx")[is_this_chain])
            is_max_idx = atom_array.get_annotation("within_chain_res_idx") == max_chain_idx
            is_max_within_chain_res_idx |= is_this_chain & is_max_idx

        # ... spread to full residue
        is_c_terminus = is_max_within_chain_res_idx & atom_array.is_polymer
        residue_segments = np.concatenate([np.where(atom_array.is_res_start)[0], [atom_array.array_length()]])
        is_c_terminus = apply_and_spread(residue_segments, is_c_terminus, np.any)

        return is_c_terminus


class Chain(ConditionBase):
    name = "chain"
    n_body = 2
    level = Level.CHAIN
    is_symmetric = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotations = atom_array.get_annotation_categories()
        chain_iid_annotation = "chain_iid" if "chain_iid" in annotations else "chain_id"
        chain_instance = atom_array.get_annotation(chain_iid_annotation)
        is_same_chain = chain_instance[None, :] == chain_instance[:, None]
        pairs = np.stack(np.where(is_same_chain), axis=0).T
        values = np.ones(pairs.shape[0], dtype=bool)
        return AnnotationList2D(atom_array.array_length(), pairs=pairs, values=values)
