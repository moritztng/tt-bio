# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Atom37 slot lookup and token-reference annotations for RFD4's vocabulary."""

import functools

import numpy as np
from tt_bio._vendor.atomworks.constants import MASKED, STANDARD_AA, UNKNOWN_AA
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_SEGMIN
from tt_bio._vendor.atomworks.ml.conditions import C_IDX
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from .datapipes_schema_proteina import TokenType
from .encoding import UNIFIED_ATOM37_ENCODING
from .utils_constants import LOCAL_LATENT_RESIDUES


@functools.lru_cache
def _build_atom37_lookups() -> tuple[np.ndarray, np.ndarray]:
    """Build sorted residue-specific atom keys for vectorized atom37 slot lookup.

    Uses ``<res_name>|<atom_name>`` keys rather than atom names alone. This is
    required for nucleic acids, where purines and pyrimidines reuse names like
    ``N1`` / ``C2`` / ``N3`` but place them in different atom37 slots.
    """
    atom_key_to_slot: dict[str, int] = {}
    for res_name in LOCAL_LATENT_RESIDUES:
        if res_name == MASKED:
            continue
        atoms = UNIFIED_ATOM37_ENCODING.token_atoms[res_name]
        for slot, atom_name in enumerate(atoms):
            atom_name = atom_name.strip()
            if atom_name:
                atom_key_to_slot[f"{res_name}|{atom_name}"] = slot
                # Standard proteins share atom-name slots, including observed atoms in unknown residues.
                if res_name in STANDARD_AA:
                    atom_key_to_slot[f"{UNKNOWN_AA}|{atom_name}"] = slot
    keys_sorted = sorted(atom_key_to_slot.keys())
    atom_keys = np.array(keys_sorted)
    atom_slots = np.array([atom_key_to_slot[key] for key in keys_sorted], dtype=np.int32)
    return atom_keys, atom_slots


def _lookup_atom37_slots(keys: np.ndarray, sorted_keys: np.ndarray, sorted_slots: np.ndarray) -> np.ndarray:
    """Lookup atom37 slots from sorted residue-specific keys."""
    positions = np.searchsorted(sorted_keys, keys)
    valid = positions < len(sorted_keys)
    matched = valid & (sorted_keys[np.clip(positions, 0, len(sorted_keys) - 1)] == keys)
    out = np.full(len(keys), -1, dtype=np.int32)
    out[matched] = sorted_slots[positions[matched]]
    return out


def build_atom37_slots(atom_array: AtomArray) -> np.ndarray:
    """Map residue/atom names to vocabulary slots, independent of chain classification.

    Explicitly atomized atoms use slot 1 instead of their residue-specific slot.

    Returns:
        Integer slot indices of shape ``[n_atoms]``. Unknown atoms get -1.
    """
    aap = as_atom_array_plus(atom_array)
    n_atoms = aap.array_length()
    atom_names = np.char.strip(aap.atom_name.astype(str))
    keys = np.char.add(np.char.add(aap.res_name.astype(str), "|"), atom_names)
    slots = _lookup_atom37_slots(keys, *_build_atom37_lookups())
    categories = set(aap.get_annotation_categories())

    if "token_type" in categories:
        is_atomized = aap.token_type == TokenType.ATOMIZED
    else:
        is_atomized = aap.atomize if "atomize" in categories else np.zeros(n_atoms, dtype=bool)

    if is_atomized.any():
        slots[is_atomized] = 1

    return slots


def annotate_token_centers(atom_array: AtomArray) -> AtomArray:
    """Mark slot-1 atoms and every atomized atom in place; return the same array.

    Call after setting atomization or changing residue/atom identities.
    """
    atom_array.set_annotation("is_token_center", build_atom37_slots(atom_array) == 1)
    return atom_array


def missing_token_reference_mask(atom_array: AtomArray) -> np.ndarray:
    """Return residue tokens missing atom37 slot 1; atomized atoms and expandable segments are exempt."""
    starts = get_token_starts(atom_array)
    if not len(starts):
        return np.zeros(0, dtype=bool)
    requires_reference = np.isin(atom_array.res_name[starts], LOCAL_LATENT_RESIDUES)
    requires_reference &= ~atom_array.atomize[starts]
    requires_reference &= ~S_SEGMIN.mask(atom_array, default="generate")[starts]
    has_reference = np.logical_or.reduceat(build_atom37_slots(atom_array) == 1, starts)
    return requires_reference & ~has_reference


def has_missing_token_reference(aa: AtomArray) -> bool:
    """Require reference atoms for indexed residues, not unindexed conditioning fragments."""
    indexed = np.logical_and.reduceat(C_IDX.mask(aa, default="generate"), get_token_starts(aa))
    return bool((missing_token_reference_mask(aa) & indexed).any())
