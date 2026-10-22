# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Protein-DNA contact-centered contiguous cropping.

The training pipeline crops protein-DNA examples around their binding
interface — keeping a contiguous stretch of the contacting protein region *and*
a contiguous stretch of the contacting DNA duplex — instead of the generic
AF3 k-nearest-neighbor spatial crop, which can slice the duplex mid-strand or
under-represent the DNA.

The public entry point is :py:class:`ProteinDNAContactContiguousCrop`. It is a
drop-in sibling of :py:class:`atomworks.ml.transforms.crop.CropSpatialLikeAF3`
and, when no protein-DNA contact can be sampled (e.g. RNA-only context,
single-stranded DNA, or a noncanonical-DNA token explosion), it falls back to a
standard spatial crop rather than raising — so it is always safe to place in a
:py:class:`atomworks.ml.transforms.base.RandomRoute` over a mixed dataset.

Requires the ``within_poly_res_idx`` annotation (add it upstream with
:py:class:`atomworks.ml.transforms.atom_array.AddWithinPolyResIdxAnnotation`)
and the standard ``chain_type`` annotation.
"""

import numpy as np
from tt_bio._vendor.atomworks.enums import ChainType
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
)
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.transforms.crop import (
    CropSpatialLikeAF3,
    resize_crop_info_if_too_many_atoms,
    sample_dynamic_crop_size,
)
from tt_bio._vendor.atomworks.ml.utils.token import get_token_count, get_token_starts
from biotite.structure import Atom, AtomArray
from biotite.structure.basepairs import (
    _check_dssr_criteria,
    _get_proximate_residues,
    get_residue_masks,
    get_residue_starts_for,
)
from scipy.spatial import cKDTree

# Base atoms used to define a protein-DNA "contact" on the DNA side: the
# major-groove hydrogen-bond donor/acceptor edge of each canonical base.
_DNA_BASE_CONTACT_ATOMS = {
    "DA": ["N7", "N6"],
    "DC": ["N4"],
    "DG": ["N7", "O6"],
    "DT": ["O4"],
}


def atom_array_from_contact_dict(atom_array: AtomArray, contact_atoms: dict) -> AtomArray:
    """Subset ``atom_array`` to the (res_name, atom_name) pairs in ``contact_atoms``."""
    mask = [row.res_name in contact_atoms and row.atom_name in contact_atoms[row.res_name] for row in atom_array]
    return atom_array[mask]


def identify_and_sample_protein_dna_contact(
    atom_array: AtomArray,
    protein_contact_atoms: dict | list | str,
    dna_contact_atoms: dict | list | str,
    contact_dist: float = 4.0,
) -> tuple[Atom, Atom]:
    """Sample one protein atom and one DNA atom that are in contact.

    Returns:
      ``(dna_contact_atom, protein_contact_atom)`` — two single-atom views.

    Raises:
      ValueError: If no protein-DNA contact within ``contact_dist`` exists.
    """
    ensure_annotations(atom_array, "is_protein", "is_dna")

    if isinstance(protein_contact_atoms, dict):
        protein = atom_array_from_contact_dict(atom_array, protein_contact_atoms)
    elif isinstance(protein_contact_atoms, list):
        protein = atom_array[atom_array.is_protein & np.isin(atom_array.atom_name, protein_contact_atoms)]
    elif protein_contact_atoms == "all":
        protein = atom_array[atom_array.is_protein]
    else:
        raise ValueError(f"Invalid protein_contact_atoms: {protein_contact_atoms!r}")

    if isinstance(dna_contact_atoms, dict):
        dna = atom_array_from_contact_dict(atom_array[atom_array.is_dna], dna_contact_atoms)
    elif isinstance(dna_contact_atoms, list):
        dna = atom_array[atom_array.is_dna & np.isin(atom_array.atom_name, dna_contact_atoms)]
    elif dna_contact_atoms == "all":
        dna = atom_array[atom_array.is_dna]
    else:
        raise ValueError(f"Invalid dna_contact_atoms: {dna_contact_atoms!r}")

    if len(dna) == 0 or len(protein) == 0:
        raise ValueError("No protein-DNA contacts found")

    # Sample one DNA-protein atom pair within ``contact_dist``. A KD-tree query
    # gives the contacting pairs without materializing the dense
    # [n_dna x n_protein] distance matrix (which can blow up memory on large
    # uncropped complexes).
    protein_neighbors = cKDTree(dna.coord).query_ball_tree(cKDTree(protein.coord), contact_dist)
    contacts = [(dna_idx, prot_idx) for dna_idx, prot_idxs in enumerate(protein_neighbors) for prot_idx in prot_idxs]
    if len(contacts) == 0:
        raise ValueError("No protein-DNA contacts found")

    dna_idx, prot_idx = contacts[np.random.choice(len(contacts))]
    return dna[dna_idx], protein[prot_idx]


def create_residue_mask(atom_array: AtomArray, first_atom_indices: list[int] | np.ndarray) -> np.ndarray:
    """Boolean atom mask selecting every atom of the residues given by their first-atom indices."""
    target_res_ids = atom_array.res_id[first_atom_indices][:, np.newaxis]
    target_chain_ids = atom_array.chain_id[first_atom_indices][:, np.newaxis]
    res_match = atom_array.res_id == target_res_ids
    chain_match = atom_array.chain_id == target_chain_ids
    return (res_match & chain_match).any(axis=0)


def expand_connected_component_mask(
    atom_array: AtomArray, origin: Atom, left_expand: int, right_expand: int
) -> np.ndarray:
    """Keep ``origin``'s chain residues in the contiguous window ``[center-left, center+right)``."""
    center = origin.within_poly_res_idx
    candidates = list(range(center - left_expand, center + right_expand))
    return (atom_array.chain_id == origin.chain_id) & np.isin(atom_array.within_poly_res_idx, candidates)


def get_dna_mask(atom_array: AtomArray, origin: Atom, left_expand: int, right_expand: int) -> np.ndarray:
    """Keep a contiguous DNA window around ``origin`` plus its base-paired partner residues."""
    one_chain_mask = expand_connected_component_mask(atom_array, origin, left_expand, right_expand)

    pairs = base_pairs(atom_array)
    other_chain_first_atom_indices = []
    for pair in pairs:
        if one_chain_mask[pair[0]]:
            other_chain_first_atom_indices.append(pair[1])
        elif one_chain_mask[pair[1]]:
            other_chain_first_atom_indices.append(pair[0])

    other_chain_mask = create_residue_mask(atom_array, other_chain_first_atom_indices)
    return np.logical_or(one_chain_mask, other_chain_mask)


def protein_dna_contact_contiguous_crop_mask(
    atom_array: AtomArray,
    protein_contact_atoms: dict | list | str,
    dna_contact_atoms: dict | list | str,
    contact_dist_cutoff: float,
    protein_expand_min: int,
    protein_expand_max: int,
    dna_expand_min: int,
    dna_expand_max: int,
    crop_size: int | None = None,
) -> dict:
    """Build a crop_info dict keeping a contiguous protein region + contiguous DNA duplex around a sampled contact.

    Args:
      crop_size: Token budget to record under ``crop_info["crop_size"]``.

    Raises:
      ValueError: If no protein-DNA contact exists, or the kept region exceeds
        300 tokens (noncanonical-DNA token explosion).
    """
    dna_contact, prot_contact = identify_and_sample_protein_dna_contact(
        atom_array, protein_contact_atoms, dna_contact_atoms, contact_dist_cutoff
    )

    left = np.random.randint(protein_expand_min, protein_expand_max)
    right = np.random.randint(protein_expand_min, protein_expand_max)
    protein_keep_mask = expand_connected_component_mask(atom_array, prot_contact, left, right)

    left = np.random.randint(dna_expand_min, dna_expand_max)
    right = np.random.randint(dna_expand_min, dna_expand_max)
    dna_keep_mask = get_dna_mask(atom_array, dna_contact, left, right)

    mask = np.logical_or(protein_keep_mask, dna_keep_mask)

    if get_token_count(atom_array[mask]) > 300:
        raise ValueError("Noncanonical DNAs are causing token count explosion, skipping...")

    # Per-token surviving indices (a token survives iff its start atom is kept),
    # matching the atomworks crop convention consumed by EnforceCropBudget.
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    return {
        "type": "ProteinDNAContactContiguousCrop",
        "requires_crop": bool(np.any(mask)),
        "crop_atom_idxs": np.where(mask)[0],
        "crop_token_idxs": np.where(mask[token_starts])[0],
        "atom_array": atom_array,
        "crop_size": crop_size,
    }


class ProteinDNAContactContiguousCrop(Transform):
    """Crop a protein-DNA example around its binding interface.

    Samples a protein-DNA contact, then keeps a contiguous stretch of the
    contacting protein region and a contiguous stretch of the contacting DNA
    duplex (including base-paired partner residues). If no contact can be
    sampled (RNA-only context, single-stranded DNA, or a noncanonical-DNA token
    explosion), falls back to a standard AF3 spatial crop so the example is
    still usable.

    Args:
      crop_size: Token budget for the spatial-crop fallback. Either this or
        `crop_size_range` must be set (not both) — see
        :py:class:`~atomworks.ml.transforms.crop.CropSpatialLikeAF3`.
      crop_size_range: If set, the spatial-crop fallback's token budget is
        resampled per-call from this ``(min, max)`` range instead of using a
        fixed `crop_size` (which must then be `None`). Forwarded verbatim to
        the fallback's `CropSpatialLikeAF3`. Defaults to `None`.
      max_crop_size: Upper bound on the sampled crop size when `crop_size_range`
        is fraction-based. Forwarded verbatim to the fallback. Defaults to `None`.
      crop_center_cutoff_distance: Cutoff (Å) for the spatial-crop fallback's
        crop-center selection.
      protein_contact_type: ``"all"`` or ``"backbone"`` (N/CA/C).
      dna_contact_type: ``"base"`` (major-groove edge atoms) or ``"all"``.
      contact_distance_cutoff: Distance (Å) defining a protein-DNA contact.
      protein_expand_min/max: Range for the contiguous protein flank (residues
        each side of the contact).
      dna_expand_min/max: Range for the contiguous DNA flank.
      keep_uncropped_atom_array: Store the pre-crop array under
        ``uncropped_atom_array``.
      max_atoms_in_crop: Atom budget; oversized crops are trimmed by distance.
    """

    def __init__(
        self,
        crop_size: int | None = None,
        crop_size_range: tuple[int, int] | tuple[float, float] | None = None,
        max_crop_size: int | None = None,
        crop_center_cutoff_distance: float = 15.0,
        protein_contact_type: str = "all",
        dna_contact_type: str = "base",
        contact_distance_cutoff: float = 10.0,
        protein_expand_min: int = 15,
        protein_expand_max: int = 40,
        dna_expand_min: int = 3,
        dna_expand_max: int = 10,
        keep_uncropped_atom_array: bool = False,
        max_atoms_in_crop: int | None = None,
    ):
        if protein_contact_type == "backbone":
            self.protein_contact_atoms = ["N", "CA", "C"]
        elif protein_contact_type == "all":
            self.protein_contact_atoms = "all"
        else:
            raise ValueError(f"Invalid protein_contact_type: {protein_contact_type!r}")

        if dna_contact_type == "backbone":
            self.dna_contact_atoms = ["P", "OP1", "OP2"]
        elif dna_contact_type == "base":
            self.dna_contact_atoms = _DNA_BASE_CONTACT_ATOMS
        else:
            self.dna_contact_atoms = "all"

        self.protein_expand_min = protein_expand_min
        self.protein_expand_max = protein_expand_max
        self.dna_expand_min = dna_expand_min
        self.dna_expand_max = dna_expand_max
        self.contact_distance_cutoff = contact_distance_cutoff
        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.max_atoms_in_crop = max_atoms_in_crop
        self.crop_size = crop_size
        self.crop_size_range = crop_size_range
        self.max_crop_size = max_crop_size
        self.crop_center_cutoff_distance = crop_center_cutoff_distance

        # Fallback used when no protein-DNA contact can be sampled. Mirrors the
        # kwargs of the main spatial crop in the design pipeline.
        self._spatial_fallback = CropSpatialLikeAF3(
            crop_size=crop_size,
            crop_size_range=crop_size_range,
            max_crop_size=max_crop_size,
            crop_center_cutoff_distance=crop_center_cutoff_distance,
            keep_uncropped_atom_array=keep_uncropped_atom_array,
            max_atoms_in_crop=max_atoms_in_crop,
            raise_if_missing_crop_center=False,
        )

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_atom_array_annotation(data, ["res_name", "within_poly_res_idx"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        try:
            # Resolve the token budget
            crop_size = self.crop_size
            if self.crop_size_range is not None:
                crop_size = sample_dynamic_crop_size(
                    self.crop_size,
                    self.crop_size_range,
                    reference_length=get_token_count(atom_array),
                    max_crop_size=self.max_crop_size,
                )

            crop_info = protein_dna_contact_contiguous_crop_mask(
                atom_array,
                self.protein_contact_atoms,
                self.dna_contact_atoms,
                self.contact_distance_cutoff,
                self.protein_expand_min,
                self.protein_expand_max,
                self.dna_expand_min,
                self.dna_expand_max,
                crop_size=crop_size,
            )
        except ValueError:
            # No sampleable protein-DNA contact (RNA-only, single-stranded, or
            # token explosion) — fall back to a spatial crop instead of crashing
            # the dataloader worker.
            return self._spatial_fallback(data)

        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info, atom_array=atom_array, max_atoms=self.max_atoms_in_crop
        )

        crop_info = {
            "crop_size": crop_size,
            "crop_size_range": self.crop_size_range,
            "crop_center_cutoff_distance": self.crop_center_cutoff_distance,
            "max_atoms_in_crop": self.max_atoms_in_crop,
        } | crop_info

        if self.keep_uncropped_atom_array:
            crop_info["atom_array"] = atom_array
        if crop_info["requires_crop"]:
            data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]
            data["crop_info"] = crop_info
        else:
            data["atom_array"] = atom_array
        return data


def fill_nan_coords_with_random(atoms: AtomArray, min_val: float = -50, max_val: float = 50) -> AtomArray:
    """Return a copy of ``atoms`` with NaN coordinates replaced by uniform random values.

    Used by :py:func:`base_pairs` so the DSSR geometry checks don't choke on
    unresolved nucleotide atoms.
    """
    filled_atoms = atoms.copy()
    coords = filled_atoms.coord
    nan_mask = np.isnan(coords)
    coords[nan_mask] = np.random.uniform(low=min_val, high=max_val, size=coords[nan_mask].shape)
    return filled_atoms


def base_pairs(atom_array: AtomArray, min_atoms_per_base: int = 3, unique: bool = True) -> np.ndarray:
    """Identify DNA base pairs via DSSR criteria.

    Returns an ``(n, 2)`` array where each row holds the first-atom indices (in
    ``atom_array``) of the two residues forming a base pair. Uses Biotite's DSSR base-pair
    detection restricted to canonical DNA bases.
    """
    dna_boolean = np.logical_and(
        atom_array.chain_type == ChainType.DNA,
        np.isin(atom_array.res_name, ["DA", "DG", "DT", "DC"]),
    )
    non_phosphate_boolean = ~np.isin(atom_array.atom_name, ["O5'", "P", "OP1", "OP2", "OP3", "HOP2", "HOP3"])
    boolean_mask = np.logical_and(non_phosphate_boolean, dna_boolean)
    nucleosides = atom_array[boolean_mask]

    if len(nucleosides) == 0:
        return np.array([])

    n_o_mask = np.isin(nucleosides.element, ["N", "O"])
    nucleosides = fill_nan_coords_with_random(nucleosides)
    basepair_candidates, n_o_matches = _get_proximate_residues(nucleosides, n_o_mask, 3.6)

    basepairs = []
    basepairs_hbonds = []
    base_masks = get_residue_masks(nucleosides, basepair_candidates.flatten())
    base_masks = base_masks.reshape((basepair_candidates.shape[0], 2, nucleosides.shape[0]))

    for (base1_index, base2_index), (base1_mask, base2_mask), n_o_pairs in zip(
        basepair_candidates, base_masks, n_o_matches, strict=False
    ):
        base1 = nucleosides[base1_mask]
        base2 = nucleosides[base2_mask]
        hbonds = _check_dssr_criteria((base1, base2), min_atoms_per_base, unique)
        if hbonds is None:
            hbonds = n_o_pairs / 2
        if hbonds != -1:
            basepairs.append((base1_index, base2_index))
            if unique:
                basepairs_hbonds.append(hbonds)

    basepair_array = np.array(basepairs)

    if unique and len(basepair_array) > 0:
        to_remove = []
        base_indices, occurrences = np.unique(basepairs, return_counts=True)
        for base_index, occurrence in zip(base_indices, occurrences, strict=False):
            if occurrence > 1:
                remove_candidates = {}
                for i, row in enumerate(np.asarray(basepair_array == base_index)):
                    if np.any(row):
                        remove_candidates[i] = basepairs_hbonds[i]
                del remove_candidates[max(remove_candidates, key=remove_candidates.get)]
                to_remove += list(remove_candidates.keys())
        basepair_array = np.delete(basepair_array, to_remove, axis=0)

    if len(basepair_array) > 0:
        basepair_array = np.where(boolean_mask)[0][basepair_array]
        for i, row in enumerate(basepair_array):
            basepair_array[i] = get_residue_starts_for(atom_array, row)
    return basepair_array
