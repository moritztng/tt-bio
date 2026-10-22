# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Utilities for copying coordinates from generated structures back to input structures.

Handles coordinate transfer for workflows where residue types may change but we want to preserve annotations.
"""

import logging
import biotite.structure as struc
import numpy as np
from tt_bio._vendor.atomworks.io.template import get_empty_ccd_template
from tt_bio._vendor.atomworks.io.utils.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import insert_atoms
from tt_bio._vendor.atomworks.io.utils.selection import get_annotation_categories
from tt_bio._vendor.atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level
from tt_bio._vendor.atomworks.ml.conditions import C_CRD
from tt_bio._vendor.atomworks.ml.transforms.filters import remove_nucleic_acid_terminal_oxygen, remove_protein_terminal_oxygen
from tt_bio._vendor.atomworks.ml.utils.condition import default_annotations_and_conditions_from_registry
from biotite.structure import AtomArray

from .conditions import C_CRDGRP
from .utils_standard_annotations import SA_SRC
from .utils_token_encoding import annotate_token_centers

logger = logging.getLogger(__name__)

# Default atom-level annotations that vary within residues
DEFAULT_ATOM_LEVEL_ANNOTATIONS = {
    "atom_name",
    "element",
    "coord",
    "hetero",
    "atom_id",
    "b_factor",
    "occupancy",
    "charge",
    "is_token_center",
} | {sa.full_name for sa in STANDARD_ANNOTATIONS if (sa.n_body == 1 and sa.level == Level.ATOM)}

# These are copied over from the input if present as residue-level annotations
# Recomputing them might lead to an input-output mismatch
ANNOTS_TO_KEEP = {
    "chain_type",
    "is_protein",
    "is_nucleic_acid",
    "is_dna",
    "is_rna",
    "is_polymer",
    "is_ligand",
    "within_chain_res_idx",
    "within_chain_source_res_idx",
    "token_id",
}


def remove_terminal_oxygen_atoms(atom_array: AtomArray) -> AtomArray:
    """Remove protein OXT and nucleic-acid terminal oxygens for refolding comparisons."""
    atom_array = remove_protein_terminal_oxygen(atom_array)
    atom_array = remove_nucleic_acid_terminal_oxygen(atom_array)
    return atom_array


def _copy_coords_by_atom_name(
    source_array: AtomArray,
    source_mask: np.ndarray,
    target_array: AtomArray,
    target_mask: np.ndarray,
) -> None:
    """Copy coordinates from source to target for all matching atom names (in-place)."""
    source_atom_names = set(source_array.atom_name[source_mask])
    target_atom_names = target_array.atom_name[target_mask]

    for atom_name in target_atom_names:
        # Only copy if atom exists in source
        if atom_name not in source_atom_names:
            continue

        source_atom_mask = source_mask & (source_array.atom_name == atom_name)
        target_atom_mask = target_mask & (target_array.atom_name == atom_name)
        target_array.coord[target_atom_mask] = source_array.coord[source_atom_mask]


def _broadcast_src_component_residue_wise(atom_array: AtomArray) -> AtomArray:
    """Ensure that the src_component annotation reflects the residue-level nature of the input component syntax.

    Examples:
        - An indexed residue with only tip-atoms fixed is fully annotated with the tip-atom source residue
        - A residue that will be mapped to unindexed in-context conditioning is still an arbitrary diffused residue
            at this point, and is annotated as such. Its `src_coordinate_conditioned_region` will be added later.
    """

    # NOTE: Operations are residue-wise rather than token-wise since atomized residues have a single src_component
    src_comp = SA_SRC.annotation(atom_array)
    if np.any(struc.apply_residue_wise(atom_array, src_comp, lambda x: len(set(x) - {"."}) > 1)):
        raise ValueError("Found multiple non-trivial src_component values within a single residue.")

    residue_level_src_component = struc.apply_residue_wise(
        atom_array, src_comp, lambda x: np.array(x[x != "."][0] if np.any(x != ".") else ".", dtype="<U10")
    )
    broadcasted_src_component = struc.spread_residue_wise(atom_array, residue_level_src_component)

    SA_SRC.set_annotation(atom_array, broadcasted_src_component)

    return atom_array


def get_residue_level_annotations(
    atom_array: AtomArray,
    token_mask: np.ndarray,
    annotations_to_exclude: set[str] | None = None,
) -> dict[str, int | float | str]:
    """Extract residue-level one-body annotations from a token.

    Returns constant annotation values that are uniform across all atoms
    in a residue/token.

    Args:
      atom_array: The atom array containing the residue.
      token_mask: Boolean mask selecting atoms from a single token/residue.
      annotations_to_exclude: Annotation names to exclude. Defaults to
        standard atom-level annotations.

    Returns:
      Mapping of annotation names to their constant values for this residue.
    """
    if annotations_to_exclude is None:
        annotations_to_exclude = DEFAULT_ATOM_LEVEL_ANNOTATIONS

    all_annots = set(get_annotation_categories(atom_array, n_body=1))
    residue_level_annots = all_annots - annotations_to_exclude

    residue_annots = {}
    for annot in residue_level_annots:
        values = atom_array.get_annotation(annot)[token_mask]

        # Skip array annotations (they indexing errors downstream)
        if isinstance(values, np.ndarray) and values.ndim > 1:
            continue

        unique_values = np.unique(values)

        # Verify annotation is actually constant across the residue
        if len(unique_values) == 1:
            value = unique_values[0]
            residue_annots[annot] = value

    return residue_annots


def copy_generated_coordinates_to_input(
    generated_atom_array: AtomArray,
    input_atom_array: AtomArray,
) -> AtomArray:
    """Copy generated coordinates back to input structure.

    Handles coordinate transfer for three cases:

    1. Single-atom tokens: Direct copy
    2. Matching residue identities and inventories: Atom-wise copy
    3. Different identities or inventories: Delete extra atoms, add missing atoms

    For case 3, preserves all residue-level annotations from input and creates
    new atoms, defaulting atom-level annotations where possible.

    Args:
      generated_atom_array: Structure with generated coordinates. Must have
        ``token_id`` annotation.
      input_atom_array: Original structure to update. A deep copy is made
        internally, so the original is not modified.

    Returns:
      Updated atom array with coordinates from generated structure.
      May have different atoms if residue identities or inventories changed.

    Raises:
      ValueError: The generated inventory omits coordinate-conditioned atoms.
    """
    ensure_annotations(generated_atom_array, "token_id")
    ensure_annotations(input_atom_array, "token_id")

    # Deep copy to avoid modifying the original input
    input_atom_array = input_atom_array.copy()

    assert set(input_atom_array.token_id) == set(
        generated_atom_array.token_id
    ), "Input and generated atom arrays must have the same tokens"

    input_atom_array.set_annotation("original_global_index", np.arange(input_atom_array.array_length()))

    extra_atoms_mask = np.zeros(len(input_atom_array), dtype=bool)
    missing_atoms_and_indices = []

    for token_id in np.unique(input_atom_array.token_id):
        # Get annotations for input token
        input_token_mask = input_atom_array.token_id == token_id
        input_res_name = input_atom_array.res_name[input_token_mask][0]
        input_atom_names = input_atom_array.atom_name[input_token_mask]

        # Get corresponding annotations for the generated token
        generated_token_mask = generated_atom_array.token_id == token_id
        generated_res_name = generated_atom_array.res_name[generated_token_mask][0]
        generated_atom_names = generated_atom_array.atom_name[generated_token_mask]

        # + ---------------------------------------------------------------------------------- +
        # CASE 1: Single-atom token (e.g., small molecule, atomized) -> copy directly
        # + ---------------------------------------------------------------------------------- +
        if np.sum(input_token_mask) == 1 and input_atom_array.atomize[input_token_mask][0]:
            assert np.sum(generated_token_mask) == 1, "Generated token should also be single-atom"
            input_atom_array.coord[input_token_mask] = generated_atom_array.coord[generated_token_mask]
            continue

        # + ---------------------------------------------------------------------------------- +
        # CASE 2: Matching residue names -> copy by atom name (to ensure order invariance)
        # + ---------------------------------------------------------------------------------- +
        if generated_res_name == input_res_name:
            input_atom_name_set = set(input_atom_names)
            generated_atom_name_set = set(generated_atom_names)
            if input_atom_name_set == generated_atom_name_set:
                _copy_coords_by_atom_name(
                    generated_atom_array, generated_token_mask, input_atom_array, input_token_mask
                )
                continue

        # + ---------------------------------------------------------------------------------- +
        # CASE 3: Residue identity or inventory changed -> reconcile atoms by name
        # + ---------------------------------------------------------------------------------- +
        missing_atoms = np.setdiff1d(generated_atom_names, input_atom_names)
        extra_atoms = np.setdiff1d(input_atom_names, generated_atom_names)

        # Keep track of atoms that we need to remove later
        extra_atoms_mask |= input_token_mask & np.isin(input_atom_array.atom_name, extra_atoms)

        # Copy matching atoms
        _copy_coords_by_atom_name(generated_atom_array, generated_token_mask, input_atom_array, input_token_mask)

        # Update residue name
        input_atom_array.res_name[input_token_mask] = generated_res_name

        # Prepare to add missing atoms later
        if len(missing_atoms) > 0:
            residue_level_annotations_to_exclude = (
                DEFAULT_ATOM_LEVEL_ANNOTATIONS | set(ANNOTATOR_REGISTRY.keys())
            ) - ANNOTS_TO_KEEP
            residue_annots = get_residue_level_annotations(
                input_atom_array, input_token_mask, annotations_to_exclude=residue_level_annotations_to_exclude
            )

            # Use the values in the template, matching what the model generated
            residue_annots.pop("res_name", None)
            residue_annots.pop("atom_name", None)
            residue_annots.pop("element", None)

            residue_annots["original_global_index"] = np.full_like(input_token_mask.sum(), -1, dtype=int)

            # Create template for new residue type with all residue-level annotations
            template = get_empty_ccd_template(
                ccd_code=generated_res_name,
                hydrogen_policy="remove",
                **residue_annots,
            )

            # Subset to missing atoms only
            template = template[np.isin(template.atom_name, missing_atoms)]
            if "is_token_center" in input_atom_array.get_annotation_categories():
                annotate_token_centers(template)

            # Spoof all annotations with defaults so they survive insert_atoms()
            if "occupancy" in get_annotation_categories(input_atom_array, n_body="all"):
                template.set_annotation("occupancy", np.ones(template.array_length(), dtype=float))
            if "b_factor" in get_annotation_categories(input_atom_array, n_body="all"):
                template.set_annotation("b_factor", np.full(template.array_length(), np.nan, dtype=float))
            annotations_to_default = get_annotation_categories(input_atom_array, n_body="all")
            existing_annotations = get_annotation_categories(template, n_body="all")
            annotations_to_default = set(annotations_to_default) - set(existing_annotations)
            # These two annotations have no obvious defaults for the template and can be removed from the input
            annotations_to_default -= {"atom_id", "coord_to_be_noised"}
            template = default_annotations_and_conditions_from_registry(template, annotations=annotations_to_default)
            SA_SRC.set_annotation(template, SA_SRC.default_annotation(template))

            # Set coordinates from generated atom array
            _copy_coords_by_atom_name(
                generated_atom_array, generated_token_mask, template, np.ones(len(template), dtype=bool)
            )

            missing_atoms_and_indices.append((int(token_id), template))

    if np.any(extra_atoms_mask & (C_CRD.mask(input_atom_array) | C_CRDGRP.is_internal(input_atom_array))):
        raise ValueError("Predicted atom inventory omits coordinate-conditioned atoms; cannot preserve the input motif")

    # Delete the extra atoms...
    input_atom_array = input_atom_array[~extra_atoms_mask]

    # ... and add the missing atoms
    if len(missing_atoms_and_indices) > 0:
        # Insert at the end of each token. Use searchsorted on the post-deletion
        # token_id array so this still works when a token had all its atoms removed.
        post_deletion_token_ids = input_atom_array.token_id
        insertion_indices = [
            int(np.searchsorted(post_deletion_token_ids, token_id, side="right"))
            for token_id, _ in missing_atoms_and_indices
        ]
        atoms_to_add = [atom_template for _, atom_template in missing_atoms_and_indices]

        # NOTE: We insert BEFORE the specified index
        # TODO: Switch to "raise" once all inserted-template annotations are equivalent.
        input_atom_array = insert_atoms(
            input_atom_array,
            atoms_to_add,
            insertion_indices,
            fill_missing_standard_annotations=True,
            on_annotation_mismatch_policy="drop",
        )

        # Standardize atom order
        input_atom_array = input_atom_array[struc.info.standardize_order(input_atom_array)]

        # Add any missing canonical bonds after atom insertion.
        # connect_via_residue_names returns indices relative to the subset it receives,
        # so we must remap them back to positions in the full array.
        non_atomize_mask = ~input_atom_array.atomize
        non_atomize_indices = np.where(non_atomize_mask)[0]
        canonical_intra_residue_bonds = struc.bonds.connect_via_residue_names(
            input_atom_array[non_atomize_mask], inter_residue=True
        )
        bonds_as_array = canonical_intra_residue_bonds.as_array()
        if len(bonds_as_array) > 0:
            bonds_as_array[:, 0] = non_atomize_indices[bonds_as_array[:, 0]]
            bonds_as_array[:, 1] = non_atomize_indices[bonds_as_array[:, 1]]
        remapped_bonds = struc.BondList(input_atom_array.array_length(), bonds_as_array)
        input_atom_array.bonds = input_atom_array.bonds.merge(remapped_bonds)

    # Broadcast src_component to generated sidechain atoms
    input_atom_array = _broadcast_src_component_residue_wise(input_atom_array)
    if "is_token_center" in input_atom_array.get_annotation_categories():
        annotate_token_centers(input_atom_array)

    return input_atom_array
