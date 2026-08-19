"""
A module for lazily adding annotations to `AtomArray` objects.

This module provides a framework for defining, registering, and applying annotations
to `AtomArray` instances. Annotations are numpy arrays of the same
length as the atom array, providing information about each atom (e.g., `is_protein`,
`is_ligand`).

The core components are:
- `ANNOTATOR_REGISTRY`: A global dictionary that maps annotation names to their
  generator functions. This registry is populated automatically at import time.
- `_register_lazy_annotator`: A decorator used to register a new annotation
  generator. The decorated function should accept an `AtomArray` and return a
  `numpy.ndarray` with the annotation values.
- `ensure_annotations`: A function to ensure that one or more annotations are
  present on an `AtomArray`. If an annotation is missing, its registered
  generator function is called to compute and add it.
- `remove_annotations` and `clear_generated_annotations`: Utility functions to
  remove specific or all registered annotations from an `AtomArray`.

Example:
    To define a new annotation, create a function and decorate it:

    >>> @_register_lazy_annotator("is_hydrophobic")
    ... def is_hydrophobic(atom_array: AtomArray) -> np.ndarray:
    ...     hydrophobic_res = ["ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET"]
    ...     return np.isin(atom_array.res_name, hydrophobic_res)

    To apply this and other annotations to an `AtomArray` in-place:

    >>> from atomworks.ml.utils.testing import cached_parse
    >>> data = cached_parse("1L2Y")
    >>> atom_array = data["atom_array"]
    >>> ensure_annotations(atom_array, "is_hydrophobic", "is_protein")
    >>> print(atom_array.get_annotation("is_hydrophobic"))
    [ True  True  True ... False False False]
"""

import functools
from collections.abc import Callable
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Int

from tt_bio._vendor.atomworks.common import listmap
from tt_bio._vendor.atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    METAL_ELEMENTS,
    NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA,
    STANDARD_AA_TIP_ATOM_NAMES,
    STANDARD_DNA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from tt_bio._vendor.atomworks.enums import ChainType
from tt_bio._vendor.atomworks.io.utils.atom_array import apply_and_spread
from tt_bio._vendor.atomworks.ml.transforms.atom_array import get_within_group_res_idx
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts

Array = np.ndarray
"""Alias for numpy.ndarray"""

ANNOTATOR_REGISTRY: dict[str, Callable[[AtomArray], None]] = {}
"""
Registry of annotation generators.

NOTE: This is a global registry and will auto-populate the annotation generator
functions as long as they are decorated with `register_lazy_annotator`. These
registration functions get called at import time.
"""


# General tooling for annotating atom arrays with simple annotations
def _register_lazy_annotator(annot_name: str) -> Callable:
    """
    Decorator that adds an annotation to AtomArray if it doesn't already exist.
    Also registers the annotation in the ANNOTATION_GENERATORS.

    Args:
        annot_name: Name of the annotation to check/add

    Returns:
        Decorator function
    """

    def decorator(fn: Callable[[AtomArray], Array]) -> Callable[[AtomArray], None]:
        @functools.wraps(fn)
        def wrapper(atom_array: AtomArray) -> None:
            if annot_name not in atom_array.get_annotation_categories():
                values = fn(atom_array)
                atom_array.set_annotation(annot_name, values)

        # Register the annotation in the ANNOTATOR_REGISTRY
        ANNOTATOR_REGISTRY[annot_name] = wrapper

        return wrapper

    return decorator


def _requires_annotations(*annot_names: str) -> Callable:
    """
    Decorator that ensures required annotations exist before function execution.
    Required annotations must be registered in the ANNOTATOR_REGISTRY.

    NOTE: When using this in conjunction with `_register_lazy_annotator`, the
    annotation order has to be:
    ```python
    @_register_lazy_annotator("is_XXX")
    @_requires_annotations("is_YYY", "is_ZZZ")
    def is_XXX(atom_array: AtomArray) -> Bool[Array, "n_atoms"]: ...
    ```
    Otherwise, the required `is_YYY` and `is_ZZZ` annotations will not be generated
    before `is_XXX` is called.

    Args:
        *annot_names: Names of required annotations

    Returns:
        Decorator function
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(atom_array: AtomArray, *args, **kwargs) -> Any:
            # Generate missing annotations
            for annot_name in annot_names:
                if annot_name not in atom_array.get_annotation_categories():
                    if annot_name not in ANNOTATOR_REGISTRY:
                        raise ValueError(f"No generator found for annotation: {annot_name}")
                    ANNOTATOR_REGISTRY[annot_name](atom_array)

            # Call the original function
            return fn(atom_array, *args, **kwargs)

        return wrapper

    return decorator


def ensure_annotations(atom_array: AtomArray, *annotation_names: str) -> None:
    """
    Ensure that specified annotations exist on the AtomArray.
    If an annotation does not exist, it will be generated according to the
    generator function registered in the `ANNOTATOR_REGISTRY` and added to
    the `AtomArray` in-place.

    Args:
        atom_array: The AtomArray to annotate
        *annotation_names: Names of annotations to ensure. Must be
            registered in the `ANNOTATOR_REGISTRY`. To register an annotation,
            decorate a function with `_register_lazy_annotator("annot_name")`.

    Raises:
        ValueError: If a requested annotation has no generator
    """
    for name in annotation_names:
        if name not in atom_array.get_annotation_categories():
            if name not in ANNOTATOR_REGISTRY:
                raise ValueError(f"No generator found for annotation: {name}")
            ANNOTATOR_REGISTRY[name](atom_array)


def remove_annotations(atom_array: AtomArray, *annotation_names: str) -> None:
    """
    Remove annotations from the AtomArray.

    Args:
        atom_array: The AtomArray to modify
        *annotation_names: Names of annotations to remove

    Note:
        Silently skips annotations that don't exist.
    """
    existing_annotations = atom_array.get_annotation_categories()
    for name in annotation_names:
        if name in existing_annotations:
            atom_array.del_annotation(name)


def clear_generated_annotations(atom_array: AtomArray) -> None:
    """
    Remove all annotations that were generated by this module.

    Args:
        atom_array: The AtomArray to modify
    """
    existing_annotations = atom_array.get_annotation_categories()
    for name in ANNOTATOR_REGISTRY:
        if name in existing_annotations:
            atom_array.del_annotation(name)


# Custom annotation generators
@_register_lazy_annotator("is_protein")
def is_protein(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein chains."""
    return np.isin(atom_array.chain_type, ChainType.get_proteins())


@_register_lazy_annotator("is_nucleic_acid")
def is_nucleic_acid(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to nucleic acid chains."""
    return np.isin(atom_array.chain_type, ChainType.get_nucleic_acids())


@_register_lazy_annotator("is_dna")
def is_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to DNA chains."""
    return np.isin(atom_array.chain_type, ChainType.DNA)


@_register_lazy_annotator("is_rna")
def is_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to RNA chains."""
    return np.isin(atom_array.chain_type, ChainType.RNA)


@_register_lazy_annotator("is_standard_aa")
def is_standard_aa(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard amino acids."""
    return np.isin(atom_array.res_name, STANDARD_AA)


@_register_lazy_annotator("is_standard_or_unknown_aa")
def is_standard_or_unknown_aa(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown amino acid residue.

    NOTE: May be different than the chain-level "is_protein" in the case of mixed-type chains (e.g., a protein with a non-canonical amino acid).
    """
    return np.isin(atom_array.res_name, [*STANDARD_AA, UNKNOWN_AA])


@_register_lazy_annotator("is_protein_backbone")
@_requires_annotations("is_protein")
def is_protein_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein backbone."""

    is_protein = atom_array.get_annotation("is_protein")
    is_backbone_atom = np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    return is_protein & is_backbone_atom


@_register_lazy_annotator("is_protein_sidechain")
@_requires_annotations("is_protein")
def is_protein_sidechain(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein sidechain."""
    is_protein = atom_array.get_annotation("is_protein")
    is_sidechain_atom = ~np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    return is_protein & is_sidechain_atom


@_register_lazy_annotator("is_rna_backbone")
@_requires_annotations("is_rna")
def is_rna_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to RNA backbone (sugar-phosphate)."""
    is_rna = atom_array.get_annotation("is_rna")
    is_backbone_atom = np.isin(atom_array.atom_name, RNA_BACKBONE_ATOM_NAMES)
    return is_rna & is_backbone_atom


@_register_lazy_annotator("is_dna_backbone")
@_requires_annotations("is_dna")
def is_dna_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to DNA backbone (sugar-phosphate)."""
    is_dna = atom_array.get_annotation("is_dna")
    is_backbone_atom = np.isin(atom_array.atom_name, DNA_BACKBONE_ATOM_NAMES)
    return is_dna & is_backbone_atom


@_register_lazy_annotator("is_nucleic_acid_backbone")
@_requires_annotations("is_nucleic_acid")
def is_nucleic_acid_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to nucleic acid backbone (sugar-phosphate)."""
    is_na = atom_array.get_annotation("is_nucleic_acid")
    is_backbone_atom = np.isin(atom_array.atom_name, NUCLEIC_ACID_BACKBONE_ATOM_NAMES)
    return is_na & is_backbone_atom


@_register_lazy_annotator("is_standard_rna")
def is_standard_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard RNA."""
    return np.isin(atom_array.res_name, STANDARD_RNA)


@_register_lazy_annotator("is_standard_dna")
def is_standard_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard DNA."""
    return np.isin(atom_array.res_name, STANDARD_DNA)


@_register_lazy_annotator("is_standard_or_unknown_dna")
def is_standard_or_unknown_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown DNA residue.

    NOTE: May be different than the chain-level "is_dna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    return np.isin(atom_array.res_name, [*STANDARD_DNA, UNKNOWN_DNA])


@_register_lazy_annotator("is_standard_or_unknown_rna")
def is_standard_or_unknown_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown RNA residue.

    NOTE: May be different than the chain-level "is_rna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    return np.isin(atom_array.res_name, [*STANDARD_RNA, UNKNOWN_RNA])


@_register_lazy_annotator("is_pyrimidine")
def is_pyrimidine(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to pyrimidine residues."""
    return np.isin(atom_array.res_name, STANDARD_PYRIMIDINE_RESIDUES)


@_register_lazy_annotator("is_purine")
def is_purine(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to purine residues."""
    return np.isin(atom_array.res_name, STANDARD_PURINE_RESIDUES)


@_register_lazy_annotator("is_tip_atom")
def is_tip_atom(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Identify tip atoms in standard amino acids."""
    is_tip_atom = np.zeros(atom_array.array_length(), dtype=bool)

    for res_name, tip_atom_names in STANDARD_AA_TIP_ATOM_NAMES.items():
        mask = (atom_array.res_name == res_name) & np.isin(atom_array.atom_name, tip_atom_names)
        is_tip_atom |= mask

    return is_tip_atom


@_register_lazy_annotator("is_res_start")
def is_res_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Mark the first atom of each residue."""
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)
    is_res_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_res_start[res_starts] = True
    return is_res_start


@_register_lazy_annotator("is_token_start")
def is_token_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Mark the first atom of each token."""
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    is_token_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_token_start[token_starts] = True
    return is_token_start


@_register_lazy_annotator("is_chain_start")
def is_chain_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to chain starts."""
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=False)
    is_chain_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_chain_start[chain_starts] = True
    return is_chain_start


@_register_lazy_annotator("is_polymer")
def is_polymer(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to polymers."""
    return np.isin(atom_array.chain_type, ChainType.get_polymers())


@_register_lazy_annotator("is_ligand")
def is_ligand(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to ligands."""
    return np.isin(atom_array.chain_type, ChainType.NON_POLYMER)


@_register_lazy_annotator("is_metal")
def is_metal(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to metal ions."""
    return np.isin(atom_array.res_name, METAL_ELEMENTS)


@_register_lazy_annotator("is_carbohydrate")
def is_carbohydrate(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to carbohydrates."""
    return struc.filter_carbohydrates(atom_array)


@_register_lazy_annotator("within_chain_res_idx")
def within_chain_res_idx(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F821
    """Get the within-chain residue index for the atom array."""
    annotations = atom_array.get_annotation_categories()
    chain_group = "chain_iid" if "chain_iid" in annotations else "chain_id"
    return get_within_group_res_idx(atom_array, group_by=chain_group)


@_register_lazy_annotator("res_min_occupancy")
@_requires_annotations("is_res_start")
def res_min_occupancy(atom_array: AtomArray) -> Float[Array, "n_atoms"]:  # noqa: F821
    """Calculate minimum occupancy for each residue."""
    is_res_start = atom_array.get_annotation("is_res_start")
    res_start_idxs = np.where(is_res_start)[0]
    res_segments = np.concatenate([res_start_idxs, [atom_array.array_length()]])
    return apply_and_spread(res_segments, atom_array.occupancy, np.min)


@_register_lazy_annotator("token_min_occupancy")
@_requires_annotations("is_token_start")
def token_min_occupancy(atom_array: AtomArray) -> Float[Array, "n_atoms"]:  # noqa: F821
    """Calculate minimum occupancy for each token."""
    is_token_start = atom_array.get_annotation("is_token_start")
    token_start_idxs = np.where(is_token_start)[0]
    token_segments = np.concatenate([token_start_idxs, [atom_array.array_length()]])
    return apply_and_spread(token_segments, atom_array.occupancy, np.min)


@_register_lazy_annotator("token_id")
@_requires_annotations("is_token_start")
def token_id(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F821
    """Assign a unique ID to each token."""
    is_token_start = atom_array.get_annotation("is_token_start")
    token_start_idxs = np.where(is_token_start)[0]
    token_id = np.arange(sum(is_token_start))
    token_segments = np.concatenate([token_start_idxs, [atom_array.array_length()]])
    return struc.segments.spread_segment_wise(token_segments, token_id)


@_register_lazy_annotator("res_has_tip_atom")
@_requires_annotations("is_tip_atom", "is_res_start")
def res_has_tip_atom(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if each residue contains at least one tip atom."""
    is_tip_atom = atom_array.get_annotation("is_tip_atom")
    is_res_start = atom_array.get_annotation("is_res_start")
    res_start_idxs = np.where(is_res_start)[0]
    res_segments = np.concatenate([res_start_idxs, [atom_array.array_length()]])
    return apply_and_spread(res_segments, is_tip_atom, np.any)


@_register_lazy_annotator("atomic_number")
def atomic_number(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F821
    """Get atomic numbers for each atom."""
    return np.array(listmap(ELEMENT_NAME_TO_ATOMIC_NUMBER.get, np.char.upper(atom_array.element)), dtype=np.int8)
