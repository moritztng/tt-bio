"""Utilities for saving and loading atom arrays with condition annotations to/from CIF files."""

import logging
import warnings
from os import PathLike

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from tt_bio._vendor.atomworks.io import parse
from tt_bio._vendor.atomworks.io.parser import STANDARD_PARSER_ARGS
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
    AtomArrayPlusStack,
    as_atom_array_plus,
)
from tt_bio._vendor.atomworks.io.utils.io_utils import suppress_logging_messages, to_cif_file
from tt_bio._vendor.atomworks.io.utils.selection import get_annotation, get_annotation_categories
from tt_bio._vendor.atomworks.ml.conditions.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from tt_bio._vendor.atomworks.ml.conditions.base import CONDITIONS, ConditionBase


def save_atom_array_with_conditions_to_cif(atom_array: AtomArray, path: PathLike) -> None:
    """
    Saves an annotated atom array to a CIF file. Uses Condition registry to get all possible
    annotations and saves them as extra_categories.

    Args:
        atom_array: The atom array to save.
        path: The path to save the CIF file to.
    """
    annotations = set(get_annotation_categories(atom_array, n_body="all"))

    extra_categories = {}
    for condition_cls in CONDITIONS:
        if condition_cls.mask_name in annotations:  # always save mask if available
            mask = condition_cls.mask(atom_array)
            idx, value = _condition_to_idxs_values(mask, mask, n_body=condition_cls.n_body)
            extra_categories[condition_cls.mask_name] = _idxs_values_to_cif_dict(idx, value)

        if condition_cls.full_name in annotations and not condition_cls.is_mask:
            annotation = condition_cls.annotation(atom_array)
            mask = condition_cls.mask(atom_array)
            idx, value = _condition_to_idxs_values(mask, annotation, n_body=condition_cls.n_body)
            extra_categories[condition_cls.full_name] = _idxs_values_to_cif_dict(idx, value)

    if "atomize" in annotations:
        idx = np.where(atom_array.atomize)[0]
        value = np.ones(len(idx), dtype=int)
        extra_categories["atomize"] = _idxs_values_to_cif_dict(idx, value)

    to_cif_file(
        atom_array,
        path=path,
        extra_categories=extra_categories,
    )


def load_atom_array_with_conditions_from_cif(
    file: PathLike,
    *,
    assembly_id: str = "1",
    cif_parser_args: dict | None = None,
    return_data_dict: bool = False,
    fill_missing_conditions: bool = False,
) -> AtomArray | dict:
    """Loads an atom array from a CIF file with condition annotations.

    Uses the Condition registry to get all possible annotations and loads them from the CIF file.

    Args:
        file: The path to the CIF file to load.
        assembly_id: The assembly ID to load.
        cif_parser_args: Additional CIF parser arguments.
        return_data_dict: Whether to return the data dictionary. If false will return the atom array.
        fill_missing_conditions: Whether to fill missing conditions (as annotations in the atom array) with defaults.

    Returns:
        The loaded atom array with condition annotations, or the full data dictionary if ``return_data_dict`` is True.

    Note:
        Warnings about missing extra fields are suppressed unless the logging level is set to DEBUG.
    """

    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if cif_parser_args.get("cache_dir"):
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    merged_cif_parser_args = {
        **STANDARD_PARSER_ARGS,
        **{
            "fix_arginines": False,
            "add_missing_atoms": False,  # this is crucial otherwise the annotations are deleted
            "remove_ccds": [],
        },
        **cif_parser_args,
    }

    # Use Condition registry to get all possible annotations
    possible_annotations = CONDITIONS.get_valid_full_names()
    possible_masks = CONDITIONS.get_valid_mask_names()

    # Get the logger for atomworks.io to check its level
    io_logger = logging.getLogger("atomworks.io")
    suppress_missing_field_warnings = io_logger.getEffectiveLevel() > logging.DEBUG

    # Filter logging warnings for missing extra fields unless logging level is DEBUG
    if suppress_missing_field_warnings:
        # Suppress only the specific "Field ... not found in file" logging warnings
        with suppress_logging_messages("atomworks.io", "not found in file"):
            result_dict = parse(
                filename=file,
                keep_cif_block=True,
                build_assembly=(assembly_id,),  # Convert list to tuple (make hashable),
                extra_fields=possible_annotations | possible_masks,
                **merged_cif_parser_args,
            )
    else:
        result_dict = parse(
            filename=file,
            keep_cif_block=True,
            build_assembly=(assembly_id,),  # Convert list to tuple (make hashable),
            extra_fields=possible_annotations | possible_masks,
            **merged_cif_parser_args,
        )

    _ = result_dict.pop("asym_unit")
    cif_block = result_dict.pop("cif_block")
    atom_array = result_dict["assemblies"][assembly_id][0]
    atom_array = _add_design_annotations_from_cif_block_metadata(atom_array, cif_block)
    if fill_missing_conditions:
        atom_array = fill_missing_conditions_with_defaults(atom_array)

    if return_data_dict:
        result_dict["atom_array"] = atom_array
        return result_dict
    else:
        return atom_array


def fill_missing_conditions_with_defaults(atom_array: AtomArray) -> AtomArray:
    """
    Checks to see if any conditions are missing from the atom array and fills them with defaults.
    """
    all_annotation_categories = set(get_annotation_categories(atom_array, n_body="all"))
    for condition_cls in CONDITIONS:
        # Check if mask / annotation exists:
        is_mask_annotated = condition_cls.mask_name in all_annotation_categories
        is_condition_annotated = condition_cls.full_name in all_annotation_categories

        if is_mask_annotated and (condition_cls.is_mask or is_condition_annotated):
            # skip if both are annotated or we only need the mask
            continue

        # if we're here, either the mask or the annotation is missing or both
        if is_mask_annotated:
            default_mask = condition_cls.mask(atom_array, default="generate")
            condition_cls.set_mask(atom_array, default_mask)

        elif is_condition_annotated:
            default_annotation = condition_cls.annotation(atom_array, default="generate")
            condition_cls.set_annotation(atom_array, default_annotation)

        else:
            default_annotation = condition_cls.annotation(atom_array, default="generate")
            default_mask = condition_cls.mask(atom_array, default="generate")
            condition_cls.set_mask(atom_array, default_mask)
            condition_cls.set_annotation(atom_array, default_annotation)

    return atom_array


def _condition_to_idxs_values(mask: np.ndarray, annotation: np.ndarray, n_body: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a condition to a pair of index arrays and a value array"""
    if n_body == 1:
        idx = np.where(mask)[0]
        value = annotation[mask]
    elif n_body == 2:
        idx = annotation.pairs
        value = annotation.values
    else:
        raise NotImplementedError(f"Condition with {n_body} bodies is not supported.")

    value = value.astype(int) if value.dtype == bool else value
    return idx, value


def _add_condition_from_idxs_values(
    idx: np.ndarray, value: np.ndarray, atom_array: AtomArray, condition_cls: ConditionBase, mask: bool
) -> np.ndarray:
    """Add a condition from a pair of index arrays and a value array"""
    if condition_cls.n_body == 1:
        if mask:
            annotation = np.zeros(atom_array.array_length(), dtype=bool)
            annotation[idx] = value
            condition_cls.set_mask(atom_array, annotation)
        else:
            annotation = condition_cls.default_annotation(atom_array)
            annotation[idx] = value
            condition_cls.set_annotation(atom_array, annotation)
    elif condition_cls.n_body == 2:
        if mask:
            mask_annot = AnnotationList2D(
                n_atoms=atom_array.array_length(),
                pairs=idx,
                values=value,
            )
            condition_cls.set_mask(atom_array, mask_annot)
        else:
            condition_cls.set_annotation(atom_array, pairs=idx, values=value)
    else:
        raise NotImplementedError(f"Condition with {condition_cls.n_body} bodies is not supported.")


def _add_design_annotations_from_cif_block_metadata(
    atom_array: AtomArray | AtomArrayStack,
    cif_block: pdbx.CIFBlock,
) -> AtomArrayPlus | AtomArrayPlusStack:
    if isinstance(atom_array, AtomArray):
        atom_array = as_atom_array_plus(atom_array)
    elif isinstance(atom_array, AtomArrayStack):
        atom_array = AtomArrayPlusStack.from_atom_array_stack(atom_array)
    else:
        raise ValueError(f"AtomArrayPlus or AtomArrayStack expected, got {type(atom_array)}")

    for key in cif_block:
        if key == "atomize":
            atomize = get_annotation(
                atom_array,
                "atomize",
                default=np.zeros(atom_array.array_length(), dtype=bool),
            )
            idxs, values = _cif_dict_to_idxs_values(cif_block[key], dtype=bool)
            atomize[idxs] = values
            atom_array.set_annotation("atomize", atomize)
            continue

        if not key.startswith("condition_") and not key.startswith("mask_"):
            # ... skip if it does not start with condition_ or mask_
            continue

        mask = key.startswith("mask_")
        condition_cls = CONDITIONS.from_full_name(key)
        idxs, values = _cif_dict_to_idxs_values(cif_block[key], dtype=bool if mask else condition_cls.dtype)

        _add_condition_from_idxs_values(idxs, values, atom_array, condition_cls, mask)

    return atom_array


def _cif_dict_to_idxs_values(cif_dict: dict, dtype: np.dtype | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dictionary of CIF categories to a pair of index arrays and a value array"""
    idxs, values = [], []
    for key in cif_dict:
        if key.startswith("idx"):
            idxs.append(cif_dict[key].as_array(int))
        elif key.startswith("val"):
            values.append(cif_dict[key].as_array(dtype))
    idxs = np.stack(idxs, axis=1).squeeze()
    values = np.stack(values, axis=1).squeeze()
    return idxs, values


def _atleast_2d_last(arr: np.ndarray) -> np.ndarray:
    """Same as np.atleast_2d but adds a dimension at the end instead of the beginning"""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return np.expand_dims(arr, axis=-1)  # Add dimension at the end
    else:
        return arr


def _idxs_values_to_cif_dict(idxs: np.ndarray, values: np.ndarray) -> dict:
    """Convert a pair of index arrays and a value array to a dictionary of CIF categories"""
    idxs = _atleast_2d_last(idxs)  # (N, n_body)
    values = _atleast_2d_last(values)  # (N, n_values)
    return {f"idx{i}": idxs[:, i] for i in range(idxs.shape[1])} | {
        f"val{i}": values[:, i] for i in range(values.shape[1])
    }


def default_annotations_and_conditions_from_registry(
    atom_array: AtomArray,
    annotations: list[str],
) -> AtomArrayPlus:
    """Add all registered annotations and conditions with default values.

    Generates defaults from both ``ANNOTATOR_REGISTRY`` and ``CONDITIONS`` registry.
    Ignores annotations without known default generation methods.
    Promotes the input ``atom_array`` to an ``AtomArrayPlus`` in order to handle n-body annotations.

    Args:
      atom_array: AtomArray to annotate (modified in-place).
      annotations: Specific annotation names to generate.
    """
    atom_array = as_atom_array_plus(atom_array)

    # Filter to requested annotations
    annotations_set = set(annotations)
    target_annotator_names = annotations_set & set(ANNOTATOR_REGISTRY.keys())
    target_condition_names = annotations_set - target_annotator_names

    # Generate ANNOTATOR_REGISTRY annotations using ensure_annotations
    if target_annotator_names:
        ensure_annotations(atom_array, *target_annotator_names)

    # Generate CONDITIONS annotations and masks
    for condition in CONDITIONS:
        # Handle mask
        mask_name = condition.mask_name
        if mask_name in target_condition_names:
            try:
                condition.set_mask(atom_array, condition.default_mask(atom_array))
            except Exception as e:
                warnings.warn(f"Failed to generate mask '{mask_name}': {e}", stacklevel=2)

        # Handle annotation (if not a mask-only condition)
        if not condition.is_mask:
            full_name = condition.full_name
            if full_name in target_condition_names:
                try:
                    condition.set_annotation(atom_array, condition.default_annotation(atom_array))
                except Exception as e:
                    warnings.warn(f"Failed to generate annotation '{full_name}': {e}", stacklevel=2)

    return atom_array
