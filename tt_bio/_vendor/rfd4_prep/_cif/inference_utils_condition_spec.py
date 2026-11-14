# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Orchestration for the YAML/JSON condition-spec format.

See ``design_yaml_condition_spec.md`` / ``docs/condition_spec_yaml.md``: a
condition spec is a shorthand for building an annotated CIF -- load input
structure(s) (§4.2), subset + insert expandable segments via one contig
string (§4.3), set Conditions/StandardAnnotations by name (§4.4), then
finalize (§4.5: pre-save validation + a "summarize changes" report, no
explicit back-fill). :py:func:`build_annotated_atom_array_from_condition_spec`
does all of this for one example; :py:func:`resolve_condition_spec_inputs`
does it for every example in a spec file and writes each to disk.
"""

import logging
import json
from dataclasses import fields as dataclass_fields
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tt_bio._vendor.atomworks.io.config import PrepareConfig
from tt_bio._vendor.atomworks.io.parser import parse, parse_atom_array
from tt_bio._vendor.atomworks.io.transforms.atom_array import add_id_and_entity_annotations, add_iid_annotations
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AtomArrayPlus, as_atom_array_plus, concatenate_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.io_utils import to_cif_file
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_IDX, ConditionBase

from .conditions import C_CATH, C_CENTROID, C_CRDGRP, C_IDXGRP, C_INTCRD
from .conditions_reporting import (
    check_index_group_molecule_consistency,
    check_index_group_res_id_gaps,
    run_pipeline_validation,
    summarize_conditions,
    sync_index_groups,
)
from .inference_utils_contig_spec import apply_contig
from .transforms_utils_conditioning_utils import (
    separate_coordinate_groups,
    set_absolute_coordinate_conditioning,
)
from .utils_common import validate_design_id
from .utils_condition_resolution import (
    coerce_enum_value,
    parse_centroid_value,
    resolve_standard_annotation,
    set_cath_condition,
)
from .utils_mask_spec import resolve_mask
from .utils_parse_presets import ANNOTATED_CIF, RCSB_WITH_METALS
from .utils_structure_files import is_structure_file

logger = logging.getLogger(__name__)

_PREPARE_CONFIG_FIELD_NAMES = {f.name for f in dataclass_fields(PrepareConfig)}

# `rfproteina.utils.parse_presets.ANNOTATED_CIF`'s prepare-relevant fields (its ParseConfig-only
# fields -- build_assembly, altloc, ... -- don't apply to `parse_atom_array`, which re-derives
# id/entity annotations and ligand/chain info without altering atoms already prepared once at
# step-1 load time). Derived directly from ANNOTATED_CIF rather than duplicated by hand, so it
# can never drift out of sync with it.
_FINALIZE_PREPARE_CONFIG = PrepareConfig(
    **{k: v for k, v in ANNOTATED_CIF.to_dict().items() if k in _PREPARE_CONFIG_FIELD_NAMES}
)

_REMINDER = (
    "\nREMINDER: Conditions not listed above (including coordinates and sequence) are NOT provided "
    "to the model and will be freely generated."
)

# Every field name a condition_spec example dict consumes. `out_dir` is deliberately absent:
# nothing reads it (the CLI owns the output directory), so accepting it silently discarded an
# author's intent -- it is now rejected as an unrecognized field like any other typo.
_CONDITION_SPEC_EXAMPLE_KEYS = frozenset({"input", "input_as_annotated_cif", "contig", "conditions"})

# Field names unique to the removed legacy `InputSpecification` (contig-JSON) dialect. The legacy
# realization path (`input_parsing.py`) has been deleted, but any example naming one of these is
# still detected here so it can be rejected with a clear "migrate to condition_spec/CIF" error
# rather than misinterpreted as condition_spec. This is the legacy dialect's frozen vocabulary --
# it never gains fields -- so it is listed explicitly (the two names it shared with condition_spec,
# `input` and `contig`, are excluded).
_LEGACY_ONLY_FIELDS = frozenset(
    {
        "add_cath_code",
        "allow_non_motif_centered",
        "atom_array_input",
        "cif_parser_args",
        "dialect",
        "extra",
        "infer_ori_strategy",
        "internal_coord_groups",
        "is_non_loopy",
        "is_partial_diffusion",
        "length",
        "ligand",
        "nucleic_ss",
        "plddt_enhanced",
        "redesign_motif_sidechains",
        "select_buried",
        "select_exposed",
        "select_fixed_atoms",
        "select_hbond_acceptor",
        "select_hbond_donor",
        "select_hotspots",
        "select_partially_buried",
        "select_unfixed_sequence",
        "symmetry",
        "unindex",
    }
)


def split_global_args(spec: dict) -> dict[str, dict]:
    """Return ``{example_id: merged_example}``, folding top-level ``global_args`` into each example.

    Raises:
      ValueError: If ``global_args`` or any example is not a mapping. Skipping such
        entries instead would silently drop a whole example on a small indentation
        mistake, producing a run with fewer designs than the author asked for.
    """
    global_args = spec.get("global_args", {})
    if not isinstance(global_args, dict):
        raise ValueError(f"condition_spec `global_args` must be a mapping, got {type(global_args).__name__}.")
    invalid_examples = {
        example_id: type(example).__name__
        for example_id, example in spec.items()
        if example_id != "global_args" and not isinstance(example, dict)
    }
    if invalid_examples:
        details = ", ".join(f"{name} ({value_type})" for name, value_type in invalid_examples.items())
        raise ValueError(f"condition_spec examples must be mappings; invalid entries: {details}.")
    return {
        example_id: {**global_args, **example} for example_id, example in spec.items() if example_id != "global_args"
    }


def is_legacy_dialect(data: dict) -> bool:
    """Whether ``data`` is written in the removed legacy contig-JSON dialect.

    There's no explicit marker to check for -- the legacy dialect predated condition_spec
    existing at all. Instead, this looks at each example's own field names (with ``global_args``
    defaults folded in): condition_spec's ``input`` key is required (missing it raises `KeyError`
    downstream), so an example lacking it can't be condition_spec; and any field name outside
    condition_spec's small vocabulary -- i.e. any of the legacy-only names in
    :py:data:`_LEGACY_ONLY_FIELDS` (``select_fixed_atoms``, ``unindex``, the per-example
    ``dialect`` int, ...) -- is unambiguous evidence of the legacy one. Every legacy example ends
    up naming at least one such field (a fully-diffused example uses ``length``/``contig`` alone,
    neither of which condition_spec needs). The legacy path itself is gone, so callers use this
    only to reject such inputs with a clear migration error.

    Args:
      data: A loaded spec file (as returned by :py:func:`load_spec_file`), ``global_args``
        key included.

    Returns:
      ``True`` if any example (or ``global_args``) looks like the legacy dialect.
    """
    global_args = data.get("global_args")
    if isinstance(global_args, dict) and set(global_args) & _LEGACY_ONLY_FIELDS:
        return True
    for example in split_global_args(data).values():
        if "input" not in example or set(example) & _LEGACY_ONLY_FIELDS:
            return True
    return False


def _validate_structure_file_suffix(path: Path) -> None:
    if not is_structure_file(path):
        raise ValueError(
            f"condition_spec inputs must be CIF or BinaryCIF (plain, gzip, or Zstandard), got {path}. PDB files don't "
            "reliably encode the bonds/formal charges the model consumes directly -- convert "
            "with atomworks.io.parser.parse + to_cif_file, and manually audit the resulting "
            "bonds/charges before using it."
        )


def load_condition_spec_inputs(paths: list[str | PathLike], *, as_annotated_cif: bool = False) -> AtomArrayPlus:
    """Parse and combine one or more structure files into a single AtomArrayPlus (§4.2).
    Loads the asymmetric unit only (``build_assembly=None``).

    Args:
      paths: One or more CIF or BinaryCIF paths, optionally gzip- or Zstandard-compressed.
      as_annotated_cif: If ``True``, load any existing Condition/StandardAnnotation
        columns already present (e.g. re-editing a previous output of this tool).
        If ``False`` (default), any existing conditioning is ignored.

    Returns:
      The combined structure.

    Raises:
      ValueError: If any input isn't a CIF/BCIF file, or if `chain_id` collides
        across multiple input files.
    """
    config = (ANNOTATED_CIF if as_annotated_cif else RCSB_WITH_METALS).replace(build_assembly=None)

    arrays: list[AtomArrayPlus] = []
    for raw_path in paths:
        path = Path(raw_path)
        _validate_structure_file_suffix(path)
        result = parse(path, config=config)
        arr = as_atom_array_plus(result["asym_unit"][0])
        if "transformation_id" not in arr.get_annotation_categories():
            # The asymmetric unit has exactly one copy per chain_id/res_id by construction
            arr.set_annotation("transformation_id", np.full(arr.array_length(), "1", dtype=str))
        arr = add_iid_annotations(arr)
        arrays.append(arr)

    if len(arrays) == 1:
        return arrays[0]

    seen_chain_ids: dict[str, str] = {}
    for raw_path, arr in zip(paths, arrays, strict=True):
        for chain_id in np.unique(arr.chain_id):
            chain_id = str(chain_id)
            if chain_id in seen_chain_ids:
                raise ValueError(
                    f"chain_id {chain_id!r} appears in both {seen_chain_ids[chain_id]!r} and "
                    f"{str(raw_path)!r} -- condition_spec does not auto-rename colliding chain ids "
                    "across input files; rename one of them so `contig` selections stay unambiguous."
                )
            seen_chain_ids[chain_id] = str(raw_path)

    combined = concatenate_atom_array_plus(
        separate_coordinate_groups(arrays),
        on_annotation_mismatch_policy="raise",
        fill_missing_standard_annotations=True,
    )
    # Each file's ids (chain/pn_unit/molecule, and their instance-id counterparts) were derived
    # relative to that file alone, so e.g. both files' first molecule would be molecule_iid 0 --
    # blindly concatenating would silently alias two unrelated molecules together (and,
    # downstream, merge their index groups). Re-derive from the combined structure so ids are
    # unique across the whole combination.
    ensure_annotations(combined, "is_polymer")
    combined = add_id_and_entity_annotations(combined)
    combined = add_iid_annotations(combined)
    return combined


def _coerce_bool(value: Any) -> bool:
    """Strictly interpret one authored condition value as a boolean.

    Deliberately *not* ``bool(value)``. JSON object keys are always strings, and
    quoting a YAML key does the same thing, so a plain truthiness test makes the
    string ``"false"`` mean **true** -- silently inverting the author's intent and
    conditioning on exactly the atoms they meant to exclude. Only real booleans and
    the exact strings ``"true"``/``"false"`` are accepted; ints, floats and ``None``
    are spec errors rather than something to guess at.

    Raises:
      ValueError: If ``value`` is not a boolean or a ``"true"``/``"false"`` string.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Expected a boolean condition value (true/false), got {value!r} ({type(value).__name__}).")


def _apply_condition_value(
    cls: type,
    atom_array: AtomArrayPlus,
    indices: np.ndarray,
    value: Any,
) -> None:
    if cls is C_CATH:
        set_cath_condition(atom_array, indices, value)
        return
    if cls.dtype is bool:
        annotation = cls.annotation(atom_array, default="generate").copy()
        annotation[indices] = _coerce_bool(value)
        cls.set_annotation(atom_array, annotation)
        return

    if cls is C_CRD:
        set_absolute_coordinate_conditioning(atom_array, indices, _coerce_bool(value))
        return

    if issubclass(cls, ConditionBase) and cls.has_ground_truth:
        annotation = cls.annotation(atom_array, default="generate").copy()
        if _coerce_bool(value):
            ground_truth = cls.ground_truth_value(atom_array)
            annotation[indices] = ground_truth[indices]
        else:
            default = cls.default_annotation(atom_array)
            annotation[indices] = default[indices]
        cls.set_annotation(atom_array, annotation)
        return

    if cls is C_CENTROID:
        resolved_value = parse_centroid_value(value)
    else:
        enum_cls = cls.enum
        if enum_cls is not None:
            resolved_value = coerce_enum_value(enum_cls, value)
        else:
            # Reject presence-only authoring for conditions with no structure-derived value;
            # assigning a bool to a short string annotation could silently truncate it.
            if isinstance(value, bool) or (isinstance(value, str) and value.lower() in ("true", "false")):
                raise ValueError(
                    f"{cls.get_full_name()} writes a value, so it cannot take {value!r}. Either pass the "
                    "value to write, or -- if this condition's value is ground truth read off the "
                    "structure -- implement ground_truth_value() on the condition class."
                )
            resolved_value = value

    annot = cls.annotation(atom_array, default="generate").copy()
    annot[indices] = resolved_value
    cls.set_annotation(atom_array, annot)


def _requires_group(cls: type, value: Any) -> bool:
    """Whether ``value`` needs an accompanying ``group:`` (§4.4)."""
    return cls is C_IDX and not _coerce_bool(value)


def _apply_grouped_condition(
    cls: type,
    atom_array: AtomArrayPlus,
    groups: dict[Any, list[np.ndarray]],
    pending_index_groups: dict[Any, list[np.ndarray]],
) -> None:
    """Apply every user-labeled group for one group-requiring condition (§4.4).

    ``groups`` maps each raw, user-chosen group label (its value never matters, only
    whether two selections share one) to the list of index arrays selected under that
    label -- possibly from more than one ``select:`` entry.
    """
    assert cls is C_IDX
    all_indices = np.concatenate([idx for index_arrays in groups.values() for idx in index_arrays])
    mask = C_IDX.mask(atom_array, default="generate").copy()
    mask[all_indices] = False
    C_IDX.set_annotation(atom_array, mask)
    for label, index_arrays in groups.items():
        pending_index_groups.setdefault(label, []).extend(index_arrays)


def _finalize_index_groups(atom_array: AtomArrayPlus, pending_index_groups: dict[Any, list[np.ndarray]]) -> None:
    """Stamp the user-labeled ``C_IDXGRP`` ids once, against the *final* ``C_IDX`` mask."""
    if not pending_index_groups:
        return
    sync_index_groups(atom_array, preserve_manual_edits=True)
    group_ids = atom_array.get_annotation(C_IDXGRP.full_name).copy()
    unindexed = ~C_IDX.mask(atom_array, default="generate")
    next_id = int(group_ids.max()) + 1 if group_ids.size else 0
    for index_arrays in pending_index_groups.values():
        group_indices = np.concatenate(index_arrays)
        # Atoms a later entry re-indexed keep the baseline id
        group_indices = group_indices[unindexed[group_indices]]
        if not group_indices.size:
            continue
        group_ids[group_indices] = next_id
        next_id += 1
    C_IDXGRP.set_annotation(atom_array, group_ids)


def _apply_conditions(atom_array: AtomArrayPlus, conditions: dict, original_res_id: np.ndarray) -> None:
    """Set every annotation in ``conditions:`` on ``atom_array``, in place (§4.4).

    Selections are resolved against ``original_res_id`` (pre-contig numbering) rather
    than the assembled structure's own (renumbered) ``res_id`` -- see §4.3.1 -- so
    ``res_id`` is temporarily swapped for the duration of selection evaluation.

    Values are applied strictly in declaration order, so a later entry overrides an
    earlier one that touches the same atoms. Grouped values are flushed as soon as
    their own value's entries are exhausted rather than being deferred to the end:
    deferring them made a grouped value always win over a plain one regardless of
    declaration order (e.g. ``C_IDX: {false: [... group: g], true: [...]}`` stayed
    false), which contradicts the documented last-write-wins rule.

    ``C_IDXGRP`` id assignment is deferred until the final ``C_IDX`` annotation has settled.
    Internal-coordinate group membership is written directly to ``C_CRDGRP`` and the
    ``C_INTCRD`` coordinate payload is filled later, before unindexing.
    """
    for name, values in conditions.items():
        cls = resolve_standard_annotation(name)
        if cls is C_INTCRD:
            raise ValueError("C_INTCRD is a saved payload; edit C_CRDGRP instead.")
        if cls is C_CRDGRP:
            for value, entries in values.items():
                if value is not None and value != "null":
                    if isinstance(value, bool) or not isinstance(value, int | str):
                        raise ValueError("C_CRDGRP requires a positive integer group ID or null.")
                    if not str(value).isdigit() or not 0 < int(value) <= C_CRDGRP.MAX_GROUP_ID:
                        raise ValueError(
                            f"C_CRDGRP requires an integer ID from 1 to {C_CRDGRP.MAX_GROUP_ID}; "
                            "reserved IDs are not editable."
                        )
                if not isinstance(entries, list) or any(
                    not isinstance(e, dict) or set(e) != {"select"} for e in entries
                ):
                    raise ValueError("C_CRDGRP entries must contain only select; the value is the group ID.")
    real_res_id = atom_array.res_id.copy()
    atom_array.set_annotation("res_id", original_res_id)
    pending_index_groups: dict[Any, list[np.ndarray]] = {}
    try:
        for annotation_name, values in conditions.items():
            cls = resolve_standard_annotation(annotation_name)
            if cls.n_body != 1:
                raise ValueError(
                    f"{annotation_name!r} resolves to {cls.name!r}, a {cls.n_body}-body condition -- "
                    "condition_spec has no selection-pair grammar, so only 1-body conditions (per-atom "
                    "or per-residue) can be set here. Use the Python API directly."
                )
            if cls is C_CRDGRP:
                groups = C_CRDGRP.annotation(atom_array, default="generate").copy()
                defaults = C_CRDGRP.default_annotation(atom_array)
                for value, entries in values.items():
                    for entry in entries:
                        mask = resolve_mask(atom_array, entry["select"])
                        groups[mask] = defaults[mask] if value is None or value == "null" else int(value)
                C_CRDGRP.set_annotation(atom_array, groups, on_internal_coord_mismatch="clear")
                continue
            for value, entries in values.items():
                # Entries sharing one `group:` label merge into a single group id, so they
                # are collected across this value's entries and applied together below.
                groups: dict[str | int, list[np.ndarray]] = {}
                for entry in entries:
                    select, group = entry["select"], entry.get("group")
                    indices = np.where(resolve_mask(atom_array, select))[0]
                    if _requires_group(cls, value):
                        if group is None:
                            raise ValueError(
                                f"{annotation_name}: value {value!r} requires a `group:` field alongside "
                                "`select:` -- see docs/how-to/input_prep/yaml-json.md."
                            )
                        groups.setdefault(group, []).append(indices)
                    else:
                        if group is not None:
                            raise ValueError(
                                f"{annotation_name}: `group:` is only meaningful for C_IDX (value false), "
                                f"not {value!r}."
                            )
                        _apply_condition_value(cls, atom_array, indices, value)
                if groups:
                    _apply_grouped_condition(
                        cls,
                        atom_array,
                        groups,
                        pending_index_groups,
                    )
        _finalize_index_groups(atom_array, pending_index_groups)
    finally:
        atom_array.set_annotation("res_id", real_res_id)


def _finalize(example_id: str, arr: AtomArrayPlus, *, strict: bool = True, validate: bool = False) -> None:
    """Log the "summarize changes" report and, if requested, run pre-save validation (§4.5)."""
    # Emitted through the logger rather than `print`: this runs per example, and during
    # online inference per replicate and per dataloader worker, so bare prints would flood
    # stdout with no way to filter or redirect them.
    summary = summarize_conditions(arr)
    report = f"\n--- condition_spec: {example_id} ---"
    if summary:
        report = f"{report}\n{summary}"
    logger.info(f"{report}\n{_REMINDER}")
    if not validate:
        return
    violations = [
        *run_pipeline_validation(arr),
        *check_index_group_res_id_gaps(arr),
        *check_index_group_molecule_consistency(arr),
    ]
    if violations:
        message = "\n".join(violations)
        if strict:
            raise ValueError(f"{example_id}: condition_spec validation failed:\n{message}")
        logger.warning(f"{example_id}: condition_spec validation warnings:\n{message}")


def _resolve_input_path(raw_path: str, spec_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (spec_dir / path).resolve()


def load_example_input(spec: dict, spec_dir: Path) -> AtomArrayPlus:
    """Load one example's ``input:`` structure(s), before ``contig`` subsetting (§4.2)."""
    input_paths = spec["input"]
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    return load_condition_spec_inputs(
        [_resolve_input_path(p, spec_dir) for p in input_paths],
        as_annotated_cif=spec.get("input_as_annotated_cif", False),
    )


def build_annotated_atom_array_from_condition_spec(
    spec: dict, *, spec_dir: Path, example_id: str = "example", strict: bool = True, validate: bool = False
) -> dict:
    """Build one example's fully assembled + annotated structure from its condition-spec dict.

    Runs the fixed order of operations (companion doc): load (§4.2), contig (§4.3),
    conditions (§4.4), finalize (§4.5).

    Args:
      spec: One example's ``{input, input_as_annotated_cif, contig, conditions}`` dict.
      spec_dir: Directory the spec file itself lives in (relative ``input`` paths
        resolve against this).
      example_id: Name used in the printed summary and validation error messages.
      strict: If ``True`` (default), a failed pre-save validation check raises
        instead of only logging a warning.
      validate: If ``True``, run the (slow) pre-save validation. Defaults to
        ``False`` -- online inference validates via the dataset transform instead;
        only the CIF-writer script and ``prevalidate_inputs=true`` opt in (see
        :py:meth:`~rfproteina.inference_utils.datasets.ConditionSpecDataset._prevalidate`).

    Returns:
      ``{atom_array, chain_info, ligand_info, metadata}``, the same shape
      :py:func:`~atomworks.io.parser.parse_atom_array` returns.
    """
    # Nothing downstream reads a field outside this vocabulary, so a typo'd `conditions:` block
    # (or a stale `global_args` key folded into every example) would otherwise build a structure
    # with none of the requested conditioning applied. Raise, matching how a typo *inside*
    # `conditions:` is handled -- see :py:func:`~rfproteina.utils.condition_resolution.resolve_standard_annotation`.
    unknown_fields = sorted(set(spec) - _CONDITION_SPEC_EXAMPLE_KEYS)
    if unknown_fields:
        raise ValueError(
            f"{example_id}: unrecognized condition_spec field(s) {unknown_fields}.\n"
            f"Recognized fields: {sorted(_CONDITION_SPEC_EXAMPLE_KEYS)}.\n"
            "These are never read, so a typo here silently applies nothing (e.g. a misspelled "
            "`conditions:` block) -- see docs/condition_spec_yaml.md."
        )

    atom_array = load_example_input(spec, spec_dir)

    contig = spec.get("contig")
    if contig:
        atom_array, original_res_id = apply_contig(atom_array, contig)
    else:
        original_res_id = atom_array.res_id.astype(np.int64).copy()

    conditions = spec.get("conditions")
    if conditions:
        _apply_conditions(atom_array, conditions, original_res_id)

    _finalize(example_id, atom_array, strict=strict, validate=validate)

    result = parse_atom_array(atom_array, config=_FINALIZE_PREPARE_CONFIG)
    built = result["assemblies"]["1"][0]
    return {
        "atom_array": built,
        "chain_info": result["chain_info"],
        "ligand_info": result["ligand_info"],
        "metadata": result["metadata"],
    }


def load_spec_file(path: Path) -> dict:
    """Load a condition-spec file (YAML or JSON, by extension) into a raw dict."""
    with open(path) as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def write_condition_spec_cif(
    example_id: str, spec: dict, out_path: Path, *, spec_dir: Path, validate: bool = True
) -> Path:
    """Build one example and write it to disk as an annotated CIF.

    Args:
      example_id: Example name (used in the printed summary).
      spec: One example's condition-spec dict.
      out_path: Destination CIF path.
      spec_dir: Directory the spec file lives in (for resolving relative ``input`` paths).
      validate: If ``True`` (default), run pre-save validation on the built structure.

    Returns:
      ``out_path``.
    """
    data = build_annotated_atom_array_from_condition_spec(
        spec, spec_dir=spec_dir, example_id=example_id, validate=validate
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(data["atom_array"], out_path, save_standard_annotations=True, chain_disambiguation="chain_iid")
    return out_path


def resolve_condition_spec_inputs(
    path: str | PathLike, out_dir: str | PathLike | None = None, *, validate: bool = True
) -> list[Path]:
    """Build every example in a condition-spec file and write each to disk as an annotated CIF.

    Args:
      path: Path to a condition_spec YAML/JSON file.
      out_dir: Directory to write the CIFs into. Defaults to ``<spec_stem>_cifs/`` beside the spec.
      validate: If ``True`` (default), run pre-save validation on each built structure.

    Returns:
      The list of written CIF paths, in file declaration order.

    Raises:
      ValueError: If ``path`` is a legacy contig-JSON file (unsupported by this writer).
    """
    spec_path = Path(path).resolve()
    spec = load_spec_file(spec_path)
    if is_legacy_dialect(spec):
        raise ValueError(
            f"{spec_path} uses the legacy contig-JSON dialect, which this writer does not support. "
            "Use a condition_spec file (docs/condition_spec_yaml.md)."
        )

    out_dir = Path(out_dir) if out_dir is not None else spec_path.parent / f"{spec_path.stem}_cifs"

    written_paths = []
    for example_id, example_spec in split_global_args(spec).items():
        out_path = out_dir / f"{validate_design_id(example_id)}.cif"
        written_paths.append(
            write_condition_spec_cif(example_id, example_spec, out_path, spec_dir=spec_path.parent, validate=validate)
        )

    return written_paths
