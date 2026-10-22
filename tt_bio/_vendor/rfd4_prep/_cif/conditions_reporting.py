# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Condition vocabulary, summaries, and validation for annotated design inputs.

Reporting derives from each condition's annotation metadata, with explicit
summaries for group-like conditions. The same validation checks apply to
annotated CIF inputs and structures assembled from YAML/JSON condition specs.
"""

from collections.abc import Callable

import numpy as np
from tt_bio._vendor.atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
)
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AtomArrayPlus
from tt_bio._vendor.atomworks.ml.conditions import (
    C_CRD,
    C_IDX,
    C_SEQ,
)
from tt_bio._vendor.atomworks.ml.conditions.base import ConditionBase
from biotite.structure import get_residue_starts

from .conditions import (
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_INTCRD,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITIONS,
)
from .datapipes_pipeline import build_design_validation_pipeline
from .transforms_invariants import find_index_group_res_id_gaps


def _index_group_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_IDXGRP.has_annotation(arr):
        return []
    groups = arr.get_annotation(C_IDXGRP.full_name)
    n_groups = len(np.unique(groups))
    lines = [f"Index groups: {n_groups} groups"]
    # Report implied res_id gaps within unindexed groups so the user can confirm
    # that non-contiguous regions were split as intended (rather than collapsed
    # into one group with fixed-size gaps).
    if C_IDX.has_annotation(arr):
        is_indexed = C_IDX.mask(arr, default="generate")
        res_id = arr.res_id
        for gid in np.unique(groups):
            in_group = groups == gid
            if is_indexed[in_group].any():
                continue  # indexed group — index offsets are externally defined
            gid_res = np.unique(res_id[in_group])
            if len(gid_res) > 1 and np.any(np.diff(np.sort(gid_res)) > 1):
                lines.append(
                    f"  unindexed group {int(gid)} contains non-sequential res_ids: "
                    "these gaps will be preserved in the output. If this is not desired, "
                    "please adjust the index groups accordingly."
                )
    return ["\n".join(lines)]


def _coord_group_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_CRDGRP.has_annotation(arr):
        return []
    n_groups = len(np.unique(arr.get_annotation(C_CRDGRP.full_name)))
    return [f"Coordinate groups: {n_groups} groups"]


def _chain_centroid_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_CENTROID.has_annotation(arr):
        return []
    mask = C_CENTROID.mask(arr, default="generate")
    if not mask.any():
        return []
    annot = C_CENTROID.annotation(arr, default="generate")
    lines = ["Chain centroid:"]
    for chain_id in np.unique(arr.chain_iid[mask]):
        xyz = annot[mask & (arr.chain_iid == chain_id)][0]
        lines.append(f"  {chain_id}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
    return ["\n".join(lines)]


def _index_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_IDX.has_annotation(arr):
        return []
    _, mask = C_IDX.annotation_and_mask_at_declared_level(arr)
    count = int((~mask).sum())
    return [f"Index: {count} {_unit_name(C_IDX, count)} unindexed"] if count else []


def _internal_coord_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_CRDGRP.has_annotation(arr):
        return []
    count = int(C_CRDGRP.is_internal(arr).sum())
    return [f"{C_INTCRD.__name__}: {count} {_unit_name(C_INTCRD, count)}"] if count else []


def _cath_summary(arr: AtomArrayPlus) -> list[str]:
    starts = get_residue_starts(arr)
    labels = C_CATH.annotation(arr, default="generate")[starts]
    mask = C_CATH.mask(arr, default="generate")[starts]
    labels = np.array([".".join(label.split(".")[:3]) for label in labels])
    lines = [f"  {label}: {int((mask & (labels == label)).sum())} residues" for label in np.unique(labels[mask])]
    return ["CAT (class / architecture / topology):\n" + "\n".join(lines)] if lines else []


def _sequence_probability_summary(arr: AtomArrayPlus, condition: type[ConditionBase]) -> list[str]:
    """Report each specified probability and its residue count, including explicit zeros."""
    values, mask = condition.annotation_and_mask_at_declared_level(arr)
    if not mask.any():
        return []
    values, counts = np.unique(values[mask], return_counts=True)
    value_counts = ", ".join(
        f"{float(value):g} ({int(count)} residues)" for value, count in zip(values, counts, strict=True)
    )
    return [f"Sequence probability {condition.token_name}: {value_counts}"]


SUPPORTED_CONDITIONS: tuple[type[ConditionBase], ...] = (
    C_SEQ,
    C_CRD,
    C_IDX,
    C_HOT,
    C_RAS,
    C_SEC,
    C_HB,
    C_IDXGRP,
    C_CRDGRP,
    C_INTCRD,
    C_CATH,
    C_PICATION,
    C_PISTACK,
    C_CENTROID,
    *SEQUENCE_PROBABILITY_CONDITIONS,
)

_CUSTOM_SUMMARIES: dict[type[ConditionBase], Callable[[AtomArrayPlus], list[str]]] = {
    C_CATH: _cath_summary,
    C_IDX: _index_summary,
    C_IDXGRP: _index_group_summary,
    C_CRDGRP: _coord_group_summary,
    C_INTCRD: _internal_coord_summary,
    C_CENTROID: _chain_centroid_summary,
}


def _unit_name(condition: type[ConditionBase], count: int) -> str:
    if condition.n_body == 2:
        return "pair" if count == 1 else "pairs"
    unit = condition.level.value
    return unit if count == 1 else f"{unit}s"


def _generic_summary(condition: type[ConditionBase], arr: AtomArrayPlus) -> list[str]:
    if not condition.has_annotation(arr):
        return []
    values, mask = condition.annotation_and_mask_at_declared_level(arr)
    if not mask.any():
        return []

    label = condition.__name__
    if condition.enum is None:
        count = int(mask.sum())
        return [f"{label}: {count} {_unit_name(condition, count)}"]

    enum_cls = condition.enum
    assert enum_cls is not None
    default_value = condition.default_value
    section = []
    for member in enum_cls:
        member_mask = mask & (values == member.value)
        count = int(member_mask.sum())
        if member.value != default_value and count:
            section.append(f"  {member.name}: {count} {_unit_name(condition, count)}")
    return [f"{label}:\n" + "\n".join(section)] if section else []


def summarize_conditions(arr: AtomArrayPlus) -> str:
    """Summarize every set condition covered by :py:data:`SUPPORTED_CONDITIONS`."""
    lines: list[str] = []
    for condition in SUPPORTED_CONDITIONS:
        if condition in SEQUENCE_PROBABILITY_CONDITIONS:
            lines.extend(_sequence_probability_summary(arr, condition))
        elif custom_summary := _CUSTOM_SUMMARIES.get(condition):
            lines.extend(custom_summary(arr))
        else:
            lines.extend(_generic_summary(condition, arr))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-save pipeline validation
# ---------------------------------------------------------------------------


def run_pipeline_validation(arr: AtomArrayPlus) -> list[str]:
    """Dry-run the design validation pipeline against ``arr``, without raising.

    Expandable segments are realized on a copy with a fixed local seed before
    featurization. The caller's placeholders and length ranges are preserved;
    validation checks one realization, not every possible combination of lengths.

    Returns:
      Empty list if valid, otherwise one message per violation.
    """
    pipeline = build_design_validation_pipeline(expandable_segment_rng=np.random.default_rng(0))
    try:
        pipeline({"atom_array": arr.copy()})
        return []
    except ValueError as e:
        msg = str(e)
        if "Batch invariant violation" in msg:
            return [line.strip() for line in msg.splitlines() if line.strip().startswith("[")]
        return [f"[pipeline_error] {msg}"]
    except Exception as e:
        return [f"[pipeline_error] {e}"]


def check_index_group_res_id_gaps(arr: AtomArrayPlus) -> list[str]:
    """Return a violation string for each internal res_id gap within an index group.

    Unlike a per-chain check, discontinuities are only flagged *within the same
    index group* (:py:data:`~rfproteina.conditions.C_IDXGRP`). Gaps that fall on
    an index-group boundary are expected (e.g. unindexed insertions start a new
    group) and are not reported. Intra-group gaps are interpreted by the model as
    breaks within a contiguously-indexed region, so they are usually unintentional.

    Atomized residues (PTMs, ligands, etc.) are skipped: they can share an index
    group with a peptide chain while carrying unrelated res_ids, since atomized
    segments use bond-graph distance -- not res_id -- for relative positional
    encoding.
    """
    messages = find_index_group_res_id_gaps(arr)
    if not messages:
        return []
    return [
        f"[res_id_gap] Residue numbering has {len(messages)} internal gap(s) within index "
        f"groups: {'; '.join(messages)}. Intra-group gaps are interpreted by the model as "
        "breaks within a contiguously-indexed region -- make sure this is intentional."
    ]


def _molecule_iids(arr: AtomArrayPlus) -> np.ndarray:
    """Return the per-atom ``molecule_iid`` annotation, deriving it if absent.

    Derivation happens on a copy so the caller's array is never mutated. Mirrors
    the molecule bookkeeping in
    :py:meth:`~rfproteina.conditions.IndexGroup.default_annotation`.
    """
    if "molecule_iid" in arr.get_annotation_categories():
        return arr.molecule_iid
    arr = arr.copy()
    ensure_annotations(arr, "is_polymer")
    arr = add_id_and_entity_annotations(arr)
    if "transformation_id" not in arr.get_annotation_categories():
        arr.set_annotation("transformation_id", np.full(arr.array_length(), "1", dtype=str))
    arr = add_molecule_iid_annotation(arr)
    return arr.molecule_iid


def check_index_group_molecule_consistency(arr: AtomArrayPlus) -> list[str]:
    """Return a violation for each molecule whose indexed atoms span multiple index groups.

    Every globally-indexed atom (``C_IDX`` mask true) belonging to the same
    molecule (``molecule_iid``) must share a single index group
    (:py:data:`~rfproteina.conditions.C_IDXGRP`). The model reasons over
    molecules, so a PTM/ligand bonded to a peptide chain is part of that chain's
    molecule and must live in its index group. A molecule whose indexed atoms are
    split across groups means part of a single contiguously-indexed entity is
    being treated as a separate region, which is almost always an annotation
    mistake.
    """
    idx_mask = C_IDX.mask(arr, default="generate")
    if not idx_mask.any():
        return []

    molecule_iids = _molecule_iids(arr)
    index_groups = C_IDXGRP.annotation(arr, default="generate")

    messages: list[str] = []
    for mol_id in np.unique(molecule_iids[idx_mask]):
        groups = np.unique(index_groups[idx_mask & (molecule_iids == mol_id)])
        if groups.size > 1:
            group_list = ", ".join(str(int(g)) for g in groups)
            messages.append(f"molecule {mol_id} spans index groups {{{group_list}}}")

    if not messages:
        return []
    return [
        f"[index_group_molecule] {len(messages)} molecule(s) have indexed atoms split across "
        f"multiple index groups: {'; '.join(messages)}. All indexed atoms of a molecule must "
        "share one index group (a PTM/ligand belongs to its chain's molecule and index group) "
        "-- make sure this is intentional."
    ]


# ---------------------------------------------------------------------------
# Index group synchronization
# ---------------------------------------------------------------------------


def sync_index_groups(arr: AtomArrayPlus, preserve_manual_edits: bool = True) -> None:
    """Recompute :py:data:`~rfproteina.conditions.C_IDXGRP` in place on ``arr``.

    Enforces mandatory split points (molecule boundaries and ``C_IDX``
    transitions) while optionally preserving any manual group subdivisions
    already present.

    Args:
      arr: Structure to update in place.
      preserve_manual_edits: If ``True`` (default), two atoms keep sharing a
        group iff they already share both the default group and the current
        group. If ``False``, groups are fully reset to the default split.
    """
    default_groups = C_IDXGRP.default_annotation(arr)
    if preserve_manual_edits and C_IDXGRP.has_annotation(arr):
        current_groups = arr.get_annotation(C_IDXGRP.full_name)
        combined = np.stack([default_groups, current_groups], axis=1)
        _, new_groups = np.unique(combined, axis=0, return_inverse=True)
    else:
        new_groups = default_groups
    C_IDXGRP.set_annotation(arr, new_groups)
