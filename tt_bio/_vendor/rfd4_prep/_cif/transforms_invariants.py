# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Batch-level invariant checking for the rfproteina pipeline."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from tt_bio._vendor.atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_POLYMER_RESIDUES, STANDARD_RNA
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.scatter import apply_segment_wise
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts, get_segments
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_ATM, S_SEGMIN, STANDARD_ANNOTATIONS
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from tt_bio._vendor.atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.transforms.filters import polymer_terminal_oxygen_mask
from tt_bio._vendor.atomworks.ml.utils.token import apply_and_spread_token_wise, get_token_starts
from biotite.structure import AtomArray, spread_residue_wise

from .conditions import C_CENTROID, C_CRDGRP, C_IDXGRP, C_INTCRD, SEQUENCE_PROBABILITY_CONDITIONS
from .datapipes_bonds import (
    get_bond_distances,
    get_polymerization_atom_masks,
    get_polymerization_atom_names,
)
from .datapipes_schema_proteina import TokenType
from .encoding import UNIFIED_ATOM37_ENCODING
from .utils_constants import LOCAL_LATENT_RESIDUES
from .utils_token_encoding import build_atom37_slots

logger = logging.getLogger(__name__)

_MAX_REASONABLE_BOND_LENGTH_ANGSTROM = 4.0
_MAX_REASONABLE_POLYMER_BOND_LENGTH_ANGSTROM = 2.0


@dataclass
class InvariantViolation:
    """A single invariant violation detected in a pipeline batch."""

    check: str
    message: str
    count: int = 1


def detect_batch_invariant_violations(data: dict, is_inference: bool) -> list[InvariantViolation]:
    """Detect invariant violations in a pipeline output batch.

    Args:
        data: Pipeline output dict containing at least ``"atom_array"`` and ``"feats"``.
        is_inference: Whether the pipeline is in inference mode.

    Returns:
        List of :py:class:`InvariantViolation` instances; empty if all checks pass.
    """
    violations: list[InvariantViolation] = []
    aa = data["atom_array"]
    feats = data["feats"]

    # --- Structural / atom-array checks ---
    violations += check_molecules_on_separate_chains(aa)
    violations += check_standard_annotation_level_violations(aa)
    violations += check_global_token_id_consistent(aa)
    violations += check_centroid_conditioning_all_or_nothing_violations(aa, is_inference=is_inference)
    violations += check_conditioning_annotation_matches_ground_truth(aa, is_inference=is_inference)
    violations += check_conditioning_coords_not_nan(aa)
    violations += check_expandable_segment_atom37_conditioning_violations(aa)
    violations += check_bond_length_violations(aa)
    violations += check_unindexed_coord_conditioning_violations(aa)

    # --- Feats / token-level checks ---
    violations += check_conditioning_violations(aa, feats)
    violations += check_atomized_indexed_violations(feats)
    violations += check_index_group_molecule_connectivity_violations(aa, feats, is_inference=is_inference)
    violations += check_coordinates_not_nan(feats, is_inference=is_inference)
    violations += check_element_encoding_violations(feats)
    violations += check_sequence_conditioning_violations(feats, is_inference)
    violations += check_sequence_probability_conditioning_violations(aa)

    # --- Cross-cutting checks (require both aa and feats) ---
    violations += check_coordinate_centering_violations(aa, feats)
    violations += check_supports_local_latent_violations(aa, feats)
    violations += check_token_features_consistent_violations(aa)
    violations += check_no_oxt_on_local_latent_tokens(aa, feats)

    return violations


# ---------------------------------------------------------------------------
# Atom-array checks
# ---------------------------------------------------------------------------


def check_expandable_segment_atom37_conditioning_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return atom37 conditioning that conversion would silently drop from expandable tokens."""
    from .datapipes_pipeline import ATOM37_FEATURES

    categories = set(aa.get_annotation_categories())
    expandable = S_SEGMIN.mask(aa, default="generate")
    if not expandable.any():
        return []

    # Element and charge only reach the flow model for atomized tokens. Raw coords are not polymer
    # conditioning (C_CRD/C_INTCRD are); they are clean targets only for training/partial diffusion.
    not_polymer_flow_conditioning = {"coords", "element", "charge"}
    coordinate_condition_payloads = {C_CRD.full_name, C_INTCRD.full_name}
    nondefault_conditions = []
    affected_atoms = np.zeros(aa.array_length(), dtype=bool)
    for annotation_name, _, default in ATOM37_FEATURES:
        if annotation_name not in categories or annotation_name in not_polymer_flow_conditioning:
            continue
        annotation = aa.get_annotation(annotation_name)
        if annotation_name in coordinate_condition_payloads:
            # Coordinate conditions use NaN as their inactive AtomArray value, whereas the
            # corresponding atom37 tensors use zero padding after conversion.
            nondefault = ~np.isnan(annotation).all(axis=tuple(range(1, annotation.ndim)))
        else:
            nondefault = annotation != default
        affected = expandable & nondefault
        if affected.any():
            nondefault_conditions.append(annotation_name)
            affected_atoms |= affected

    if not affected_atoms.any():
        return []
    count = int(np.unique(aa.token_id[affected_atoms]).size)
    return [
        InvariantViolation(
            check="expandable_segment_atom37_conditioning",
            message=(
                f"{count} expandable-segment token(s) have atom37 conditioning that would be discarded: "
                f"{', '.join(nondefault_conditions)}. Remove the conditioning or provide ordinary polymer residues."
            ),
            count=count,
        )
    ]


def check_bond_length_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Check input bonds in both modes: recognized polymerization bonds ≤2 Å, others ≤4 Å.

    Check existing bonds with resolved, non-placeholder endpoints, using the bond featurizer's
    polymer-atom lookup in either endpoint order.
    """
    if aa.bonds is None:
        return []
    bonds = aa.bonds.as_array()
    if len(bonds) == 0:
        return []

    atom1, atom2 = bonds[:, :2].T
    distances = get_bond_distances(aa)
    resolved = np.isfinite(aa.coord).all(axis=-1) & ~S_SEGMIN.mask(aa, default="generate")
    if "occupancy" in aa.get_annotation_categories():
        resolved &= aa.occupancy > 0
    measured = resolved[atom1] & resolved[atom2]

    chain = aa.chain_iid if "chain_iid" in aa.get_annotation_categories() else aa.chain_id
    inter_residue = (
        (chain[atom1] != chain[atom2])
        | (aa.res_id[atom1] != aa.res_id[atom2])
        | (aa.ins_code[atom1] != aa.ins_code[atom2])
    )
    outgoing = np.zeros(len(aa), dtype=bool)
    incoming = np.zeros(len(aa), dtype=bool)
    token_type = (
        aa.token_type if "token_type" in aa.get_annotation_categories() else np.full(len(aa), TokenType.UNKNOWN)
    )
    endpoints = bonds[measured & inter_residue, :2].ravel()
    for res_name, res_type in set(zip(aa.res_name[endpoints], token_type[endpoints], strict=True)):
        exit_atom, entry_atom = get_polymerization_atom_names(str(res_name), int(res_type))
        residue = (aa.res_name == res_name) & (token_type == res_type)
        if exit_atom:
            outgoing |= residue & (aa.atom_name == exit_atom)
        if entry_atom:
            incoming |= residue & (aa.atom_name == entry_atom)
    polymer = inter_residue & ((outgoing[atom1] & incoming[atom2]) | (incoming[atom1] & outgoing[atom2]))
    limits = np.where(polymer, _MAX_REASONABLE_POLYMER_BOND_LENGTH_ANGSTROM, _MAX_REASONABLE_BOND_LENGTH_ANGSTROM)
    too_long = measured & (distances > limits)
    if not too_long.any():
        return []

    count = int(too_long.sum())
    details = []
    for bond_idx in np.flatnonzero(too_long)[:5]:
        endpoints = [
            f"{chain[i]}:{aa.res_name[i]}{aa.res_id[i]}{aa.ins_code[i]}:{aa.atom_name[i]}"
            for i in (atom1[bond_idx], atom2[bond_idx])
        ]
        details.append(f"{' -- '.join(endpoints)} = {distances[bond_idx]:.2f} Å (limit {limits[bond_idx]:.1f} Å)")
    return [
        InvariantViolation(
            check="bond_length",
            message=(
                f"{count} bond(s) exceed the polymerization limit of "
                f"{_MAX_REASONABLE_POLYMER_BOND_LENGTH_ANGSTROM:.1f} Å "
                f"or general limit of {_MAX_REASONABLE_BOND_LENGTH_ANGSTROM:.1f} Å; "
                f"maximum length is {distances[too_long].max():.2f} Å. "
                f"First {len(details)}: {'; '.join(details)}. "
                "Check source coordinates and residue numbering; "
                "preserve missing-residue gaps instead of bonding across them."
            ),
            count=count,
        )
    ]


def check_conditioning_annotation_matches_ground_truth(aa: AtomArray, is_inference: bool) -> list[InvariantViolation]:
    """Training-only: C_CRD/C_SEQ annotations equal the ground-truth array at masked positions."""
    if is_inference:
        return []

    violations: list[InvariantViolation] = []

    # Check that C_CRD.annotation matches the ground-truth coordinates at every atom with a finite C_CRD.mask
    crd_mask = C_CRD.mask(aa, default="generate")
    if crd_mask.any():
        crd_annot = C_CRD.annotation(aa, default="generate")
        # Compare against coord_to_be_noised: both C_CRD.annotation and it are re-centered together.
        cats = aa.get_annotation_categories()
        gt_coord = aa.get_annotation("coord_to_be_noised") if "coord_to_be_noised" in cats else aa.coord
        matches = np.isclose(crd_annot[crd_mask], gt_coord[crd_mask], equal_nan=True).all(axis=-1)
        if not matches.all():
            violations.append(
                InvariantViolation(
                    check="crd_annotation_matches_ground_truth",
                    message="C_CRD annotation does not equal the conditioning coords at conditioned positions",
                    count=int((~matches).sum()),
                )
            )

    # Check that C_SEQ.annotation matches the ground-truth residue identity at every residue with a finite C_SEQ.mask
    seq_mask = C_SEQ.mask(aa, default="generate")
    if seq_mask.any():
        seq_annot = C_SEQ.annotation(aa, default="generate")
        mismatched = seq_annot[seq_mask] != aa.res_name[seq_mask]
        if mismatched.any():
            violations.append(
                InvariantViolation(
                    check="seq_annotation_matches_ground_truth",
                    message="C_SEQ annotation does not equal atom_array.res_name at sequence-conditioned positions",
                    count=int(mismatched.sum()),
                )
            )

    return violations


def check_conditioning_coords_not_nan(aa: AtomArray) -> list[InvariantViolation]:
    """Coordinate conditioning must never target a non-finite coordinate.

    A NaN coordinate under absolute ``C_CRD`` or a non-reserved ``C_CRDGRP`` means
    we are conditioning on an unresolved atom; it must be stripped upstream, not fed to the model.
    """
    violations: list[InvariantViolation] = []
    internal_groups = C_CRDGRP.is_internal(aa)
    internal_coords = C_INTCRD.annotation(aa, default="generate")
    orphan_payload = np.isfinite(internal_coords).any(axis=-1) & ~internal_groups
    if orphan_payload.any():
        violations.append(
            InvariantViolation(
                check="internal_coords_require_coord_group",
                message=(
                    f"{int(orphan_payload.sum())} atom(s) have a C_INTCRD payload but no internal C_CRDGRP; "
                    "coordinate groups are the source of truth for internal-coordinate conditioning"
                ),
                count=int(orphan_payload.sum()),
            )
        )
    coordinate_conditions = (
        (C_CRD.mask(aa, default="generate"), C_CRD.annotation(aa, default="generate"), "C_CRD"),
        (internal_groups, internal_coords, "C_INTCRD"),
    )
    for mask, annotation, label in coordinate_conditions:
        if not mask.any():
            continue
        bad = mask & ~np.isfinite(annotation).all(axis=-1)
        n_bad = int(bad.sum())
        if n_bad:
            violations.append(
                InvariantViolation(
                    check="conditioning_coord_not_finite",
                    message=f"{n_bad} {label}-conditioned atom(s) have non-finite conditioning coordinates",
                    count=n_bad,
                )
            )
    return violations


def check_molecules_on_separate_chains(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where a chain contains multiple molecules"""
    categories = set(aa.get_annotation_categories())
    if "chain_iid" in categories:
        chain_identifier = aa.chain_iid
        molecule_identifier = aa.molecule_iid if "molecule_iid" in categories else aa.molecule_id
    else:
        chain_identifier = aa.chain_id
        molecule_identifier = aa.molecule_id

    violations = []
    for chain_id in np.unique(chain_identifier):
        molecule_identifiers = molecule_identifier[chain_identifier == chain_id]
        n_molecules = len(np.unique(molecule_identifiers))
        if n_molecules > 1:
            violations.append(
                InvariantViolation(
                    check="multiple_molecules_on_chain",
                    message=(f"Chain {chain_id} contains {n_molecules} different molecules"),
                    count=1,
                )
            )
    return violations


def check_centroid_conditioning_all_or_nothing_violations(
    aa: AtomArray, is_inference: bool
) -> list[InvariantViolation]:
    """Return violations where centroid conditioning covers some but not all of an example.

    Centroid conditioning (``C_CENTROID``) is a per-chain value, and must be all-or-nothing
    per example: either every chain has a finite centroid on every one of its atoms, or none
    do, matching :py:class:`~rfproteina.datapipes.pipeline.FeaturizeChainCentroid`. Two ways
    to break that are reported separately: conditioning only *some* chains, and conditioning
    only part of a chain (which leaves the rest of it with a NaN centroid the model reads as
    "not conditioned").

    At train time, a chain with no resolved coordinates (fully unresolved) is exempt from this,
    as its ground-truth centroid would be undefined.
    """
    categories = set(aa.get_annotation_categories())
    if C_CENTROID.full_name not in categories:
        return []

    chain_identifier = aa.chain_iid if "chain_iid" in categories else aa.chain_id
    centroid_mask = C_CENTROID.mask(aa, default="generate")

    aap = as_atom_array_plus(aa)
    coords = aap.coord_to_be_noised if "coord_to_be_noised" in aap.get_annotation_categories() else None

    n_chains = 0
    n_conditioned = 0
    partially_conditioned: list[str] = []
    for chain_id in np.unique(chain_identifier):
        chain_mask = chain_identifier == chain_id
        if not is_inference and coords is not None and np.isnan(coords[chain_mask]).all():
            continue
        n_chains += 1
        if centroid_mask[chain_mask].any():
            n_conditioned += 1
            if not centroid_mask[chain_mask].all():
                partially_conditioned.append(str(chain_id))

    violations = []
    if partially_conditioned:
        violations.append(
            InvariantViolation(
                check="centroid_conditioning_partial_chain",
                message=(
                    f"chain(s) {partially_conditioned} have a centroid on only some of their atoms; "
                    "the centroid is a per-chain value and must cover every atom of the chain"
                ),
                count=len(partially_conditioned),
            )
        )
    if n_chains and 0 < n_conditioned < n_chains:
        violations.append(
            InvariantViolation(
                check="centroid_conditioning_not_all_or_nothing",
                message=(
                    f"{n_conditioned}/{n_chains} chain(s) have centroid conditioning; "
                    "must be applied to all chains or none"
                ),
                count=n_chains,
            )
        )
    return violations


def _segment_has_one_value(segment: np.ndarray) -> bool:
    """Whether every entry of one segment of a StandardAnnotation carries the same value."""
    if segment.ndim == 1:
        return np.unique(segment).size <= 1
    first = segment[:1]
    equal = segment == first
    if np.issubdtype(segment.dtype, np.floating):
        equal |= np.isnan(segment) & np.isnan(first)
    return bool(equal.all())


def check_standard_annotation_level_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where a StandardAnnotation has inconsistent values within its segment.

    For each StandardAnnotation present on the atom array, values must be uniform
    within every segment defined by the annotation's declared
    :py:class:`~atomworks.io.utils.standard_annotations.base.Level`.

    ``ATOM``-level and multi-body (``n_body > 1``) annotations are skipped.
    """
    categories = set(aa.get_annotation_categories())
    violations = []

    for sa in STANDARD_ANNOTATIONS:
        if sa.n_body != 1 or sa.level == Level.ATOM or sa.full_name not in categories:
            continue

        values = aa.get_annotation(sa.full_name)

        if sa.level == Level.RESIDUE:
            starts = get_residue_starts(aa, add_exclusive_stop=True)
        elif sa.level == Level.TOKEN:
            starts = get_token_starts(aa, add_exclusive_stop=True)
        elif sa.level == Level.CHAIN:
            chain_identifier = aa.chain_iid if "chain_iid" in categories else aa.chain_id
            starts = get_segments(chain_identifier, add_exclusive_stop=True)
        elif sa.level == Level.MOLECULE:
            molecule_identifier = aa.molecule_id if "molecule_id" in categories else aa.chain_id
            starts = get_segments(molecule_identifier, add_exclusive_stop=True)
        else:
            continue

        uniform_per_seg = np.asarray(apply_segment_wise(starts, values, _segment_has_one_value), dtype=bool)
        n_bad = int((~uniform_per_seg).sum())
        if n_bad:
            violations.append(
                InvariantViolation(
                    check="standard_annotation_level_violated",
                    message=(
                        f"'{sa.full_name}' (level={sa.level.value}) has inconsistent values in {n_bad} segment(s)"
                    ),
                    count=n_bad,
                )
            )

    return violations


def check_global_token_id_consistent(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where the token_id annotation does not match the annotation implied by the AtomArray."""
    old_token_id = aa.get_annotation("token_id").copy()
    aa_copy = aa.copy()
    aa_copy.del_annotation("token_id")
    aa_copy = add_global_token_id_annotation(aa_copy)
    difference_mask = aa_copy.token_id != old_token_id
    n_bad = int(difference_mask.sum())
    if n_bad:
        return [
            InvariantViolation(
                check="global_token_id_inconsistent",
                message=f"'token_id' annotation is inconsistent with recomputed values in {n_bad} atom(s)",
                count=n_bad,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Feats / token-level checks
# ---------------------------------------------------------------------------


def check_conditioning_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where atomized/coord-conditioned tokens lack sequence conditioning."""
    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]
    seq_mask = feats[C_SEQ.full_name] != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    coord_groups = feats[C_CRDGRP.full_name]
    has_internal_coords = (coord_groups != C_CRDGRP.UNKNOWN_COORDS) & (coord_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
    crd_mask = torch.isfinite(feats[C_CRD.full_name]).any(dim=-1) | has_internal_coords

    violations = []

    is_atomized = token_type == TokenType.ATOMIZED
    atomized_not_seq = is_atomized & ~seq_mask
    if atomized_not_seq.any():
        count = int(atomized_not_seq.sum())
        violations.append(
            InvariantViolation(
                check="atomized_must_be_seq_conditioned",
                message=f"{count} atomized token(s) lack sequence conditioning",
                count=count,
            )
        )

    # Build [n_tokens, 37] backbone mask by scattering lazy-annotator results to atom37.
    # Tokens with coord conditioning only on backbone atoms don't need seq conditioning.
    aap = as_atom_array_plus(aa)
    ensure_annotations(aap, "is_protein_backbone", "is_rna_backbone", "is_dna_backbone")
    is_backbone = aap.is_protein_backbone | aap.is_rna_backbone | aap.is_dna_backbone  # [n_atoms]

    atom_slots = build_atom37_slots(atom_array=aap)

    n_tokens = int(atom37_mask.shape[0])
    bb_atom37 = np.zeros((n_tokens, 37), dtype=bool)
    valid = atom_slots >= 0
    bb_atom37[aap.token_id[valid], atom_slots[valid]] = is_backbone[valid]
    bb_mask = torch.from_numpy(bb_atom37).to(crd_mask.device)

    has_non_bb_crd = (crd_mask & atom37_mask & ~bb_mask).any(dim=1)  # [n_tokens]
    non_bb_crd_not_seq = has_non_bb_crd & ~seq_mask
    if non_bb_crd_not_seq.any():
        count = int(non_bb_crd_not_seq.sum())
        violations.append(
            InvariantViolation(
                check="coord_conditioned_must_be_seq_conditioned",
                message=f"{count} token(s) with non-backbone coordinate conditioning lack sequence conditioning",
                count=count,
            )
        )

    return violations


def check_atomized_indexed_violations(feats: dict) -> list[InvariantViolation]:
    """Return violations where atomized tokens are unindexed."""
    # Lazy import to avoid circular dependency with pipeline.py
    from .datapipes_pipeline import TokenType

    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]
    idx_mask_raw: torch.Tensor | None = feats.get(C_IDX.full_name)  # [n_tokens, 37]

    if idx_mask_raw is None:
        return []

    is_atomized = token_type == TokenType.ATOMIZED
    if not is_atomized.any():
        return []

    is_indexed = idx_mask_raw.any(dim=1)
    atomized_unindexed = is_atomized & ~is_indexed
    if atomized_unindexed.any():
        count = int(atomized_unindexed.sum())
        return [
            InvariantViolation(
                check="atomized_must_be_indexed",
                message=f"{count} atomized token(s) are unindexed",
                count=count,
            )
        ]
    return []


def check_coordinates_not_nan(feats: dict, is_inference: bool = False) -> list[InvariantViolation]:
    """Return violations where coordinates feeding the model contain NaN/inf, or padding is non-zero.

    At training time, every valid atom slot (per ``atom37_mask``) must have finite coordinates.
    At inference time, the binder chain has not been generated yet so its ``coords`` may be NaN
    legitimately. Instead, only the atoms that are explicitly coordinate-conditioned —
    finite values in ``condition_coordinate_1_atom`` — OR atomized (because
    chirality features need to be calculated from coordinate information) are checked, since
    those are the only coordinates that actually reach the model.
    The padding non-zero check is applied in both modes.
    """
    # lazy import to avoid circular dependency with pipeline.py
    from .datapipes_pipeline import TokenType

    if "coords" not in feats or "atom37_mask" not in feats:
        return []

    coords: torch.Tensor = feats["coords"]  # [n_tokens, 37, 3]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]

    atomized_mask = feats["token_type"] == TokenType.ATOMIZED  # [n_tokens,]
    atomized_mask_atom37 = torch.zeros_like(atom37_mask, dtype=torch.bool)
    ATOMIZED_SLOT_IDX = next(i for i, a in enumerate(UNIFIED_ATOM37_ENCODING.token_atoms["<A>"]) if a.strip())
    atomized_mask_atom37[:, ATOMIZED_SLOT_IDX] = atomized_mask

    violations = []

    if is_inference:
        conditioned_coords = feats.get(C_CRD.full_name)
        coordinate_mask = (
            torch.isfinite(conditioned_coords).any(dim=-1)
            if conditioned_coords is not None
            else torch.zeros_like(atom37_mask)
        )
        mask_to_check = (coordinate_mask | atomized_mask_atom37) & atom37_mask
    else:
        mask_to_check = atom37_mask

    coords_to_check = coords[mask_to_check]  # [n_valid_atoms, 3]
    if torch.isnan(coords_to_check).any():
        violations.append(
            InvariantViolation(
                check="nan_in_valid_coords",
                message="NaN found in atom coordinates",
                count=1,
            )
        )
    if torch.isinf(coords_to_check).any():
        violations.append(
            InvariantViolation(
                check="inf_in_valid_coords",
                message="Inf found in atom coordinates",
                count=1,
            )
        )

    invalid_mask = ~atom37_mask
    if invalid_mask.any():
        n_nonzero = int((coords[invalid_mask] != 0.0).any(dim=-1).sum())
        if n_nonzero:
            violations.append(
                InvariantViolation(
                    check="nonzero_padding_coords",
                    message=f"{n_nonzero} invalid atom slot(s) have non-zero coordinates",
                    count=n_nonzero,
                )
            )

    return violations


def check_element_encoding_violations(feats: dict) -> list[InvariantViolation]:
    """Return violations where element encoding leaks info to non-seq-conditioned tokens."""
    if "element" not in feats or "atom37_mask" not in feats or C_SEQ.full_name not in feats:
        return []

    element: torch.Tensor = feats["element"]  # [n_tokens, 37]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    seq_mask = feats[C_SEQ.full_name] != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]

    violations = []
    non_seq = ~seq_mask

    if non_seq.any():
        n_leaked = int((element[non_seq] != -1).sum())
        if n_leaked:
            violations.append(
                InvariantViolation(
                    check="element_leakage",
                    message=f"{n_leaked} atom slot(s) in non-seq-conditioned tokens expose element info",
                    count=n_leaked,
                )
            )

    if seq_mask.any():
        valid_in_seq = atom37_mask[seq_mask]
        element_in_seq = element[seq_mask]
        if valid_in_seq.any():
            n_missing = int((element_in_seq[valid_in_seq] < 0).sum())
            if n_missing:
                violations.append(
                    InvariantViolation(
                        check="missing_element_in_seq_conditioned",
                        message=f"{n_missing} valid atom(s) in seq-conditioned tokens lack element info",
                        count=n_missing,
                    )
                )

    return violations


def check_sequence_probability_conditioning_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Validate sparse canonical-polymer probability conditions residue by residue.

    A unified sequence-token probability is present wherever its value is not NaN. Present
    values must be finite and in [0, 1], positive mass may only target identities
    of the residue's polymer type, and the specified mass may not exceed one;
    unspecified identities retain the remaining probability mass. A
    ``C_SEQ``-conditioned residue must carry the complete exact distribution;
    identities outside the canonical vocabulary have zero mass in these channels.
    """

    aap = as_atom_array_plus(aa)
    residue_starts = get_residue_starts(aap)
    n_residues = len(residue_starts)
    probabilities = np.zeros((n_residues, len(SEQUENCE_PROBABILITY_CONDITIONS)), dtype=np.float32)
    specified = np.zeros_like(probabilities, dtype=bool)
    violations: list[InvariantViolation] = []

    for channel, condition in enumerate(SEQUENCE_PROBABILITY_CONDITIONS):
        if not condition.has_annotation(aap):
            continue

        atom_probabilities = condition.annotation(aap, default="raise")
        atom_mask = condition.mask(aap, default="raise")
        expected_mask = spread_residue_wise(aap, atom_mask[residue_starts])
        n_nonuniform_masks = int(
            np.add.reduceat((atom_mask != expected_mask).astype(np.int64), residue_starts).astype(bool).sum()
        )
        if n_nonuniform_masks:
            violations.append(
                InvariantViolation(
                    check="sequence_probability_nonuniform_residue_mask",
                    message=f"{condition.name} has a nonuniform mask in {n_nonuniform_masks} residue(s)",
                    count=n_nonuniform_masks,
                )
            )

        expected_values = spread_residue_wise(aap, atom_probabilities[residue_starts])
        inconsistent = atom_mask & ~np.isclose(atom_probabilities, expected_values, equal_nan=True)
        n_nonuniform_values = int(np.maximum.reduceat(inconsistent, residue_starts).sum())
        if n_nonuniform_values:
            violations.append(
                InvariantViolation(
                    check="sequence_probability_nonuniform_residue_value",
                    message=f"{condition.name} has nonuniform values in {n_nonuniform_values} residue(s)",
                    count=n_nonuniform_values,
                )
            )

        probabilities[:, channel] = atom_probabilities[residue_starts]
        specified[:, channel] = atom_mask[residue_starts]

    seq_conditioned = C_SEQ.mask(aap, default="generate")[residue_starts]
    if not specified.any() and not seq_conditioned.any():
        return violations

    finite = np.isfinite(probabilities)
    n_nonfinite = int((specified & ~finite).sum())
    if n_nonfinite:
        violations.append(
            InvariantViolation(
                check="sequence_probability_nonfinite",
                message=f"{n_nonfinite} specified sequence probability value(s) are not finite",
                count=n_nonfinite,
            )
        )

    out_of_range = specified & finite & ((probabilities < 0.0) | (probabilities > 1.0))
    n_out_of_range = int(out_of_range.sum())
    if n_out_of_range:
        violations.append(
            InvariantViolation(
                check="sequence_probability_out_of_range",
                message=f"{n_out_of_range} specified sequence probability value(s) fall outside [0, 1]",
                count=n_out_of_range,
            )
        )

    ensure_annotations(
        aap,
        "is_standard_unknown_or_masked_aa",
        "is_standard_unknown_or_masked_rna",
        "is_standard_unknown_or_masked_dna",
    )
    compatibility_by_token = {
        token: residue_mask
        for tokens, residue_mask in (
            (STANDARD_AA, aap.is_standard_unknown_or_masked_aa[residue_starts]),
            (STANDARD_RNA, aap.is_standard_unknown_or_masked_rna[residue_starts]),
            (STANDARD_DNA, aap.is_standard_unknown_or_masked_dna[residue_starts]),
        )
        for token in tokens
    }
    compatible = np.column_stack(
        [compatibility_by_token[condition.token_name] for condition in SEQUENCE_PROBABILITY_CONDITIONS]
    )
    wrong_token_type = specified & finite & (probabilities > 0.0) & ~compatible
    n_wrong_token_type = int(wrong_token_type.sum())
    if n_wrong_token_type:
        violations.append(
            InvariantViolation(
                check="sequence_probability_wrong_token_type",
                message=(
                    f"{n_wrong_token_type} positive sequence probability value(s) target identities "
                    "from a different polymer type"
                ),
                count=n_wrong_token_type,
            )
        )

    specified_mass = np.where(specified & finite, probabilities, 0.0).sum(axis=-1)
    exceeds_one = specified_mass > 1.0 + 1e-6
    n_exceeds_one = int(exceeds_one.sum())
    if n_exceeds_one:
        violations.append(
            InvariantViolation(
                check="sequence_probability_sum_exceeds_one",
                message=f"specified sequence probabilities sum above 1 on {n_exceeds_one} residue(s)",
                count=n_exceeds_one,
            )
        )

    fully_specified = compatible.any(axis=-1) & (specified | ~compatible).all(axis=-1)
    sequence_targets = C_SEQ.annotation(aap, default="generate")[residue_starts]
    noncanonical_exact = seq_conditioned & ~np.isin(sequence_targets, list(STANDARD_POLYMER_RESIDUES))
    missing_mass = fully_specified & ~noncanonical_exact & (specified_mass < 1.0 - 1e-6)
    n_missing_mass = int(missing_mass.sum())
    if n_missing_mass:
        violations.append(
            InvariantViolation(
                check="sequence_probability_sum_below_one",
                message=f"fully specified sequence probabilities sum below 1 on {n_missing_mass} residue(s)",
                count=n_missing_mass,
            )
        )

    incomplete = seq_conditioned & ~specified.all(axis=-1)
    n_incomplete = int(incomplete.sum())
    if n_incomplete:
        violations.append(
            InvariantViolation(
                check="sequence_probability_incomplete_for_c_seq",
                message=(
                    f"{n_incomplete} C_SEQ-conditioned residue(s) do not have all canonical-polymer "
                    "probabilities explicitly specified"
                ),
                count=n_incomplete,
            )
        )

    expected = np.zeros_like(probabilities)
    for channel, condition in enumerate(SEQUENCE_PROBABILITY_CONDITIONS):
        expected[:, channel] = sequence_targets == condition.token_name
    conflicts = seq_conditioned[:, None] & specified & ~np.isclose(probabilities, expected)
    conflicting_residues = conflicts.any(axis=-1)
    n_conflicting = int(conflicting_residues.sum())
    if n_conflicting:
        violations.append(
            InvariantViolation(
                check="sequence_probability_conflicts_with_c_seq",
                message=f"{n_conflicting} residue(s) have sequence probabilities that conflict with C_SEQ",
                count=n_conflicting,
            )
        )

    return violations


def check_index_group_molecule_connectivity_violations(
    aa: AtomArray, feats: dict, is_inference: bool
) -> list[InvariantViolation]:
    """Return violations where two tokens of one molecule were featurized as having no path between them."""
    if not is_inference or "token_bond_path_length" not in feats:
        return []

    groups = C_IDXGRP.annotation(aa, default="generate")
    indexed = C_IDX.mask(aa, default="generate") & ~aa.atomize
    if aa.bonds is not None:
        i, j = aa.bonds.as_array()[:, :2].T
        leaving, entering = get_polymerization_atom_masks(aa, np.arange(len(aa)))
        backbone = (leaving[i] & entering[j]) | (entering[i] & leaving[j])
        bad_bonds = backbone & indexed[i] & indexed[j] & (groups[i] != groups[j])
        if bad_bonds.any():
            a, b = i[bad_bonds][0], j[bad_bonds][0]
            return [
                InvariantViolation(
                    check="backbone_bond_index_group",
                    message=f"Backbone-bonded {aa.chain_id[a]}{aa.res_id[a]}-{aa.chain_id[b]}{aa.res_id[b]} "
                    f"have different index groups ({groups[a]}, {groups[b]}), hiding connectivity from the model. "
                    "Use one index group for the continuous indexed scaffold.",
                    count=int(bad_bonds.sum()),
                )
            ]

    path_length = np.asarray(feats["token_bond_path_length"])  # [n_tokens, n_tokens]
    token_starts = get_token_starts(aa)

    # Cross-group pairs are zeroed wholesale by the featurizer, so the check is scoped to a group.
    index_group = C_IDXGRP.annotation(aa, default="generate")[token_starts]  # [n_tokens]
    molecule_iid = aa.molecule_iid[token_starts]  # [n_tokens]
    bad = (
        (index_group[:, None] == index_group[None, :])
        & (molecule_iid[:, None] == molecule_iid[None, :])
        & (path_length == 0)
    )  # [n_tokens, n_tokens]
    np.fill_diagonal(bad, False)
    if not bad.any():
        return []

    # Report the upper triangle only, so a pair is named once.
    tok_i, tok_j = np.where(np.triu(bad))
    labels = [
        f"{aa.chain_id[token_starts[i]]}/{int(aa.res_id[token_starts[i]])}"
        f"-{aa.chain_id[token_starts[j]]}/{int(aa.res_id[token_starts[j]])}"
        for i, j in zip(tok_i[:10], tok_j[:10], strict=True)
    ]
    count = len(tok_i)
    return [
        InvariantViolation(
            check="index_group_molecule_must_be_connected",
            message=(
                f"{count} token pair(s) of one index group have no path between them in the bond "
                f"graph though they are one molecule: {', '.join(labels)}"
                f"{' ...' if count > len(labels) else ''}"
            ),
            count=count,
        )
    ]


def check_unindexed_coord_conditioning_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where an unindexed residue carries no coordinate conditioning.

    In the future we could consider supporting sequence-only unindexing, but for now that is
    explicitly disallowed to avoid subtle errors.
    """
    is_unindexed = ~C_IDX.mask(aa, default="generate")
    if not is_unindexed.any():
        return []

    has_coord = C_CRD.mask(aa, default="generate") | C_INTCRD.mask(aa, default="generate")
    starts = get_residue_starts(aa, add_exclusive_stop=True)
    residue_is_unindexed = np.asarray(apply_segment_wise(starts, is_unindexed, np.any), dtype=bool)
    residue_has_coord = np.asarray(apply_segment_wise(starts, has_coord, np.any), dtype=bool)

    bad = residue_is_unindexed & ~residue_has_coord
    if not bad.any():
        return []

    residue_starts = starts[:-1]
    labels = [f"{aa.chain_id[s]}/{aa.res_name[s]}/{int(aa.res_id[s])}" for s in residue_starts[bad][:10]]
    count = int(bad.sum())
    return [
        InvariantViolation(
            check="unindexed_must_be_coord_conditioned",
            message=(
                f"{count} unindexed residue(s) have no coordinate conditioning on any atom: "
                f"{', '.join(labels)}{' ...' if count > len(labels) else ''}. "
                "To pad an index group without conditioning it, leave a res_id gap instead of "
                "including an unconditioned spacer residue."
            ),
            count=count,
        )
    ]


def find_index_group_res_id_gaps(aa: AtomArray) -> list[str]:
    """Describe each internal res_id gap within a C_IDXGRP index group (atomized residues skipped).

    Returns one ``"index group <g>: res_id jumps from <a> to <b>"`` per gap; gaps on index-group
    boundaries are expected and not reported.
    """
    all_starts = get_residue_starts(aa)
    index_groups = C_IDXGRP.annotation(aa, default="generate")
    present: dict[int, set[int]] = {}  # res_ids per group, atomized residues included
    for start in all_starts:
        present.setdefault(int(index_groups[start]), set()).add(int(aa.res_id[start]))
    starts = all_starts[~S_ATM.mask(aa, default="generate")[all_starts]]
    messages: list[str] = []
    prev_group: int | None = None
    prev_res_id: int | None = None
    for start in starts:
        group = int(index_groups[start])
        res_id = int(aa.res_id[start])
        if (
            group == prev_group
            and prev_res_id is not None
            and (res_id - prev_res_id) not in (0, 1)
            # an atomized residue is skipped as an anchor but still occupies its res_id
            and not all(r in present[group] for r in range(prev_res_id + 1, res_id))
        ):
            messages.append(f"index group {group}: res_id jumps from {prev_res_id} to {res_id}")
        prev_group, prev_res_id = group, res_id
    return messages


# ---------------------------------------------------------------------------
# Cross-cutting checks (require both aa and feats)
# ---------------------------------------------------------------------------


def check_coordinate_centering_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where coordinates are not centered on coord-conditioned atoms."""
    aap = as_atom_array_plus(aa)
    if "coord_to_be_noised" not in aap.get_annotation_categories():
        return []

    crd_mask = C_CRD.mask(aap, default="generate")
    if not crd_mask.any():
        return []

    centroid_norm = float(np.linalg.norm(np.mean(aap.coord_to_be_noised[crd_mask], axis=0)))
    violations = []
    if centroid_norm >= 1e-3:
        violations.append(
            InvariantViolation(
                check="coordinates_not_centered",
                message=f"Atom-array centroid norm {centroid_norm:.2e} (threshold 1e-3)",
                count=1,
            )
        )

    if "coords" not in feats or C_CRD.full_name not in feats or "atom37_mask" not in feats:
        return violations

    coords_token: torch.Tensor = feats["coords"]  # [n_tokens, 37, 3]
    crd_mask_token = torch.isfinite(feats[C_CRD.full_name]).any(dim=-1)
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    valid_and_conditioned = atom37_mask & crd_mask_token
    if valid_and_conditioned.any():
        feats_centroid_norm = float(coords_token[valid_and_conditioned].mean(dim=0).norm().item())
        if feats_centroid_norm >= 1e-3:
            violations.append(
                InvariantViolation(
                    check="feats_coordinates_not_centered",
                    message=f"Feats centroid norm {feats_centroid_norm:.2e} (threshold 1e-3)",
                    count=1,
                )
            )

    return violations


def check_sequence_conditioning_violations(feats: dict, is_inference: bool) -> list[InvariantViolation]:
    """Return violations where residue_type encoding is inconsistent with conditioning masks."""
    required = {C_SEQ.full_name, "residue_type", "atom37_mask"}
    if not required.issubset(feats):
        return []

    # Lazy import to avoid circular dependency with pipeline.py
    from .datapipes_pipeline import TokenType

    seq_mask = feats[C_SEQ.full_name] != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
    res_type_gt: torch.Tensor = feats["residue_type"]  # [n_tokens]
    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]

    mask_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
    atomized_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"]
    max_valid_idx = UNIFIED_ATOM37_ENCODING.n_tokens

    violations = []

    if (res_type_gt < 0).any() or (res_type_gt >= max_valid_idx).any():
        violations.append(
            InvariantViolation(
                check="residue_type_out_of_range",
                message="Ground-truth residue_type contains out-of-range indices",
                count=1,
            )
        )

    # This check only applies in training, since residue_type will be passed through the encoder
    supports_local_latent = feats["supports_local_latent"]
    if not is_inference and (supports_local_latent & (res_type_gt == mask_token_idx)).any():
        violations.append(
            InvariantViolation(
                check="residue_type_contains_mask_token",
                message="Ground-truth residue_type contains <M> (mask) token",
                count=int((res_type_gt == mask_token_idx).sum()),
            )
        )

    cond_seq_key = C_SEQ.full_name
    if cond_seq_key in feats:
        res_type_cond: torch.Tensor = feats[cond_seq_key]

        if seq_mask.any():
            mismatch = (res_type_gt[seq_mask] != res_type_cond[seq_mask]).sum()
            if mismatch:
                violations.append(
                    InvariantViolation(
                        check="seq_conditioned_residue_type_mismatch",
                        message=f"{int(mismatch)} seq-conditioned token(s): condition_sequence != residue_type",
                        count=int(mismatch),
                    )
                )

        not_seq = ~seq_mask
        if not_seq.any():
            n_not_mask = int((res_type_cond[not_seq] != mask_token_idx).sum())
            if n_not_mask:
                violations.append(
                    InvariantViolation(
                        check="non_seq_conditioned_not_masked",
                        message=f"{n_not_mask} non-seq-conditioned token(s) have condition_sequence != <M>",
                        count=n_not_mask,
                    )
                )

        is_atomized = token_type == TokenType.ATOMIZED
        if is_atomized.any():
            n_not_a = int((res_type_cond[is_atomized] != atomized_token_idx).sum())
            if n_not_a:
                violations.append(
                    InvariantViolation(
                        check="atomized_condition_sequence_not_A",
                        message=f"{n_not_a} atomized token(s) have condition_sequence != <A>",
                        count=n_not_a,
                    )
                )

    return violations


def check_supports_local_latent_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where local-latent support disagrees with token eligibility."""
    if "supports_local_latent" not in feats:
        return []

    token_starts = get_token_starts(aa)
    res_names = aa.res_name[token_starts]
    categories = set(aa.get_annotation_categories())

    is_atomized = aa.atomize[token_starts] if "atomize" in categories else np.zeros(len(token_starts), dtype=bool)
    is_in_context_conditioning = apply_and_spread_token_wise(aa, ~C_IDX.mask(aa, default="generate"), np.any)[
        token_starts
    ]
    expected = np.isin(res_names, LOCAL_LATENT_RESIDUES) & ~is_atomized & ~is_in_context_conditioning

    supports_local_latent = feats["supports_local_latent"].cpu().numpy()
    violations = []

    if "atom37_mask" in feats:
        missing_reference = supports_local_latent & ~feats["atom37_mask"][:, 1].cpu().numpy().astype(bool)
        missing_reference &= ~S_SEGMIN.mask(aa, default="generate")[token_starts]
        n_missing_reference = int(missing_reference.sum())
        if n_missing_reference:
            violations.append(
                InvariantViolation(
                    check="local_latent_reference_atom_missing",
                    message=f"{n_missing_reference} local-latent token(s) lack their atom37 slot-1 reference atom",
                    count=n_missing_reference,
                )
            )

    n_invalid = int((supports_local_latent & ~expected).sum())
    if n_invalid:
        violations.append(
            InvariantViolation(
                check="supports_local_latent_ineligible",
                message=f"{n_invalid} token(s) have supports_local_latent=True but are not eligible",
                count=n_invalid,
            )
        )

    n_missing = int((~supports_local_latent & expected).sum())
    if n_missing:
        violations.append(
            InvariantViolation(
                check="supports_local_latent_missing",
                message=f"{n_missing} eligible token(s) have supports_local_latent=False",
                count=n_missing,
            )
        )

    return violations


def check_no_oxt_on_local_latent_tokens(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Reject local-latent OXT atoms that violate the shared terminal-oxygen policy."""

    categories = set(aa.get_annotation_categories())

    supports_local_latent = feats["supports_local_latent"].cpu().numpy()  # [n_tokens]
    token_id_per_atom = aa.token_id  # [n_atoms]
    is_atomized = aa.atomize if "atomize" in categories else np.zeros(len(aa), dtype=bool)
    is_oxt = aa.atom_name == "OXT"  # [n_atoms]

    supports_local_latent_per_atom = supports_local_latent[token_id_per_atom]  # [n_atoms]
    bad = is_oxt & supports_local_latent_per_atom & ~is_atomized
    bad &= polymer_terminal_oxygen_mask(aa)
    n_bad = int(bad.sum())
    if n_bad:
        return [
            InvariantViolation(
                check="oxt_on_local_latent_token",
                message=f"{n_bad} OXT atom(s) found on supports_local_latent non-atomized token(s)",
                count=n_bad,
            )
        ]
    return []


def check_token_features_consistent_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where token-level features have inconsistent values within a token."""
    # Lazy import to avoid circular dependency with pipeline.py
    from .datapipes_pipeline import ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL

    categories = set(aa.get_annotation_categories())
    if "token_id" not in categories:
        return []

    token_starts = get_token_starts(aa, add_exclusive_stop=False)  # [n_tokens]
    n_atoms = aa.array_length()
    atom_token_idx = aa.token_id  # [n_atoms], values 0..n_tokens-1

    violations = []
    for feat_name in ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL:
        if feat_name not in categories:
            continue
        feat_values = aa.get_annotation(feat_name)

        first = feat_values[token_starts][atom_token_idx]  # token-first value per atom
        differs = feat_values != first
        if np.issubdtype(feat_values.dtype, np.floating):
            # NaN == NaN (all-NaN token is consistent); NaN != any real value.
            differs &= ~(np.isnan(feat_values) & np.isnan(first))
        atom_differs = differs.reshape(n_atoms, -1).any(axis=1)  # [n_atoms]

        # A token is bad if any of its atoms differ from the token's first atom.
        n_bad = int((np.add.reduceat(atom_differs.astype(np.int64), token_starts) > 0).sum())
        if n_bad:
            violations.append(
                InvariantViolation(
                    check="token_feature_inconsistent",
                    message=f"'{feat_name}' has inconsistent values in {n_bad} token(s)",
                    count=n_bad,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class ValidateBatchInvariants(Transform):
    """Validate batch-level invariants at the end of the pipeline.

    Calls :py:func:`detect_batch_invariant_violations` on every example and
    handles violations according to ``on_violation``.

    Args:
        on_violation: Action taken when violations are found.
            ``"raise"`` (default) - raise :py:exc:`ValueError`.
            ``"warn"`` - emit a warning via :mod:`logging` and continue.
            ``"ignore"`` - silently continue (useful when used as a probe).
    """

    def __init__(self, is_inference: bool, on_violation: str = "raise"):
        super().__init__()
        if on_violation not in ("raise", "warn", "ignore"):
            raise ValueError(f"on_violation must be 'raise', 'warn', or 'ignore', got {on_violation!r}")
        self.on_violation = on_violation
        self.is_inference = is_inference

    def forward(self, data: dict) -> dict:
        violations = detect_batch_invariant_violations(data, is_inference=self.is_inference)
        if not violations or self.on_violation == "ignore":
            return data

        example_id = data.get("example_id", "unknown")
        lines = "\n".join(f"  [{v.check}] {v.message}" for v in violations)
        message = f"Batch invariant violation(s) for example '{example_id}':\n{lines}"

        if self.on_violation == "raise":
            raise ValueError(message)
        else:
            logger.warning(message)

        return data
