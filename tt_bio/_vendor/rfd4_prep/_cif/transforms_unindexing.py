# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import logging
import copy
import itertools

import biotite.structure as struc
import numpy as np
from tt_bio._vendor.atomworks.io.utils.annotator import remove_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import concatenate_any
from tt_bio._vendor.atomworks.io.utils.scatter import safe_scatter
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_SEGMIN
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_CTR, C_IDX, C_NTR
from tt_bio._vendor.atomworks.ml.conditions.base import CONDITIONS, ConditionConflictError
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_contains_keys,
    check_is_instance,
)
from tt_bio._vendor.atomworks.ml.transforms.atom_array import (
    add_global_atom_id_annotation,
    add_global_token_id_annotation,
    get_within_group_source_res_idx,
)
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.utils.annotator import (
    ensure_annotations,
)
from tt_bio._vendor.atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    apply_token_wise,
    get_token_count,
)
from biotite.structure import AtomArray
from biotite.structure.residues import get_residue_starts, spread_residue_wise
from scipy.optimize import linear_sum_assignment

from .conditions import C_CRDGRP, C_IDXGRP, C_INTCRD
from .datapipes_bonds import bond_flanking_residues
from .transforms_utils_conditioning_utils import (
    set_specified_conditions_to_default,
)
from .utils_standard_annotations import SA_CRDSRC, SA_CWI, SA_IDXGRPSRC, SA_SRC, SA_UII

global_logger = logging.getLogger(__name__)


class UnindexFlaggedTokens(Transform):
    """Create in-context conditioning for tokens with unknown indices."""

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def expand_unindexed_motifs(self, atom_array: AtomArray, is_inference: bool) -> AtomArray:
        """
        Takes atom array with C_IDX annotation and padds it to include unindexed motif atoms.

        During training, the original coordinates are left behind for the model to learn to diffuse, with all Conditions
        set to their defaults. During inference, the original tokens are removed.
        """

        # Assemble necessary annotations
        is_motif_atom_unindexed = ~C_IDX.mask(atom_array, default="generate")
        ensure_annotations(atom_array, "is_polymer", "within_chain_source_res_idx")

        if not np.any(is_motif_atom_unindexed):
            atom_array.set_annotation("insertion_source_idx", np.full(atom_array.array_length(), -1, dtype=np.int32))
            return atom_array

        unique_chain_iids = np.unique(atom_array.chain_iid)
        max_resid_by_chain_iid = {
            chain_iid: np.max(atom_array.res_id[atom_array.chain_iid == chain_iid]) for chain_iid in unique_chain_iids
        }

        # Each unindexed copy gets a fresh token_id (the source id is recorded in insertion_source_idx)
        next_copy_token_id = int(np.max(atom_array.token_id)) + 1
        max_atom_idx = (
            atom_array.array_length() - 1  # start one less because of the way we increment it
        )

        # Identify tokens which have at least one unindexed atom
        token_to_unindex = struc.spread_residue_wise(
            atom_array,
            struc.apply_residue_wise(
                atom_array,
                is_motif_atom_unindexed,
                function=lambda x: np.any(x),
            ),
        )
        assert token_to_unindex.sum() > 0, "No tokens to unindex!"

        # Record original annotations, to be added back later where appropriate.
        original_1d_conditions = {}
        original_2d_conditions = {}
        for cond in CONDITIONS:
            # Add to the appropriate dict for the current n_body
            if cond.n_body == 0:
                continue
            elif cond.n_body == 1:
                dict_to_use = original_1d_conditions
            elif cond.n_body == 2:
                dict_to_use = original_2d_conditions
            else:
                raise NotImplementedError(f"Unindexing does not yet support conditions with n_body={cond.n_body}")

            if cond.has_annotation(atom_array):
                if cond.name not in dict_to_use:
                    dict_to_use[cond.name] = {}
                dict_to_use[cond.name]["annotation"] = cond.annotation(atom_array, default="raise")

        # Canonicalize condition storage before copying the motif.
        # Condition accessors validate aliases before any redundant columns are removed.
        for conditions in (original_1d_conditions, original_2d_conditions):
            for cond_name, condition_dict in conditions.items():
                cond = CONDITIONS[cond_name]
                cond.set_annotation(atom_array, condition_dict["annotation"])
                remove_annotations(atom_array, *cond.aliases)

        idxs = np.arange(atom_array.array_length())
        if not C_IDXGRP.has_annotation(atom_array):
            C_IDXGRP.set_annotation(atom_array, C_IDXGRP.default_annotation(atom_array))
        index_groups = C_IDXGRP.annotation(atom_array, default="raise")
        unindexed_tokens = []
        old_to_new_atom_idx_map = {}
        next_free_res_id = {chain: int(max_resid) + 1 for chain, max_resid in max_resid_by_chain_iid.items()}
        for index_group in np.unique(index_groups[is_motif_atom_unindexed]):
            in_group = (index_groups == index_group) & is_motif_atom_unindexed
            motif = copy.deepcopy(atom_array[in_group])

            assert motif.is_polymer.all(), "Cannot unindex non-polymers!"

            # Break in contiguity between indexed and unindexed regions is fine since they never get relative indices.
            # Offsets are per chain and res_ids are strictly increasing within each chain.
            res_id_offsets = np.zeros(motif.array_length(), dtype=motif.res_id.dtype)
            for chain_iid in np.unique(motif.chain_iid):
                in_chain = motif.chain_iid == chain_iid
                offset = next_free_res_id[chain_iid] - int(motif.res_id[in_chain].min())
                res_id_offsets[in_chain] = offset
                next_free_res_id[chain_iid] = int(motif.res_id[in_chain].max()) + offset + 2
            motif.res_id = motif.res_id + res_id_offsets

            # A residue is still a token: the group is copied in one piece, but its residues must not
            # be fused into one. Ids are assigned before either array is written back, so the source
            # ids are read unmodified.
            residue_starts = struc.get_residue_starts(motif, add_exclusive_stop=True)
            source_token_ids = np.empty(motif.array_length(), dtype=np.int32)
            copy_token_ids = np.empty(motif.array_length(), dtype=motif.token_id.dtype)
            for start, end in itertools.pairwise(residue_starts):
                # At inference there is no indexed source to point back to, so leave it at -1.
                source_token_ids[start:end] = -1 if is_inference else int(motif.token_id[start])
                copy_token_ids[start:end] = next_copy_token_id
                next_copy_token_id += 1
            motif.set_annotation("insertion_source_idx", source_token_ids)
            motif.token_id = copy_token_ids

            # TODO: When we introduce the is_canonical annotation, we should enforce that here instead
            assert motif.is_polymer.all(), f"Cannot unindex non-polymer token: {motif}"

            unindexed_tokens.append(motif)

            # update index map for future 2d conditioning updates
            old_to_new_atom_idx_map.update({old_idx: (max_atom_idx := max_atom_idx + 1) for old_idx in idxs[in_group]})

        assert len(unindexed_tokens) > 0, "No unindexed tokens found!"

        unindexed = concatenate_any(unindexed_tokens)

        # Do not leak index through termini features
        false_mask = np.zeros(unindexed.array_length(), dtype=bool)
        if C_NTR.has_annotation(unindexed):
            C_NTR.set_annotation(unindexed, false_mask)
        if C_CTR.has_annotation(unindexed):
            C_CTR.set_annotation(unindexed, false_mask)

        # ... Remove original tokens e.g. during inference
        if is_inference:
            # some indexed atoms might change location when we delete the unindexed tokens
            original_indexed_positions = np.arange(atom_array.array_length())[~token_to_unindex]
            new_indexed_positions = np.arange((~token_to_unindex).sum())

            # update old_to_new_atom_idx_map to reflect shortened atom array
            old_to_new_atom_idx_map = {
                old_idx: new_idx - token_to_unindex.sum() for old_idx, new_idx in old_to_new_atom_idx_map.items()
            }

            # update old_to_new_atom_idx_map to reflect moving indexed atoms
            old_to_new_atom_idx_map.update(dict(zip(original_indexed_positions, new_indexed_positions, strict=False)))

            # Removing an unindexed motif region leaves a gap. If at least one side of this is
            # an expandable segment, the gap is assumed to be unintended and is filled.
            # Otherwise, the gap is preserved with a warning.
            residue_starts = get_residue_starts(atom_array)
            is_removed_res = token_to_unindex[residue_starts]
            is_expandable_res = S_SEGMIN.mask(atom_array, default="generate")[residue_starts]
            chain_of_res = atom_array.chain_iid[residue_starts]
            res_id_of_res = atom_array.res_id[residue_starts]

            n_collapsed_before = np.zeros(len(residue_starts), dtype=np.int64)
            for chain_iid in np.unique(chain_of_res):
                in_chain = np.where(chain_of_res == chain_iid)[0]
                is_removed = is_removed_res[in_chain]
                collapse = is_removed.copy()
                edges = np.diff(np.concatenate([[0], is_removed.astype(np.int8), [0]]))
                for run_start, run_stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
                    if run_start == 0 or run_stop == len(in_chain):
                        continue
                    if is_expandable_res[in_chain[[run_start - 1, run_stop]]].any():
                        continue
                    collapse[run_start:run_stop] = False
                    global_logger.warning(
                        f"Unindexed residues {chain_iid}/{res_id_of_res[in_chain[run_start]]}-"
                        f"{res_id_of_res[in_chain[run_stop - 1]]} lie between two indexed regions taken "
                        "from the input, so the design keeps that gap between the flanking indexed regions."
                    )
                n_collapsed_before[in_chain] = np.cumsum(collapse) - collapse
            collapsed_before_atom = spread_residue_wise(atom_array, n_collapsed_before)
            atom_array.res_id = atom_array.res_id - collapsed_before_atom.astype(atom_array.res_id.dtype)

            atom_array = atom_array[~token_to_unindex]

            # Re-join the residues an excision left adjacent.
            res_bounds = get_residue_starts(atom_array, add_exclusive_stop=True)
            starts = res_bounds[:-1]
            collapsed_before = collapsed_before_atom[~token_to_unindex][starts]
            chains, res_ids = atom_array.chain_iid[starts], atom_array.res_id[starts].astype(np.int64)
            for prev_res, next_res in itertools.pairwise(range(len(starts))):
                if (
                    chains[prev_res] == chains[next_res]
                    and res_ids[next_res] - res_ids[prev_res] == 1
                    and collapsed_before[next_res] > collapsed_before[prev_res]
                ):
                    bond_flanking_residues(atom_array, prev_res, next_res, res_bounds)

            new_within_chain_source_res_idx = get_within_group_source_res_idx(atom_array, group_by="chain_iid")
            atom_array.set_annotation("within_chain_source_res_idx", new_within_chain_source_res_idx)

        else:
            # Temporarily set all previously-existing conditions to their default values
            conditions_to_remove = [CONDITIONS[cond_name] for cond_name in original_1d_conditions]
            set_specified_conditions_to_default(atom_array, conditions=conditions_to_remove)

            # TODO: Consider stochastic condition resets to avoid leaking which source tokens were copied.

            # Restore conditions on tokens whose atoms were not copied for unindexing.
            # Restore coordinate-group membership before its saved internal-coordinate payload.
            for cond_name, condition_dict in sorted(
                original_1d_conditions.items(), key=lambda item: item[0] == C_INTCRD.name
            ):
                cond = CONDITIONS[cond_name]
                if "annotation" in condition_dict:
                    current_annotation = condition_dict["annotation"].copy()
                    default_annotation = cond.annotation(atom_array, default="raise")
                    current_annotation[token_to_unindex] = default_annotation[token_to_unindex]
                    cond.set_annotation(atom_array, current_annotation)

            # Index groups present an edge case -- resetting to defaults only in the regions to be
            # unindexed may cause the (now-indexed) source residues to have different index groups from
            # their same-chain indexed neighbors. To avoid this, we reset the full indexed region
            C_IDXGRP.set_annotation(atom_array, C_IDXGRP.default_annotation(atom_array))

        # Indexed (source) tokens are never insertion queries; set insertion_source_idx=-1 there.
        if is_inference:
            atom_array.set_annotation("insertion_source_idx", np.full(atom_array.array_length(), -1, dtype=np.int32))
        else:
            # Cast token_id to signed int for NumPy 2 (NEP 50) compatibility: otherwise np.where
            # resolves to uint32 and rejects the -1 sentinel with an OverflowError.
            atom_array.set_annotation(
                "insertion_source_idx",
                np.where(C_IDX.mask(atom_array, default="raise"), -1, atom_array.token_id.astype(np.int32)),
            )

        # Concatenate unindexed parts to the end
        atom_array_full = concatenate_any([atom_array, unindexed])

        # Ensure that the newly-created index groups in the indexed region do not overlap with
        # those in the tokens that were copied for unindexing
        index_groups = C_IDXGRP.annotation(atom_array_full, default="raise")
        is_indexed = C_IDX.mask(atom_array_full, default="raise")
        max_index_group_in_indexed_region = index_groups[is_indexed].max()
        min_index_group_in_unindexed_region = index_groups[~is_indexed].min()
        index_groups[~is_indexed] += max_index_group_in_indexed_region - min_index_group_in_unindexed_region + 1
        C_IDXGRP.set_annotation(atom_array_full, index_groups)

        # update pairs in place using the old_to_new_atom_idx_map
        # vectorize for efficiency and one-shot update
        vectorized_map = np.vectorize(lambda x: old_to_new_atom_idx_map.get(x, x))

        # Update all 2D conditions to reflect movement of unindexed atoms
        for cond_name, condition_dict in original_2d_conditions.items():
            cond = CONDITIONS[cond_name]
            if "annotation" in condition_dict:
                annotation = condition_dict["annotation"]
                pairs, values = annotation.pairs, annotation.values
                if len(pairs) > 0:
                    pairs = vectorized_map(pairs)
                    cond.set_annotation(atom_array_full, pairs=pairs, values=values)

        return atom_array_full

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "feats" not in data:
            data["feats"] = {}

        _check_index_group_consistency(atom_array)

        is_unindexed_atom = ~C_IDX.mask(atom_array, default="generate")

        is_token_with_unindexed_atoms = apply_and_spread_token_wise(
            atom_array,
            is_unindexed_atom.copy(),
            function=lambda x: np.any(x),
        )
        is_token_fully_unindexed = apply_and_spread_token_wise(
            atom_array,
            is_unindexed_atom.copy(),
            function=lambda x: np.all(x),
        )
        if not np.array_equal(is_token_with_unindexed_atoms, is_token_fully_unindexed):
            raise ValueError("C_IDX must be uniform across every token; unindexed polymer tokens are full residues.")
        has_any_unindexed = bool(np.any(is_token_with_unindexed_atoms))

        # Expand unindexed motifs if necessary
        atom_array_expanded = self.expand_unindexed_motifs(
            atom_array,
            is_inference=data["is_inference"],
        )

        # Per-atom mask over the expanded array: True for atoms of the original
        # ground-truth tokens whose atoms were copied into the unindexed conditioning slice.
        n_original_atoms = atom_array.array_length()
        n_expanded_atoms = atom_array_expanded.array_length() - n_original_atoms
        mask = np.concatenate([is_token_with_unindexed_atoms, np.zeros(n_expanded_atoms)])

        # Expose the same mask as an atom_array annotation for downstream bookkeeping
        # (These are the ORIGINAl ground-truth tokens, not e.g., the appended in-context copies)
        if data.get("is_inference", False):
            is_unindexed_source = np.zeros(atom_array_expanded.array_length(), dtype=bool)
        else:
            is_unindexed_source = mask.astype(bool)
        atom_array_expanded.set_annotation("is_unindexed_source", is_unindexed_source)

        # Reset global token IDs and atom IDs after possible expansion

        atom_array_expanded = add_global_token_id_annotation(atom_array_expanded)
        atom_array_expanded = add_global_atom_id_annotation(atom_array_expanded)

        # Ensure newly-added tokens are recognised as separate
        if has_any_unindexed and not data["is_inference"]:
            n_unindexed_tokens = int(apply_token_wise(atom_array, is_unindexed_atom, np.any).sum())
            tc_orig = get_token_count(atom_array)
            tc_exp = get_token_count(atom_array_expanded)
            assert tc_exp == tc_orig + n_unindexed_tokens, (
                "UnindexFlaggedTokens failed to create uniquely recognised tokens after "
                f"concatenation: expanded tokens={tc_exp}, original={tc_orig}, "
                f"unindexed={n_unindexed_tokens} (expected expanded == original + unindexed)."
            )

        data["atom_array"] = atom_array_expanded
        return data


def _check_index_group_consistency(atom_array: AtomArray) -> None:
    """Raise if any ``C_IDXGRP`` index group spans both indexed and unindexed tokens."""
    if not C_IDXGRP.has_annotation(atom_array):
        return
    index_groups = C_IDXGRP.annotation(atom_array, default="raise")
    is_unindexed = ~C_IDX.mask(atom_array, default="generate")
    groups_with_both = sorted(
        int(group)
        for group in np.unique(index_groups)
        if is_unindexed[index_groups == group].any() and not is_unindexed[index_groups == group].all()
    )
    if groups_with_both:
        raise ValueError(
            f"Index group(s) {groups_with_both} contain both indexed and unindexed tokens. Each index group must be "
            "wholly indexed or wholly unindexed."
        )


def _get_target_indices_corresponding_to_inserted(
    atom_array: AtomArray,
    insertion_mask: np.ndarray,
    target_mask: np.ndarray,
    allow_element_fallback: bool = False,
) -> np.ndarray:
    """Find indices into the target token giving the same atom names/ordering as the inserted token.

    ``insertion_mask`` and ``target_mask`` select the inserted and target atoms from
    ``atom_array``. If ``allow_element_fallback`` is True, atom names not found in the
    target token are matched by element, with ties broken by minimizing total distance.
    """
    # Get insertion atom names
    insertion_atom_names = atom_array.atom_name[insertion_mask]

    # Get target token data
    target_atoms = atom_array[target_mask]
    target_sorted_indices = np.argsort(target_atoms.atom_name)
    target_sorted_atom_names = target_atoms.atom_name[target_sorted_indices]

    # Attempt to match insertion atoms to target atoms by name
    search_positions = np.searchsorted(target_sorted_atom_names, insertion_atom_names)
    unmatched_insertion_atoms_mask = ~np.isin(insertion_atom_names, target_sorted_atom_names)

    if np.any(unmatched_insertion_atoms_mask):
        if not allow_element_fallback:
            raise ValueError("Inserted atom names not found in target residue atom names.")
        else:
            # Identify which target atoms don't match any insertion atom names
            unmatched_target_atoms_mask = ~np.isin(target_sorted_atom_names, insertion_atom_names)
            unmatched_target_indices = np.where(unmatched_target_atoms_mask)[0]
            unmatched_insertion_indices = np.where(unmatched_insertion_atoms_mask)[0]

            # Initialize output array and fill in name-based matches
            target_indices_corresponding_to_inserted = np.full(len(insertion_atom_names), np.nan)
            target_indices_corresponding_to_inserted[~unmatched_insertion_atoms_mask] = target_sorted_indices[
                search_positions[~unmatched_insertion_atoms_mask]
            ]

            # Get coordinates and elements for unmatched atoms
            unmatched_target_coords = target_atoms.coord[target_sorted_indices][unmatched_target_atoms_mask]
            unmatched_insertion_coords = atom_array.coord[insertion_mask, :][unmatched_insertion_atoms_mask]

            unmatched_target_elements = target_atoms.element[target_sorted_indices][unmatched_target_atoms_mask]
            unmatched_insertion_elements = atom_array.element[insertion_mask][unmatched_insertion_atoms_mask]
            same_element_mask = unmatched_insertion_elements[:, np.newaxis] == unmatched_target_elements[np.newaxis, :]

            # Compute distance matrix, setting inf for different elements
            dists = np.linalg.norm(
                unmatched_insertion_coords[:, np.newaxis, :] - unmatched_target_coords[np.newaxis, :, :],
                axis=-1,
            )
            dists[~same_element_mask] = np.inf

            # Use linear sum assignment to match by element, minimizing total distance
            row_indices, col_indices = linear_sum_assignment(dists)

            # Fill in element-based matches
            for r, c in zip(row_indices, col_indices, strict=False):
                if not np.isfinite(dists[r, c]):
                    raise ValueError("Could not find matching target atom for inserted atom based on element type.")
                assert np.isnan(
                    target_indices_corresponding_to_inserted[unmatched_insertion_indices[r]]
                ), "Duplicate target element assignemnt"
                target_indices_corresponding_to_inserted[unmatched_insertion_indices[r]] = target_sorted_indices[
                    unmatched_target_indices[c]
                ]

            if np.any(np.isnan(target_indices_corresponding_to_inserted)):
                raise ValueError(
                    "Some inserted atoms could not be matched to target atoms even after element fallback."
                )

    else:
        # All insertion atoms matched by name
        target_indices_corresponding_to_inserted = target_sorted_indices[search_positions]

    # Convert to int (may be float64 if initialized with NaN)
    return target_indices_corresponding_to_inserted.astype(int)


def annotate_unindexed_insertion(
    atom_array: AtomArray,
    token_mapping: dict[int, int],
) -> AtomArray:
    """Annotate generated tokens provided a token mapping.

    For each ``{unindexed_token_id: target_token_id}`` entry the unindexed in-context token is
    transferred onto its mapped generated token:

    - Provenance / metric annotations are staged over the coord-conditioned atoms (``SA_CRDSRC`` source
      id, ``SA_CWI`` reference coords) plus, over the whole mapped token, ``SA_UII`` and ``SA_IDXGRPSRC``
      (the motif's original unindexed index group, since ``C_IDXGRP`` is re-derived).
    - Every per-position (atom/token/residue-level) 1-body condition (coords, sequence, coord group,
      SSE, RASA, h-bond, interactions, ...) is additionally re-applied with the ORIGINAL-input value,
      copied through one per-token atom correspondence (source and target atoms line up 1:1).

    The mapped token is then marked indexed (``C_IDX`` = True): once an unindexed motif atom has a
    definite position it becomes a normal indexed atom, so a further round of partial diffusion treats
    it as fixed rather than re-placing it.

    A mapping that cannot be applied (e.g. unmatched atoms) leaves the unindexed region unmapped. A
    mapped token already conditioned with a *different* value raises :py:exc:`ConditionConflictError`.

    Args:
        atom_array: AtomArray with the unindexed in-context conditioning appended to the
            generated tokens.
        token_mapping: ``{unindexed_token_id: target_token_id}`` (``token_id`` is globally
            unique)

    Returns:
        Annotated generated AtomArray with the unindexed conditioning slice removed.
    """
    # Copy every per-position (atom/token/residue-level), 1-body condition by default
    # NOTE: index/termini are position-derived (re-derived, not carried); pairwise have no 1:1 mapping.
    # C_CRDGRP is copied too, but with overwrite semantics: every target atom already carries a coord group
    # (default value UNKNOWN -1, mask always present), so the raise-loop would treat that default as a conflict.
    not_copied = (C_IDX, C_IDXGRP, C_CRDGRP, C_NTR, C_CTR)
    conditions_to_copy = tuple(
        cond
        for cond in CONDITIONS
        if cond not in not_copied and cond.n_body == 1 and cond.level in (Level.ATOM, Level.TOKEN, Level.RESIDUE)
    )

    has_index = C_IDX.mask(atom_array, default="generate")
    model_generated_atom_array = atom_array[has_index].copy()
    in_context_conditioning = atom_array[~has_index].copy()

    # Materialize the output schema before early returns for indexed or unmapped inputs.
    if not C_CRDGRP.has_annotation(model_generated_atom_array):
        C_CRDGRP.set_annotation(model_generated_atom_array, C_CRDGRP.default_annotation(model_generated_atom_array))

    n_atoms = model_generated_atom_array.array_length()

    def _set_empty_unindexed_annotations() -> AtomArray:
        """Set all unindexed SAs to empty defaults and merge indexed conditioning into SA_CRDSRC.

        Used both when there are no unindexed tokens and when the token mapping fails.
        """
        SA_CRDSRC.set_annotation(model_generated_atom_array, SA_CRDSRC.default_annotation(model_generated_atom_array))
        SA_IDXGRPSRC.set_annotation(
            model_generated_atom_array, SA_IDXGRPSRC.default_annotation(model_generated_atom_array)
        )
        SA_UII.set_annotation(model_generated_atom_array, SA_UII.default_annotation(model_generated_atom_array))
        return _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    # Short-circuit: no unindexed tokens at all — populate indexed conditioning and return.
    if in_context_conditioning.array_length() == 0:
        return _set_empty_unindexed_annotations()

    # Union of absolute and internal coord conditioning
    has_3d_conditioning = C_CRD.mask(in_context_conditioning, default="generate") | C_CRDGRP.is_internal(
        in_context_conditioning
    )
    # Track every generated atom mapped from an unindexed token (whole token, drives SA_UII / the source
    # index group) and the coord-conditioned subset (drives SA_CRDSRC / SA_CWI). The coord group itself
    # is copied live onto C_CRDGRP; only the source *index* group is kept as provenance (SA_IDXGRPSRC).
    mapped_to_unindexed_insertion = np.zeros(n_atoms, dtype=bool)
    mapped_to_coordinate_conditioned_region = np.zeros(n_atoms, dtype=bool)
    # An atom carries at most one coord mode: absolute -> C_CRD, internal -> C_INTCRD.
    assert not (
        C_CRD.mask(in_context_conditioning, default="generate") & C_CRDGRP.is_internal(in_context_conditioning)
    ).any(), "Atom carries both absolute (C_CRD) and internal (C_INTCRD) coord conditioning; modes are exclusive."
    remapped_coord = np.full_like(model_generated_atom_array.coord, np.nan)
    reference_coord = C_INTCRD.annotation(model_generated_atom_array, default="generate").copy()
    coordinate_groups = C_CRDGRP.annotation(model_generated_atom_array, default="generate").copy()
    src_coordinate_conditioned_region = SA_CRDSRC.default_annotation(model_generated_atom_array)
    src_index_group = SA_IDXGRPSRC.default_annotation(model_generated_atom_array)
    generated_indices = np.arange(n_atoms)

    unindexed_internal_coords = C_INTCRD.annotation(in_context_conditioning, default="generate")
    unindexed_absolute_coords = C_CRD.annotation(in_context_conditioning, default="generate")
    unindexed_is_absolute = C_CRD.mask(in_context_conditioning, default="generate")
    unindexed_index_groups = C_IDXGRP.annotation(atom_array, default="generate")[~has_index]

    # Only coordinate-conditioned atoms feed the SA_* provenance / SA_CWI reference; the rest of a
    # token (e.g. a sequence-free sidechain) has no coord partner to match.
    is_coord_conditioned_in_context = np.zeros(atom_array.array_length(), dtype=bool)
    is_coord_conditioned_in_context[~has_index] = has_3d_conditioning

    try:
        for unindexed_token_id, target_token_id in token_mapping.items():
            unindexed_token_mask = in_context_conditioning.token_id == unindexed_token_id
            unindexed_token = in_context_conditioning[unindexed_token_mask]
            token_src_id = (
                SA_SRC.annotation(unindexed_token)[0]
                if SA_SRC.has_annotation(unindexed_token)
                else str(unindexed_token_id)
            )
            target_mask_generated = model_generated_atom_array.token_id == target_token_id
            mapped_to_unindexed_insertion[target_mask_generated] = True

            # Copy each allowlisted 1-body condition from the unindexed source onto the mapped token.
            # Values come from the source's stored ANNOTATION (the frozen original input), not the live
            # coord / res_name. Source and target atoms line up (same residue, same atoms), so one name
            # correspondence carries every condition -- atom- and residue-level alike.
            target_mask_indexed = (atom_array.token_id == target_token_id) & has_index
            src_token_mask = ~has_index & (atom_array.token_id == unindexed_token_id)
            src_token_indices = np.where(src_token_mask)[0]
            token_target_indices = generated_indices[target_mask_generated][
                _get_target_indices_corresponding_to_inserted(
                    atom_array, src_token_mask, target_mask_indexed, allow_element_fallback=True
                )
            ]
            for cond in conditions_to_copy:
                carries = cond.mask(atom_array, default="generate")[src_token_indices]
                if not carries.any():
                    continue
                cond.transfer(
                    atom_array,
                    model_generated_atom_array,
                    src_token_indices[carries],
                    token_target_indices[carries],
                    on_mismatch="raise",
                )

            # The motif's coord group is authoritative post-insertion: overwrite the target's default
            # UNKNOWN group rather than treating it as a conflict. C_CRDGRP's mask is always present, so
            # gate on the value -- skip tokens whose source carries no real group (all UNKNOWN).
            src_group = C_CRDGRP.annotation(atom_array, default="generate")[src_token_indices]
            if (src_group != C_CRDGRP.UNKNOWN_COORDS).any():
                coordinate_groups = safe_scatter(src_group, token_target_indices, coordinate_groups)

            # Coord provenance / metric staging: reuse the token correspondence restricted to the
            # coord-conditioned atoms, so conditions and coords land on the same target atoms.
            coord_target = token_target_indices[is_coord_conditioned_in_context[src_token_indices]]
            token_coord_mask = has_3d_conditioning[unindexed_token_mask]
            mapped_to_coordinate_conditioned_region[coord_target] = True
            src_coordinate_conditioned_region[coord_target] = token_src_id
            # Record the motif's original unindexed index group over the whole mapped token; C_IDXGRP
            # itself is re-derived into the indexed frame, so this is the only index-group provenance.
            src_index_group[target_mask_generated] = unindexed_index_groups[unindexed_token_mask][0]
            # Absolute motif coords come from the frozen C_CRD annotation (matching structure_export and
            # the internal path), not the live coord; internal-only atoms keep the live coord as a
            # placeholder that _restage_internal_coord_cwi_per_group overwrites.
            token_is_absolute = unindexed_is_absolute[unindexed_token_mask]
            token_ref_coord = np.where(
                token_is_absolute[:, None], unindexed_absolute_coords[unindexed_token_mask], unindexed_token.coord
            )
            remapped_coord[coord_target] = token_ref_coord[token_coord_mask]
            reference_coord[coord_target] = unindexed_internal_coords[unindexed_token_mask][token_coord_mask]

        # Insertion restores membership and frozen targets together, in the source groups' frames.
        internal_targets = C_INTCRD.annotation(model_generated_atom_array, default="generate").copy()
        if np.any(
            C_CRD.mask(model_generated_atom_array, default="generate")
            & (coordinate_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
        ) or np.any(~np.isnan(internal_targets).all(axis=1) & (coordinate_groups <= C_CRDGRP.KNOWN_ABSOLUTE_COORDS)):
            raise ConditionConflictError("Unindexed insertion conflicts with existing coordinate conditioning modes.")
        C_CRDGRP.set_annotation(model_generated_atom_array, coordinate_groups, on_internal_coord_mismatch="clear")
        C_INTCRD.set_annotation(model_generated_atom_array, internal_targets)

        SA_CRDSRC.set_annotation(model_generated_atom_array, src_coordinate_conditioned_region)
        SA_IDXGRPSRC.set_annotation(model_generated_atom_array, src_index_group)
        SA_UII.set_annotation(model_generated_atom_array, mapped_to_unindexed_insertion)

        # Mapped tokens now have a definite position -> mark C_IDX=True in case of further rounds of sampling.
        C_IDX.set_annotation(
            model_generated_atom_array,
            C_IDX.mask(model_generated_atom_array, default="generate") | mapped_to_unindexed_insertion,
        )

        # Stage the unindexed coordinates as the conditioned-reference coords (consumed by metrics).
        if mapped_to_coordinate_conditioned_region.any():
            if np.isnan(remapped_coord[mapped_to_coordinate_conditioned_region]).any():
                raise ValueError("NaN values found in remapped coordinates")
            cwi = SA_CWI.annotation(model_generated_atom_array)
            cwi[mapped_to_coordinate_conditioned_region] = remapped_coord[mapped_to_coordinate_conditioned_region]
            SA_CWI.set_annotation(model_generated_atom_array, np.asarray(cwi, dtype=np.float32))

    except ValueError as e:
        # Soft-fail: an unappliable mapping (e.g. unmatched atoms) drops the unindexed region. A
        # ConditionConflictError from transfer is not a ValueError, so contradictory conditioning
        # deliberately propagates and hard-fails rather than being silently dropped here.
        global_logger.warning(
            f"Could not apply the insertion mapping ({e}); the unindexed region will not be annotated."
        )
        # Discard any partial condition transfer -- re-derive the clean generated array before emitting.
        model_generated_atom_array = atom_array[has_index].copy()
        if not C_CRDGRP.has_annotation(model_generated_atom_array):
            C_CRDGRP.set_annotation(model_generated_atom_array, C_CRDGRP.default_annotation(model_generated_atom_array))
        return _set_empty_unindexed_annotations()

    # Fold directly-indexed coord conditioning into SA_CRDSRC so the per-group re-staging below
    # sees indexed and unindexed atoms of the same coordinate group together.
    model_generated_atom_array = _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    # Re-stage the conditioned-reference coords for internal-coord groups with a single per-group
    # superposition
    model_generated_atom_array = _restage_internal_coord_cwi_per_group(model_generated_atom_array, reference_coord)

    if has_3d_conditioning.any():
        expected_unindexed_count = int(has_3d_conditioning.sum())
        actual_unindexed_count = int(mapped_to_coordinate_conditioned_region.sum())
        if actual_unindexed_count != expected_unindexed_count:
            global_logger.warning(
                f"Model reproduced only {actual_unindexed_count} of {expected_unindexed_count} "
                "coord-conditioned motif atoms; invalidating the unindexed motif reference coords."
            )
            # Invalidate only the unindexed motif coords; leave any directly-indexed groups intact.
            cwi = SA_CWI.annotation(model_generated_atom_array)
            cwi[mapped_to_coordinate_conditioned_region] = np.nan
            SA_CWI.set_annotation(model_generated_atom_array, np.asarray(cwi, dtype=np.float32))

    if not mapped_to_unindexed_insertion.any():
        global_logger.warning("Did not map any unindexed regions to generated tokens.")

    return model_generated_atom_array


def _restage_internal_coord_cwi_per_group(generated_atom_array: AtomArray, reference_coord: np.ndarray) -> AtomArray:
    """Re-stage the conditioned-reference coords for each internal-coord group with one rigid superposition.

    Indexed and unindexed atoms of the same internal coordinate group are otherwise staged into
    the conditioned-reference coords in different frames — indexed atoms are fit to the generated structure by
    :py:func:`~rfproteina.transforms.utils.conditioning_utils.insert_internal_coord_conditioned_groups`,
    while unindexed atoms are staged in the raw input frame by
    :py:func:`annotate_unindexed_insertion`. Fitting the whole group's ``reference_coord``
    (the ``C_INTCRD`` internal coords, which share one frame per group and so carry correct
    relative geometry across indexed and unindexed atoms) onto the model-generated coords in one
    superposition puts the group in a common frame, which the per-coordinate-group RMSD metrics
    (e.g. ``design_motif_allatom_rmsd``) assume. Absolute (``KNOWN_ABSOLUTE_COORDS``) and
    unconditioned groups are left untouched.

    Args:
      generated_atom_array: Generated AtomArray with ``SA_CRDSRC`` / live ``C_CRDGRP`` populated.
      reference_coord: Per-atom internal-coord (``C_INTCRD``) reference of shape ``(n_atoms, 3)``;
        NaN for atoms without internal-coord conditioning.
    """
    crd_src = SA_CRDSRC.mask(generated_atom_array)
    groups = C_CRDGRP.annotation(generated_atom_array, default="generate")
    is_internal = crd_src & (groups != C_CRDGRP.UNKNOWN_COORDS) & (groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
    if not is_internal.any():
        return generated_atom_array

    cwi = SA_CWI.annotation(generated_atom_array)
    generated_coord = generated_atom_array.coord
    for group_id in np.unique(groups[is_internal]):
        group_mask = is_internal & (groups == group_id)
        reference = reference_coord[group_mask]
        generated = generated_coord[group_mask]
        if np.isnan(reference).any() or np.isnan(generated).any():
            global_logger.warning(
                f"Skipping conditioned-reference re-staging for internal coord group {group_id}: NaN coords present."
            )
            continue
        aligned, _ = struc.superimpose(generated, reference)
        cwi[group_mask] = aligned
    SA_CWI.set_annotation(generated_atom_array, np.asarray(cwi, dtype=np.float32))
    return generated_atom_array


def _merge_indexed_coord_conditioning_into_src(generated_atom_array: AtomArray) -> AtomArray:
    """Fold directly-indexed ``C_CRD``/``C_INTCRD`` atoms into ``SA_CRDSRC``.

    Two kinds of atoms qualify as "their coords came from input conditioning":
      (a) atoms remapped from unindexed in-context tokens — already flagged in
          ``SA_CRDSRC`` by :py:func:`annotate_unindexed_insertion`.
      (b) atoms whose ``C_CRD`` / ``C_INTCRD`` was set directly in the input
          (the indexed motif scaffold case) — NOT yet in ``SA_CRDSRC``.
    """
    # (b) — atoms with directly-indexed coordinate conditioning in the input
    indexed_coord_conditioned_mask = C_CRD.mask(generated_atom_array) | C_CRDGRP.is_internal(generated_atom_array)

    indexed_coord_conditioned_mask &= ~SA_CRDSRC.mask(generated_atom_array)

    # For the (b) atoms only, copy SA_SRC ids over so the source-region label
    # exists for every CRDSRC-flagged atom (the (a) atoms got their ids set
    # inside annotate_unindexed_insertion).
    src_crd = SA_CRDSRC.annotation(generated_atom_array)
    src_crd[indexed_coord_conditioned_mask] = SA_SRC.annotation(generated_atom_array)[indexed_coord_conditioned_mask]
    SA_CRDSRC.set_annotation(generated_atom_array, src_crd)

    # Token-level mask: True for every token whose atoms include at least one
    # SA_CRDSRC-flagged atom. Consumed by motif-RMSD / internal-coord metrics.
    generated_atom_array.set_annotation(
        "mapped_to_coordinate_conditioned_region_token_lvl",
        apply_and_spread_token_wise(generated_atom_array, SA_CRDSRC.mask(generated_atom_array), np.any),
    )

    return generated_atom_array
