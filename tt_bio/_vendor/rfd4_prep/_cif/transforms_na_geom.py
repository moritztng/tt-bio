# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
from typing import Any

import numpy as np
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AnnotationList2D, as_atom_array_plus
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray

from .conditions import C_BP, C_NA_SS_STATE, BasePairEnum
from .transforms_utils_conditioning_utils import sample_island_tokens
from .transforms_utils_na_geom_utils import annotate_na_ss, annotate_na_ss_from_data_specification


def sparse_pair_and_state_from_conditions(
    atom_array: AtomArray,
    token_starts: np.ndarray | None = None,
) -> tuple[AnnotationList2D, np.ndarray]:
    """Read sparse pair edges plus 1D state values from canonical conditions.

    Args:
        atom_array: AtomArray carrying canonical NA-SS conditions.
        token_starts: Optional token-start indices for ``atom_array``.

    Returns:
        pair_annotation: sparse pair-only annotation in local token coordinates.
        state_values: [n_tokens] int, ``BasePairEnum`` values per token.
    """
    atom_array = as_atom_array_plus(atom_array)
    if token_starts is None:
        token_starts = get_token_starts(atom_array)

    n_tokens = len(token_starts)
    state_values = C_NA_SS_STATE.annotation(atom_array, default="raise")[token_starts].astype(np.int16)

    if C_BP.full_name in atom_array.get_annotation_categories(n_body=2):
        atom_to_token = spread_token_wise(atom_array, np.arange(n_tokens, dtype=np.int32))
        atom_pair_annotation = C_BP.annotation(atom_array, default="raise")
        token_pairs = atom_to_token[atom_pair_annotation.pairs]
        valid = np.all(token_pairs >= 0, axis=1)
        pair_inds_arr = token_pairs[valid].astype(np.int32)
        pair_values = np.asarray(atom_pair_annotation.values)[valid]
    else:
        pair_inds_arr = np.empty((0, 2), dtype=np.int32)
        pair_values = np.empty((0,), dtype=np.int64)

    pair_annotation = AnnotationList2D(
        n_atoms=n_tokens,
        pairs=pair_inds_arr,
        values=pair_values,
    )
    return pair_annotation, state_values


def _set_canonical_na_ss_conditions(
    atom_array: AtomArray,
    *,
    token_starts: np.ndarray,
    pair_annotation: AnnotationList2D,
    state_values: np.ndarray,
) -> AtomArray:
    """Write canonical NA-SS conditions from token-level data.

    Notes:
        ``C_BP`` is stored sparsely on token-start atom pairs only, not on all
        atom-atom pairs spanning two paired residues. ``C_NA_SS_STATE`` is
        still written densely to all atoms via token spreading.
    """
    atom_level_state = spread_token_wise(atom_array, state_values)
    C_NA_SS_STATE.set_annotation(atom_array, atom_level_state.astype(np.int16))

    atom_pairs = (
        token_starts[pair_annotation.pairs] if len(pair_annotation.pairs) > 0 else np.empty((0, 2), dtype=np.int32)
    )
    C_BP.set_annotation(
        atom_array,
        pairs=atom_pairs.astype(np.int16),
        values=pair_annotation.values,
    )
    return atom_array


def _sample_where_to_show_ss(
    n_tokens: int,
    *,
    is_nucleic_ss_example: bool = True,
    give_partial_feats: bool = True,
    partner_sym_map: dict[int, list[int]] | None = None,
    nucleic_ss_min_shown: float = 0.2,
    nucleic_ss_max_shown: float = 1.0,
    n_islands_min: int = 1,
    n_islands_max: int = 6,
) -> np.ndarray:
    """Sample which token rows/columns of NA secondary structure to reveal.

    The returned mask is symmetric with respect to annotated partners so that if one
    side of a base pair is shown, the paired token is shown as well.

    Args:
        n_tokens: Number of token positions in the example.
        is_nucleic_ss_example: Whether this example should receive NA-SS conditioning.
        give_partial_feats: Whether to reveal only a subset of the annotated structure.
        partner_sym_map: Mapping from token index to partner token indices used to
            preserve pair symmetry in the shown mask.
        nucleic_ss_min_shown: Minimum fraction of tokens to reveal when sampling
            a partial mask.
        nucleic_ss_max_shown: Maximum fraction of tokens to reveal when sampling
            a partial mask.
        n_islands_min: Minimum number of contiguous revealed islands.
        n_islands_max: Maximum number of contiguous revealed islands.

    Returns:
        Boolean token-level mask indicating which NA-SS annotations are shown.
    """
    if not is_nucleic_ss_example:
        return np.zeros((n_tokens,), dtype=bool)

    if not give_partial_feats:
        return np.ones((n_tokens,), dtype=bool)

    frac_shown = nucleic_ss_min_shown + (nucleic_ss_max_shown - nucleic_ss_min_shown) * np.random.rand()
    frac_shown = float(np.clip(frac_shown, 0.0, 1.0))
    max_length = int(np.ceil(frac_shown * n_tokens))
    if max_length <= 0:
        return np.zeros((n_tokens,), dtype=bool)

    island_len_min = max(1, int(frac_shown * n_tokens // max(int(n_islands_max), 1)))
    island_len_max = max(1, int(frac_shown * n_tokens // max(int(n_islands_min), 1)))
    island_len_min = min(island_len_min, n_tokens)
    island_len_max = min(island_len_max, n_tokens)
    island_len_max = max(island_len_max, island_len_min)

    token_mask_to_show = sample_island_tokens(
        n_tokens,
        island_len_min=island_len_min,
        island_len_max=island_len_max,
        n_islands_min=n_islands_min,
        n_islands_max=n_islands_max,
        max_length=max_length,
    )

    partner_mask_to_show = np.ones_like(token_mask_to_show)
    for token_i, partner_ind_list in (partner_sym_map or {}).items():
        for partner_ind in partner_ind_list:
            partner_mask_to_show[partner_ind] = token_mask_to_show[token_i]

    return token_mask_to_show & partner_mask_to_show


def apply_training_na_ss_condition(
    atom_array: AtomArray,
    *,
    p_show_partial_feats: float = 0.7,
    nucleic_ss_min_shown: float = 0.2,
    nucleic_ss_max_shown: float = 1.0,
    n_islands_min: int = 1,
    n_islands_max: int = 6,
    p_canonical_bp_filter: float = 0.0,
    NA_only: bool = False,
    planar_only: bool = True,
) -> AtomArray:
    """Apply task-driven NA secondary-structure conditioning during training.

    Args:
        atom_array: Structure to annotate for training-time NA-SS conditioning.
        p_show_partial_feats: Probability of revealing only a subset of the
            annotated secondary structure.
        nucleic_ss_min_shown: Minimum fraction of annotated tokens to reveal when
            sampling partial features.
        nucleic_ss_max_shown: Maximum fraction of annotated tokens to reveal when
            sampling partial features.
        n_islands_min: Minimum number of contiguous revealed islands.
        n_islands_max: Maximum number of contiguous revealed islands.
        p_canonical_bp_filter: Probability of filtering to canonical base pairs
            during geometry annotation.
        NA_only: Whether geometry annotation should only consider nucleic residues.
        planar_only: Whether geometry annotation should require planar residues.

    Returns:
        AtomArray with partially shown canonical NA-SS conditions.
    """
    token_starts = get_token_starts(atom_array)
    n_tokens = len(token_starts)

    atom_array = annotate_na_ss(
        atom_array,
        NA_only=NA_only,
        planar_only=planar_only,
        p_canonical_bp_filter=p_canonical_bp_filter,
    )

    pair_annotation, state_values = sparse_pair_and_state_from_conditions(
        atom_array,
        token_starts=token_starts,
    )
    partner_sym_map = {i: [] for i in range(n_tokens)}
    for i, j in np.asarray(pair_annotation.pairs, dtype=np.int32):
        partner_sym_map[int(i)].append(int(j))

    give_partial_feats = bool(np.random.rand() < p_show_partial_feats)
    token_mask_to_show = _sample_where_to_show_ss(
        n_tokens,
        is_nucleic_ss_example=True,
        give_partial_feats=give_partial_feats,
        partner_sym_map=partner_sym_map,
        nucleic_ss_min_shown=nucleic_ss_min_shown,
        nucleic_ss_max_shown=nucleic_ss_max_shown,
        n_islands_min=n_islands_min,
        n_islands_max=n_islands_max,
    )

    filtered_pair_mask = (
        token_mask_to_show[pair_annotation.pairs[:, 0]] & token_mask_to_show[pair_annotation.pairs[:, 1]]
        if len(pair_annotation.pairs) > 0
        else np.zeros((0,), dtype=bool)
    )
    filtered_pair_annotation = AnnotationList2D(
        n_atoms=n_tokens,
        pairs=pair_annotation.pairs[filtered_pair_mask] if len(pair_annotation.pairs) > 0 else pair_annotation.pairs,
        values=pair_annotation.values[filtered_pair_mask]
        if len(pair_annotation.values) > 0
        else pair_annotation.values,
    )
    state_values = np.asarray(state_values, dtype=np.int16).copy()
    state_values[~token_mask_to_show] = BasePairEnum.UNSPECIFIED.value
    atom_array = _set_canonical_na_ss_conditions(
        atom_array,
        token_starts=token_starts,
        pair_annotation=filtered_pair_annotation,
        state_values=state_values,
    )
    return atom_array


class AnnotateNucleicAcidSecondaryStructure(Transform):
    """Materialize canonical NA-SS conditions from the inference/validation specification.

    Training-time NA-SS conditioning now lives in ``NucleicSecondaryStructureTask``.
    When used with ``is_inference=False``, this transform is a no-op for backward
    compatibility.
    """

    def __init__(
        self,
        is_inference: bool,
        p_is_nucleic_ss_example: float = 0.3,
        p_show_partial_feats: float = 0.7,
        nucleic_ss_min_shown: float = 0.2,
        nucleic_ss_max_shown: float = 1.0,
        n_islands_min: int = 1,
        n_islands_max: int = 6,
        p_canonical_bp_filter: float = 0.0,
        NA_only: bool = False,
        planar_only: bool = True,
    ):
        self.is_inference = is_inference
        self.p_is_nucleic_ss_example = p_is_nucleic_ss_example
        self.p_show_partial_feats = p_show_partial_feats
        self.nucleic_ss_min_shown = nucleic_ss_min_shown
        self.nucleic_ss_max_shown = nucleic_ss_max_shown
        self.n_islands_min = n_islands_min
        self.n_islands_max = n_islands_max
        self.p_canonical_bp_filter = p_canonical_bp_filter
        self.NA_only = NA_only
        self.planar_only = planar_only

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if not self.is_inference:
            data["atom_array"] = atom_array
            return data
        else:
            atom_array = annotate_na_ss_from_data_specification(data)

        data["atom_array"] = atom_array
        return data


class FeaturizeNucleicAcidSecondaryStructure(Transform):
    """Ensure canonical NA-SS conditions are present on the AtomArray."""

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array = as_atom_array_plus(atom_array)
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)
        if (
            C_BP.full_name not in atom_array.get_annotation_categories(n_body=2)
            or C_NA_SS_STATE.full_name not in atom_array.get_annotation_categories()
        ):
            pair_annotation = AnnotationList2D(
                n_atoms=n_tokens,
                pairs=np.empty((0, 2), dtype=np.int32),
                values=np.asarray(C_BP.default_annotation(atom_array).values, dtype=np.int64),
            )
            state_values = C_NA_SS_STATE.default_annotation(atom_array)[token_starts].astype(np.int16)
            atom_array = _set_canonical_na_ss_conditions(
                atom_array,
                token_starts=token_starts,
                pair_annotation=pair_annotation,
                state_values=state_values,
            )

        data.setdefault("log_dict", {})
        data["atom_array"] = atom_array
        return data
