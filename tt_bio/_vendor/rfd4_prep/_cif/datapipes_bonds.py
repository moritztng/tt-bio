# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Atom-level bond utilities (bond order, aromaticity, distances).

Extracted from :py:mod:`~rfproteina.datapipes.pipeline` to keep that
module focused on transform composition.
"""

import itertools
import logging
from collections.abc import Iterator

import numpy as np
from tt_bio._vendor.atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER, MASKED, UNKNOWN_AA, UNKNOWN_DNA, UNKNOWN_RNA
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.ccd import get_polymerization_atoms
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_SEGMIN
from tt_bio._vendor.atomworks.ml.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.utils.expandable_segment import insert_expandable_segment, realize_expandable_segments
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from biotite.structure.bonds import BondType
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .conditions import C_IDXGRP
from .datapipes_schema_proteina import TokenType
from .encoding import GLYCAN_NAMES, NCAA_NAMES

logger = logging.getLogger(__name__)

_UNKNOWN_RES_BY_TOKEN_TYPE = {
    TokenType.PROTEIN: UNKNOWN_AA,
    TokenType.NCAA: UNKNOWN_AA,
    TokenType.RNA: UNKNOWN_RNA,
    TokenType.DNA: UNKNOWN_DNA,
}

_AROMATIC_BOND_TYPES = (
    BondType.AROMATIC_SINGLE,
    BondType.AROMATIC_DOUBLE,
    BondType.AROMATIC_TRIPLE,
    BondType.AROMATIC,
)

# Marks the residues :py:func:`_fill_res_id_gaps` adds, on its own copy. An input can arrive with
# expandable segments of its own already realised, so the segment annotations cannot tell them apart.
_GAP_FILLER = "_res_id_gap_filler"


def _build_symmetric_bond_matrix(
    atom_array: AtomArray, per_bond_values: np.ndarray, dtype: np.dtype, fill: float | bool | int = 0
) -> np.ndarray:
    """Scatter ``per_bond_values`` into a symmetric ``(n_atoms, n_atoms)`` matrix."""
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    n_atoms = atom_array.array_length()
    matrix = np.full((n_atoms, n_atoms), fill, dtype=dtype)
    matrix[atom1_idxs, atom2_idxs] = per_bond_values
    matrix[atom2_idxs, atom1_idxs] = per_bond_values
    return matrix


def _token_type_at(atom_array: AtomArray, atom_idxs: np.ndarray) -> np.ndarray:
    """Return token types at given atom positions, with atomization taking precedence."""
    # Use token type annotation, if it exists
    if "token_type" in atom_array.get_annotation_categories():
        return np.asarray(atom_array.token_type)[atom_idxs].astype(np.int16)
    ensure_annotations(
        atom_array,
        "is_standard_unknown_or_masked_aa",
        "is_standard_unknown_or_masked_dna",
        "is_standard_unknown_or_masked_rna",
    )
    types = np.full(len(atom_idxs), TokenType.UNKNOWN, dtype=np.int16)
    types[atom_array.get_annotation("is_standard_unknown_or_masked_dna")[atom_idxs]] = TokenType.DNA
    types[atom_array.get_annotation("is_standard_unknown_or_masked_rna")[atom_idxs]] = TokenType.RNA
    types[atom_array.get_annotation("is_standard_unknown_or_masked_aa")[atom_idxs]] = TokenType.PROTEIN
    types[np.isin(atom_array.res_name[atom_idxs], NCAA_NAMES)] = TokenType.NCAA
    types[np.isin(atom_array.res_name[atom_idxs], GLYCAN_NAMES)] = TokenType.GLYCAN
    types[atom_array.atomize[atom_idxs]] = TokenType.ATOMIZED
    return types


def get_polymerization_atom_names(res_name: str, token_type: int = TokenType.UNKNOWN) -> tuple[str, str]:
    """Return outgoing/incoming backbone atoms, with ``""`` for an absent side.

    If neither atom is recognized, use the unknown residue of the supplied protein/RNA/DNA
    token type. A recognized one-sided terminus is preserved without filling its absent side.
    """
    leaving, entering = get_polymerization_atoms(res_name)
    if leaving is None and entering is None:
        unknown = _UNKNOWN_RES_BY_TOKEN_TYPE.get(token_type)
        if unknown is not None:
            leaving, entering = get_polymerization_atoms(unknown)
    return leaving or "", entering or ""


def _link_atom_names(atom_array: AtomArray, atom_idxs: np.ndarray, token_type: np.ndarray) -> np.ndarray:
    """Per-atom ``[leaving, entering]`` names expected of that atom's residue."""
    res, types = atom_array.res_name[atom_idxs], token_type[atom_idxs]
    pairs = {(r, t): get_polymerization_atom_names(str(r), int(t)) for r, t in set(zip(res, types, strict=False))}
    return np.array([pairs[k] for k in zip(res, types, strict=False)]).reshape(-1, 2)


def get_polymerization_atom_masks(
    atom_array: AtomArray,
    atom_idxs: np.ndarray,
    token_type: np.ndarray | None = None,
    stands_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify leaving/entering polymerization atoms; expandable placeholders stand in for both."""
    if token_type is None:
        token_type = _token_type_at(atom_array, np.arange(len(atom_array)))
    if stands_in is None:
        stands_in = S_SEGMIN.mask(atom_array, default="generate")
    link = _link_atom_names(atom_array, atom_idxs, token_type)
    names, stand_in = atom_array.atom_name[atom_idxs], stands_in[atom_idxs]
    return (names == link[:, 0]) | stand_in, (names == link[:, 1]) | stand_in


def _inter_token_bond_edges(
    atom_array: AtomArray,
    atom_to_token: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return inter-token bond edges as a dict with keys ``tok1``, ``tok2``, ``bond_types``, ``atom1``, ``atom2``.

    Args:
        atom_array: Atom array with bond information.
        atom_to_token: Per-atom token ID (shape ``[n_atoms]``).
        atom_mask: If provided, only bonds where both atoms are ``True`` are kept.

    Returns:
        Dict of five arrays, each of length n_inter_token_bonds.
    """
    a1, a2, btypes = atom_array.bonds.as_array().T
    if atom_mask is not None:
        keep = atom_mask[a1] & atom_mask[a2]
        a1, a2, btypes = a1[keep], a2[keep], btypes[keep]
    t1 = atom_to_token[a1]
    t2 = atom_to_token[a2]
    inter = t1 != t2
    return {
        "tok1": t1[inter].astype(np.int32),
        "tok2": t2[inter].astype(np.int32),
        "bond_types": btypes[inter],
        "atom1": a1[inter],
        "atom2": a2[inter],
    }


def _compute_token_path_lengths(
    tok1: np.ndarray,
    tok2: np.ndarray,
    source_indices: np.ndarray,
    path_length_cutoff: int | None,
) -> np.ndarray:
    """Compute inter-token shortest-path distances via BFS.

    Pairs above the length cutoff get ``path_length_cutoff + 1``, and disconnected pairs get 0.

    Args:
        tok1: Source token indices for inter-token bond edges.
        tok2: Target token indices for inter-token bond edges.
        source_indices: Token positions to run BFS from.
        path_length_cutoff: Longest hop count reported exactly. Connected pairs beyond it are
            stored as ``path_length_cutoff + 1``. ``None`` keeps all distances.

    Returns:
        int16 array, shape ``[len(source_indices), len(source_indices)]``.
        0 = no path (or self); ``path_length_cutoff + 1`` = connected but beyond the cutoff.
    """
    n_sources = len(source_indices)
    dist = np.zeros((n_sources, n_sources), dtype=np.int16)
    if n_sources == 0 or len(tok1) == 0:
        return dist

    # Multi-source BFS via scipy csgraph (same hop counts as the old networkx loop, much faster).
    n_nodes = max(int(tok1.max()), int(tok2.max()), int(source_indices.max())) + 1
    rows = np.concatenate([tok1, tok2])
    cols = np.concatenate([tok2, tok1])
    adj = coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    d = shortest_path(adj, method="D", unweighted=True, directed=False, indices=source_indices)  # [n_sources, n_nodes]
    sub = d[:, source_indices]  # [n_sources, n_sources]

    connected = np.isfinite(sub) & (sub > 0)
    within_cutoff = connected if path_length_cutoff is None else connected & (sub <= path_length_cutoff)
    dist[within_cutoff] = sub[within_cutoff].astype(np.int16)
    if path_length_cutoff is not None:
        dist[connected & ~within_cutoff] = path_length_cutoff + 1
    return dist


def _closes_ring(edges: dict[str, np.ndarray], res_id_per_tok: np.ndarray, mol_per_tok: np.ndarray) -> np.ndarray:
    """Whether each bond closes a ring: joins two non-consecutive residues of the same molecule.

    NOTE: Fallible heuristic: only consulted during inference, when both endpoints are expandable-segment
    stand-ins (S_SEGMIN), so chemistry cannot orient the bond.
    """
    tok1, tok2 = edges["tok1"], edges["tok2"]
    same_mol = mol_per_tok[tok1] == mol_per_tok[tok2]
    return same_mol & (np.abs(res_id_per_tok[tok1] - res_id_per_tok[tok2]) > 1)


def _upstream_direction(
    atom_array: AtomArray, edges: dict[str, np.ndarray], idx_tok_starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bond ``(fwd, rev)`` masks: which endpoint of each inter-token bond is upstream."""
    # Chemistry: a bond runs from a residue's leaving atom into the next residue's entering atom.
    atom_type = _token_type_at(atom_array, np.arange(atom_array.array_length()))

    # Expandable-segment sentinels carry no backbone atoms, so they stand in for either role.
    stands_in = S_SEGMIN.mask(atom_array, default="generate")
    leaves1, enters1 = get_polymerization_atom_masks(atom_array, edges["atom1"], atom_type, stands_in)
    leaves2, enters2 = get_polymerization_atom_masks(atom_array, edges["atom2"], atom_type, stands_in)
    fwd, rev = leaves1 & enters2, leaves2 & enters1
    if not stands_in.any():
        return fwd, rev  # no expandable-segment stand-ins (all training, most inference): chemistry decides.

    # No chemistry (both endpoints stand-ins): fall back to chain order (lower token upstream),
    # except a ring-closing bond runs high->low, so flip the tie-break there.
    res_id_per_tok, mol_per_tok = atom_array.res_id[idx_tok_starts], atom_array.molecule_iid[idx_tok_starts]
    ring = _closes_ring(edges, res_id_per_tok, mol_per_tok)
    blind = stands_in[edges["atom1"]] & stands_in[edges["atom2"]]
    tok1_upstream = (edges["tok1"] < edges["tok2"]) ^ ring

    return (fwd & ~blind) | (blind & tok1_upstream), (rev & ~blind) | (blind & ~tok1_upstream)


def _iter_unbonded_implied_neighbours(atom_array: AtomArray, starts: np.ndarray) -> Iterator[tuple[int, int, int]]:
    """Yield ``(prev_residue, next_residue, res_id_gap)`` for adjacent same-chain, disconnected polymer residues."""
    groups = C_IDXGRP.annotation(atom_array, default="generate")[starts]
    chain_iids = atom_array.chain_iid[starts]
    res_ids = atom_array.res_id[starts].astype(np.int64)
    is_polymer = atom_array.is_polymer[starts]

    # Bond-graph components over residues, merged in place as breaks are yielded.
    atom_to_residue = np.searchsorted(starts, np.arange(atom_array.array_length()), side="right") - 1
    atom1, atom2, _ = atom_array.bonds.as_array().T
    n_residues = len(starts)
    adjacency = coo_matrix(
        (np.ones(len(atom1), dtype=np.int8), (atom_to_residue[atom1], atom_to_residue[atom2])),
        shape=(n_residues, n_residues),
    )
    _, component = connected_components(adjacency, directed=False)

    for group in np.unique(groups):
        for chain_iid in np.unique(chain_iids[groups == group]):
            members = np.where(is_polymer & (groups == group) & (chain_iids == chain_iid))[0]
            members = members[np.argsort(res_ids[members], kind="stable")]
            for prev_res, next_res in itertools.pairwise(members):
                if component[prev_res] == component[next_res]:
                    continue
                yield int(prev_res), int(next_res), max(int(res_ids[next_res] - res_ids[prev_res]), 1)
                component[component == component[next_res]] = component[prev_res]


def polymerisation_atom_idx(
    array: AtomArray, res_start: int, res_stop: int, token_type: int, *, exiting: bool
) -> int | None:
    """Index of the atom a residue polymerises through, or ``None`` if it is not present.

    A residue standing in for one whose atoms are not known yet -- a realised expandable segment
    stub, or the filler this module splices into a ``res_id`` gap -- carries a single placeholder
    atom, and polymerises through it in both directions.
    """
    if res_stop - res_start == 1 and array.res_name[res_start] == MASKED:
        return int(res_start)

    exiting_name, entering_name = get_polymerization_atom_names(str(array.res_name[res_start]), int(token_type))
    atom_name = exiting_name if exiting else entering_name
    span = np.arange(res_start, res_stop)
    match = span[array.atom_name[span] == atom_name]
    if len(match) != 1:
        logger.warning(
            f"no unique {'exiting' if exiting else 'entering'} polymerisation atom {atom_name!r} on "
            f"{array.chain_id[res_start]}/{array.res_name[res_start]}/{int(array.res_id[res_start])}: "
            f"matches {len(match)} of its atoms"
        )
        return None
    return int(match[0])


def bond_flanking_residues(array: AtomArray, prev_res: int, next_res: int, res_bounds: np.ndarray) -> bool:
    """Bond two residues through their polymerisation atoms. Returns whether a bond was added."""
    starts, stops = res_bounds[:-1], res_bounds[1:]
    token_type = _token_type_at(array.copy(), starts)
    left = polymerisation_atom_idx(array, starts[prev_res], stops[prev_res], token_type[prev_res], exiting=True)
    right = polymerisation_atom_idx(array, starts[next_res], stops[next_res], token_type[next_res], exiting=False)
    if left is None or right is None:
        return False
    array.bonds.add_bond(left, right, BondType.SINGLE)
    return True


def _bond_bridge_ends(filled: AtomArray, is_filler: np.ndarray) -> None:
    """Join each run of filler residues to its flanking residues through their linkage atoms.

    :py:func:`insert_expandable_segment` wires its sentinel in by the chain type's *standard*
    linkage atom names, which be wrong for certain nonstandard residues.
    """
    bounds = get_residue_starts(filled, add_exclusive_stop=True)
    starts, stops = bounds[:-1], bounds[1:]
    is_filler_res = is_filler[starts]
    token_type = _token_type_at(filled, starts)

    # Each maximal run of filler residues is one bridge, with a real residue on either side.
    edges = np.diff(np.concatenate([[0], is_filler_res.astype(np.int8), [0]]))
    for run_start, run_stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
        for real_res, filler_res, is_upstream in ((run_start - 1, run_start, True), (run_stop, run_stop - 1, False)):
            link_atom = polymerisation_atom_idx(
                filled, starts[real_res], stops[real_res], token_type[real_res], exiting=is_upstream
            )
            if link_atom is None:
                continue
            filler_atom = int(starts[filler_res])

            for partner in filled.bonds.get_bonds(filler_atom)[0]:
                if starts[real_res] <= partner < stops[real_res] and partner != link_atom:
                    filled.bonds.remove_bond(filler_atom, int(partner))
            filled.bonds.add_bond(filler_atom, link_atom, BondType.SINGLE)


def _fill_res_id_gaps(atom_array: AtomArray) -> tuple[AtomArray, np.ndarray]:
    """Realise each ``res_id`` gap of a polymer chain as that many residues, spliced into place.

    Args:
        atom_array: Array to fill. Not modified; a filled copy is returned.

    Returns:
        ``(filled, source_token_idxs)``: the filled array, and the token index that each of
        ``atom_array``\\ 's tokens occupies within it.
    """
    filled = as_atom_array_plus(atom_array.copy())
    assert not (
        S_SEGMIN.has_annotation(filled) and (S_SEGMIN.mask(filled) & (filled.res_name != MASKED)).any()
    ), "input still holds unrealised expandable segments; realising the gaps below would realise those too"
    n_input_tokens = len(get_token_starts(filled))
    res_starts = get_residue_starts(filled, add_exclusive_stop=True)
    starts, stops = res_starts[:-1], res_starts[1:]

    gaps = [
        (prev_res, next_res, int(gap) - 1)
        for prev_res, next_res, gap in _iter_unbonded_implied_neighbours(filled, starts)
        if gap > 1
    ]
    if not gaps:
        return filled, np.arange(n_input_tokens)

    # Right to left, so the atom indices of the gaps still to come stay valid
    n_inserted = 0
    for prev_res, next_res, n_fillers in sorted(gaps, reverse=True):
        if stops[prev_res] != starts[next_res]:
            # Something else sits between them in the array (a non-polymer residue on the same
            # chain, or a residue the input put in another IndexGroup), so a segment spliced in
            # here would bond to the wrong residues. Leave the gap unfilled.
            logger.warning(
                f"residues {prev_res} and {next_res} are consecutive by res_id but not in the array; "
                "leaving the implied gap between them unfilled for the purposes of bond featurization."
            )
            continue
        filled = insert_expandable_segment(
            filled, seg_min=n_fillers, seg_max=n_fillers, atom_idx=int(stops[prev_res]) - 1, insert_before=False
        )
        n_inserted += 1

    if not n_inserted:
        return filled, np.arange(n_input_tokens)

    # Flag the sentinels we placed, which are the only unrealized expandable segments in the array.
    filled.set_annotation(_GAP_FILLER, S_SEGMIN.mask(filled) & (filled.res_name != MASKED))

    filled = realize_expandable_segments(filled)
    is_filler = filled.get_annotation(_GAP_FILLER)
    _bond_bridge_ends(filled, is_filler)

    return filled, np.where(~is_filler[get_token_starts(filled)])[0]


def compute_token_bond_features(
    atom_array: AtomArray,
    path_length_cutoff: int | None = None,
    source_atom_array: AtomArray | None = None,
    source_token_idxs: np.ndarray | None = None,
    fill_res_id_gaps: bool = False,
) -> dict[str, np.ndarray]:
    """Compute token-level bond features for all tokens, including unindexed expansion copies.

    Features are computed in the direct block [n_direct x n_direct] from the bond
    graph of tokens without a source reference. Training copies borrow their source
    features within each IndexGroup; inference motifs are featurized directly.

    Path lengths are special: when ``source_atom_array`` and ``source_token_idxs`` are provided
    (the training path), the BFS is run on a wider bond graph from earlier in the pipeline, so
    that shortest paths through tokens dropped since then -- by the crop, or by
    :py:class:`RemoveTokensWithAnyNaNAtom` -- are preserved. The result is then subsetted to
    ``source_token_idxs``. Bond mask, order, aromaticity, and canonical ordering are computed on
    the surviving indexed block only (a direct bond between two surviving tokens survives). At
    inference there is no such graph, so ``fill_res_id_gaps`` synthesises one from the input's
    ``res_id`` spacing.

    Args:
        atom_array: Post-expansion atom array with ``insertion_source_idx`` annotation.
        path_length_cutoff: Longest hop count reported exactly. Connected pairs beyond it
            are encoded as cutoff + 1; disconnected pairs are zero. ``None`` keeps all distances.
        source_atom_array: A wider atom array from earlier in the pipeline. When provided
            together with ``source_token_idxs``, path lengths are computed here and
            subsetted, preserving paths through tokens dropped since.
        source_token_idxs: Token-level indices into ``source_atom_array`` of the tokens
            that survive in ``atom_array``, in order. Must be provided iff
            ``source_atom_array`` is provided.
        fill_res_id_gaps: Inference-only. Derive the source atom array by realising the
            ``res_id`` gaps of each IndexGroup as residues, standing in for the wider graph
            training gets from before the crop. See :py:func:`_fill_res_id_gaps`. Mutually
            exclusive with ``source_atom_array``.

    Returns:
        Dict with five ``[n_total, n_total]`` arrays. **Every one is blank across IndexGroup
        boundaries**: two groups have no known index offset between them, and a bond -- or a
        hop count, or an ordering sign -- would assert one.

        - ``"bond_mask"`` (bool): True where the input contains a direct bond.
        - ``"bond_order"`` (int16): Kekulé order (0=none, 1-3, 4=coord).
          Poly-poly pairs, cross-group pairs and the self-diagonal are set to 0.
        - ``"is_aromatic"`` (bool): True if any inter-token bond is aromatic.
          Poly-poly and cross-group pairs are False.
        - ``"token_bond_path_length"`` (int16): Shortest bond-graph hop count within
          IndexGroup. 0 = disconnected / cross-group / self. Paths above
          ``path_length_cutoff`` use cutoff + 1; disconnected pairs remain zero.
        - ``"canonical_polymer_ordering"`` (int8): ``+1`` at ``[downstream, upstream]`` of a
          backbone linkage (C-N / O3'-P) between same-type canonical polymer residues
          (protein/RNA/DNA), ``-1`` at the transpose, else 0.
    """
    assert (source_atom_array is None) == (
        source_token_idxs is None
    ), "source_atom_array and source_token_idxs must be provided together"
    assert not (
        fill_res_id_gaps and source_atom_array is not None
    ), "fill_res_id_gaps stands in for the wider graph; the two must not be combined"
    assert (
        "insertion_source_idx" in atom_array.get_annotation_categories()
    ), "atom_array must have insertion_source_idx annotation (set by UnindexFlaggedTokens)"

    if fill_res_id_gaps:
        source_atom_array, source_token_idxs = _fill_res_id_gaps(atom_array)

    token_starts = get_token_starts(atom_array)
    n_total = len(token_starts)

    # ── Partition based on whether tokens were copied from others (i.e. unindexed during trianing) ─
    src_per_tok = atom_array.insertion_source_idx[token_starts]  # [n_total]
    is_tok_with_src = src_per_tok != -1  # [n_total]
    n_direct = int((~is_tok_with_src).sum())
    assert source_token_idxs is None or len(source_token_idxs) == n_direct, (
        f"source_token_idxs names {len(source_token_idxs)} tokens but {n_direct} are directly featurized; "
        "a transform between the source array and here dropped tokens without re-indexing them"
    )

    # Positions (in 0..n_total-1) of the borrowers, and the source token (in 0..n_direct-1) each borrows from.
    toks_with_src = np.where(is_tok_with_src)[0]
    src_toks = src_per_tok[toks_with_src].astype(np.int32)

    # IndexGroup per token from the full (post-expansion) atom array.
    tok_groups = C_IDXGRP.annotation(atom_array, default="generate")[token_starts]

    # ── Inter-token bond edges filtered to directly-featurized atoms only ─────
    is_direct_atom = spread_token_wise(atom_array, ~is_tok_with_src)
    atom_to_token = spread_token_wise(atom_array, np.arange(n_total, dtype=np.int32))
    edges = _inter_token_bond_edges(atom_array, atom_to_token, atom_mask=is_direct_atom)

    # ── Bond mask [n_direct x n_direct] ───────────────────────────────────
    direct_bond_mask = np.zeros((n_direct, n_direct), dtype=bool)
    direct_bond_mask[edges["tok1"], edges["tok2"]] = True
    direct_bond_mask[edges["tok2"], edges["tok1"]] = True
    np.fill_diagonal(direct_bond_mask, False)

    # ── Bond orders [n_direct x n_direct] ──────────────────────────────────
    _lookup = np.ones(max(int(bt) for bt in BondType) + 1, dtype=np.int16)
    for bt, order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
        _lookup[int(bt)] = order
    _lookup[BondType.QUADRUPLE] = 1
    _lookup[BondType.COORDINATION] = 4
    _lookup[BondType.AROMATIC] = 1
    bond_orders = _lookup[edges["bond_types"]]

    _ORDER_SENTINEL = np.iinfo(np.int16).max
    _order_work = np.full((n_direct, n_direct), _ORDER_SENTINEL, dtype=np.int16)
    np.minimum.at(_order_work, (edges["tok1"], edges["tok2"]), bond_orders)
    np.minimum.at(_order_work, (edges["tok2"], edges["tok1"]), bond_orders)
    direct_bond_order = np.where(_order_work < _ORDER_SENTINEL, _order_work, np.int16(0))
    np.fill_diagonal(direct_bond_order, 0)

    # ── Aromaticity [n_direct x n_direct] ──────────────────────────────────
    _aromatic_ints = [int(bt) for bt in _AROMATIC_BOND_TYPES]
    is_aromatic_feat = np.isin(edges["bond_types"], _aromatic_ints).astype(np.int8)
    _arom_work = np.zeros((n_direct, n_direct), dtype=np.int8)
    np.maximum.at(_arom_work, (edges["tok1"], edges["tok2"]), is_aromatic_feat)
    np.maximum.at(_arom_work, (edges["tok2"], edges["tok1"]), is_aromatic_feat)
    direct_is_aromatic = _arom_work.astype(bool)
    np.fill_diagonal(direct_is_aromatic, False)

    # Polymer-polymer pair mask. Used both by canonical_polymer_ordering (which
    # wants the unmasked bond graph including peptide bonds) and by the
    # subsequent suppression of poly-poly bond features.
    direct_tok_starts = token_starts[~is_tok_with_src]
    is_poly = ~atom_array.atomize[direct_tok_starts]
    is_poly_poly = np.outer(is_poly, is_poly)

    # Same-IndexGroup pair mask. Two groups have unknown relative indices, so
    # no features derived from the bond graph can relate them
    direct_tok_groups = tok_groups[~is_tok_with_src]
    is_same_group = direct_tok_groups[:, None] == direct_tok_groups[None, :]

    # ── Canonical polymer ordering [n_direct x n_direct] ───────────────────
    # Computed BEFORE the bond-feature suppression below so this feature can
    # use the unmasked bond graph (peptide bonds visible). Same logic as #205.
    token_type = _token_type_at(atom_array, direct_tok_starts)
    token_type = np.where(token_type == TokenType.NCAA, TokenType.PROTEIN, token_type)
    is_canonical = np.isin(token_type, (TokenType.PROTEIN, TokenType.RNA, TokenType.DNA))
    same_type = token_type[:, None] == token_type[None, :]
    eligible = is_poly_poly & np.outer(is_canonical, is_canonical) & same_type & is_same_group & direct_bond_mask

    # Orient each inter-token bond: which endpoint is upstream (chemistry, chain-order tie-break).
    fwd, rev = _upstream_direction(atom_array, edges, direct_tok_starts)

    up_tok = np.concatenate([edges["tok1"][fwd], edges["tok2"][rev]])
    dn_tok = np.concatenate([edges["tok2"][fwd], edges["tok1"][rev]])
    direct_canonical_ordering = np.zeros((n_direct, n_direct), dtype=np.int8)
    direct_canonical_ordering[dn_tok, up_tok] = 1
    direct_canonical_ordering[up_tok, dn_tok] = -1
    direct_canonical_ordering[~eligible] = 0

    # ── Suppress backbone-linkage bond-feature signal ────────────────────────
    # Standard residue connectivity is already carried by token_bond_path_length /
    # canonical_polymer_ordering, so bond features are reserved for bonds outside the chain.
    suppress = np.zeros((n_direct, n_direct), dtype=bool)
    suppress[up_tok, dn_tok] = suppress[dn_tok, up_tok] = True
    suppress &= is_poly_poly
    suppress |= ~is_same_group
    direct_bond_mask[suppress] = False
    direct_bond_order[suppress] = 0
    direct_is_aromatic[suppress] = False

    # ── Path lengths [n_direct x n_direct] ─────────────────────────────────
    if source_atom_array is not None:
        # BFS only from the surviving tokens through the full source graph, so we never
        # allocate an [n_source x n_source] dense matrix.
        src_token_starts = get_token_starts(source_atom_array)
        n_source = len(src_token_starts)
        src_atom_to_token = spread_token_wise(source_atom_array, np.arange(n_source, dtype=np.int32))
        src_edges = _inter_token_bond_edges(source_atom_array, src_atom_to_token)
        direct_path_length = _compute_token_path_lengths(
            src_edges["tok1"],
            src_edges["tok2"],
            source_indices=source_token_idxs,
            path_length_cutoff=path_length_cutoff,
        )  # [n_direct, n_direct]
    else:
        # No wider source array: compute on the direct block itself.
        direct_path_length = _compute_token_path_lengths(
            edges["tok1"],
            edges["tok2"],
            source_indices=np.arange(n_direct),
            path_length_cutoff=path_length_cutoff,
        )

    # Mask out inter-group paths and self-paths
    direct_path_length[~is_same_group] = 0
    np.fill_diagonal(direct_path_length, 0)

    # ── Assemble [n_total x n_total] output per IndexGroup ───────────────────
    out_bond_mask = np.zeros((n_total, n_total), dtype=bool)
    out_bond_order = np.zeros((n_total, n_total), dtype=np.int16)
    out_is_aromatic = np.zeros((n_total, n_total), dtype=bool)
    out_path_length = np.zeros((n_total, n_total), dtype=np.int16)
    out_canonical = np.zeros((n_total, n_total), dtype=np.int8)

    direct_feats = (
        direct_bond_mask,
        direct_bond_order,
        direct_is_aromatic,
        direct_path_length,
        direct_canonical_ordering,
    )
    out_feats = (out_bond_mask, out_bond_order, out_is_aromatic, out_path_length, out_canonical)

    for direct_f, out_f in zip(direct_feats, out_feats, strict=False):
        out_f[:n_direct, :n_direct] = direct_f

    tok_with_src_groups = tok_groups[toks_with_src]
    for grp in np.unique(tok_with_src_groups):
        grp_mask = tok_with_src_groups == grp
        grp_toks = toks_with_src[grp_mask]
        grp_srcs = src_toks[grp_mask]
        for direct_f, out_f in zip(direct_feats, out_feats, strict=False):
            out_f[np.ix_(grp_toks, grp_toks)] = direct_f[np.ix_(grp_srcs, grp_srcs)]

    return {
        "bond_mask": out_bond_mask,
        "bond_order": out_bond_order,
        "is_aromatic": out_is_aromatic,
        "token_bond_path_length": out_path_length,
        "canonical_polymer_ordering": out_canonical,
    }


def get_bond_order_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get the bond order matrix using Kekulé assignments for aromatic bonds.

    Values: 0=no bond, 1=single, 2=double, 3=triple, 4=coordination.
    Aromatic bonds get their Kekulé order (AROMATIC_SINGLE->1, AROMATIC_DOUBLE->2, AROMATIC_TRIPLE->3).
    QUADRUPLE and generic AROMATIC (no Kekulé info) are mapped to single (1).
    """
    _, _, bond_types = atom_array.bonds.as_array().T

    # Build lookup from BIOTITE_BOND_TYPE_TO_BOND_ORDER constant (sized to the biotite enum)
    lookup = np.ones(max(int(bt) for bt in BondType) + 1, dtype=np.int16)  # default: unknown -> single
    for bt, order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
        lookup[int(bt)] = order
    lookup[BondType.QUADRUPLE] = 1  # QUADRUPLE -> single
    lookup[BondType.COORDINATION] = 4  # COORDINATION -> own class
    lookup[BondType.AROMATIC] = 1  # generic AROMATIC (no Kekulé info) -> single

    bond_orders = lookup[bond_types]
    assert np.all(bond_orders >= 0) and np.all(bond_orders <= 4), "Invalid bond order detected"

    return _build_symmetric_bond_matrix(atom_array, bond_orders, dtype=np.int16, fill=0)


def get_is_aromatic_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get boolean aromaticity matrix for an atom array.

    A bond is aromatic if its biotite BondType is any of AROMATIC_{SINGLE,DOUBLE,TRIPLE}
    or the generic AROMATIC.
    """
    _, _, bond_types = atom_array.bonds.as_array().T
    is_aromatic = np.isin(bond_types, [int(bt) for bt in _AROMATIC_BOND_TYPES])
    return _build_symmetric_bond_matrix(atom_array, is_aromatic, dtype=bool, fill=False)


def get_bond_distances(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond distance (adjacency) list as a 1D array."""
    coords = atom_array.coord
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    return np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)


def get_bond_distance_matrix(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond adjacency matrix with bond distances as values."""
    return _build_symmetric_bond_matrix(atom_array, get_bond_distances(atom_array), dtype=float, fill=np.inf)
