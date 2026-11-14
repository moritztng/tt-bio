# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Contig-string subsetting + expandable-segment insertion for condition specs.

Implements §4.3 of ``design_yaml_condition_spec.md``: one comma-separated
contig string does both "subset the input structure" and "insert expandable
segments" in a single pass, replacing the two separate ``subset:``/
``expandable_segments:`` fields from the original draft.

Assembly is **chain-run-first, subset-then-expand** -- not a single
left-to-right walk over the token string -- because a wholly-expandable
chain-run's chain id (and default chain type) can depend on a preserved
segment token that appears *later* in the string within the same run (see
:py:func:`apply_contig`'s docstring for the worked example from the design
doc's feedback round).
"""

import numpy as np
from tt_bio._vendor.atomworks.enums import ChainType
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
    concatenate_atom_array_plus,
)
from tt_bio._vendor.atomworks.io.utils.chain import create_chain_id_generator
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
from tt_bio._vendor.atomworks.ml.utils.expandable_segment import insert_expandable_segment

# Snapshot value for `original_res_id` on expandable-segment atoms (§4.3.1). Chosen over a
# small magic number (e.g. -100): real author-supplied res_ids are occasionally negative but
# always small in magnitude, so int32-min is unreachable by any genuine value while still
# fitting the `res_id` column's int64 dtype with no overflow/promotion risk.
EXPANDABLE_SEGMENT_RES_ID_SENTINEL = -(2**31)

# The grammar -- tokens, chain runs and the checks that need no atoms -- lives in
# `contig_grammar`, which imports nothing but the standard library, so a caller that
# only wants to know what a contig asks for does not pay numpy, atomworks and torch
# (1.2 s and ~376 MB). Re-exported here because this module is where upstream's call
# sites look for them.
from .contig_grammar import (  # noqa: F401  (re-exported)
    ChainRun,
    ExpandableSegment,
    PreservedSegment,
    _parse_token,
    _validate_no_overlapping_preserved_segments,
    chain_runs,
)


def _partition_and_assign_chain_ids(contig: str) -> list[ChainRun]:
    """`chain_runs` plus the output chain ids, which is the half that needs atomworks."""
    runs = chain_runs(contig)

    claimed_chain_ids = {t.chain_id for run in runs for t in run if isinstance(t, PreservedSegment)}
    synthetic_chain_id_gen = create_chain_id_generator(unavailable_chain_ids=claimed_chain_ids)

    chain_runs_out = []
    seen_chain_ids: set[str] = set()
    for run_tokens in runs:
        preserved = [t for t in run_tokens if isinstance(t, PreservedSegment)]
        if preserved:
            run_chain_ids = {t.chain_id for t in preserved}
            if len(run_chain_ids) > 1:
                raise ValueError(
                    "An output chain mixes preserved segments from multiple original chains "
                    f"{sorted(run_chain_ids)} with no `/0` between them -- insert a `/0` to "
                    "split them into separate output chains."
                )
            chain_id = next(iter(run_chain_ids))
        else:
            chain_id = next(synthetic_chain_id_gen)
        if chain_id in seen_chain_ids:
            raise ValueError(f"Contig produces two separate output chains both named {chain_id!r}. ")
        seen_chain_ids.add(chain_id)
        chain_runs_out.append(ChainRun(tokens=run_tokens, chain_id=chain_id))
    return chain_runs_out


def _anchor_chain_type(atom_array_in: AtomArrayPlus, run: ChainRun) -> ChainType | None:
    preserved = next((t for t in run.tokens if isinstance(t, PreservedSegment)), None)
    if preserved is None:
        return None
    chain_mask = atom_array_in.chain_id == preserved.chain_id
    chain_types = np.unique(atom_array_in.chain_type[chain_mask])
    if len(chain_types) > 1:
        names = sorted(ChainType(int(c)).name for c in chain_types)
        raise ValueError(
            f"Input chain {preserved.chain_id!r} contains multiple ChainTypes {names} -- a "
            "contig run anchored on this chain can't unambiguously default an unsuffixed "
            "expandable segment's type. Split this chain into separately-named chains upstream."
        )
    return ChainType(int(chain_types[0]))


def _resolve_run_chain_type(run: ChainRun, anchor_chain_type: ChainType | None) -> ChainType:
    """Resolve the one ChainType shared by every expandable segment in a chain run.

    A run becomes a single output chain, so it has exactly one chain type: every
    explicit ``P``/``R``/``D`` suffix in the run must agree with the others, and with
    the run's anchor when a preserved segment supplies one. Resolving this per token
    instead meant a run with no preserved anchor -- e.g. ``/0,X:1P,Y:1R`` -- silently
    dropped ``Y``'s RNA request and emitted it as polypeptide.

    Raises:
      ValueError: If the run's expandable segments request more than one chain type,
        or if the requested type disagrees with the run's anchor.
    """
    requested = {
        tok.chain_type for tok in run.tokens if isinstance(tok, ExpandableSegment) and tok.chain_type is not None
    }
    if len(requested) > 1:
        names = sorted(chain_type.name for chain_type in requested)
        raise ValueError(
            f"Chain run requests conflicting chain types {names} for its expandable segments, but a "
            "run becomes a single output chain and can only have one. Separate them with `/0`."
        )
    if not requested:
        return anchor_chain_type if anchor_chain_type is not None else ChainType.POLYPEPTIDE_L

    chain_type = requested.pop()
    if anchor_chain_type is not None and chain_type != anchor_chain_type:
        raise ValueError(
            f"An expandable segment explicitly specifies chain type {chain_type.name}, which "
            f"disagrees with its run's anchor chain type {anchor_chain_type.name}."
        )
    return chain_type


# ---------------------------------------------------------------------------
# Per-run preserved atoms (pass 2 of 2, part a: "subset")
# ---------------------------------------------------------------------------


def _build_run_preserved_atoms(
    atom_array_in: AtomArrayPlus, run: ChainRun
) -> tuple[AtomArrayPlus | None, np.ndarray, dict[int, tuple[int, int]]]:
    """Concatenate this run's preserved-segment atoms (written order), renumbering
    ``res_id`` to a fresh contiguous per-run counter starting at 1.

    Returns:
      ``(preserved_atoms, original_res_id, token_offsets)``. ``token_offsets`` maps a
      preserved-segment token's index within ``run.tokens`` to its local
      ``(start, end)`` atom-index range within ``preserved_atoms``. ``preserved_atoms``
      is ``None`` if the run has no preserved-segment tokens.
    """
    pieces: list[AtomArrayPlus] = []
    original_res_id_pieces: list[np.ndarray] = []
    token_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    next_res_id = 1

    for tok_idx, tok in enumerate(run.tokens):
        if not isinstance(tok, PreservedSegment):
            continue
        chain_mask = atom_array_in.chain_id == tok.chain_id
        res_mask = (atom_array_in.res_id >= tok.start) & (atom_array_in.res_id <= tok.end)
        selected = chain_mask & res_mask
        if not selected.any():
            raise ValueError(
                f"Preserved segment '{tok.chain_id}{tok.start}-{tok.end}' matched no atoms in the input structure."
            )
        # A range running off the end of the chain is a typ
        chain_res_ids = atom_array_in.res_id[chain_mask]
        chain_min, chain_max = int(chain_res_ids.min()), int(chain_res_ids.max())
        if tok.start < chain_min or tok.end > chain_max:
            raise ValueError(
                f"Preserved segment '{tok.chain_id}{tok.start}-{tok.end}' runs outside chain "
                f"{tok.chain_id}, whose residues are {chain_min}-{chain_max}."
            )
        segment = as_atom_array_plus(atom_array_in[selected].copy())

        res_starts = get_residue_starts(segment, add_exclusive_stop=True)
        new_res_id = np.empty(segment.array_length(), dtype=np.int64)
        for i in range(len(res_starts) - 1):
            new_res_id[res_starts[i] : res_starts[i + 1]] = next_res_id + i
        segment.set_annotation("res_id", new_res_id)

        pieces.append(segment)
        original_res_id_pieces.append(atom_array_in.res_id[selected].astype(np.int64))
        token_offsets[tok_idx] = (cursor, cursor + segment.array_length())
        cursor += segment.array_length()
        next_res_id += len(res_starts) - 1

    if not pieces:
        return None, np.empty(0, dtype=np.int64), {}
    preserved_atoms = pieces[0] if len(pieces) == 1 else concatenate_atom_array_plus(pieces)
    original_res_id = np.concatenate(original_res_id_pieces)
    return preserved_atoms, original_res_id, token_offsets


def _next_preserved_token_idx(tokens: list, tok_idx: int) -> int | None:
    for i in range(tok_idx + 1, len(tokens)):
        if isinstance(tokens[i], PreservedSegment):
            return i
    return None


def _ensure_res_name_width(arr: AtomArrayPlus, min_width: int) -> None:
    current_width = arr.res_name.dtype.itemsize // 4
    if current_width < min_width:
        arr.set_annotation("res_name", arr.res_name.astype(f"<U{min_width}"))


def _process_run(
    atom_array_in: AtomArrayPlus,
    run: ChainRun,
    run_preserved_atoms: AtomArrayPlus | None,
    run_original_res_id: np.ndarray,
    token_offsets: dict[int, tuple[int, int]],
    assembled: AtomArrayPlus | None,
    original_res_id: np.ndarray,
) -> tuple[AtomArrayPlus, np.ndarray]:
    """Merge one chain-run's preserved atoms (if any) onto ``assembled``, then insert its
    expandable segments (§4.3, part b: "expand").

    Requires ``assembled`` to already be real (non-``None``) if ``run_preserved_atoms`` is
    also ``None`` -- ``insert_expandable_segment`` always needs a non-empty array to derive a
    new chain's sentinel schema from, even for ``new_chain=True``. :py:func:`apply_contig`
    defers any wholly-expandable run that would otherwise violate this until a later run
    supplies real preserved atoms, rather than bootstrapping from a throwaway copy of
    ``atom_array_in``.
    """
    # One chain type for the whole run -- see `_resolve_run_chain_type`.
    chain_type = _resolve_run_chain_type(run, _anchor_chain_type(atom_array_in, run))

    run_start = assembled.array_length() if assembled is not None else 0
    if run_preserved_atoms is not None:
        # Expandable-segment insertion adds its private UID annotation to `assembled`.
        # Preserve existing identities and mark untouched atoms with its missing-value sentinel.
        if (
            assembled is not None
            and "_expseg_uid" in assembled.get_annotation_categories()
            and "_expseg_uid" not in run_preserved_atoms.get_annotation_categories()
        ):
            run_preserved_atoms = run_preserved_atoms.copy()
            run_preserved_atoms.set_annotation(
                "_expseg_uid",
                np.full(
                    run_preserved_atoms.array_length(),
                    -1,
                    dtype=assembled.get_annotation("_expseg_uid").dtype,
                ),
            )
        assembled = (
            run_preserved_atoms
            if assembled is None
            else concatenate_atom_array_plus(
                [assembled, run_preserved_atoms],
                fill_missing_standard_annotations=True,
                on_annotation_mismatch_policy="raise",
            )
        )
        original_res_id = np.concatenate([original_res_id, run_original_res_id])
    run_tail = run_start + (run_preserved_atoms.array_length() if run_preserved_atoms is not None else 0)

    # Live offsets: global (not run-local) positions of each preserved token's
    # (start, end) atom range, updated in lockstep as sentinels are inserted.
    live_offsets = {idx: (run_start + s, run_start + e) for idx, (s, e) in token_offsets.items()}

    first_expandable_in_run = True
    for tok_idx, tok in enumerate(run.tokens):
        if not isinstance(tok, ExpandableSegment):
            continue
        next_idx = _next_preserved_token_idx(run.tokens, tok_idx)

        if next_idx is not None:
            atom_idx = live_offsets[next_idx][0]
            assembled = insert_expandable_segment(
                assembled, tok.seg_min, tok.seg_max, atom_idx=atom_idx, insert_before=True
            )
            insert_position = atom_idx
        elif live_offsets:
            atom_idx = run_tail - 1
            assembled = insert_expandable_segment(
                assembled, tok.seg_min, tok.seg_max, atom_idx=atom_idx, insert_before=False
            )
            insert_position = atom_idx + 1
        elif first_expandable_in_run:
            assembled = insert_expandable_segment(
                assembled,
                tok.seg_min,
                tok.seg_max,
                new_chain=True,
                new_chain_id=run.chain_id,
                new_chain_type=chain_type,
            )
            insert_position = run_tail
        else:
            assembled = insert_expandable_segment(
                assembled, tok.seg_min, tok.seg_max, atom_idx=run_tail - 1, insert_before=False
            )
            insert_position = run_tail

        first_expandable_in_run = False
        original_res_id = np.insert(original_res_id, insert_position, EXPANDABLE_SEGMENT_RES_ID_SENTINEL)
        live_offsets = {
            i: (s + 1 if s >= insert_position else s, e + 1 if e >= insert_position else e)
            for i, (s, e) in live_offsets.items()
        }
        run_tail += 1

        # `insert_expandable_segment` has no custom res_name parameter -- it always derives
        # `res_name = f"{seg_min}-{seg_max}"`. The sentinel's array index is exactly
        # `insert_position` (the value we already computed to make the call above), so no
        # lookup is needed to overwrite it with the user's chosen name.
        name = tok.name if tok.name is not None else f"{tok.seg_min}-{tok.seg_max}"
        _ensure_res_name_width(assembled, len(name))
        assembled.res_name[insert_position] = name

    return assembled, original_res_id


def _shift_chain_res_ids_to_start_at_one(assembled: AtomArrayPlus) -> None:
    """Uniformly offset each chain's ``res_id`` so its lowest value is 1.

    Handles the edge case where a leading expandable segment before res_id 1
    would simply get res_id 0, rather than shifting the rest of the chain.
    """
    res_id = assembled.res_id
    new_res_id = res_id.copy()
    for chain_id in np.unique(assembled.chain_id):
        chain_mask = assembled.chain_id == chain_id
        offset = 1 - int(res_id[chain_mask].min())
        if offset != 0:
            new_res_id[chain_mask] = res_id[chain_mask] + offset
    assembled.set_annotation("res_id", new_res_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_variable_length_segments(contig: str) -> bool:
    """Whether any expandable segment in ``contig`` has a length *range* rather than a fixed length."""
    return any(
        isinstance(tok, ExpandableSegment) and tok.seg_min != tok.seg_max
        for run in _partition_and_assign_chain_ids(contig)
        for tok in run.tokens
    )


def apply_contig(atom_array_in: AtomArrayPlus, contig: str) -> tuple[AtomArrayPlus, np.ndarray]:
    """Subset ``atom_array_in`` and insert expandable segments, per one contig string.

    Assembly happens chain-run-first, subset-then-expand (§4.3): tokens are
    first partitioned into chain-runs by splitting on ``/0``, each run's
    output chain id (and default chain type for unsuffixed expandable
    segments) is resolved from any preserved-segment anchor *before* anything
    is built, preserved-segment atoms are concatenated per run, and only then
    are expandable segments inserted per run. This ordering is required
    because a wholly-expandable run's chain id can depend on a preserved
    segment that appears *later* in the string within the same run -- e.g.
    ``"A1-10, /0, E1:1-20, B1-10"`` has two runs, ``[A1-10]`` and
    ``[E1:1-20, B1-10]`` (only one ``/0``), so ``E1`` ends up on chain ``B``
    even though it precedes ``B1-10`` in the string: a single left-to-right
    walk could not know that in time to assign ``E1``'s chain id.

    A run of leading, wholly-expandable chain-runs (no preserved segment at all, so nothing
    to build atoms from yet) is deferred until a later run supplies real preserved atoms,
    processed onto that structure, then moved back in front of it -- see
    :py:func:`_process_run` -- so the assembled chain order still matches the contig's
    declared order. Every contig must therefore have at least one preserved segment
    somewhere: a contig with none generates everything from scratch and has no reason to
    reference an ``input`` structure at all.

    Args:
      atom_array_in: Structure to subset (loaded per §4.2, before any renumbering).
      contig: Comma-separated contig string (see module docstring / §4.3).

    Returns:
      ``(assembled, original_res_id)`` -- the assembled structure, and a
      parallel per-atom array of pre-contig ``res_id`` values for preserved
      atoms (:py:data:`EXPANDABLE_SEGMENT_RES_ID_SENTINEL` for
      expandable-segment sentinel atoms). See §4.3.1: used to resolve
      ``conditions:`` selections against the structure's original numbering.

    Raises:
      ValueError: If the contig has no preserved segments at all.
    """
    chain_runs = _partition_and_assign_chain_ids(contig)
    if not any(isinstance(tok, PreservedSegment) for run in chain_runs for tok in run.tokens):
        raise ValueError(
            "Contig has no preserved segments -- condition_spec's `input` structure exists "
            "only to be referenced by preserved segments, so a contig that generates "
            "everything from scratch doesn't need an `input` file at all."
        )

    assembled: AtomArrayPlus | None = None
    original_res_id = np.empty(0, dtype=np.int64)
    pending_runs: list[tuple[ChainRun, np.ndarray, dict[int, tuple[int, int]]]] = []

    for run in chain_runs:
        # Build preserved atoms first: its per-token "matched no atoms" check gives a much
        # clearer error for a typo'd chain id/res_id than `_anchor_chain_type` indexing an
        # empty chain-id mask would.
        run_preserved_atoms, run_original_res_id, token_offsets = _build_run_preserved_atoms(atom_array_in, run)

        if assembled is None and run_preserved_atoms is None:
            # Wholly-expandable run with no preserved atoms of its own, and nothing built yet
            # to derive a new chain's sentinel schema from -- look ahead instead of bootstrapping
            # from a throwaway copy of atom_array_in.
            pending_runs.append((run, run_original_res_id, token_offsets))
            continue

        assembled, original_res_id = _process_run(
            atom_array_in, run, run_preserved_atoms, run_original_res_id, token_offsets, assembled, original_res_id
        )
        unblocked_run_len = assembled.array_length()
        for pending_run, pending_original_res_id, pending_token_offsets in pending_runs:
            assembled, original_res_id = _process_run(
                atom_array_in,
                pending_run,
                None,
                pending_original_res_id,
                pending_token_offsets,
                assembled,
                original_res_id,
            )
        if pending_runs:
            # Pending runs just got appended after the run that unblocked them -- move them
            # back in front of it so the assembled order matches the contig's declared order.
            reorder = np.concatenate(
                [np.arange(unblocked_run_len, assembled.array_length()), np.arange(unblocked_run_len)]
            )
            assembled = assembled[reorder]
            original_res_id = original_res_id[reorder]
        pending_runs = []

    assert assembled is not None  # guaranteed by the no-preserved-segments check above

    # Offset each chain so its res_id starts at 1 (a leading expandable sentinel otherwise sits
    # at res_id 0 -- see `_shift_chain_res_ids_to_start_at_one`).
    _shift_chain_res_ids_to_start_at_one(assembled)
    return assembled, original_res_id
