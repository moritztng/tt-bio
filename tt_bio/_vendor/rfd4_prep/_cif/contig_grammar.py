# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""The contig grammar alone: tokens, chain runs, and the checks that need no structure.

Split out of ``inference_utils_contig_spec`` so that reading a contig costs nothing but
the standard library. That module assembles atoms and therefore imports numpy and
atomworks, which import torch: 1.2 s and about 376 MB of RSS, paid by anything that only
wanted to know what a contig asks for. Two callers want exactly that and no atoms --
``tt_bio.size_limits.scan_rfd4_total``, which refuses an oversized spec before a device
opens, and JapanFold's web process, which refuses one at submit, and whose own
requirements say the API imports only stdlib-only engine code.

Nothing here is a second copy of the rules: ``_partition_and_assign_chain_ids`` in the
module next door is this module's :func:`chain_runs` plus the chain-id assignment, which
is the only part that needs atomworks.

The two atomworks values the grammar does refer to -- the chain-type suffix enum and the
reserved ``MASKED`` name -- are looked up lazily, so a contig that uses neither (every
binder and scaffold contig does not) never reaches them.
"""

import itertools
from dataclasses import dataclass

_SUFFIX_LETTERS = "PRD"


def _chain_type(letter: str):
    """``ChainType`` for a contig token's explicit chain-type suffix."""
    from tt_bio._vendor.atomworks.enums import ChainType

    return {"P": ChainType.POLYPEPTIDE_L, "R": ChainType.RNA, "D": ChainType.DNA}[letter]


def _masked() -> str:
    from tt_bio._vendor.atomworks.constants import MASKED

    return MASKED


@dataclass(frozen=True)
class PreservedSegment:
    """A contig token referencing a range of the input structure, kept verbatim."""

    chain_id: str
    start: int
    end: int


@dataclass(frozen=True)
class ExpandableSegment:
    """A contig token inserting a collapsed expandable-segment sentinel."""

    name: str | None
    seg_min: int
    seg_max: int
    chain_type: object | None  # explicit suffix; None means "use the run's default"


@dataclass
class ChainRun:
    """One output chain: a contiguous run of tokens between two chainbreaks."""

    tokens: list
    chain_id: str


# ---------------------------------------------------------------------------
# Token parsing
# ---------------------------------------------------------------------------

_LENGTH_RANGE_CHARS = set("0123456789-")


def _parse_length_and_suffix(body: str, token: str) -> tuple[int, int, object | None]:
    chain_type = None
    if body and body[-1] in _SUFFIX_LETTERS:
        chain_type = _chain_type(body[-1])
        body = body[:-1]
    if not body or not set(body) <= _LENGTH_RANGE_CHARS:
        raise ValueError(f"Malformed expandable-segment token: {token!r}")
    parts = body.split("-")
    if len(parts) == 1:
        seg_min = seg_max = int(parts[0])
    elif len(parts) == 2:
        seg_min, seg_max = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Malformed expandable-segment token: {token!r}")
    if seg_max < seg_min:
        raise ValueError(f"Expandable segment {token!r} has end < start ({seg_max} < {seg_min}).")
    return seg_min, seg_max, chain_type


def _parse_preserved_segment(token: str) -> PreservedSegment:
    chain_chars = 0
    for ch in token:
        if not ch.isalpha():
            break
        chain_chars += 1
    if chain_chars == 0:
        raise ValueError(f"Malformed preserved-segment token: {token!r}")
    chain_id, rest = token[:chain_chars], token[chain_chars:]
    parts = rest.split("-")
    if len(parts) == 1 and parts[0].isdigit():
        start = end = int(parts[0])
    elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        start, end = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Malformed preserved-segment token: {token!r}")
    if end < start:
        raise ValueError(f"Preserved segment {token!r} has end < start ({end} < {start}).")
    return PreservedSegment(chain_id=chain_id, start=start, end=end)


def _parse_token(token: str):
    """Classify and parse one contig token, unambiguously by shape.

    Returns ``"chainbreak"`` for ``/0``, otherwise a :py:class:`PreservedSegment`
    or :py:class:`ExpandableSegment`.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty contig token (check for stray/trailing commas).")
    if token == "/0":
        return "chainbreak"
    if ":" in token:
        name, _, body = token.partition(":")
        if not name:
            raise ValueError(f"Malformed named expandable-segment token (empty name): {token!r}")
        if name == _masked():
            raise ValueError(
                f"Expandable segment name cannot be {name!r} -- that string is reserved "
                "internally to mark an already-realized segment sentinel."
            )
        seg_min, seg_max, chain_type = _parse_length_and_suffix(body, token)
        return ExpandableSegment(name=name, seg_min=seg_min, seg_max=seg_max, chain_type=chain_type)
    if token[0].isalpha():
        return _parse_preserved_segment(token)
    seg_min, seg_max, chain_type = _parse_length_and_suffix(token, token)
    return ExpandableSegment(name=None, seg_min=seg_min, seg_max=seg_max, chain_type=chain_type)


def _validate_no_overlapping_preserved_segments(runs: list[list]) -> None:
    """Reject preserved segments on the same original chain that reference overlapping residues.

    An overlap would otherwise be silently duplicated into the assembled structure
    (the shared residues appear twice, contiguously renumbered), which is essentially
    always a typo rather than something intentional.
    """
    by_chain: dict[str, list[PreservedSegment]] = {}
    for run in runs:
        for tok in run:
            if isinstance(tok, PreservedSegment):
                by_chain.setdefault(tok.chain_id, []).append(tok)

    for chain_id, segments in by_chain.items():
        segments = sorted(segments, key=lambda s: s.start)
        for prev, curr in itertools.pairwise(segments):
            if curr.start <= prev.end:
                raise ValueError(
                    f"Preserved segments on chain {chain_id!r} overlap: "
                    f"{chain_id}{prev.start}-{prev.end} and {chain_id}{curr.start}-{curr.end} "
                    "both reference some of the same residues."
                )


def chain_runs(contig: str) -> list[list]:
    """The contig's tokens, split into one list per output chain, fully checked.

    Everything ``_partition_and_assign_chain_ids`` does except assigning the chain ids,
    which is the only part that needs an atom array's vocabulary. A caller that wants to
    know what a contig asks for -- how much is kept, how much is designed, which chains
    it reads -- wants this and stops here.
    """
    tokens = [_parse_token(t) for t in contig.split(",")]

    names = [t.name for t in tokens if isinstance(t, ExpandableSegment) and t.name is not None]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate expandable-segment name(s) in contig: {dupes}")

    runs: list[list] = [[]]
    for tok in tokens:
        if tok == "chainbreak":
            runs.append([])
        else:
            runs[-1].append(tok)
    runs = [r for r in runs if r]
    if not runs:
        raise ValueError("Contig string produced no chain runs (only chainbreaks/empty tokens).")

    _validate_no_overlapping_preserved_segments(runs)
    return runs
