"""RFD3 InputSpecification front-end: contig-string + InputSelection mini-language.

This is the parser front door every RFD3 design mode shares (protein binder,
NA binder, small-mol binder, enzyme, motif scaffolding, symmetry). It turns a
user's JSON/YAML InputSpecification into a structured representation that the
(host-side) featurizer will expand into the `f` feature dict the on-device
TokenInitializer + DiffusionModule consume.

Grammar (verified against RosettaCommons/foundry models/rfd3/docs/input.md,
production branch, 2026-07-23 — the source of truth, not memory of p1):

Contig string (comma-separated components):
  - ``/0``                          chain break (no peptide bond across it)
  - ``A1-80``                       residues 1..80 of chain A, taken from the input
                                    structure (fixed coord + fixed seq by default)
  - ``70``                          a designed region of exactly 70 residues
  - ``60-80``                       a designed region of random length in [60, 80]
  - ``A203``                        a single indexed residue (A171-A202 are dropped,
                                    a bond is created across the gap to A203)
  - ``A11-12``                      unindexed components tied together (0 offset)
  - ``A11,0,A12`` / ``A11,3,A12``   documented, but VERIFIED BROKEN in the real
                                    reference itself (`rc-foundry==0.2.0`, both
                                    input dialects — p22, see
                                    ``rfd3_featurize._plan_unindexed_tokens``);
                                    parsed here as ``UnindexedOffset`` for
                                    completeness, the featurizer intentionally
                                    still raises NotImplementedError rather
                                    than match a reference crash

InputSelection (boolean | contig string | dict):
  - ``True`` / ``False``
  - a contig string (see above)
  - a dict ``{ "<contig>": "<atom spec>" }`` (``<contig>`` may be a single
    residue or a range, e.g. ``"A2-10"`` — expanded per-residue, same value
    applied to each) where ``<atom spec>`` is one of:
      ``ALL``  all atoms in the residue(s)
      ``TIP``  the common tip atom for the residue (per upstream constants.py)
      ``BKBN`` backbone atoms (N, CA, C, O)
      ``N,CA,C,O,CB``  an explicit comma-joined atom-name list
      ``""``           no atoms (unfix)
  - dict-form ``unindex`` is a SEPARATE mechanism from `select_fixed_atoms`
    (p22): its own dict value additionally subsets which real atoms enter the
    unindexed token, composed as an intersection with any `select_fixed_atoms`
    restriction on the same residue — see
    ``rfd3_featurize._unindex_dict_atom_names``.

Only the grammar lives here. Resolving ``TIP``/``BKBN`` to concrete atom names,
building atom14, RASA, hbond, hotspot, ori_token, symmetry and the rest of the
`f` dict is the featurizer's job (N6, in progress) — it needs atomworks
structure-I/O conventions and is parity-gated against a real design `f` capture,
so it is NOT faked here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence, Union


# --- contig components ------------------------------------------------------
@dataclass(frozen=True)
class ChainBreak:
    """A ``/0`` chain break between two contig components."""

    def __repr__(self) -> str:
        return "ChainBreak(/0)"


@dataclass(frozen=True)
class Indexed:
    """Residues taken from the input structure: ``<chain><start>-<end>`` or
    ``<chain><num>``. These have fixed coordinates and (by default) fixed
    sequence."""

    chain: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Indexed({self.chain}{self.start}-{self.end})"


@dataclass(frozen=True)
class Designed:
    """A designed region of fixed length (a bare integer)."""

    length: int

    def __repr__(self) -> str:
        return f"Designed({self.length})"


@dataclass(frozen=True)
class DesignedRange:
    """A designed region of random length within [min, max] (a bare ``min-max``)."""

    lo: int
    hi: int

    def __repr__(self) -> str:
        return f"DesignedRange({self.lo}-{self.hi})"


@dataclass(frozen=True)
class UnindexedOffset:
    """A sequence offset between two unindexed indexed components: the ``N`` in
    ``A11,N,A12``. ``0`` ties the components together (equivalent to ``A11-12``)."""

    offset: int

    def __repr__(self) -> str:
        return f"UnindexedOffset({self.offset})"


ContigComponent = Union[ChainBreak, Indexed, Designed, DesignedRange, UnindexedOffset]


# --- InputSelection atom specs ---------------------------------------------
# Shorthand atom sets (per input.md / upstream constants.py). BKBN is fixed; TIP
# is residue-dependent and resolved by the featurizer against the residue's
# CCD entry, so it is kept as a symbolic marker here.
ATOM_SHORTHANDS = {"ALL", "TIP", "BKBN"}


@dataclass(frozen=True)
class AtomSelection:
    """A parsed InputSelection dict value: either a symbolic shorthand (ALL/TIP/
    BKBN), an explicit atom-name set, or the empty set (unfix)."""

    shorthand: str | None = None  # one of ALL/TIP/BKBN, or None
    atoms: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def all_(cls) -> "AtomSelection":
        return cls(shorthand="ALL")

    @classmethod
    def tip(cls) -> "AtomSelection":
        return cls(shorthand="TIP")

    @classmethod
    def bkbn(cls) -> "AtomSelection":
        return cls(shorthand="BKBN")

    @classmethod
    def explicit(cls, names: Iterable[str]) -> "AtomSelection":
        return cls(atoms=frozenset(n.strip().upper() for n in names if n.strip()))

    @classmethod
    def none_(cls) -> "AtomSelection":
        return cls()

    def is_none(self) -> bool:
        return self.shorthand is None and not self.atoms

    def __repr__(self) -> str:
        if self.shorthand:
            return f"AtomSelection({self.shorthand})"
        if not self.atoms:
            return "AtomSelection(none)"
        return f"AtomSelection({','.join(sorted(self.atoms))})"


# --- contig parser ---------------------------------------------------------
_INDEXED_RE = re.compile(r"^([A-Za-z]+)(\d+)(?:-(\d+))?$")
_CHAIN_BREAK = "/0"


def parse_contig(s: str, *, unindex: bool = False) -> List[ContigComponent]:
    """Parse a contig string into an ordered list of components.

    Raises ``ValueError`` on a malformed component.

    A bare integer is interpreted differently by field (per input.md — the
    offset syntax lives under "Unindexing Specifics", NOT "Contig Strings"):
      - ``contig`` field (``unindex=False``, default): a bare integer is a
        ``Designed`` region of that length (e.g. the ``70`` in
        ``A40-60,70,A120-170`` — "design a chain with exactly 70 residues").
      - ``unindex`` field (``unindex=True``): a bare integer between two indexed
        components is an ``UnindexedOffset`` (the ``N`` in ``A11,N,A12``);
        ``A11-12`` / ``A11,0,A12`` tie components (0 offset), ``A11,3,A12`` is a
        3-residue separation. Unindexed motifs have no designed regions.
    """
    if not isinstance(s, str):
        raise TypeError(f"contig must be a string, got {type(s).__name__}")
    raw = [c.strip() for c in s.split(",") if c.strip() != ""]
    out: List[ContigComponent] = []
    for i, comp in enumerate(raw):
        if comp == _CHAIN_BREAK:
            out.append(ChainBreak())
            continue
        m = _INDEXED_RE.match(comp)
        if m:
            chain, a, b = m.group(1), int(m.group(2)), m.group(3)
            out.append(Indexed(chain, a, int(b) if b is not None else a))
            continue
        if comp.isdigit():
            n = int(comp)
            if unindex:
                # unindex has no designed regions: a bare int is the offset
                # between two unindexed components.
                out.append(UnindexedOffset(n))
            else:
                out.append(Designed(n))
            continue
        if not unindex:
            mm = re.match(r"^(\d+)-(\d+)$", comp)
            if mm:
                lo, hi = int(mm.group(1)), int(mm.group(2))
                if lo > hi:
                    raise ValueError(f"designed range lo>hi in {comp!r}")
                out.append(DesignedRange(lo, hi))
                continue
        raise ValueError(f"unparseable contig component {comp!r} in {s!r}")
    return out


def contig_summary(comps: Sequence[ContigComponent]) -> dict:
    """A compact, human-readable summary of a parsed contig (counts + indexed
    residues), useful for logging the sampled contig back into the output JSON
    (upstream logs `sampled_contig` + counts)."""
    n_indexed = sum(1 for c in comps if isinstance(c, Indexed))
    n_breaks = sum(1 for c in comps if isinstance(c, ChainBreak))
    n_designed_fixed = sum(1 for c in comps if isinstance(c, Designed))
    n_designed_range = sum(1 for c in comps if isinstance(c, DesignedRange))
    n_offsets = sum(1 for c in comps if isinstance(c, UnindexedOffset))
    return {
        "n_components": len(comps),
        "n_indexed": n_indexed,
        "n_chain_breaks": n_breaks,
        "n_designed_fixed": n_designed_fixed,
        "n_designed_range": n_designed_range,
        "n_unindexed_offsets": n_offsets,
    }


# --- InputSelection parser -------------------------------------------------
def parse_input_selection(
    value: Union[bool, str, Mapping, None],
) -> Union[bool, List[ContigComponent], dict[str, AtomSelection], None]:
    """Parse an InputSelection field value.

    Returns one of:
      - ``None``                          (the field was not set)
      - ``bool``                          (the value was a bare boolean)
      - ``list[ContigComponent]]``        (the value was a contig string)
      - ``dict[str, AtomSelection]``     (the value was a dict)

    The dict keys are kept as the raw contig-string keys (e.g. ``"A1-2"``);
    resolving them to concrete residues/atoms is the featurizer's job (it needs
    the parsed input structure). Dict values are normalized to ``AtomSelection``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_contig(value)
    if isinstance(value, Mapping):
        out: dict[str, AtomSelection] = {}
        for key, spec in value.items():
            out[str(key)] = _parse_atom_spec(spec)
        return out
    raise TypeError(f"InputSelection must be bool/str/dict/None, got {type(value).__name__}")


def _parse_atom_spec(spec: Union[str, bool, None]) -> AtomSelection:
    if spec is None or (isinstance(spec, bool) and spec is False):
        return AtomSelection.none_()
    if isinstance(spec, bool) and spec is True:
        return AtomSelection.all_()
    if not isinstance(spec, str):
        raise TypeError(f"atom spec must be str/bool/None, got {type(spec).__name__}")
    s = spec.strip()
    if s == "":
        return AtomSelection.none_()
    up = s.upper()
    if up in ATOM_SHORTHANDS:
        return AtomSelection(shorthand=up)
    return AtomSelection.explicit(s.split(","))


# --- InputSpecification dataclass -----------------------------------------
@dataclass
class InputSpecification:
    """A structured, in-memory view of one RFD3 design spec (one JSON/YAML key).

    Only the fields the parser can validate without a structure are typed here;
    everything else passes through as-is for the featurizer. This mirrors the
    upstream InputSpecification field table (input.md) — completeness is the
    goal, not a partial subset.
    """

    input: str | None = None
    contig: str | None = None
    unindex: str | Mapping | None = None
    length: str | int | None = None
    ligand: str | None = None
    cif_parser_args: dict | None = None
    extra: dict | None = None
    dialect: int = 2
    select_fixed_atoms: Union[bool, str, Mapping] = True
    select_unfixed_sequence: Union[bool, str, Mapping] = True
    select_buried: Union[bool, str, Mapping] | None = None
    select_partially_buried: Union[bool, str, Mapping] | None = None
    select_exposed: Union[bool, str, Mapping] | None = None
    select_hbond_donor: Union[bool, str, Mapping] | None = None
    select_hbond_acceptor: Union[bool, str, Mapping] | None = None
    select_hotspots: Union[bool, str, Mapping] | None = None
    redesign_motif_sidechains: bool = False
    symmetry: Mapping | None = None
    ori_token: Sequence[float] | None = None
    infer_ori_strategy: str | None = None  # "com" | "hotspots"
    plddt_enhanced: bool = True
    is_non_loopy: bool | None = None
    partial_t: float | None = None
    # passthrough for fields the parser doesn't validate (allow_ligand_on_existing_chain, etc.)
    extra_fields: dict = field(default_factory=dict)

    _SELECT_FIELDS = (
        "select_fixed_atoms", "select_unfixed_sequence", "select_buried",
        "select_partially_buried", "select_exposed", "select_hbond_donor",
        "select_hbond_acceptor", "select_hotspots",
    )

    @classmethod
    def from_dict(cls, d: Mapping) -> "InputSpecification":
        known = {f for f in (
            "input", "contig", "unindex", "length", "ligand", "cif_parser_args",
            "extra", "dialect", *cls._SELECT_FIELDS, "redesign_motif_sidechains",
            "symmetry", "ori_token", "infer_ori_strategy", "plddt_enhanced",
            "is_non_loopy", "partial_t",
        )}
        spec = cls()
        for k, v in d.items():
            if k in known:
                setattr(spec, k, v)
            else:
                spec.extra_fields[k] = v
        return spec

    def validate(self) -> None:
        """Early, informative errors (input.md "Safer parsing"): unknown keys
        already separated into extra_fields by from_dict; here we check the
        contig/unindex grammar and mutually-exclusive RASA bins are well-formed
        at the string level (overlap checks need the structure -> featurizer)."""
        if self.contig is not None:
            parse_contig(self.contig)  # raises on malformed
        if isinstance(self.unindex, str):
            parse_contig(self.unindex, unindex=True)
        elif isinstance(self.unindex, Mapping):
            # p22: dict-form unindex -- same component/tie grammar as the
            # string form, applied to the joined keys (verified vs the real
            # reference's own `break_unindexed`: `",".join(unindex.raw.keys())`
            # re-parsed by the identical `get_motif_components_and_breaks`).
            parse_contig(",".join(str(k) for k in self.unindex.keys()), unindex=True)
        if self.dialect not in (1, 2):
            raise ValueError(f"dialect must be 1 or 2, got {self.dialect}")
        if self.infer_ori_strategy is not None and self.infer_ori_strategy not in ("com", "hotspots"):
            raise ValueError(f"infer_ori_strategy must be com|hotspots, got {self.infer_ori_strategy!r}")
        if self.partial_t is not None and self.partial_t <= 0:
            raise ValueError(f"partial_t must be > 0, got {self.partial_t}")
        for fld in ("select_buried", "select_partially_buried", "select_exposed"):
            v = getattr(self, fld)
            if isinstance(v, str):
                parse_contig(v)
        # parse the dict-form selections to validate atom specs early
        for fld in self._SELECT_FIELDS:
            parse_input_selection(getattr(self, fld))

    def parsed_contig(self) -> List[ContigComponent]:
        return parse_contig(self.contig) if self.contig is not None else []

    def parsed_unindex(self) -> List[ContigComponent]:
        if self.unindex is None:
            return []
        if isinstance(self.unindex, str):
            return parse_contig(self.unindex, unindex=True)
        if isinstance(self.unindex, Mapping):
            # p22: same component/tie grammar as the string form, over the
            # joined dict keys (see `validate()`) -- the reference's own
            # `break_unindexed` does the identical join-then-reparse. Per-
            # residue atom subsetting from the dict VALUES is a separate
            # concern, resolved by the featurizer (`_unindex_dict_atom_names`).
            return parse_contig(",".join(str(k) for k in self.unindex.keys()), unindex=True)
        return []
