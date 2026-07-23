"""RFD3 host featurizer: build the ``f`` feature dict + initial token/pair state
metadata from a real user PDB/CIF + a parsed :class:`InputSpecification`.

This is the N6 core that turns a from-PDB design input into the ``f`` dict the
on-device TokenInitializer + DiffusionModule consume. It is grounded in the
real RosettaCommons/foundry featurizer
(``models/rfd3/src/rfd3/transforms/design_transforms.py`` + ``virtual_atoms.py``
+ ``pipelines.py``, production branch, 2026-07-23) and the ``f`` contract the
TokenInitializer reads.

Status (p15): ATOM-LEVEL parity landed for the protein-binder/motif-scaffold
case (F1/F6) AND the nucleic-acid-binder case (F2/F8). The reference does NOT
pad every token to a fixed 14 atoms —
``PadTokensWithVirtualAtoms`` only pads DESIGNED (sequence-unknown) tokens to
14; MOTIF (fixed-seq, indexed) tokens keep exactly their real observed heavy
atoms, looked up via the "dense" association scheme
(``rfd3.constants.association_schemes["dense"]``, vendored below as
``_DENSE_ATOM14_SCHEME``) which assigns each residue's real atoms to a
per-residue-type slot (with symmetry-reserved gaps, e.g. GLU's OE2 lands at
slot 9 not 8). Beyond backbone (N/CA/C/O/CB), atom NAMES are relabeled to
generic ``V0..V8`` for BOTH motif and designed atoms (``ATOM14_ATOM_NAMES``) —
this hides side-chain chemical identity from the atom-name channel while still
conditioning on real 3D geometry via ``motif_pos``. Verified against a local
CPU capture of the real reference featurizer (``rc-foundry[rfd3]``, no ckpt
needed): see ``scripts/rfd3_port/parity_artifacts/``.

Protein-specific reference-feature semantics (from
``CreateDesignReferenceFeatures.forward``, where ``has_sequence`` excludes
protein under ``generate_conformers_for_non_protein_only``): ``ref_pos``,
``ref_mask``, ``ref_pos_is_ground_truth``, ``ref_charge`` are all-zero/False
for EVERY protein atom (motif or designed) — real motif coordinates flow only
through ``motif_pos``. ``ref_element`` is likewise never filled for protein,
so its one-hot is the constant index-0 row for every atom (not real chemical
identity). ``motif_pos`` is centered: the whole design is translated so the
center of mass of the real (motif) atoms sits at the origin.

Coverage: protein-binder (F1) + motif-scaffolding (F6, indexed AND unindexed)
on protein input, nucleic-acid-binder design (F2/F8: a fixed-sequence DNA/RNA
target chain + a designed protein binder chain, e.g. the ``dsDNA_basic``/
``RNA_basic`` reference examples), PLUS small-molecule-binder design (F3: a
real ligand named by CCD code via the separate `ligand` spec field, e.g. the
real ``sm_binder_design.md`` "buried"/"partial" examples). Enzyme (F4, catalytic-
residue conditioning on TOP of a ligand) and symmetry (F5) input still raise
NotImplementedError with a pointer to the reference transform; so does NA as
an *indexed motif inside a protein chain*'s unindex field, and the unindex
numeric-offset-tie syntax / dict-form per-atom fixing (see
``_plan_unindexed_tokens``).

Ligand (F3) grounding (same design_transforms.py/virtual_atoms.py + the real
``rfd3.inference.input_parsing.py``/``rfd3.inference.parsing.py`` select-field
resolution, verified against a real local CPU capture of IAI.pdb/the
``sm_binder_design.md`` "buried" example via ``capture_ref_f_spec.py`` — no
ckpt needed, same method as F2/F8):
- A ligand is ATOMIZED: each real heavy atom is its OWN token (not grouped
  into one multi-atom token like protein/NA) — verified: ``PadTokensWithVirtualAtoms``'s
  ``is_residue`` gate (``is_protein & ~atomize``) excludes a ligand entirely,
  so it's never padded/grouped. Each atomized token's single atom is trivially
  its own representative: ``is_ca``/``is_central``/``is_backbone`` are all
  True and ``is_sidechain`` False for every ligand atom (verified).
- A ligand ALWAYS has known chemical identity (``is_motif``/"class 1", same
  as an indexed motif) even when its COORDINATE is diffused (unfixed) — the
  reference's ``select_unfixed_sequence`` field explicitly excludes ligands
  ("ligands / DNA always have fixed sequence"); `is_fixed_coord` and
  `is_fixed_seq` are therefore tracked as SEPARATE flags on `_Token` (unlike
  protein/NA, where they always coincide) — see the `_Token.fixed_coord`/
  `fixed_seq` properties and the ``select_fixed_atoms`` resolution below.
- ``ref_pos``/``ref_element``/``ref_charge``/intra-ligand ``token_bonds`` come
  from a REAL CCD template — this port reuses its OWN existing bundled CCD
  rdkit-mol library (``tt_bio.data.mol.load_molecules``, ``~/.boltz/mols``,
  the same one Boltz-2/Protenix-v2 already ship) rather than re-vendoring
  RDKit/CCD conformer generation. ``ref_pos`` is reference-CONFORMER geometry,
  not identity: the real reference itself draws a fresh, unseeded random
  RDKit ETKDG conformer + random rigid augmentation every run (verified in
  the reference source), so no single captured reference run's `ref_pos` is
  "the" bit-exact target — this port's OWN Protenix-v2 host featurizer
  already documents and relies on exactly this invariance
  (``tt_bio/protenix_data.py:466``).
- ``ref_atom_name_chars`` is overridden to encode the ELEMENT symbol, not the
  real atom name, for every ligand atom (the reference's
  ``use_element_for_atom_names_of_atomized_tokens=True`` default — verified: a
  real capture's ligand rows decode to "C   "/"N   ", not "C22 "/"N9  ").
- ``select_buried``/``select_exposed`` become the ``ref_atomwise_rasa`` one-hot
  bin DIRECTLY (0=buried, 2=exposed) — a user-specified per-atom LABEL at
  inference, not a computed SASA value (the real Shrake-Rupley RASA transform
  is training-only, never invoked at inference — verified in the reference
  source). A ligand-code dict key (e.g. ``{"IAI": "C1,C2"}``) is a SEPARATE
  convention from protein/NA's ``{chain}{res_id}`` key.
- ``restype`` for a ligand token is the protein-UNK slot (index 20, NOT the
  GAP/no-sequence slot 31) — verified against a real capture.
- ``residue_index``/``ref_space_uid`` are RESIDUE-level, not token-level: all
  of a ligand's atomized tokens share ONE residue_index (0, on the ligand's
  own fresh chain) and ONE ref_space_uid (the first ligand token's index) —
  verified against a real capture (getting this wrong assigns each ligand
  atom its own distinct "residue", silently misconditioning symmetry/pair
  features that key off `ref_space_uid`).
- A pure ``length``-only spec (no ``contig`` at all, e.g. a small-molecule
  binder with nothing but a fresh designed chain + a ``ligand``) is treated
  as a bare designed-length contig string (``parse_contig`` already parses a
  bare "180-180"/"180" as Designed/DesignedRange).
- Scoped out this pass: multiple ligand instances of the same/different CCD
  code (single instance only), the ``TIP``/``BKBN`` atom-selection shorthands
  applied to a ligand, and a contig-string (rather than dict-form) select_*
  value targeting a ligand — all raise NotImplementedError rather than guess.

NA (F2/F8) grounding (``rfd3.transforms.design_transforms.py`` +
``virtual_atoms.py`` + ``util_transforms.py``, verified against real local CPU
captures of ``1bna.pdb``/dsDNA and ``1q75.pdb``/RNA via ``capture_ref_f.py`` —
no ckpt needed, same method as F1/F6):
- DNA/RNA atoms are NEVER renamed to generic ``V0..V8`` labels and NEVER
  padded — ``PadTokensWithVirtualAtoms``'s ``is_residue`` gate is
  ``is_protein & ~atomize`` (plus unindexed, N/A here), so non-protein tokens
  never enter that transform at all. Real atom names (``O5'``, ``C1'``,
  ``N9``, ...) are kept verbatim, in the input structure's real order — no
  scheme lookup needed (unlike the protein "dense" scheme).
- ``ref_element`` IS filled for NA (unlike protein, whose ``has_sequence``
  is unconditionally excluded by ``generate_conformers_for_non_protein_only``)
  — one-hot atomic number of the real parsed element. ``ref_charge`` is 0 and
  ``ref_mask`` is True for every NA atom (verified against both a real capture
  AND the persisted p4/p10 dsDNA_basic ckpt golden).
- ``ref_pos`` (the reference-conformer 3D geometry) is NOT reproduced this
  pass — the real pipeline calls into RDKit/CCD-template conformer generation
  (``get_af3_reference_molecule_features``), which this port does not vendor;
  left at 0 (documented gap, same simplification protein already uses since
  fixed atoms get real geometry via ``motif_pos`` regardless).
- ``is_ca``/``is_central`` (the one "representative atom" per token) is the
  base's ring-center atom: ``C4`` for purines (DA/DG/A/G), ``C2`` for
  pyrimidines (DC/DT/C/U) — verified against both captures.
- ``is_backbone``/``is_sidechain`` are never set for NA (that split is
  protein-only in the reference); ``terminus_type`` (5'/3') is likewise never
  set for NA in this contract (verified all-zero on both captures).
- ``restype``'s 32-dim one-hot follows the real AF3 vocabulary
  (``atomworks.ml.encoding_definitions.AF3_TOKENS``): 0-19 the 20 AA, 20
  unknown-AA, 21-24 RNA A/C/G/U, 25 unknown-RNA, 26-29 DNA DA/DC/DG/DT, 30
  unknown-DNA, 31 GAP (designed/no-sequence) — verified index-for-index
  against both captures.
- ``entity_id``/``sym_id``: chains are grouped into the same entity by their
  FULL real-chain residue-name sequence (not just the contig-selected
  subset) — e.g. dsDNA_basic's chain A and chain B are the same 12-mer
  palindrome, so they share ``entity_id`` and get distinct ``sym_id`` replica
  indices. A synthetic (designed) chain always starts a fresh entity. A
  ``Designed``/``DesignedRange`` segment immediately after a contig chain
  break (``/0``) gets a brand-new synthetic chain letter rather than
  inheriting the preceding indexed block's chain (verified: dsDNA_basic's
  designed protein segment is its own chain/entity, not chain "B").
"""
from __future__ import annotations

import copy
import os as _os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .rfd3_input import (
    InputSpecification, parse_contig, ChainBreak, Indexed, Designed, DesignedRange,
    AtomSelection, _parse_atom_spec,
)

# -- atom14 generic name template (rfd3.constants.ATOM14_ATOM_NAMES) --------
# Slots 0..4 keep real backbone/CB names; slots 5..13 are always the generic
# "V{i}" placeholder, whether the atom is a real (renamed) side-chain atom of
# a motif residue or a synthetic virtual pad atom of a designed residue.
ATOM14_ATOM_NAMES = ["N", "CA", "C", "O", "CB"] + [f"V{i}" for i in range(9)]
BACKBONE_NAMES = {"N", "CA", "C", "O"}

# The "dense" association scheme (rfd3.constants.association_schemes["dense"],
# stripped variant used by map_to_association_scheme): for each residue type,
# the REAL atom name occupying each of the 14 atom14 slots (None = unused
# slot for that residue — note the gaps, e.g. GLU's OE2 sits at slot 9, not
# 8, reserved for symmetry-consistent packing across residue types). A real
# atom's generic name is ATOM14_ATOM_NAMES[slot]; a motif residue emits only
# the slots that are both non-None here AND actually present in the input
# structure (no padding — this is what makes L variable per token).
_DENSE_ATOM14_SCHEME: dict[str, list[str | None]] = {
    "ALA": ["N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", None, None, None],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", None, None, None, None, None, None],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", None, "OD2", None, None, None, None, None],
    "CYS": ["N", "CA", "C", "O", "CB", None, "SG", None, None, None, None, None, None, None],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", None, None, None, None, None],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", None, "OE2", None, None, None, None],
    "GLY": ["N", "CA", "C", "O", None, None, None, None, None, None, None, None, None, None],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", None, None, None, None],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None, None, None, None, None, None],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", None, None, None, None, None, None],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", None, None, None, None, None],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", None, None, None, None, None, None],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", None, None, None],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", None, None, None, None, None, None, None],
    "SER": ["N", "CA", "C", "O", "CB", "OG", None, None, None, None, None, None, None, None],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", None, None, None, None, None, None, None],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", None, None],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", None, None, None, None, None, None, None],
}

# Real AF3 sequence vocabulary (atomworks.ml.encoding_definitions.AF3_TOKENS):
# 20 AA + unknown-AA, 4 RNA + unknown-RNA, 4 DNA + unknown-DNA, GAP. restype
# is a 32-dim one-hot over this exact order (index-verified vs real local
# captures of dsDNA_basic-style and RNA_basic-style inputs, see module
# docstring). Designed (sequence-unknown) tokens use the GAP slot (31).
_RESTYPE_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
    "A", "C", "G", "U", "N",
    "DA", "DC", "DG", "DT", "DN",
    "GAP",
]
_RESTYPE_TO_IDX = {n: i for i, n in enumerate(_RESTYPE_ORDER)}
DESIGNED_RESTYPE_IDX = _RESTYPE_TO_IDX["GAP"]
RESTYPE_DIM = 32
assert DESIGNED_RESTYPE_IDX == 31 and RESTYPE_DIM == len(_RESTYPE_ORDER)

PROTEIN_RES = set(_RESTYPE_ORDER[:20])
RNA_RES = {"A", "C", "G", "U"}
DNA_RES = {"DA", "DC", "DG", "DT"}
PURINE_RES = {"DA", "DG", "A", "G"}
PYRIMIDINE_RES = {"DC", "DT", "C", "U"}
_ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53,
}


# -- encoders ---------------------------------------------------------------
def _encode_atom_names_like_af3(names: Sequence[str]) -> np.ndarray:
    """AF3 atom-name encoding: each name padded to 4 chars, each char one-hot
    over 64 bins where bin = ord(c) - 32 (printable ASCII starting at space).
    Returns [N, 4, 64] float32. Matches atomworks._encode_atom_names_like_af3."""
    out = np.zeros((len(names), 4, 64), dtype=np.float32)
    for i, name in enumerate(names):
        s = (name or "")[:4].ljust(4)
        for j, ch in enumerate(s):
            b = ord(ch) - 32
            if 0 <= b < 64:
                out[i, j, b] = 1.0
    return out


def _restype_onehot(res_names: Sequence[str]) -> np.ndarray:
    """[I, 32] one-hot restype (designed slot at DESIGNED_RESTYPE_IDX=31)."""
    out = np.zeros((len(res_names), RESTYPE_DIM), dtype=np.float32)
    for i, r in enumerate(res_names):
        idx = _RESTYPE_TO_IDX.get(str(r).strip().upper(), DESIGNED_RESTYPE_IDX)
        out[i, idx] = 1.0
    return out


# -- structure loading (biotite; atomworks-free) -----------------------------
def load_structure(path: str | Path):
    """Parse a PDB/CIF into a biotite AtomArray with the annotations the
    featurizer needs. This replaces atomworks.io.parser.parse for the
    protein/NA case; it does NOT reproduce atomworks' bond perception,
    assembly building, or CCD normalisation (parity-ungated).

    Heavy atoms only (matches the reference's universal heavy-atom-only
    convention): the protein path already gets this implicitly (hydrogens
    aren't in the "dense" atom14 scheme so they're silently skipped), but an
    NA input file that models explicit hydrogens (e.g. an NMR structure)
    needs an explicit drop here — verified vs a real reference capture
    (1q75.pdb/RNA_basic, which is H-explicit; without this filter L came out
    550 instead of the reference's 386)."""
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    p = Path(path)
    if p.suffix.lower() in (".cif", ".mmcif"):
        cf = CIFFile.read(str(p))
        arr = get_structure(cf, model=1)
    else:
        pf = PDBFile.read(str(p))
        arr = pf.get_structure(model=1)
    arr = arr[arr.element != ""] if hasattr(arr, "element") else arr
    arr = arr[arr.element != "H"] if hasattr(arr, "element") else arr
    return arr


@dataclass
class _Residue:
    chain: str
    res_id: int
    res_name: str
    atom_names: list[str]
    coord: np.ndarray  # [n_atoms, 3]
    elements: list[str]


def _group_residues(arr) -> list[_Residue]:
    """Group a biotite AtomArray into per-residue records (one token per
    residue for protein/NA; ligands are handled separately)."""
    import biotite.structure as struc
    starts = list(struc.get_residue_starts(arr))
    stops = list(struc.get_residue_starts(arr, add_exclusive_stop=True))
    res = []
    for k, s in enumerate(starts):
        e = stops[k + 1] if k + 1 < len(stops) else len(arr)
        sub = arr[s:e]
        res.append(_Residue(
            chain=str(sub.chain_id[0]), res_id=int(sub.res_id[0]),
            res_name=str(sub.res_name[0]).strip().upper(),
            atom_names=[str(a).strip() for a in sub.atom_name],
            coord=np.asarray(sub.coord, dtype=np.float32),
            elements=[str(e).strip().upper() for e in sub.element],
        ))
    return res


def _is_protein(r: _Residue) -> bool:
    return r.res_name in PROTEIN_RES


def _is_na(r: _Residue) -> bool:
    return r.res_name in DNA_RES or r.res_name in RNA_RES


def _central_atom_name(res_name: str) -> str | None:
    """The single "representative atom" name for a DNA/RNA residue (the
    reference's ``is_ca``/``is_central`` flag) — the base ring-center atom:
    C4 for purines, C2 for pyrimidines. Verified vs real captures (dsDNA_basic
    -style and RNA_basic-style)."""
    if res_name in PURINE_RES:
        return "C4"
    if res_name in PYRIMIDINE_RES:
        return "C2"
    return None


def _motif_atom_layout(r: _Residue):
    """Real heavy atoms of a motif (fixed-seq) residue, in the INPUT
    STRUCTURE'S real atom order, renamed to the generic atom14 template via
    the dense-scheme slot lookup — NOT padded, NOT slot-sorted. Missing atoms
    (absent from the input structure) are simply skipped, matching the
    reference (it only ever sees the atoms present in the parsed structure).

    Emission order matters and is NOT the same as slot order: a residue whose
    real PDB atom listing doesn't already happen to match the canonical
    dense-scheme slot order (e.g. a TRP with CE2/CE3 listed before NE1) keeps
    its real order in the reference — only the per-atom NAME is remapped to
    the generic V-slot label. Verified vs a real reference capture (p14,
    ``scripts/rfd3_port/parity_artifacts/parity_unindex.py``, residue A100
    TRP): getting this wrong silently permutes ``motif_pos``/atom-name
    features for any residue with non-canonical side-chain atom ordering in
    its input file (didn't manifest on the p12 F1/F6 fixture by coincidence).
    Returns (names, coord[n,3], is_virtual[n]=False, elements=None) — elements
    is always None for protein: ``ref_element`` is never filled for protein
    atoms regardless (see module docstring), so no per-atom element is needed."""
    scheme = _DENSE_ATOM14_SCHEME.get(r.res_name)
    if scheme is None:
        raise NotImplementedError(f"no dense atom14 scheme for motif residue {r.res_name!r}")
    slot_by_name = {name: slot for slot, name in enumerate(scheme) if name is not None}
    names: list[str] = []
    coord: list[np.ndarray] = []
    seen: set[str] = set()
    for real_name, c in zip(r.atom_names, r.coord):
        slot = slot_by_name.get(real_name)
        if slot is None or real_name in seen:
            continue
        seen.add(real_name)
        names.append(ATOM14_ATOM_NAMES[slot])
        coord.append(c)
    coord_arr = np.asarray(coord, dtype=np.float32) if coord else np.zeros((0, 3), dtype=np.float32)
    return names, coord_arr, np.zeros(len(names), dtype=bool), None


def _na_atom_layout(r: _Residue):
    """Real heavy atoms of a DNA/RNA motif residue, verbatim: real names, real
    order, no scheme lookup, no renaming (``PadTokensWithVirtualAtoms``'s
    ``is_residue`` gate is protein-only, so non-protein tokens never get
    V-slot-relabeled or padded — see module docstring). Elements are the real
    parsed per-atom element (needed for ``ref_element``, which — unlike
    protein — IS filled for NA). Returns (names, coord[n,3], is_virtual[n]
    =False, elements[n])."""
    return list(r.atom_names), r.coord.copy(), np.zeros(len(r.atom_names), dtype=bool), list(r.elements)


def _designed_atom_layout():
    """Full 14-slot template for a designed (sequence-unknown) residue: the
    5 backbone+CB slots are real (undetermined, coord 0); V0..V8 are virtual
    pad atoms (PadTokensWithVirtualAtoms). Returns (names, coord[14,3],
    is_virtual[14], elements=None)."""
    names = list(ATOM14_ATOM_NAMES)
    coord = np.zeros((14, 3), dtype=np.float32)
    is_virtual = np.zeros(14, dtype=bool)
    is_virtual[5:] = True
    return names, coord, is_virtual, None


# -- contig -> token plan ----------------------------------------------------
@dataclass
class _Token:
    chain: str
    res_id: int          # input PDB res_id (motif) or assigned (designed)
    res_name: str        # real name (motif) or "" (designed -> DESIGNED_RESTYPE_IDX)
    is_motif: bool       # BROAD: has known identity (fixed coord OR fixed seq OR
                         # unindexed) -> drives ref_motif_token_type/ref_plddt/
                         # motif_token_class. For protein/NA this always coincides
                         # with "fully fixed coord+seq" (both True together); a
                         # ligand can be is_motif=True (known chemical identity)
                         # while is_fixed_coord=False (diffused position) -- see
                         # is_fixed_coord/is_fixed_seq below.
    is_designed: bool    # diffused
    is_unindexed: bool    # unindexed motif (from the unindex field)
    is_chain_break_before: bool  # /0 precedes this token
    residue: _Residue | None  # source residue for motif tokens, else None
    unindex_new_island: bool = False  # this unindexed token starts a new RPE-leak island
    is_ligand: bool = False  # F3/F4: one atom == one token (atomize)
    is_fixed_coord: bool | None = None  # None => derive from is_motif (protein/NA)
    is_fixed_seq: bool | None = None    # None => derive from is_motif (protein/NA)
    ligand_atom_name: str | None = None  # real atom name (ligand tokens only)

    @property
    def fixed_coord(self) -> bool:
        return self.is_motif if self.is_fixed_coord is None else self.is_fixed_coord

    @property
    def fixed_seq(self) -> bool:
        return self.is_motif if self.is_fixed_seq is None else self.is_fixed_seq


def _plan_unindexed_tokens(spec: InputSpecification, residues: list[_Residue]) -> list[_Token]:
    """Unindexed motif tokens (``spec.unindex``), appended at the END of the
    token list — matches the reference (``accumulate_components`` places
    unindexed components after the main contig; ``UnindexFlaggedTokens``
    reorders/expands them, but at inference they are already physically last).

    Scoped this pass (p14, grounded via a real local reference capture —
    ``scripts/rfd3_port/parity_artifacts/parity_unindex.py``): plain contig
    components only (a single indexed residue, or an indexed ``-`` RANGE which
    ties the residues together / "leaks" their relative order to the model).
    The doc-described numeric-offset-tie syntax (``A11,0,A12`` / ``A11,3,A12``)
    and dict-form per-atom fixing are NOT implemented — both are genuinely
    ambiguous from the reference source alone (the offset digit does not
    obviously survive into ``get_motif_components_and_breaks``'s breaks array
    the way the docs describe) and were not capture-verified this pass; they
    raise NotImplementedError with a pointer here rather than guess.

    A residue is "tied" (leaked) to the PRECEDING residue in the same ``-``
    range (RPE may reveal their relative sequence position); it is masked
    (never leaked) from every other token — indexed motif, designed, AND any
    other unindexed island — per the captured ``unindexing_pair_mask`` (see
    ``UnindexFlaggedTokens.create_unindexed_masks``: group id = cumsum of
    per-token "new island" flags; same group -> leak allowed, else masked).
    """
    if spec.unindex is None:
        return []
    if not isinstance(spec.unindex, str):
        raise NotImplementedError("dict-form unindex (per-atom fixing) — p14+")
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_unindex()
    tokens: list[_Token] = []
    for c in comps:
        if not isinstance(c, Indexed):
            raise NotImplementedError(
                f"unindex component {c!r} not supported this pass (p14): only "
                "plain indexed residues/ranges ('A244' or 'A11-12'); the "
                "numeric-offset-tie syntax and '/0' inside unindex are out of scope"
            )
        for k, rid in enumerate(range(c.start, c.end + 1)):
            r = by_key.get((c.chain, rid))
            if r is None:
                raise ValueError(f"unindex references {c.chain}{rid} not present in input structure")
            if not _is_protein(r):
                raise NotImplementedError("non-protein unindexed motif (NA/ligand) — p15+")
            tokens.append(_Token(c.chain, rid, r.res_name, True, False, True,
                                 False, r, unindex_new_island=(k == 0)))
    return tokens


def _fresh_chain_letter(used: set[str]) -> str:
    """Allocate a synthetic chain letter for a Designed/DesignedRange segment
    that does not reuse any real input chain or previously-assigned synthetic
    chain. Verified vs a real reference capture (dsDNA_basic-style): a
    designed segment immediately after a contig chain break (``/0``) gets its
    own new chain/entity, NOT the preceding indexed block's chain letter."""
    import string
    for letter in string.ascii_uppercase:
        if letter not in used:
            used.add(letter)
            return letter
    raise NotImplementedError("more than 26 chains — p15+")


def _plan_tokens_from_contig(spec: InputSpecification, residues: list[_Residue]) -> list[_Token]:
    """Map a parsed contig + the parsed structure's residues into an ordered
    token plan. Indexed components pull residues from the input structure
    (motif, fixed coord+seq, protein OR DNA/RNA); Designed/DesignedRange
    components become diffused (protein) tokens; ChainBreak marks the next
    token. Unindexed motif tokens (from ``spec.unindex``, see
    ``_plan_unindexed_tokens``) are appended last."""
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_contig()
    tokens: list[_Token] = []
    break_before_next = False
    indexed_keys: set[tuple[str, int]] = set()
    used_chains: set[str] = {r.chain for r in residues}
    for c in comps:
        if isinstance(c, ChainBreak):
            break_before_next = True
            continue
        if isinstance(c, Indexed):
            for rid in range(c.start, c.end + 1):
                r = by_key.get((c.chain, rid))
                if r is None:
                    raise ValueError(f"contig indexes {c.chain}{rid} not present in input structure")
                if not (_is_protein(r) or _is_na(r)):
                    raise NotImplementedError("ligand/enzyme indexed motif (F3/F4) — p15+")
                tokens.append(_Token(c.chain, rid, r.res_name, True, False, False,
                                     break_before_next, r))
                indexed_keys.add((c.chain, rid))
                break_before_next = False
            continue
        if isinstance(c, (Designed, DesignedRange)):
            n = c.length if isinstance(c, Designed) else (c.lo + c.hi) // 2
            # A designed segment continues the preceding block's chain UNLESS
            # a chain break (or nothing yet) precedes it, in which case it
            # gets a brand-new synthetic chain (verified vs a real capture).
            if break_before_next or not tokens:
                chain = _fresh_chain_letter(used_chains)
            else:
                chain = tokens[-1].chain
            base = (tokens[-1].res_id + 1) if tokens and tokens[-1].chain == chain else 1
            for k in range(n):
                tokens.append(_Token(chain, base + k, "", False, True, False,
                                     break_before_next, None))
                break_before_next = False
            continue
        raise NotImplementedError(f"contig component {c!r} not supported this pass (p12)")
    unindexed = _plan_unindexed_tokens(spec, residues)
    overlap = indexed_keys & {(tk.chain, tk.res_id) for tk in unindexed}
    if overlap:
        raise ValueError(f"contig and unindex must not overlap, got: {overlap}")
    tokens.extend(unindexed)
    return tokens


def _token_kind(tk: "_Token") -> str:
    """'protein' | 'dna' | 'rna' | 'ligand' for a token (designed tokens,
    res_name=="", are always protein in this port's scope — NA is never
    diffused/designed in the documented use cases, only ever a fixed binder
    target)."""
    if tk.is_ligand:
        return "ligand"
    if not tk.res_name or tk.res_name in PROTEIN_RES:
        return "protein"
    if tk.res_name in DNA_RES:
        return "dna"
    if tk.res_name in RNA_RES:
        return "rna"
    raise NotImplementedError(f"unrecognized residue {tk.res_name!r} (enzyme catalytic-residue conditioning, F4) — p17+")


# -- ligand (F3/F4) -----------------------------------------------------------
# Real per-atom CCD template (name -> element/charge/reference-conformer coord)
# + intra-ligand bond graph, reused from tt_bio's EXISTING Boltz-2/Protenix-v2
# bundled CCD rdkit-mol library (tt_bio.data.mol.load_molecules, default
# ~/.boltz/mols) rather than re-vendoring RDKit/CCD conformer generation.
#
# `ref_pos` is reference-CONFORMER geometry, not ground-truth identity: the
# real reference (rfd3.transforms.design_transforms.CreateDesignReferenceFeatures
# -> atomworks.ml.transforms.af3_reference_molecule.get_af3_reference_molecule_features)
# generates it via a STOCHASTIC RDKit ETKDG embed (a fresh random seed/rotation
# per run, verified: `ccd_code_to_rdkit_with_conformers`/`random_rigid_augmentation`
# both draw an unseeded random value absent an explicit seed) — so no single
# reference capture's `ref_pos` is "the" bit-exact target. This port's own
# Protenix-v2 host featurizer already documents and relies on exactly this
# invariance (tt_bio/protenix_data.py:466, "the reference uses a STOCHASTIC
# RDKit conformer, so any valid one folds correctly"); the same principle is
# applied here rather than re-derived.
def _ligand_template(ccd_code: str, mol_dir: str | None = None) -> dict:
    from .data.mol import load_molecules
    mol_dir = mol_dir or _os.path.expanduser("~/.boltz/mols")
    mols = load_molecules(mol_dir, [ccd_code])
    mol = mols[ccd_code]
    conf = mol.GetConformer(0)
    names: list[str] = []
    elements: list[str] = []
    charges: list[int] = []
    coords: list[tuple[float, float, float]] = []
    idx_by_rdkit_idx: dict[int, int] = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() <= 1:  # heavy atoms only, matches this port's convention
            continue
        nm = a.GetProp("name") if a.HasProp("name") else a.GetSymbol().upper()
        p = conf.GetAtomPosition(a.GetIdx())
        idx_by_rdkit_idx[a.GetIdx()] = len(names)
        names.append(nm)
        elements.append(a.GetSymbol().upper())
        charges.append(int(a.GetFormalCharge()))
        coords.append((p.x, p.y, p.z))
    bonds: list[tuple[int, int]] = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in idx_by_rdkit_idx and v in idx_by_rdkit_idx:
            bonds.append((idx_by_rdkit_idx[u], idx_by_rdkit_idx[v]))
    return {
        "names": names, "elements": elements, "charges": charges,
        "coord": np.asarray(coords, dtype=np.float32), "bonds": bonds,
    }


def _resolve_ligand_atom_selection(sel_value, code: str) -> AtomSelection:
    """Resolve a select_fixed_atoms/select_buried/select_exposed field value to
    an ``AtomSelection`` for a ligand, keyed by CCD code (e.g. ``{"IAI": "C1,C2"}``)
    — a SEPARATE key convention from protein/NA's ``{chain}{res_id}`` (verified
    vs a real reference capture + rfd3.inference.parsing.canonicalize_, which
    resolves a bare ligand-code dict key via `unravel_components` to the
    ligand's actual chain+res_id before the same per-atom-name lookup protein/
    NA use). A dict WITHOUT the ligand's code present defaults to "no atoms"
    (matches the reference: providing the field at all means only the given
    keys are set, unlisted residues keep the annotation's init default)."""
    if sel_value is None:
        return AtomSelection.none_()
    if isinstance(sel_value, bool):
        return AtomSelection.all_() if sel_value else AtomSelection.none_()
    if isinstance(sel_value, dict):
        for k, v in sel_value.items():
            if str(k).strip().upper() == code:
                return _parse_atom_spec(v)
        return AtomSelection.none_()
    raise NotImplementedError(
        f"select_* value {sel_value!r} not supported for ligand atoms "
        "(a contig-string selection targeting a ligand) — p17+"
    )


def _atom_selection_mask(sel: AtomSelection, names: Sequence[str]) -> np.ndarray:
    if sel.shorthand == "ALL":
        return np.ones(len(names), dtype=bool)
    if sel.shorthand in ("TIP", "BKBN"):
        raise NotImplementedError(f"{sel.shorthand} shorthand for ligand atoms — p17+")
    if not sel.atoms:
        return np.zeros(len(names), dtype=bool)
    return np.isin(np.asarray(names), np.asarray(sorted(sel.atoms)))


def _plan_ligand_tokens(spec: InputSpecification, all_residues: list[_Residue],
                         used_chains: set[str]) -> list[_Token]:
    """One token PER ATOM (AF3/RFD3 "atomize" convention — verified vs a real
    reference capture: PadTokensWithVirtualAtoms's `is_residue` gate excludes
    every non-protein-non-unindexed token, so a ligand is never grouped into a
    per-residue token like protein/NA are). Real coordinates/atom identity come
    from the actual input-structure ligand residue (matched by CCD code); a
    fresh synthetic chain (never colliding with a real or prior chain letter)
    matches the reference's own chain re-numbering for an appended ligand."""
    code = spec.ligand.strip().upper()
    matches = [r for r in all_residues if r.res_name == code]
    if not matches:
        raise ValueError(f"ligand {code!r} not found in input structure")
    if len(matches) > 1:
        raise NotImplementedError(f"multiple {code!r} ligand instances — p17+ (single instance only this pass)")
    residue = matches[0]
    chain = _fresh_chain_letter(used_chains)
    return [
        _Token(chain, k + 1, code, True, False, False, False, residue,
               is_ligand=True, ligand_atom_name=nm)
        for k, nm in enumerate(residue.atom_names)
    ]


def _ligand_atom_layout(tk: "_Token"):
    """Single real heavy atom, verbatim (no renaming/padding — see module
    docstring for the reference's atomize gate). Returns (names, coord[1,3],
    is_virtual[1]=False, elements[1])."""
    idx = tk.residue.atom_names.index(tk.ligand_atom_name)
    return ([tk.ligand_atom_name], tk.residue.coord[idx:idx + 1].copy(),
             np.zeros(1, dtype=bool), [tk.residue.elements[idx]])


def featurize(structure_path: str | Path, spec: InputSpecification) -> dict[str, torch.Tensor]:
    """Build the ``f`` feature dict for one design spec from a real PDB/CIF.

    Protein-binder (F1) + motif scaffolding (F6) + nucleic-acid-binder design
    (F2/F8: a fixed-sequence DNA/RNA target + a designed protein binder), with
    atom-level parity vs the real reference (see module docstring).
    """
    arr = load_structure(structure_path)
    all_residues = _group_residues(arr)
    # Non-polymer residues NOT named by `spec.ligand` (solvent, ions, an
    # unreferenced HETATM) are simply invisible to this featurizer, exactly
    # like a real PDB's crystallographic waters are to a contig that never
    # references them. A contig that DOES try to index one fails with a
    # ValueError ("not present in input structure") since it's filtered out
    # here; the ligand (if any) is planned separately below (it is never
    # indexed via `contig`, only via the `ligand` spec field).
    residues = [r for r in all_residues if _is_protein(r) or _is_na(r)]
    # A pure `length`-only spec (no `contig` at all — e.g. a small-molecule
    # binder design with nothing but a fresh designed chain + a `ligand`) is
    # equivalent to a bare designed-length contig string (`parse_contig`
    # already parses a bare "180-180"/"180" as Designed/DesignedRange).
    contig_spec = spec
    if spec.contig is None and spec.length is not None:
        contig_spec = copy.copy(spec)
        contig_spec.contig = str(spec.length)
    tokens = _plan_tokens_from_contig(contig_spec, residues)
    if spec.ligand:
        # Fresh chain letters must avoid BOTH real input chains AND any
        # synthetic chain the designed segments above already claimed.
        used_chains = {r.chain for r in all_residues} | {tk.chain for tk in tokens}
        tokens = tokens + _plan_ligand_tokens(spec, all_residues, used_chains)
    I = len(tokens)
    token_kind = [_token_kind(tk) for tk in tokens]

    # Per-token atom layout (variable count: motif = real heavy atoms only,
    # designed = full 14-slot template, ligand = one real heavy atom). NA/
    # ligand motif tokens keep real atom names/order verbatim (no scheme
    # lookup — see module docstring).
    layouts = [
        _ligand_atom_layout(tk) if kind == "ligand" else
        (_na_atom_layout(tk.residue) if kind in ("dna", "rna") else _motif_atom_layout(tk.residue))
        if tk.is_motif else _designed_atom_layout()
        for tk, kind in zip(tokens, token_kind)
    ]
    L = sum(len(nm) for nm, _, _, _ in layouts)

    # The whole design is centered at the center of mass of the real, FIXED-
    # COORD atoms (verified vs a real reference capture: motif_pos ==
    # real_coord - com, where com = mean over every atom whose coordinate is
    # actually held fixed — for protein/NA this coincides with `is_motif`, but
    # a ligand can be `is_motif` (known chemical identity) while NOT
    # fixed-coord (its position is diffused), so it must NOT contribute to
    # centering in that case; gate on `tk.fixed_coord`, not the broader
    # `tk.is_motif`).
    motif_coords = [c for tk, (nm, c, _, _) in zip(tokens, layouts) if tk.fixed_coord and len(nm)]
    if motif_coords:
        com = np.concatenate(motif_coords, axis=0).mean(axis=0)
    else:
        com = np.zeros(3, dtype=np.float32)

    atom_names: list[str] = []
    atom_elements: list[str | None] = []
    atom_coord = np.zeros((L, 3), dtype=np.float32)
    is_virtual = np.zeros(L, dtype=bool)
    is_backbone = np.zeros(L, dtype=bool)
    is_sidechain = np.zeros(L, dtype=bool)
    is_ca = np.zeros(L, dtype=bool)
    is_central = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_coord = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_seq = np.zeros(L, dtype=bool)
    is_motif_atom_unindexed = np.zeros(L, dtype=bool)
    is_na_atom = np.zeros(L, dtype=bool)
    motif_pos = np.zeros((L, 3), dtype=np.float32)
    ref_space_uid = np.zeros(L, dtype=np.int64)
    atom_to_token_map = np.zeros(L, dtype=np.int32)

    token_res_name = []
    token_chain = []
    token_res_id = []
    token_is_motif = np.zeros(I, dtype=bool)
    token_is_unindexed = np.zeros(I, dtype=bool)
    token_break_before = np.zeros(I, dtype=bool)
    token_is_fully_fixed_coord = np.zeros(I, dtype=bool)
    is_ligand_atom = np.zeros(L, dtype=bool)

    # Ligand-only per-atom selections (resolved once, outside the loop): the
    # ligand's real atom-name order (== emission order, since `_ligand_atom_layout`
    # emits one real atom per token in the SAME order `_plan_ligand_tokens`
    # created them) lets a single boolean mask line up with the per-atom arrays
    # built below.
    if spec.ligand:
        code = spec.ligand.strip().upper()
        lig_names = [tk.ligand_atom_name for tk in tokens if tk.is_ligand]
        lig_fixed_mask = _atom_selection_mask(
            _resolve_ligand_atom_selection(spec.select_fixed_atoms, code), lig_names)
        lig_buried_mask = _atom_selection_mask(
            _resolve_ligand_atom_selection(spec.select_buried, code), lig_names)
        lig_exposed_mask = _atom_selection_mask(
            _resolve_ligand_atom_selection(spec.select_exposed, code), lig_names)
        lig_template = _ligand_template(code)
        _lig_pos_by_name = {nm: i for i, nm in enumerate(lig_template["names"])}
        _lig_atom_counter = 0
    lig_first_ti = next((ti for ti, tk in enumerate(tokens) if tk.is_ligand), None)

    pos = 0
    for ti, (tk, kind, (names, coord, tk_is_virtual, elements)) in enumerate(zip(tokens, token_kind, layouts)):
        n = len(names)
        s, e = pos, pos + n
        atom_names.extend(names)
        atom_elements.extend(elements if elements is not None else [None] * n)
        is_virtual[s:e] = tk_is_virtual
        fixed_coord = tk.fixed_coord
        fixed_seq = tk.fixed_seq
        if kind == "protein":
            has_cb = "CB" in names
            for j, nm in enumerate(names):
                if nm in BACKBONE_NAMES:
                    is_backbone[s + j] = True
                    if nm == "CA":
                        is_ca[s + j] = True
                        if not has_cb:
                            is_central[s + j] = True  # GLY: CA is the representative atom
                else:
                    is_sidechain[s + j] = True
                    if nm == "CB":
                        is_central[s + j] = True
        elif kind == "ligand":
            # A single-atom token IS its own representative atom (verified vs
            # a real reference capture: is_ca/is_central/is_backbone all True,
            # is_sidechain False, for every ligand atom — the AF3 "atomized
            # token" convention, distinct from protein/NA's multi-atom tokens).
            is_ligand_atom[s:e] = True
            is_ca[s:e] = True
            is_central[s:e] = True
            is_backbone[s:e] = True
            fixed_coord = bool(lig_fixed_mask[_lig_atom_counter])
            _lig_atom_counter += 1
        else:  # dna/rna: never backbone/sidechain-flagged; representative
            # atom is the base ring-center (C4 purine / C2 pyrimidine) —
            # verified vs a real reference capture (see module docstring).
            is_na_atom[s:e] = True
            central = _central_atom_name(tk.res_name)
            if central is not None and central in names:
                j = names.index(central)
                is_ca[s + j] = True
                is_central[s + j] = True
        if fixed_coord:
            is_motif_atom_fixed_coord[s:e] = True
            atom_coord[s:e] = coord
            motif_pos[s:e] = coord - com
        if fixed_seq:
            is_motif_atom_fixed_seq[s:e] = True
        if tk.is_unindexed:
            is_motif_atom_unindexed[s:e] = True
            # Reference override for unindexed tokens: `is_ca` is forced onto
            # the token's FIRST atom regardless of its real name (design_
            # transforms.py: "Ensure is_ca represents one and the first atom
            # only for unindexed tokens") — verified vs a real capture.
            is_ca[s:e] = False
            if n:
                is_ca[s] = True
        # ref_space_uid is a RESIDUE-level index (biotite's get_residue_starts
        # grouping), not a token-level one: a ligand's per-atom tokens all
        # belong to the SAME underlying residue, so they all share the ligand
        # residue's index rather than each atom claiming its own (verified vs
        # a real reference capture: all 33 ligand atoms -> ref_space_uid=180,
        # the token-index of the FIRST ligand token, not 180..212).
        ref_space_uid[s:e] = lig_first_ti if (tk.is_ligand and lig_first_ti is not None) else ti
        atom_to_token_map[s:e] = ti
        token_res_name.append("UNK" if tk.is_ligand else (tk.res_name or "GAP"))
        token_chain.append(tk.chain)
        token_res_id.append(tk.res_id)
        token_is_motif[ti] = tk.is_motif
        token_is_unindexed[ti] = tk.is_unindexed
        token_break_before[ti] = tk.is_chain_break_before
        token_is_fully_fixed_coord[ti] = bool(is_motif_atom_fixed_coord[s:e].all()) if e > s else False
        pos = e

    # --- reference-conformer features: ALL-ZERO/False for protein (motif or
    # designed alike) — CreateDesignReferenceFeatures.has_sequence excludes
    # protein entirely under generate_conformers_for_non_protein_only, so
    # ref_pos/ref_charge/ref_pos_is_ground_truth never get filled for protein;
    # real motif geometry flows only through motif_pos. For NA (not excluded),
    # ref_mask=True and ref_element is the real per-atom atomic-number one-hot
    # (verified vs real reference captures — see module docstring); ref_pos
    # stays 0 (the real reference-conformer 3D geometry needs RDKit/CCD
    # embedding this port does not vendor — documented gap) and ref_charge
    # stays 0 (matches both real captures: no formally-charged atoms in the
    # standard-nucleotide neutral conformer).
    ref_pos = np.zeros((L, 3), dtype=np.float32)
    ref_mask = np.array(is_na_atom | is_ligand_atom, dtype=bool)
    ref_pos_is_ground_truth = np.zeros(L, dtype=bool)
    ref_charge = np.zeros(L, dtype=np.int8)
    ref_element = np.zeros((L, 128), dtype=np.float32)
    ref_element[:, 0] = 1.0
    for i, (elem, is_na) in enumerate(zip(atom_elements, is_na_atom)):
        if is_na and elem in _ELEMENT_TO_ATOMIC_NUMBER:
            ref_element[i, 0] = 0.0
            ref_element[i, _ELEMENT_TO_ATOMIC_NUMBER[elem]] = 1.0
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_names)  # [L,4,64] f32 (live-pipeline dtype)
    has_zero_occupancy = np.zeros(L, dtype=bool)  # forced False at inference regardless of input

    # atomworks-ungated defaults (all-zero in the default inference config —
    # verified vs a real reference capture: FeaturizeAtoms' rasa_binned default
    # bin is excluded from the one-hot; no hbond/hotspot annotation present).
    ref_atomwise_rasa = np.zeros((L, 3), dtype=np.int64)
    active_donor = np.zeros(L, dtype=np.int64)
    active_acceptor = np.zeros(L, dtype=np.int64)
    is_atom_level_hotspot = np.zeros((L, 1), dtype=np.float32)

    # --- ligand (F3/F4) reference-conformer + RASA-conditioning features ---
    # ref_pos/ref_charge/ref_element come from the real CCD template (matched
    # by real atom name, verified vs a real reference capture: element/charge
    # ARE real per-atom values for a ligand, unlike protein's always-zero
    # placeholder). ref_atom_name_chars is overridden to encode the ELEMENT
    # symbol, not the real atom name — `use_element_for_atom_names_of_
    # atomized_tokens=True` in the reference's default inference config
    # (verified: a real capture's ligand rows decode to "C   "/"N   ", not
    # "C22 "/"N9  "). select_buried/select_exposed become the one-hot
    # `ref_atomwise_rasa` bin DIRECTLY (bin 0=buried, 2=exposed) — this is a
    # user-specified per-atom label at inference, NOT a computed SASA value
    # (rfd3.inference.input_parsing assigns `rasa_bin` straight from the
    # selection; `rfd3.transforms.rasa.CalculateRASA`'s real Shrake-Rupley
    # computation is training-only, never invoked at inference).
    if spec.ligand:
        lig_atom_idx = np.where(is_ligand_atom)[0]
        for k, atom_i in enumerate(lig_atom_idx):
            nm = atom_names[atom_i]
            tmpl_i = _lig_pos_by_name.get(nm)
            if tmpl_i is not None:
                ref_pos[atom_i] = lig_template["coord"][tmpl_i]
                ref_charge[atom_i] = lig_template["charges"][tmpl_i]
                elem = lig_template["elements"][tmpl_i]
                if elem in _ELEMENT_TO_ATOMIC_NUMBER:
                    ref_element[atom_i, 0] = 0.0
                    ref_element[atom_i, _ELEMENT_TO_ATOMIC_NUMBER[elem]] = 1.0
                ref_atom_name_chars[atom_i] = _encode_atom_names_like_af3([elem])[0]
            if lig_buried_mask[k]:
                ref_atomwise_rasa[atom_i] = [1, 0, 0]
            elif lig_exposed_mask[k]:
                ref_atomwise_rasa[atom_i] = [0, 0, 1]

    # --- token-level features ---
    restype = _restype_onehot(token_res_name).astype(np.int64)  # [I,32] one-hot int64
    motif_token_class = np.zeros(I, dtype=np.int8)
    motif_token_class[token_is_motif] = 1
    motif_token_class[token_is_unindexed] = 2
    ref_motif_token_type = np.eye(3, dtype=np.int8)[motif_token_class]
    ref_plddt = np.where(token_is_motif, 0, 1).astype(np.int64)
    is_non_loopy = np.zeros((I, 1), dtype=np.float32)
    is_motif_token_unindexed = token_is_unindexed.copy()
    is_motif_token_with_fully_fixed_coord = token_is_fully_fixed_coord

    is_protein_tok = np.array([k == "protein" for k in token_kind], dtype=bool)
    is_rna_tok = np.array([k == "rna" for k in token_kind], dtype=bool)
    is_dna_tok = np.array([k == "dna" for k in token_kind], dtype=bool)
    is_ligand_tok = np.array([k == "ligand" for k in token_kind], dtype=bool)
    _POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "LYS", "ARG", "ASP", "GLU"}
    is_polar = is_protein_tok & np.isin(np.array(token_res_name), np.array(list(_POLAR)))

    is_N_term = np.zeros(I, dtype=bool)
    is_C_term = np.zeros(I, dtype=bool)
    for ti in range(I):
        first_seg = (ti == 0) or token_break_before[ti] or (token_chain[ti] != token_chain[ti - 1])
        is_N_term[ti] = first_seg
        # The real chain's C-terminus lands on the last non-unindexed token
        # even though the array continues with the appended unindexed block
        # (verified vs a real reference capture: the token right before the
        # first unindexed token IS flagged C-terminus).
        entering_unindexed = token_is_unindexed[ti + 1] and not token_is_unindexed[ti] if ti + 1 < I else False
        last_in_seg = (ti == I - 1) or token_break_before[ti + 1] or (token_chain[ti + 1] != token_chain[ti]) or entering_unindexed
        is_C_term[ti] = last_in_seg
    # Unindexed tokens never carry a terminus flag (verified vs a real
    # reference capture: terminus_type is all-zero for every unindexed token,
    # regardless of island boundaries — add_protein_termini_annotation is not
    # re-applied to them). Same for NA (5'/3' ends): terminus_type is
    # protein-only in the reference contract — verified all-zero for DNA/RNA
    # on both real captures.
    is_N_term[token_is_unindexed | ~is_protein_tok] = False
    is_C_term[token_is_unindexed | ~is_protein_tok] = False
    terminus_type = np.zeros((I, 2), dtype=np.int64)
    terminus_type[is_C_term, 0] = 1
    terminus_type[is_N_term, 1] = 1

    chain_to_asym = {}
    for c in token_chain:
        if c not in chain_to_asym:
            chain_to_asym[c] = len(chain_to_asym)  # 0-based
    asym_id = np.array([chain_to_asym[c] for c in token_chain], dtype=np.int64)

    # entity_id/sym_id: chains sharing the SAME full real-chain residue-name
    # sequence (not just the contig-selected subset) are the same entity,
    # with sym_id enumerating replica copies — verified vs a real reference
    # capture (dsDNA_basic: chain A and B are the same 12-mer palindrome, so
    # they share entity_id with distinct sym_id). A synthetic (designed)
    # chain has no real sequence to match and always starts a fresh entity.
    chain_full_seq: dict[str, tuple] = {}
    for r in residues:
        chain_full_seq.setdefault(r.chain, []).append((r.res_id, r.res_name))
    chain_full_seq = {c: tuple(name for _, name in sorted(v)) for c, v in chain_full_seq.items()}
    entity_of_seq: dict[tuple, int] = {}
    chain_entity: dict[str, int] = {}
    next_entity = 0
    for c in chain_to_asym:  # insertion order == order of first appearance among tokens
        seq = chain_full_seq.get(c)
        if seq is None or seq not in entity_of_seq:
            eid = next_entity
            next_entity += 1
            if seq is not None:
                entity_of_seq[seq] = eid
        else:
            eid = entity_of_seq[seq]
        chain_entity[c] = eid
    entity_id = np.array([chain_entity[c] for c in token_chain], dtype=np.int64)
    sym_counter: dict[int, int] = {}
    chain_sym: dict[str, int] = {}
    for c in chain_to_asym:
        e = chain_entity[c]
        chain_sym[c] = sym_counter.get(e, 0)
        sym_counter[e] = chain_sym[c] + 1
    sym_id = np.array([chain_sym[c] for c in token_chain], dtype=np.int32)
    residue_index = np.zeros(I, dtype=np.int32)
    _per_chain_ctr = {}
    for ti, c in enumerate(token_chain):
        _per_chain_ctr.setdefault(c, 0)
        if tokens[ti].is_ligand:
            # All of a ligand's per-atom tokens are the SAME underlying
            # residue (one CCD instance) -> they all share residue_index=0 on
            # their (fresh, ligand-only) chain rather than each incrementing
            # it — verified vs a real reference capture.
            residue_index[ti] = 0
        else:
            residue_index[ti] = _per_chain_ctr[c]  # 0-based per chain
            _per_chain_ctr[c] += 1
    token_index = np.arange(I, dtype=np.int64)

    # token_bonds: ALL FALSE for standard contiguous protein (not the peptide-
    # bond graph — encodes inter-token bonds for modified residues/crosslinks/
    # ligands only). A ligand IS one such case: since each atom is its own
    # token, the real intra-ligand covalent bond graph (from the CCD template)
    # becomes real inter-TOKEN bonds — verified vs a real reference capture
    # (a 33-heavy-atom ligand token block has a real, non-trivial token_bonds
    # submatrix, not all-zero).
    token_bonds = np.zeros((I, I), dtype=bool)
    if spec.ligand:
        lig_token_idx = np.where(is_ligand_tok)[0]
        name_to_tok = {tokens[ti].ligand_atom_name: ti for ti in lig_token_idx}
        for u, v in lig_template["bonds"]:
            tu, tv = name_to_tok.get(lig_template["names"][u]), name_to_tok.get(lig_template["names"][v])
            if tu is not None and tv is not None:
                token_bonds[tu, tv] = token_bonds[tv, tu] = True

    # unindexing_pair_mask: True = RPE must NOT leak relative position between
    # this token pair (UnindexFlaggedTokens.create_unindexed_masks). Indexed<->
    # unindexed is ALWAYS masked; unindexed<->unindexed is masked unless the
    # two tokens are in the same "island" (contiguous '-' range in the unindex
    # spec). Verified vs a real reference capture (scripts/rfd3_port/
    # parity_artifacts/parity_unindex.py).
    unindexing_pair_mask = np.zeros((I, I), dtype=bool)
    ui = token_is_unindexed
    if ui.any():
        unindexing_pair_mask = ui[:, None] ^ ui[None, :]
        idx_ui = np.where(ui)[0]
        island = np.cumsum([tokens[i].unindex_new_island for i in idx_ui])
        sub_mask = island[:, None] != island[None, :]
        unindexing_pair_mask[np.ix_(idx_ui, idx_ui)] = sub_mask

    bf = lambda a: torch.from_numpy(a)
    f = {
        # atom-level
        "ref_atom_name_chars": bf(ref_atom_name_chars),                # [L,4,64] f32
        "ref_pos": bf(ref_pos),                                        # [L,3] f32
        "ref_mask": torch.from_numpy(ref_mask),                        # [L] bool
        "ref_element": bf(ref_element),                                # [L,128] f32
        "ref_charge": torch.from_numpy(ref_charge),                    # [L] int8
        "ref_space_uid": torch.from_numpy(ref_space_uid),              # [L] int64
        "ref_pos_is_ground_truth": torch.from_numpy(ref_pos_is_ground_truth),  # [L] bool
        "has_zero_occupancy": torch.from_numpy(has_zero_occupancy),     # [L] bool
        "ref_is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "ref_is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "ref_atomwise_rasa": torch.from_numpy(ref_atomwise_rasa),      # [L,3] int64
        "active_donor": torch.from_numpy(active_donor),                # [L] int64
        "active_acceptor": torch.from_numpy(active_acceptor),          # [L] int64
        "is_atom_level_hotspot": bf(is_atom_level_hotspot),            # [L,1] f32
        "is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "is_motif_atom_with_fixed_seq": torch.from_numpy(is_motif_atom_fixed_seq),
        "is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "motif_pos": bf(motif_pos),                                  # [L,3] f32
        "is_ca": torch.from_numpy(is_ca),
        "is_central": torch.from_numpy(is_central),
        "is_backbone": torch.from_numpy(is_backbone),
        "is_sidechain": torch.from_numpy(is_sidechain),
        "is_virtual": torch.from_numpy(is_virtual),
        "atom_to_token_map": torch.from_numpy(atom_to_token_map),     # [L] int32
        # token-level
        "restype": torch.from_numpy(restype),                          # [I,32] int64 one-hot
        "ref_motif_token_type": torch.from_numpy(ref_motif_token_type),  # [I,3] int8 one-hot
        "ref_plddt": torch.from_numpy(ref_plddt),                      # [I] int64
        "is_non_loopy": bf(is_non_loopy),                              # [I,1] f32
        "is_motif_token_unindexed": torch.from_numpy(is_motif_token_unindexed),  # [I] bool
        "is_motif_token_with_fully_fixed_coord": torch.from_numpy(is_motif_token_with_fully_fixed_coord),
        "is_protein": torch.from_numpy(is_protein_tok),                # [I] bool
        "is_rna": torch.from_numpy(is_rna_tok),                        # [I] bool
        "is_dna": torch.from_numpy(is_dna_tok),                        # [I] bool
        "is_ligand": torch.from_numpy(is_ligand_tok),                  # [I] bool
        "is_polar": torch.from_numpy(is_polar),                        # [I] bool
        "terminus_type": torch.from_numpy(terminus_type),              # [I,2] int64 one-hot
        "asym_id": torch.from_numpy(asym_id),
        "entity_id": torch.from_numpy(entity_id),
        "sym_id": torch.from_numpy(sym_id),                          # [I] int32
        "residue_index": torch.from_numpy(residue_index),              # [I] int32
        "token_index": torch.from_numpy(token_index),
        "token_bonds": torch.from_numpy(token_bonds),                  # [I,I] bool
        "unindexing_pair_mask": torch.from_numpy(unindexing_pair_mask),
    }
    return f
