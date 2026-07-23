"""RFD3 host featurizer: build the ``f`` feature dict + initial token/pair state
metadata from a real user PDB/CIF + a parsed :class:`InputSpecification`.

This is the N6 core that turns a from-PDB design input into the ``f`` dict the
on-device TokenInitializer + DiffusionModule consume. It is grounded in the
real RosettaCommons/foundry featurizer
(``models/rfd3/src/rfd3/transforms/design_transforms.py`` + ``virtual_atoms.py``
+ ``pipelines.py``, production branch, 2026-07-23) and the ``f`` contract the
TokenInitializer reads.

Status (p12): ATOM-LEVEL parity landed for the protein-binder/motif-scaffold
case (F1/F6). The reference does NOT pad every token to a fixed 14 atoms —
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
on a protein-only input (contig with indexed-motif + designed regions + chain
breaks; ``unindex`` with plain/`-`-range components). Ligand (F3), NA (F2/F8),
enzyme (F4) and symmetry (F5) raise NotImplementedError with a pointer to the
reference transform; so does the unindex numeric-offset-tie syntax and
dict-form per-atom fixing (see ``_plan_unindexed_tokens``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .rfd3_input import InputSpecification, parse_contig, ChainBreak, Indexed, Designed, DesignedRange

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

# AF3 standard 20-AA restype order (index 0..19); restype=32 channels, designed
# (sequence-unknown) tokens use class index 31 (confirmed vs a real reference
# capture, see module docstring / state notes).
_RESTYPE_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
_RESTYPE_TO_IDX = {n: i for i, n in enumerate(_RESTYPE_ORDER)}
DESIGNED_RESTYPE_IDX = 31
RESTYPE_DIM = 32

PROTEIN_RES = set(_RESTYPE_ORDER)
DNA_RES = {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "DX"}
RNA_RES = {"A", "C", "G", "U"}


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
    protein-only case; it does NOT reproduce atomworks' bond perception,
    assembly building, or CCD normalisation (parity-ungated)."""
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
    return arr


@dataclass
class _Residue:
    chain: str
    res_id: int
    res_name: str
    atom_names: list[str]
    coord: np.ndarray  # [n_atoms, 3]


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
        ))
    return res


def _is_protein(r: _Residue) -> bool:
    return r.res_name in PROTEIN_RES


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
    Returns (names, coord[n,3], is_virtual[n]=False)."""
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
    return names, coord_arr, np.zeros(len(names), dtype=bool)


def _designed_atom_layout():
    """Full 14-slot template for a designed (sequence-unknown) residue: the
    5 backbone+CB slots are real (undetermined, coord 0); V0..V8 are virtual
    pad atoms (PadTokensWithVirtualAtoms). Returns (names, coord[14,3], is_virtual[14])."""
    names = list(ATOM14_ATOM_NAMES)
    coord = np.zeros((14, 3), dtype=np.float32)
    is_virtual = np.zeros(14, dtype=bool)
    is_virtual[5:] = True
    return names, coord, is_virtual


# -- contig -> token plan ----------------------------------------------------
@dataclass
class _Token:
    chain: str
    res_id: int          # input PDB res_id (motif) or assigned (designed)
    res_name: str        # real name (motif) or "" (designed -> DESIGNED_RESTYPE_IDX)
    is_motif: bool       # fixed coord + fixed seq (indexed OR unindexed motif)
    is_designed: bool    # diffused
    is_unindexed: bool    # unindexed motif (from the unindex field)
    is_chain_break_before: bool  # /0 precedes this token
    residue: _Residue | None  # source residue for motif tokens, else None
    unindex_new_island: bool = False  # this unindexed token starts a new RPE-leak island


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
                raise NotImplementedError("non-protein unindexed motif (NA/ligand) — p14+")
            tokens.append(_Token(c.chain, rid, r.res_name, True, False, True,
                                 False, r, unindex_new_island=(k == 0)))
    return tokens


def _plan_tokens_from_contig(spec: InputSpecification, residues: list[_Residue]) -> list[_Token]:
    """Map a parsed contig + the parsed structure's residues into an ordered
    token plan. Indexed components pull residues from the input structure
    (motif, fixed coord+seq); Designed/DesignedRange components become diffused
    tokens; ChainBreak marks the next token. Unindexed motif tokens (from
    ``spec.unindex``, see ``_plan_unindexed_tokens``) are appended last."""
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_contig()
    tokens: list[_Token] = []
    break_before_next = False
    indexed_keys: set[tuple[str, int]] = set()
    for c in comps:
        if isinstance(c, ChainBreak):
            break_before_next = True
            continue
        if isinstance(c, Indexed):
            for rid in range(c.start, c.end + 1):
                r = by_key.get((c.chain, rid))
                if r is None:
                    raise ValueError(f"contig indexes {c.chain}{rid} not present in input structure")
                if not _is_protein(r):
                    raise NotImplementedError("non-protein indexed motif (NA/ligand) — p12")
                tokens.append(_Token(c.chain, rid, r.res_name, True, False, False,
                                     break_before_next, r))
                indexed_keys.add((c.chain, rid))
                break_before_next = False
            continue
        if isinstance(c, (Designed, DesignedRange)):
            n = c.length if isinstance(c, Designed) else (c.lo + c.hi) // 2
            # designed residues continue on the same chain as the preceding indexed block
            chain = tokens[-1].chain if tokens else "A"
            base = (tokens[-1].res_id + 1) if tokens else 1
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


def featurize(structure_path: str | Path, spec: InputSpecification) -> dict[str, torch.Tensor]:
    """Build the ``f`` feature dict for one design spec from a real PDB/CIF.

    Protein-binder (F1) + motif scaffolding (F6) on protein-only input, with
    atom-level parity vs the real reference (see module docstring).
    """
    arr = load_structure(structure_path)
    residues = _group_residues(arr)
    if any(not _is_protein(r) for r in residues):
        raise NotImplementedError("non-protein input (NA/ligand) — p12")
    tokens = _plan_tokens_from_contig(spec, residues)
    I = len(tokens)

    # Per-token atom layout (variable count: motif = real heavy atoms only,
    # designed = full 14-slot template).
    layouts = [
        _motif_atom_layout(tk.residue) if tk.is_motif else _designed_atom_layout()
        for tk in tokens
    ]
    L = sum(len(nm) for nm, _, _ in layouts)

    # The whole design is centered at the center of mass of the real (motif)
    # atoms (verified vs a real reference capture: motif_pos == real_coord -
    # com, where com = mean over every motif atom actually emitted).
    motif_coords = [c for tk, (nm, c, _) in zip(tokens, layouts) if tk.is_motif and len(nm)]
    if motif_coords:
        com = np.concatenate(motif_coords, axis=0).mean(axis=0)
    else:
        com = np.zeros(3, dtype=np.float32)

    atom_names: list[str] = []
    atom_coord = np.zeros((L, 3), dtype=np.float32)
    is_virtual = np.zeros(L, dtype=bool)
    is_backbone = np.zeros(L, dtype=bool)
    is_sidechain = np.zeros(L, dtype=bool)
    is_ca = np.zeros(L, dtype=bool)
    is_central = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_coord = np.zeros(L, dtype=bool)
    is_motif_atom_fixed_seq = np.zeros(L, dtype=bool)
    is_motif_atom_unindexed = np.zeros(L, dtype=bool)
    motif_pos = np.zeros((L, 3), dtype=np.float32)
    ref_space_uid = np.zeros(L, dtype=np.int64)
    atom_to_token_map = np.zeros(L, dtype=np.int32)

    token_res_name = []
    token_chain = []
    token_res_id = []
    token_is_motif = np.zeros(I, dtype=bool)
    token_is_unindexed = np.zeros(I, dtype=bool)
    token_break_before = np.zeros(I, dtype=bool)

    pos = 0
    for ti, (tk, (names, coord, tk_is_virtual)) in enumerate(zip(tokens, layouts)):
        n = len(names)
        s, e = pos, pos + n
        atom_names.extend(names)
        is_virtual[s:e] = tk_is_virtual
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
        if tk.is_motif:
            is_motif_atom_fixed_coord[s:e] = True
            is_motif_atom_fixed_seq[s:e] = True
            atom_coord[s:e] = coord
            motif_pos[s:e] = coord - com
        if tk.is_unindexed:
            is_motif_atom_unindexed[s:e] = True
            # Reference override for unindexed tokens: `is_ca` is forced onto
            # the token's FIRST atom regardless of its real name (design_
            # transforms.py: "Ensure is_ca represents one and the first atom
            # only for unindexed tokens") — verified vs a real capture.
            is_ca[s:e] = False
            if n:
                is_ca[s] = True
        ref_space_uid[s:e] = ti
        atom_to_token_map[s:e] = ti
        token_res_name.append(tk.res_name or "UNK")
        token_chain.append(tk.chain)
        token_res_id.append(tk.res_id)
        token_is_motif[ti] = tk.is_motif
        token_is_unindexed[ti] = tk.is_unindexed
        token_break_before[ti] = tk.is_chain_break_before
        pos = e

    # --- reference-conformer features: ALL-ZERO/False for protein (motif or
    # designed alike) — CreateDesignReferenceFeatures.has_sequence excludes
    # protein entirely under generate_conformers_for_non_protein_only, so
    # ref_pos/ref_mask/ref_charge/ref_pos_is_ground_truth never get filled;
    # real motif geometry flows only through motif_pos. ref_element's one-hot
    # is therefore the constant index-0 row (its scalar source stays 0), not
    # real chemical identity. Verified vs a real reference capture.
    ref_pos = np.zeros((L, 3), dtype=np.float32)
    ref_mask = np.zeros(L, dtype=bool)
    ref_pos_is_ground_truth = np.zeros(L, dtype=bool)
    ref_charge = np.zeros(L, dtype=np.int8)
    ref_element = np.zeros((L, 128), dtype=np.float32)
    ref_element[:, 0] = 1.0
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_names)  # [L,4,64] f32 (live-pipeline dtype)
    has_zero_occupancy = np.zeros(L, dtype=bool)  # forced False at inference regardless of input

    # atomworks-ungated defaults (all-zero in the default inference config —
    # verified vs a real reference capture: FeaturizeAtoms' rasa_binned default
    # bin is excluded from the one-hot; no hbond/hotspot annotation present).
    ref_atomwise_rasa = np.zeros((L, 3), dtype=np.int64)
    active_donor = np.zeros(L, dtype=np.int64)
    active_acceptor = np.zeros(L, dtype=np.int64)
    is_atom_level_hotspot = np.zeros((L, 1), dtype=np.float32)

    # --- token-level features ---
    restype = _restype_onehot(token_res_name).astype(np.int64)  # [I,32] one-hot int64
    motif_token_class = np.zeros(I, dtype=np.int8)
    motif_token_class[token_is_motif] = 1
    motif_token_class[token_is_unindexed] = 2
    ref_motif_token_type = np.eye(3, dtype=np.int8)[motif_token_class]
    ref_plddt = np.where(token_is_motif, 0, 1).astype(np.int64)
    is_non_loopy = np.zeros((I, 1), dtype=np.float32)
    is_motif_token_unindexed = token_is_unindexed.copy()
    is_motif_token_with_fully_fixed_coord = token_is_motif.copy()

    is_protein_tok = np.ones(I, dtype=bool)
    is_rna_tok = np.zeros(I, dtype=bool)
    is_dna_tok = np.zeros(I, dtype=bool)
    is_ligand_tok = np.zeros(I, dtype=bool)
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
    # re-applied to them).
    is_N_term[token_is_unindexed] = False
    is_C_term[token_is_unindexed] = False
    terminus_type = np.zeros((I, 2), dtype=np.int64)
    terminus_type[is_C_term, 0] = 1
    terminus_type[is_N_term, 1] = 1

    chain_to_asym = {}
    for c in token_chain:
        if c not in chain_to_asym:
            chain_to_asym[c] = len(chain_to_asym)  # 0-based
    asym_id = np.array([chain_to_asym[c] for c in token_chain], dtype=np.int64)
    entity_id = asym_id.copy()
    sym_id = np.zeros(I, dtype=np.int32)
    residue_index = np.zeros(I, dtype=np.int32)
    _per_chain_ctr = {}
    for ti, c in enumerate(token_chain):
        _per_chain_ctr.setdefault(c, 0)
        residue_index[ti] = _per_chain_ctr[c]  # 0-based per chain
        _per_chain_ctr[c] += 1
    token_index = np.arange(I, dtype=np.int64)

    # token_bonds: ALL FALSE for standard contiguous protein (not the peptide-
    # bond graph — encodes inter-token bonds for modified residues/crosslinks/
    # ligands only).
    token_bonds = np.zeros((I, I), dtype=bool)

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
