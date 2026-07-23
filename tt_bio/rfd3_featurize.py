"""RFD3 host featurizer: build the ``f`` feature dict + initial token/pair state
metadata from a real user PDB/CIF + a parsed :class:`InputSpecification`.

This is the N6 core that turns a from-PDB design input into the ``f`` dict the
on-device TokenInitializer + DiffusionModule consume. It is grounded in the
real RosettaCommons/foundry featurizer
(``models/rfd3/src/rfd3/transforms/design_transforms.py`` + ``pipelines.py``,
production branch, 2026-07-23) and the ``f`` contract the TokenInitializer
reads (``model/layers/encoders.py`` TokenInitializer.forward + the
``token_1d_features`` / ``atom_1d_features`` lists vendored in
``scripts/rfd3_port/rfd3_ref.py``).

Status (p10): the assembly core is landed and unit-verified for STRUCTURAL
invariants (shapes/dtypes/key relationships). It is NOT yet parity-gated
against a real reference ``f`` capture. The atomworks-dependent annotation
layer (RASA via surface, hbond donor/acceptor, atom-level hotspots, virtual
atom padding to atom14, reference-molecule conformer features for ligands,
ori_token inference) is reproduced here from the reference's documented
behaviour but its exact tensor encodings (one-hot vs class-index, index
assignment) MUST be confirmed against a captured design-``f`` golden on
vast.ai before any accuracy claim. The parity-compare harness lives at
``scripts/rfd3_port/parity_compare_f.py``; the capture script at
``scripts/rfd3_port/capture_binder_f.sh``. Both are ready to run; the gate
itself is owed to p11.

Coverage this pass: protein-binder (F1) + motif-scaffolding (F6) on a
protein-only input (contig with indexed-motif + designed regions + chain
breaks). Ligand (F3), NA (F2/F8), enzyme (F4) and symmetry (F5) raise
NotImplementedError with a pointer to the reference transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .rfd3_input import InputSpecification, parse_contig, ChainBreak, Indexed, Designed, DesignedRange

# -- atom14 layout (from rfd3/constants.py) ---------------------------------
ATOM14_ATOM_NAMES = ["N", "CA", "C", "O", "CB"] + [f"V{i}" for i in range(14 - 5)]
ATOM14_ATOM_ELEMENTS = ["N", "C", "C", "O", "C"] + ["VX" for _ in range(14 - 5)]
VIRTUAL_ATOM_ELEMENT_NAME = "VX"
BACKBONE_NAMES = {"N", "CA", "C", "O"}

# AF3 element vocabulary: index = atomic_number - 1 (H=1 -> idx 0), 128 bins.
# Matches atomworks ELEMENT_NAME_TO_ATOMIC_NUMBER / ref_element=128 contract.
_ELEMENT_TO_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53, "VX": 0,  # VX (virtual) -> 0 / all-zero one-hot
}

# AF3 standard 20-AA restype order (index 0..19) + UNK at 20; restype=32 channels
# leaves a margin. Exact index mapping must be confirmed vs the captured golden.
_RESTYPE_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
_RESTYPE_TO_IDX = {n: i for i, n in enumerate(_RESTYPE_ORDER)}
UNK_IDX = 20
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


def _element_onehot(elements: Sequence[str]) -> np.ndarray:
    """[N, 128] one-hot over atomic_number (idx = Z - 1). VX -> all-zero row."""
    out = np.zeros((len(elements), 128), dtype=np.float32)
    for i, e in enumerate(elements):
        z = _ELEMENT_TO_Z.get(str(e).strip().upper(), 0)
        if z > 0 and z <= 128:
            out[i, z - 1] = 1.0
    return out


def _restype_onehot(res_names: Sequence[str]) -> np.ndarray:
    """[I, 32] one-hot restype (UNK at idx 20; 21..31 unused -> zero)."""
    out = np.zeros((len(res_names), RESTYPE_DIM), dtype=np.float32)
    for i, r in enumerate(res_names):
        idx = _RESTYPE_TO_IDX.get(str(r).strip().upper(), UNK_IDX)
        out[i, idx] = 1.0
    return out


# -- structure loading (biotite; atomworks-free) -----------------------------
def load_structure(path: str | Path):
    """Parse a PDB/CIF into a biotite AtomArray with the annotations the
    featurizer needs. This replaces atomworks.io.parser.parse for the
    protein-only case; it does NOT reproduce atomworks' bond perception,
    assembly building, or CCD normalisation (parity-ungated)."""
    import biotite.structure as struc
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
    # biotite gives chain_id, res_id, res_name, atom_name, element, coord, occupancy, b_factor
    return arr


@dataclass
class _Residue:
    chain: str
    res_id: int
    res_name: str
    atom_names: list[str]
    elements: list[str]
    coord: np.ndarray  # [n_atoms, 3]
    occupancy: np.ndarray  # [n_atoms]
    b_factor: np.ndarray  # [n_atoms]


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
            elements=[str(e_).strip().upper() for e_ in sub.element],
            coord=np.asarray(sub.coord, dtype=np.float32),
            occupancy=np.asarray(getattr(sub, "occupancy", np.ones(len(sub))), dtype=np.float32),
            b_factor=np.asarray(getattr(sub, "b_factor", np.zeros(len(sub))), dtype=np.float32),
        ))
    return res


def _is_protein(r: _Residue) -> bool:
    return r.res_name in PROTEIN_RES


def _pad_protein_to_atom14(r: _Residue):
    """Return atom14 names/elements/coord/is_virtual for a protein residue.
    Real atoms N,CA,C,O,CB are placed in slots 0..4; missing real atoms and
    slots 5..13 are virtual (VX, coord 0). For GLY, CB is virtual."""
    names = list(ATOM14_ATOM_NAMES)
    elements = list(ATOM14_ATOM_ELEMENTS)
    coord = np.zeros((14, 3), dtype=np.float32)
    is_virtual = np.zeros(14, dtype=bool)
    real_by_name = {n: (e, c) for n, e, c in zip(r.atom_names, r.elements, r.coord)}
    for slot, nm in enumerate(["N", "CA", "C", "O", "CB"]):
        if nm in real_by_name:
            e, c = real_by_name[nm]
            elements[slot] = e
            coord[slot] = c
            is_virtual[slot] = False
        else:
            # missing real backbone/CB atom -> virtual placeholder
            elements[slot] = VIRTUAL_ATOM_ELEMENT_NAME
            is_virtual[slot] = True
    # slots 5..13 are always virtual
    is_virtual[5:] = True
    return names, elements, coord, is_virtual


# -- contig -> token plan ----------------------------------------------------
@dataclass
class _Token:
    chain: str
    res_id: int          # input PDB res_id (motif) or assigned (designed)
    res_name: str        # real name (motif) or "" (designed -> UNK)
    is_motif: bool       # fixed coord + fixed seq (indexed motif)
    is_designed: bool    # diffused
    is_unindexed: bool    # unindexed motif (from the unindex field)
    is_chain_break_before: bool  # /0 precedes this token
    # atom14 layout (filled for protein)
    atom_names: list[str]
    elements: list[str]
    coord: np.ndarray     # [14, 3]
    is_virtual: np.ndarray  # [14]


def _plan_tokens_from_contig(spec: InputSpecification, residues: list[_Residue]) -> list[_Token]:
    """Map a parsed contig + the parsed structure's residues into an ordered
    token plan. Indexed components pull residues from the input structure
    (motif, fixed coord+seq); Designed/DesignedRange components become diffused
    tokens (UNK restype, zero ref_pos); ChainBreak marks the next token."""
    # index residues by (chain, res_id) for motif lookup
    by_key = {(r.chain, r.res_id): r for r in residues}
    comps = spec.parsed_contig()
    tokens: list[_Token] = []
    break_before_next = False
    # sample a designed length for DesignedRange (deterministic: midpoint)
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
                    raise NotImplementedError("non-protein indexed motif (NA/ligand) — p11")
                nm, el, co, iv = _pad_protein_to_atom14(r)
                tokens.append(_Token(c.chain, rid, r.res_name, True, False, False,
                                     break_before_next, nm, el, co, iv))
                break_before_next = False
            continue
        if isinstance(c, (Designed, DesignedRange)):
            n = c.length if isinstance(c, Designed) else (c.lo + c.hi) // 2
            # designed tokens: assign a fresh chain (B for a binder scaffold) + continuing res_id
            # For the simple F1/F6 case, designed residues share the LAST motif chain if the
            # contig is contiguous, else a new chain. We follow the common binder convention:
            # designed region continues on the same chain as the preceding indexed block.
            chain = tokens[-1].chain if tokens else "A"
            base = (tokens[-1].res_id + 1) if tokens else 1
            for k in range(n):
                nm = list(ATOM14_ATOM_NAMES); el = list(ATOM14_ATOM_ELEMENTS)
                co = np.zeros((14, 3), dtype=np.float32); iv = np.ones(14, dtype=bool)
                iv[:5] = False  # N,CA,C,O,CB slots exist but are diffused (coord 0)
                tokens.append(_Token(chain, base + k, "", False, True, False,
                                     break_before_next, nm, el, co, iv))
                break_before_next = False
            continue
        raise NotImplementedError(f"contig component {c!r} not supported this pass (p11)")
    # unindex field (F6 unindexed motif) -> mark those tokens unindexed
    if spec.unindex is not None:
        raise NotImplementedError("unindex field (unindexed motif) -> p11")
    return tokens


def featurize(structure_path: str | Path, spec: InputSpecification) -> dict[str, torch.Tensor]:
    """Build the ``f`` feature dict for one design spec from a real PDB/CIF.

    Returns the atom-level + token-level tensors the TokenInitializer reads
    (see module docstring for the contract). Protein-binder (F1) + motif
    scaffolding (F6) on protein-only input this pass.

    NOT parity-gated: the atomworks-dependent encodings (rasa, donor/acceptor,
    hotspot, ref conformer, exact index assignment) use the reference's
    documented default-when-absent behaviour. Confirm against a captured
    design-``f`` golden before any accuracy claim (parity_compare_f.py).
    """
    arr = load_structure(structure_path)
    residues = _group_residues(arr)
    # reject ligand/NA inputs this pass
    if any(not _is_protein(r) for r in residues):
        raise NotImplementedError("non-protein input (NA/ligand) — p11")
    tokens = _plan_tokens_from_contig(spec, residues)
    I = len(tokens)
    L = I * 14  # atom14

    # --- flatten atom14 across tokens ---
    atom_names: list[str] = []
    atom_elements: list[str] = []
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
    occupancy = np.ones(L, dtype=np.float32)
    b_factor = np.zeros(L, dtype=np.float32)
    token_res_name = []
    token_chain = []
    token_res_id = []
    token_is_motif = np.zeros(I, dtype=bool)
    token_is_unindexed = np.zeros(I, dtype=bool)
    token_break_before = np.zeros(I, dtype=bool)

    for ti, tk in enumerate(tokens):
        s = ti * 14
        atom_names.extend(tk.atom_names)
        atom_elements.extend(tk.elements)
        atom_coord[s:s + 14] = tk.coord
        is_virtual[s:s + 14] = tk.is_virtual
        for j, nm in enumerate(tk.atom_names):
            if nm in BACKBONE_NAMES and not tk.is_virtual[j]:
                is_backbone[s + j] = True
                if nm == "CA":
                    is_ca[s + j] = True
                    is_central[s + j] = True  # CA is the protein token representative
            elif nm == "CB" and not tk.is_virtual[j]:
                is_sidechain[s + j] = True
        if tk.is_motif:
            is_motif_atom_fixed_coord[s:s + 14] = True
            is_motif_atom_fixed_seq[s:s + 14] = True
            motif_pos[s:s + 14] = tk.coord
        if tk.is_unindexed:
            is_motif_atom_unindexed[s:s + 14] = True
        token_res_name.append(tk.res_name or "UNK")
        token_chain.append(tk.chain)
        token_res_id.append(tk.res_id)
        token_is_motif[ti] = tk.is_motif
        token_is_unindexed[ti] = tk.is_unindexed
        token_break_before[ti] = tk.is_chain_break_before

    # ref_pos: motif atoms keep their real coord; designed/virtual -> 0
    ref_pos = motif_pos.copy()  # motif_pos is already 0 for non-motif
    # ref_mask: bool, 1 where the reference conformer is provided (motif with seq) else 0
    ref_mask = is_motif_atom_fixed_seq.copy()
    # ref_element / ref_charge / ref_atom_name_chars (golden dtypes: bf16 one-hot / int8 / bf16 flat)
    ref_element = _element_onehot(atom_elements)            # [L,128] float -> bf16
    ref_charge = np.zeros(L, dtype=np.int8)
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_names).reshape(L, 256)  # [L,256]
    # has_zero_occupancy
    has_zero_occupancy = (occupancy == 0.0)
    # ref_space_uid: per-residue segment id (same value for all 14 atoms of a token)
    ref_space_uid = np.repeat(np.arange(I, dtype=np.int64), 14)
    # atom_to_token_map (golden dtype int32)
    atom_to_token_map = np.repeat(np.arange(I, dtype=np.int32), 14)

    # --- atomworks-ungated defaults (confirm vs golden p11) ---
    # ref_atomwise_rasa: one-hot int64 [L,3]; all-zero when no rasa annotation (the dsDNA_basic
    # golden is all-zero; a protein-binder WOULD have non-zero rasa computed from surface area,
    # which needs atomworks -> owed p11. Set all-zero (the no-annotation default) and flag.
    ref_atomwise_rasa = np.zeros((L, 3), dtype=np.int64)
    # active_donor / active_acceptor: int64, zeros when no hbond annotation (reference default).
    active_donor = np.zeros(L, dtype=np.int64)
    active_acceptor = np.zeros(L, dtype=np.int64)
    # is_atom_level_hotspot: bf16 [L,1], zeros when no hotspot annotation.
    is_atom_level_hotspot = np.zeros((L, 1), dtype=np.float32)

    # --- token-level features ---
    restype = _restype_onehot(token_res_name).astype(np.int64)  # [I,32] one-hot int64
    # ref_motif_token_type: [I,3] one-hot int8 (0 non-motif, 1 indexed motif, 2 unindexed motif)
    motif_token_class = np.zeros(I, dtype=np.int8)
    motif_token_class[token_is_motif] = 1
    motif_token_class[token_is_unindexed] = 2
    ref_motif_token_type = np.eye(3, dtype=np.int8)[motif_token_class]
    # ref_plddt: int64 [I], 0/1 (golden shows 0/1, not -1/0/+1; inference default 0).
    ref_plddt = np.zeros(I, dtype=np.int64)
    # is_non_loopy: bf16 [I,1], default 0.
    is_non_loopy = np.zeros((I, 1), dtype=np.float32)
    # is_motif_token_unindexed: bool [I] (token-level spread of is_motif_atom_unindexed).
    is_motif_token_unindexed = token_is_unindexed.copy()
    # is_motif_token_with_fully_fixed_coord: bool [I] (all 14 atoms fixed-coord -> motif tokens).
    is_motif_token_with_fully_fixed_coord = token_is_motif.copy()

    # --- molecule-type token masks (bool [I]) ---
    is_protein_tok = np.ones(I, dtype=bool)   # protein-only input this pass
    is_rna_tok = np.zeros(I, dtype=bool)
    is_dna_tok = np.zeros(I, dtype=bool)
    is_ligand_tok = np.zeros(I, dtype=bool)
    # is_polar: is_protein & res_name in polar set (util_transforms.py:446).
    _POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "LYS", "ARG", "ASP", "GLU"}
    is_polar = is_protein_tok & np.isin(np.array(token_res_name), np.array(list(_POLAR)))

    # --- terminus_type [I,2] int64 one-hot: col0 = C-terminus, col1 = N-terminus ---
    # A token is N-term if it starts a chain-segment (first, or break_before, or chain change);
    # C-term if it ends a segment (last, or next has break, or next is different chain).
    is_N_term = np.zeros(I, dtype=bool)
    is_C_term = np.zeros(I, dtype=bool)
    for ti in range(I):
        first_seg = (ti == 0) or token_break_before[ti] or (token_chain[ti] != token_chain[ti - 1])
        is_N_term[ti] = first_seg
        last_in_seg = (ti == I - 1) or token_break_before[ti + 1] or (token_chain[ti + 1] != token_chain[ti])
        is_C_term[ti] = last_in_seg
    terminus_type = np.zeros((I, 2), dtype=np.int64)
    terminus_type[is_C_term, 0] = 1
    terminus_type[is_N_term, 1] = 1

    # --- indices (golden dtypes: asym_id/entity_id/token_index int64; residue_index/sym_id int32) ---
    chain_to_asym = {}
    for c in token_chain:
        if c not in chain_to_asym:
            chain_to_asym[c] = len(chain_to_asym) + 1
    asym_id = np.array([chain_to_asym[c] for c in token_chain], dtype=np.int64)
    entity_id = asym_id.copy()
    sym_id = np.zeros(I, dtype=np.int32)
    residue_index = np.zeros(I, dtype=np.int32)
    _per_chain_ctr = {}
    for ti, c in enumerate(token_chain):
        _per_chain_ctr.setdefault(c, 0)
        _per_chain_ctr[c] += 1
        residue_index[ti] = _per_chain_ctr[c]
    token_index = np.arange(I, dtype=np.int64)

    # --- token_bonds [I, I] bool: peptide bond between consecutive same-chain tokens
    #     with no chain break between them and contiguous res_id. ---
    token_bonds = np.zeros((I, I), dtype=bool)
    for ti in range(I - 1):
        if (token_chain[ti] == token_chain[ti + 1]
                and not token_break_before[ti + 1]
                and token_res_id[ti + 1] == token_res_id[ti] + 1):
            token_bonds[ti, ti + 1] = True
            token_bonds[ti + 1, ti] = True

    # --- unindexing_pair_mask [I, I] bool: all-False this pass (no unindex field). ---
    unindexing_pair_mask = np.zeros((I, I), dtype=bool)

    bf = lambda a: torch.from_numpy(a).to(torch.bfloat16)
    f = {
        # atom-level (golden dtypes from the captured dsDNA_basic `f`)
        "ref_atom_name_chars": bf(ref_atom_name_chars),                # [L,256] bf16
        "ref_pos": bf(ref_pos),                                        # [L,3] bf16
        "ref_mask": torch.from_numpy(ref_mask),                        # [L] bool
        "ref_element": bf(ref_element),                               # [L,128] bf16
        "ref_charge": torch.from_numpy(ref_charge),                    # [L] int8
        "ref_space_uid": torch.from_numpy(ref_space_uid),              # [L] int64
        "ref_pos_is_ground_truth": torch.from_numpy(is_motif_atom_fixed_seq),  # [L] bool
        "has_zero_occupancy": torch.from_numpy(has_zero_occupancy),     # [L] bool
        "ref_is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "ref_is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "ref_atomwise_rasa": torch.from_numpy(ref_atomwise_rasa),      # [L,3] int64
        "active_donor": torch.from_numpy(active_donor),                # [L] int64
        "active_acceptor": torch.from_numpy(active_acceptor),          # [L] int64
        "is_atom_level_hotspot": bf(is_atom_level_hotspot),            # [L,1] bf16
        "is_motif_atom_with_fixed_coord": torch.from_numpy(is_motif_atom_fixed_coord),
        "is_motif_atom_with_fixed_seq": torch.from_numpy(is_motif_atom_fixed_seq),
        "is_motif_atom_unindexed": torch.from_numpy(is_motif_atom_unindexed),
        "motif_pos": bf(motif_pos),                                  # [L,3] bf16
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
        "is_non_loopy": bf(is_non_loopy),                              # [I,1] bf16
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



