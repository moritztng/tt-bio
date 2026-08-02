"""OpenDDE structural-token featurizer (opendde/data/tokenizer.py + featurizer.py port).

Builds the residue-token -> structural-token feature dict StructuralTokenExpander.__call__
and the diffusion module's atom<->token broadcast need, from tt-bio's own residue-token-native
data pipeline (tt_bio.protenix_data). No biotite AtomArray dependency: tt-bio's per-residue
atom names already come from tt_bio.data.const.ref_atoms (the same table protein_atom_features
uses to build ref_pos), so the backbone/sidechain split and atom<->structural-token maps are
derived independently from the identical source, in the identical iteration order.

Scope: protein chains + ligand chains (the covalent-inhibitor case). Protein residues
expand to backbone/sidechain structural tokens (glycine and any sidechain-less residue degenerate
to a single protein_bb token -- upstream's own fallback). Ligand atoms are already tokenized
per-atom by build_complex_features (one residue-token per ligand atom, restype UNK), so each
expands to a single "atom"-role structural token -- mirroring upstream AtomArrayTokenizer's
_get_atom_token_from_parent (opendde/data/tokenizer.py). Nucleic-acid backbone/base splitting
follows the identical pattern (NUCLEIC_BACKBONE_ATOMS) and is left for when a nucleic co-folding
target is on the critical path.
"""
import torch

from .data import const
from .protenix_data import RESTYPE_ORDER

# opendde/data/tokenizer.py STRUCTURAL_TOKEN_ROLES + PROTEIN_BACKBONE_ATOMS (verified
# 2026-07-12 against /tmp/opendde-src a0d5134, both dicts copied verbatim).
STRUCTURAL_TOKEN_ROLES = {
    "atom": 0, "protein_bb": 1, "protein_sc": 2,
    "dna_bb": 3, "dna_base": 4, "rna_bb": 5, "rna_base": 6,
}
PROTEIN_BACKBONE_ATOMS = frozenset(["N", "CA", "C", "O", "OXT"])
MOL_TYPE_LIGAND = 3  # feats["mol_type"] value for a ligand atom-token (MOL_TYPE_IDS["ligand"])

_LETTER_TO_RES = {v: k for k, v in const.prot_token_to_letter.items()}


def _residue_atom_names(res, is_c_terminal):
    """Same atom list + OXT-append rule as protenix_data.protein_atom_features, so the
    resulting atom_to_structural_token(atom)_idx lines up 1:1 with ref_pos's atom order."""
    atoms = list(const.ref_atoms[res])
    if is_c_terminal:
        atoms = atoms + ["OXT"]
    return atoms


def _residue_adjacency(asym_id, res_id):
    """Strict same-chain, adjacent-res_id fallback (opendde/data/core/featurizer.py
    get_polymer_residue_graph's fallback path -- the only path that applies here, since
    tt-bio's residue-token features carry no explicit inter-residue bond graph)."""
    n = asym_id.shape[0]
    prev = torch.full((n,), -1, dtype=torch.long)
    nxt = torch.full((n,), -1, dtype=torch.long)
    for i in range(n - 1):
        if int(asym_id[i]) == int(asym_id[i + 1]) and int(res_id[i + 1]) - int(res_id[i]) == 1:
            nxt[i] = i + 1
            prev[i + 1] = i
    return prev, nxt


def build_structural_token_features(feats):
    """feats: a tt_bio.protenix_data residue-token feature dict (restype one-hot, asym_id,
    residue_index) for a single- or multi-chain PROTEIN complex (n_res == n_token, one token
    per residue -- true for protein-only inputs; ligand/NA chains are not yet split, see
    module docstring).

    Returns the dict StructuralTokenExpander.__call__ consumes directly
    (parent_residue_idx, subtoken_role_id, asym_id, prev/next_parent_residue_idx) plus
    atom_to_structural_token_idx / atom_to_structural_tokatom_idx for the diffusion module's
    atom<->structural-token broadcast (opendde/model/opendde.py expand_to_structural_tokens).
    """
    asym_id = feats["asym_id"]
    res_id = feats["residue_index"]
    aatype = feats["restype"].argmax(-1)
    n_res = aatype.shape[0]
    mol_type = feats["mol_type"]
    n_atom = feats["atom_to_token_idx"].shape[0]

    # C-terminal per CHAIN: each asym_id's last residue carries OXT, matching
    # protenix_data.protein_atom_features (called once per chain). A per-chain vs
    # global mismatch is N_chain-1 OXT atoms off and breaks atom_to_structural_token
    # alignment for multi-chain input (the 4969-vs-4967 failure on the 9dsg Fab+RBD).
    asym = asym_id.tolist()
    is_c_term = [r == n_res - 1 or asym[r] != asym[r + 1] for r in range(n_res)]

    parent, role, twin = [], [], []
    atom_tok, atom_tokatom = [], []
    for r in range(n_res):
        # Ligand atom-token: one "atom"-role structural token per ligand atom
        # (upstream AtomArrayTokenizer._get_atom_token_from_parent). build_complex_features
        # tokenizes a ligand PER ATOM, so this parent token IS the atom (1:1); tokatom=0.
        if int(mol_type[r]) == MOL_TYPE_LIGAND:
            st_idx = len(parent)
            parent.append(r); role.append(STRUCTURAL_TOKEN_ROLES["atom"]); twin.append(-1)
            atom_tok.append(st_idx); atom_tokatom.append(0)
            continue
        aa = int(aatype[r])
        res = _LETTER_TO_RES[RESTYPE_ORDER[aa]] if aa < len(RESTYPE_ORDER) else "UNK"
        names = _residue_atom_names(res, is_c_terminal=is_c_term[r])
        is_bb = [nm in PROTEIN_BACKBONE_ATOMS for nm in names]
        has_sidechain = not all(is_bb)

        if res == "GLY" or not has_sidechain:
            bb_idx = len(parent)
            parent.append(r); role.append(STRUCTURAL_TOKEN_ROLES["protein_bb"]); twin.append(-1)
            for k in range(len(names)):
                atom_tok.append(bb_idx); atom_tokatom.append(k)
            continue

        bb_idx = len(parent)
        parent.append(r); role.append(STRUCTURAL_TOKEN_ROLES["protein_bb"]); twin.append(bb_idx + 1)
        sc_idx = len(parent)
        parent.append(r); role.append(STRUCTURAL_TOKEN_ROLES["protein_sc"]); twin.append(bb_idx)

        bb_k = sc_k = 0
        for nm in names:
            if nm in PROTEIN_BACKBONE_ATOMS:
                atom_tok.append(bb_idx); atom_tokatom.append(bb_k); bb_k += 1
            else:
                atom_tok.append(sc_idx); atom_tokatom.append(sc_k); sc_k += 1

    parent_t = torch.tensor(parent, dtype=torch.long)
    prev_res, next_res = _residue_adjacency(asym_id, res_id)
    # atom_to_structural_token_idx is indexed by ATOM (feats atom order). The protein
    # branch appends per-residue atoms in ref_atoms order and the ligand branch appends
    # one atom per ligand token in token order -- both match feats' atom order within
    # each chain, so the concatenated list is in feats atom order iff the count matches.
    assert len(atom_tok) == n_atom, (
        f"structural-token atom map misaligned: {len(atom_tok)} atoms built vs "
        f"{n_atom} in feats (ligand atom<->token order drift?)")
    return {
        "structural_token_index": torch.arange(len(parent), dtype=torch.long),
        "parent_residue_idx": parent_t,
        "subtoken_role_id": torch.tensor(role, dtype=torch.long),
        "twin_token_idx": torch.tensor(twin, dtype=torch.long),
        "prev_parent_residue_idx": prev_res.index_select(0, parent_t),
        "next_parent_residue_idx": next_res.index_select(0, parent_t),
        "asym_id": asym_id,
        "atom_to_structural_token_idx": torch.tensor(atom_tok, dtype=torch.long),
        "atom_to_structural_tokatom_idx": torch.tensor(atom_tokatom, dtype=torch.long),
    }
