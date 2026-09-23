"""OpenDDE structural-token featurizer (opendde/data/tokenizer.py + featurizer.py port).

Builds the residue-token -> structural-token feature dict StructuralTokenExpander.__call__
and the diffusion module's atom<->token broadcast need, from tt-bio's own residue-token-native
data pipeline (tt_bio.protenix_data). No biotite AtomArray dependency: the atom names are read
back off the feats' own ``ref_atom_name_chars``, so the split can never disagree with the atom
order it indexes into.

The rule is upstream AtomArrayTokenizer.tokenize_structural's. A token that owns one atom (a
ligand atom, an atom of a `modifications:` residue) stays a single "atom"-role structural
token. A standard polymer residue splits into a backbone token and a second token holding the
rest: its sidechain for protein, its base for DNA/RNA. A residue with nothing outside the
backbone set (glycine, the unknown nucleotide N/DN) stays a single backbone token.
"""
import torch

from .protenix_data import MOL_TYPE_IDS

# opendde/data/tokenizer.py STRUCTURAL_TOKEN_ROLES, PROTEIN_BACKBONE_ATOMS and
# NUCLEIC_BACKBONE_ATOMS, copied verbatim from opendde==1.0.3.
STRUCTURAL_TOKEN_ROLES = {
    "atom": 0, "protein_bb": 1, "protein_sc": 2,
    "dna_bb": 3, "dna_base": 4, "rna_bb": 5, "rna_base": 6,
}
PROTEIN_BACKBONE_ATOMS = frozenset(["N", "CA", "C", "O", "OXT"])
NUCLEIC_BACKBONE_ATOMS = frozenset([
    "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "O2*", "C1*", "O5T", "O3T",
])

# mol_type id -> (backbone atom set, backbone role, second role)
_SPLIT = {
    MOL_TYPE_IDS["protein"]: (PROTEIN_BACKBONE_ATOMS, "protein_bb", "protein_sc"),
    MOL_TYPE_IDS["dna"]: (NUCLEIC_BACKBONE_ATOMS, "dna_bb", "dna_base"),
    MOL_TYPE_IDS["rna"]: (NUCLEIC_BACKBONE_ATOMS, "rna_bb", "rna_base"),
}


def _atom_names(feats):
    """Per-atom names decoded from ref_atom_name_chars (one-hot of ord(c) - 32, 4 chars)."""
    codes = feats["ref_atom_name_chars"].reshape(-1, 4, 64).argmax(-1) + 32
    return ["".join(map(chr, row)).rstrip() for row in codes.tolist()]


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
    """feats: a tt_bio.protenix_data residue-token feature dict (build_complex_features).

    Returns the dict StructuralTokenExpander.__call__ consumes directly
    (parent_residue_idx, subtoken_role_id, asym_id, prev/next_parent_residue_idx) plus
    atom_to_structural_token_idx / atom_to_structural_tokatom_idx for the diffusion module's
    atom<->structural-token broadcast (opendde/model/opendde.py expand_to_structural_tokens).
    """
    asym_id = feats["asym_id"]
    n_res = asym_id.shape[0]
    a2t = feats["atom_to_token_idx"].long()
    names = _atom_names(feats)
    mol_type = feats["mol_type"].tolist()
    # Atoms of each residue token, in feats atom order.
    atoms_of = [[] for _ in range(n_res)]
    for a, t in enumerate(a2t.tolist()):
        atoms_of[t].append(a)

    parent, role, twin = [], [], []
    atom_tok = torch.empty(len(names), dtype=torch.long)
    atom_tokatom = torch.empty(len(names), dtype=torch.long)
    for r, atoms in enumerate(atoms_of):
        split = _SPLIT.get(mol_type[r]) if len(atoms) > 1 else None
        if split is None:
            # One structural token for the whole residue token: upstream's
            # _get_atom_token_from_parent. Every atomized token lands here.
            groups = [(STRUCTURAL_TOKEN_ROLES["atom"], atoms)]
        else:
            bb_set, bb_role, other_role = split
            bb = [a for a in atoms if names[a] in bb_set]
            rest = [a for a in atoms if names[a] not in bb_set]
            groups = [(STRUCTURAL_TOKEN_ROLES[bb_role], bb or atoms)]
            if bb and rest:
                groups.append((STRUCTURAL_TOKEN_ROLES[other_role], rest))
        first = len(parent)
        for g, (rl, members) in enumerate(groups):
            st = first + g
            parent.append(r); role.append(rl)
            twin.append(first + 1 - g if len(groups) == 2 else -1)
            atom_tok[members] = st
            atom_tokatom[members] = torch.arange(len(members))

    parent_t = torch.tensor(parent, dtype=torch.long)
    prev_res, next_res = _residue_adjacency(asym_id, feats["residue_index"])
    return {
        "structural_token_index": torch.arange(len(parent), dtype=torch.long),
        "parent_residue_idx": parent_t,
        "subtoken_role_id": torch.tensor(role, dtype=torch.long),
        "twin_token_idx": torch.tensor(twin, dtype=torch.long),
        "prev_parent_residue_idx": prev_res.index_select(0, parent_t),
        "next_parent_residue_idx": next_res.index_select(0, parent_t),
        "asym_id": asym_id,
        "atom_to_structural_token_idx": atom_tok,
        "atom_to_structural_tokatom_idx": atom_tokatom,
    }
