# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

"""Constants and lookup tables for Nesso."""

# Chains
chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}

# Residues & Tokens
canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
]

tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

# Atoms
num_elements = 128

chirality_types = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER",
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}
unk_chirality_type = "CHI_OTHER"

hybridization_map = [
    "S",
    "SP",
    "SP2",
    "SP2D",
    "SP3",
    "SP3D",
    "SP3D2",
    "OTHER",
    "UNSPECIFIED",
]
hybridization_type_ids = {hybrid: i for i, hybrid in enumerate(hybridization_map)}
unk_hybridization_type = "UNSPECIFIED"

# fmt: off
ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # noqa: E501
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],  # noqa: E501
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],  # noqa: E501
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],  # noqa: E501
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
}

res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'"
}

res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'"
}

res_to_center_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_center_atom.items()
}

res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_disto_atom.items()
}
# fmt: on

# Bonds
bond_types = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
unk_bond_type = "OTHER"

# Contacts
pocket_contact_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}

# Chunking
chunk_size_threshold = 384
