# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import numpy as np
import torch
from tt_bio._vendor.atomworks.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, UNKNOWN_AA

from .encoding import UNIFIED_ATOM37_ENCODING

LOCAL_LATENT_RESIDUES = tuple(name for name in UNIFIED_ATOM37_ENCODING.tokens if name not in ("<A>", "<G>"))
"""Canonical residue tokens represented by the autoencoder's local latents."""

DEFAULT_UNDESIRED_RES_NAMES = [name for name in AF3_EXCLUDED_LIGANDS if name not in LOCAL_LATENT_RESIDUES]
"""Default training exclusions, preserving residues supported by the autoencoder."""

af2_atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

atom_order = {atom_type: i for i, atom_type in enumerate(af2_atom_types)}

_atom_to_slot = {}
for res_type in [*list(STANDARD_AA), UNKNOWN_AA]:
    if res_type in UNIFIED_ATOM37_ENCODING.token_atoms:
        atom37_layout = UNIFIED_ATOM37_ENCODING.token_atoms[res_type]
        for slot_idx, atom_name in enumerate(atom37_layout):
            atom_name_stripped = atom_name.strip()
            if atom_name_stripped and atom_name_stripped not in _atom_to_slot:
                _atom_to_slot[atom_name_stripped] = slot_idx
atom_types = [atom for atom, _ in sorted(_atom_to_slot.items(), key=lambda x: x[1])]

# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# OpenFold ordering: https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L556

ATOM_NUMBERING = {
    atom: i
    for i, atom in enumerate(
        [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OD1",
            "ND2",
            "CG1",
            "CG2",
            "CD",
            "CE",
            "NZ",
            "OD2",
            "OE1",
            "NE2",
            "OE2",
            "OH",
            "NE",
            "NH1",
            "NH2",
            "OG1",
            "SD",
            "ND1",
            "SG",
            "NE1",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
            "OXT",
        ]
    )
}
"""Default ordering of atoms in (dimension 1 of) a protein structure tensor."""
PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor([ATOM_NUMBERING[atom] for atom in atom_types])
OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor([atom_types.index(atom) for atom in ATOM_NUMBERING])

# Amino acid constants (standard 20 AAs + unknown)
restypes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # 20

# Amino acid 3-letter to 1-letter conversion (standard 20 AAs)
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
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
}

# Amino acid 1-letter to 3-letter conversion (inverse mapping)
restype_3to1 = {v: k for k, v in restype_1to3.items()}

AA_CHARACTER_PROTORP = {
    "ALA": "A",
    "CYS": "P",
    "GLU": "C",
    "ASP": "C",
    "GLY": "A",
    "PHE": "A",
    "ILE": "A",
    "HIS": "P",
    "LYS": "C",
    "MET": "A",
    "LEU": "A",
    "ASN": "P",
    "GLN": "P",
    "PRO": "A",
    "SER": "P",
    "ARG": "C",
    "THR": "P",
    "TRP": "P",
    "VAL": "A",
    "TYR": "P",
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CE", "NZ"],
    "MET": ["CG", "CE", "SD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CA", "CB", "CG", "CD", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}

# AME (Active site Motif Elements) - multiple possible atom configurations per residue
# Randomly choose one configuration per residue instance
AME_ATOMS = {
    "PHE": [["C", "O"], ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ"]],
    "ALA": [["C", "CA", "CB", "N"], ["CA", "N"], ["CA", "CB"], ["C", "CA", "O"], ["C", "O"]],
    "HIS": [
        ["CD2", "CE1", "CG", "ND1", "NE2"],
        ["CB", "CD2", "CE1", "CG", "ND1", "NE2"],
        ["CD2", "CE1", "NE2"],
        ["CE1", "CG", "ND1"],
        ["CE1", "ND1", "NE2"],
    ],
    "GLY": [["CA", "N"], ["C", "CA", "N"]],
    "TRP": [
        ["CD1", "CD2", "CE2", "CG", "CZ2", "NE1"],
        ["CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "NE1"],
    ],
    "ASP": [
        ["CB", "CG", "OD1", "OD2"],
        ["CG", "OD2"],
        ["CG", "OD1"],
        ["C", "CA", "CB", "N"],
        ["C", "CA", "CB", "CG", "N", "O"],
    ],
    "ARG": [
        ["CZ", "NE", "NH1", "NH2"],
        ["C", "CA", "CB", "N"],
        ["CZ", "NH1"],
        ["CD", "CZ", "NE"],
        ["CD", "CG", "CZ", "NE", "NH1", "NH2"],
        ["CB", "CD", "CG", "CZ", "NE"],
        ["CZ", "NH2"],
        ["C", "O"],
        ["CD", "CZ", "NE", "NH1", "NH2"],
    ],
    "LYS": [["CD", "CE", "NZ"], ["CE", "NZ"]],
    "THR": [["CA", "CB", "CG2", "OG1"], ["C", "CA", "CB", "N"], ["CB", "OG1"]],
    "GLU": [["C", "CA", "O"], ["CA", "N"], ["CD", "OE2"], ["CD", "OE1"], ["CD", "CG", "OE1", "OE2"]],
    "ILE": [["C", "CA", "O"], ["C", "CA", "CB", "N"], ["CB", "CD1", "CG1", "CG2"]],
    "SER": [["CB", "OG"], ["C", "CA", "O"], ["CA", "CB", "OG"], ["C", "O"], ["CA", "N"], ["C", "CA", "CB", "N"]],
    "ASN": [["CB", "CG", "ND2", "OD1"], ["CA", "N"], ["CG", "ND2"]],
    "CYS": [["CA", "CB", "SG"], ["CB", "SG"]],
    "GLN": [["CA", "N"], ["CD", "OE1"], ["CD", "CG", "NE2", "OE1"]],
    "TYR": [["CE1", "CE2", "CZ", "OH"], ["CZ", "OH"], ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "OH"]],
    "PRO": [["CD", "CG", "N"], ["CA", "CB", "CD", "CG", "N"]],
    "LEU": [["CB", "CD1", "CD2", "CG"]],
    "MET": [["CE", "CG", "SD"]],
    "VAL": [["CB", "CG1", "CG2"]],
}


def _build_unified_atom37_mask() -> np.ndarray:
    """Build the legal Atom37 slots for each residue in the unified vocabulary."""
    token_names = list(UNIFIED_ATOM37_ENCODING.tokens)
    n_tokens = len(token_names)
    n_atoms = 37  # Fixed size for atom37 encoding
    mask = np.zeros((n_tokens, n_atoms), dtype=np.float32)

    for token_idx, token_name in enumerate(token_names):
        if token_name not in UNIFIED_ATOM37_ENCODING.token_atoms:
            # Token not in encoding (e.g., unknown types) - leave as all zeros
            continue

        # Get the atom names for this token from the encoding
        encoding_atom_names = UNIFIED_ATOM37_ENCODING.token_atoms[token_name]

        # For each position in the atom37 representation
        for pos_idx in range(min(n_atoms, len(encoding_atom_names))):
            encoding_atom = encoding_atom_names[pos_idx].strip()

            if encoding_atom:  # Non-empty position
                mask[token_idx, pos_idx] = 1.0

    return mask


# Unified mask for the canonical 147-class encoding.
UNIFIED_RESTYPE_ATOM37_MASK = _build_unified_atom37_mask()

TIP_BY_RESTYPE = {
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # fix both rings
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],  # fixed ring
    "TYR": ["CZ", "OH"],  # keeps ring dihedral flexible
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "GLN": ["CG", "CD", "OE1", "NE2"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "CYS": ["CB", "SG"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "VAL": ["CG1", "CG2"],
    "ILE": ["CB", "CG2"],
    "MET": ["SD", "CE"],
    "LYS": ["CE", "NZ"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "PRO": None,
    "ALA": None,
    "GLY": None,
    "UNK": None,
    "MSK": None,
}
"""
For convenience, we allow BKBN or TIP to be used as a shortcut when specifying the inference config.
TIP is the largest set of fixed atom given at least 2 tip atoms
"""
# fmt: on


# Mapping from residue type to its backbone and sidechain atoms (for convenience)
ATOM_REGION_BY_RESI = {
    "ALA": {"bb": ("N", "CA", "C", "O"), "sc": ("CB",)},
    "ARG": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2")},
    "ASN": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "OD1", "ND2")},
    "ASP": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "OD1", "OD2")},
    "CYS": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "SG")},
    "GLN": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD", "OE1", "NE2")},
    "GLU": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD", "OE1", "OE2")},
    "GLY": {"bb": ("N", "CA", "C", "O"), "sc": ()},
    "HIS": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "ND1", "CD2", "CE1", "NE2")},
    "ILE": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG1", "CG2", "CD1")},
    "LEU": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD1", "CD2")},
    "LYS": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD", "CE", "NZ")},
    "MET": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "SD", "CE")},
    "PHE": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")},
    "PRO": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD")},
    "SER": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "OG")},
    "THR": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "OG1", "CG2")},
    "TRP": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2")},
    "TYR": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")},
    "VAL": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG1", "CG2")},
    "UNK": {"bb": ("N", "CA", "C", "O"), "sc": ("CB",)},
    "MAS": {"bb": ("N", "CA", "C", "O"), "sc": ("CB",)},
    "DA": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"),
        "sc": ("N9", "C4", "N3", "C2", "N1", "C6", "C5", "N7", "C8", "N6"),
    },
    "DC": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"),
        "sc": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    },
    "DG": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"),
        "sc": ("N9", "C4", "N3", "C2", "N1", "C6", "C5", "N7", "C8", "N2", "O6"),
    },
    "DT": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"),
        "sc": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
    },
    "DX": {"bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"), "sc": ()},
    "A": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"),
        "sc": ("N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"),
    },
    "C": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"),
        "sc": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    },
    "G": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"),
        "sc": ("N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"),
    },
    "U": {
        "bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"),
        "sc": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    },
    "X": {"bb": ("O4'", "C1'", "C2'", "OP1", "P", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "O2'"), "sc": ()},
    "HIS_D": {"bb": ("N", "CA", "C", "O"), "sc": ("CB", "CG", "NE2", "CD2", "CE1", "ND1")},
}


# Known planar sidechain atoms for each canonical residue type:
PLANAR_ATOMS_BY_RESI = {
    "ALA": [],
    "ARG": ["NH1", "NH2", "CZ", "NE", "CD"],
    "ASN": ["OD1", "ND2", "CG", "CB"],
    "ASP": ["OD1", "OD2", "CG", "CB"],
    "CYS": [],
    "GLN": ["OE1", "NE2", "CD", "CG"],
    "GLU": ["OE1", "OE2", "CD", "CG"],
    "GLY": [],
    "HIS": ["ND1", "CE1", "NE2", "CD2", "CG", "CB"],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PHE": ["CZ", "CE1", "CE2", "CD1", "CD2", "CG", "CB"],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": ["CH2", "CZ3", "CZ2", "CE3", "CE2", "CD2", "NE1", "CD1", "CG", "CB"],
    "TYR": ["OH", "CZ", "CE1", "CE2", "CD1", "CD2", "CG", "CB"],
    "VAL": [],
    "UNK": [],
    "MAS": [],
    "DA": ["N6", "C6", "N1", "C2", "N3", "C4", "C5", "N7", "C8", "N9"],
    "DC": ["N4", "C4", "N3", "O2", "C2", "C5", "C6", "N1"],
    "DG": ["O6", "C6", "N1", "N2", "C2", "N3", "C4", "C5", "N7", "C8", "N9"],
    "DT": ["O4", "O2", "N3", "C4", "C2", "C5", "C6", "N1", "C7"],
    "DX": [],
    "A": ["N6", "C6", "N1", "C2", "N3", "C4", "C5", "N7", "C8", "N9"],
    "C": ["N4", "C4", "N3", "O2", "C2", "C5", "C6", "N1"],
    "G": ["O6", "C6", "N1", "N2", "C2", "N3", "C4", "C5", "N7", "C8", "N9"],
    "U": ["O4", "O2", "N3", "C4", "C2", "C5", "C6", "N1"],
    "X": [],
    "HIS_D": ["ND1", "CD2", "CE1", "NE2", "CG", "CB"],
}
