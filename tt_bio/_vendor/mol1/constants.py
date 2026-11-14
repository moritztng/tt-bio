# Vendored from Mol-1 (https://github.com/molence/Mol-1) @ e6e64866ee89640a91cef7ee777a85f1d8be03bb, Apache-2.0.
# Copyright 2024 ByteDance and/or its affiliates; modifications Copyright 2026 Molence Bio.
# Regenerate with scripts/mol1_io/vendor_msa.py; do not edit by hand.
from typing import Final, Set

PRO_STD_RESIDUES = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}

RNA_STD_RESIDUES = {
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
}

DNA_STD_RESIDUES = {
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
}

RNA_START_INDEX = 21
DNA_START_INDEX = 26

GAP = {"-": 31}
STD_RESIDUES = PRO_STD_RESIDUES | RNA_STD_RESIDUES | DNA_STD_RESIDUES
STD_RESIDUES_WITH_GAP = STD_RESIDUES | GAP

MSA_PROTEIN_SEQ_TO_ID = {
    "A": 0,
    "B": 3,  # Same as D.
    "C": 4,
    "D": 3,
    "E": 6,
    "F": 13,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 20,  # Same as unknown (X).
    "K": 11,
    "L": 10,
    "M": 12,
    "N": 2,
    "O": 20,  # Same as unknown (X).
    "P": 14,
    "Q": 5,
    "R": 1,
    "S": 15,
    "T": 16,
    "U": 4,  # Same as C.
    "V": 19,
    "W": 17,
    "X": 20,
    "Y": 18,
    "Z": 6,  # Same as E.
    "-": 31,
}
MSA_RNA_SEQ_TO_ID = {
    # Map non-standard residues to UNK_NUCLEIC (N) -> 25
    **{chr(i): 25 for i in range(ord("A"), ord("Z") + 1)},
    # Continue the RNA indices from where Protein indices left off.
    "-": 31,
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
}
MSA_DNA_SEQ_TO_ID = {
    # Map non-standard residues to UNK_NUCLEIC (N) -> 30
    **{chr(i): 30 for i in range(ord("A"), ord("Z") + 1)},
    # Continue the DNA indices from where DNA indices left off.
    "-": 31,
    "A": 26,
    "G": 27,
    "C": 28,
    "T": 29,
}

PROTEIN_CHAIN: Final[str] = "polypeptide(L)"
RNA_CHAIN: Final[str] = "polyribonucleotide"
DNA_CHAIN: Final[str] = "polydeoxyribonucleotide"
BRANCHED_CHAIN: Final[str] = "branched"
MACROLIDE_CHAIN: Final[str] = "macrolide"
NON_POLYMER_CHAIN: Final[str] = "non-polymer"
LIGAND_CHAIN_TYPES: Final[Set[str]] = {
    BRANCHED_CHAIN,
    MACROLIDE_CHAIN,
    NON_POLYMER_CHAIN,
}
# Most common _entity_poly.types.
STANDARD_POLYMER_CHAIN_TYPES: Final[Set[str]] = {
    PROTEIN_CHAIN,
    DNA_CHAIN,
    RNA_CHAIN,
}

