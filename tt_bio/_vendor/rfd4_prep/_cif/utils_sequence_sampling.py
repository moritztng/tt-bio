# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Constants used by sequence-conditioning sampling policies."""

import numpy as np
from tt_bio._vendor.atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA

# fmt: off
_UNIPROT_AA_PERCENT = {
    "ALA": 8.75, "ARG": 5.90, "ASN": 3.79, "ASP": 5.48, "CYS": 1.46,
    "GLN": 3.95, "GLU": 6.34, "GLY": 7.07, "HIS": 2.33, "ILE": 5.14,
    "LEU": 9.72, "LYS": 5.01, "MET": 2.27, "PHE": 3.78, "PRO": 5.26,
    "SER": 7.30, "THR": 5.59, "TRP": 1.28, "TYR": 2.79, "VAL": 6.78,
}
# fmt: on

UNIPROT_AA_PRIOR = np.asarray([_UNIPROT_AA_PERCENT[token] for token in STANDARD_AA], dtype=np.float64)
UNIPROT_AA_PRIOR /= UNIPROT_AA_PRIOR.sum()
UNIPROT_AA_PRIOR.setflags(write=False)
"""UniProtKB amino-acid composition in :data:`atomworks.constants.STANDARD_AA` order."""

UNIFORM_RNA_PRIOR = np.full(len(STANDARD_RNA), 1 / len(STANDARD_RNA), dtype=np.float64)
UNIFORM_RNA_PRIOR.setflags(write=False)
UNIFORM_DNA_PRIOR = np.full(len(STANDARD_DNA), 1 / len(STANDARD_DNA), dtype=np.float64)
UNIFORM_DNA_PRIOR.setflags(write=False)
