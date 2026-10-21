# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""RFD4's checkpoint vocabulary, loaded into AtomWorks' generic token encoding."""

from importlib.resources import files

import numpy as np
import yaml
from tt_bio._vendor.atomworks.constants import AA_LIKE_CHEM_TYPES, DNA_LIKE_CHEM_TYPES, RNA_LIKE_CHEM_TYPES
from tt_bio._vendor.atomworks.ml.encoding_definitions import TokenEncoding

TOKEN_TYPE_IDS = {"aa": 0, "rna": 1, "dna": 2, "atomized": 3, "unknown": 4, "glycan": 5, "ncaa": 6, "mask": 4}


def _load_vocabulary() -> tuple[TokenEncoding, dict[str, tuple[str, ...]]]:
    """Read and validate the ordered 37-slot vocabulary packaged with RFD4."""
    resource = files("tt_bio._vendor.rfd4_prep").joinpath("resources/aa_rna_dna_glycan_ncaa.yaml")
    spec = yaml.safe_load(resource.read_text(encoding="utf-8"))
    groups = spec["residues"]
    token_atoms = {}
    for group, residues in groups.items():
        if group not in TOKEN_TYPE_IDS:
            raise ValueError(f"Unknown vocabulary token type: {group}")
        for name, layout in residues.items():
            atoms = [atom if atom != "-" else "" for atom in layout.split()]
            occupied = [atom for atom in atoms if atom]
            if not 1 <= len(atoms) <= 37 or len(set(occupied)) != len(occupied):
                raise ValueError(f"Invalid 37-slot atom layout for {name!r}")
            token_atoms[name] = np.array(atoms + [""] * (37 - len(atoms)))
    encoding = TokenEncoding(
        token_atoms=token_atoms,
        chemcomp_type_to_unknown=(
            dict.fromkeys(AA_LIKE_CHEM_TYPES, "UNK")
            | dict.fromkeys(DNA_LIKE_CHEM_TYPES, "DN")
            | dict.fromkeys(RNA_LIKE_CHEM_TYPES, "N")
        ),
    )
    return encoding, {group: tuple(residues) for group, residues in groups.items()}


UNIFIED_ATOM37_ENCODING, RESIDUES_BY_TOKEN_TYPE = _load_vocabulary()
NCAA_NAMES, GLYCAN_NAMES = RESIDUES_BY_TOKEN_TYPE["ncaa"], RESIDUES_BY_TOKEN_TYPE["glycan"]
