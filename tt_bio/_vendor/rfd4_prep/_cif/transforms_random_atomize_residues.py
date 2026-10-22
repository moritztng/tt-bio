# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Vocabulary-aware atomization policy for RFD4 training."""

from collections.abc import Mapping

import numpy as np
from tt_bio._vendor.atomworks.ml.transforms.random_atomize_residues import RandomAtomizeResidues as AtomWorksRandomAtomizeResidues

from .encoding import RESIDUES_BY_TOKEN_TYPE

_DEFAULT_PROBABILITIES = {"aa": 0.005, "rna": 0.005, "dna": 0.005, "ncaa": 0.05, "glycan": 0.05}


class RandomAtomizeResidues(AtomWorksRandomAtomizeResidues):
    """Atomize by token class; unlisted classes retain their current atomization."""

    def __init__(
        self, probabilities: Mapping[str, float] | None = None, *, rng: np.random.Generator | None = None
    ) -> None:
        probabilities = _DEFAULT_PROBABILITIES if probabilities is None else probabilities
        unknown = set(probabilities) - _DEFAULT_PROBABILITIES.keys()
        if unknown:
            raise ValueError(f"Unknown residue classes: {sorted(unknown)}")
        super().__init__(
            {residue: p for group, p in probabilities.items() for residue in RESIDUES_BY_TOKEN_TYPE[group]},
            rng=rng,
        )
