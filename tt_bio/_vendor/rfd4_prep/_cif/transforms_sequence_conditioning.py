# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Finalize sequence conditioning consistently for training and inference."""

import numpy as np
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_SEQ
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.utils.token import apply_and_spread_token_wise
from biotite.structure import spread_residue_wise

from .conditions import C_CRDGRP, SEQUENCE_PROBABILITY_CONDITIONS
from .encoding import RESIDUES_BY_TOKEN_TYPE


class FinalizeSequenceConditioning(Transform):
    """Require identities for restricted types, atomized tokens, and fixed non-backbone atoms.

    Args:
        token_types: Vocabulary groups allowed to retain task-selected conditioning.
            Existing identities are preserved within these groups; all other groups
            use ground-truth identities. Defaults to standard amino acids and RNA.
    """

    def __init__(self, token_types: tuple[str, ...] = ("aa", "rna")) -> None:
        super().__init__()
        invalid = set(token_types) - RESIDUES_BY_TOKEN_TYPE.keys()
        if invalid:
            raise ValueError(f"Unknown token types: {sorted(invalid)}")
        self.residue_names = tuple(name for group in token_types for name in RESIDUES_BY_TOKEN_TYPE[group])

    def forward(self, data: dict) -> dict:
        aa = as_atom_array_plus(data["atom_array"])
        allowed = np.isin(aa.res_name, self.residue_names)
        existing = C_SEQ.mask(aa, default="generate")
        newly_exact = ~allowed & ~existing
        # Training task probabilities yield to mandatory identities; inference conflicts remain errors.
        if not data.get("is_inference", False) and newly_exact.any():
            for condition in SEQUENCE_PROBABILITY_CONDITIONS:
                if condition.has_annotation(aa):
                    probability = condition.annotation(aa).copy()
                    probability[newly_exact] = np.nan
                    condition.set_annotation(aa, probability)

        sequence = np.where(allowed, C_SEQ.annotation(aa, default="generate"), aa.res_name)
        conditioned = existing | ~allowed
        required = conditioned.copy()
        if "atomize" in aa.get_annotation_categories():
            required |= aa.atomize
        crd_mask = C_CRD.mask(aa, default="generate") | C_CRDGRP.is_internal(aa)
        if crd_mask.any():
            ensure_annotations(aa, "is_protein_backbone", "is_rna_backbone", "is_dna_backbone")
            is_backbone = aa.is_protein_backbone | aa.is_rna_backbone | aa.is_dna_backbone
            required |= crd_mask & ~is_backbone
        required = apply_and_spread_token_wise(aa, required, np.any)
        newly_conditioned = required & ~conditioned
        sequence[newly_conditioned] = C_SEQ.ground_truth_value(aa)[newly_conditioned]
        C_SEQ.set_annotation(aa, sequence)
        data["atom_array"] = aa
        return data


class CanonicalizeSequenceProbabilityConditions(Transform):
    """Normalize sequence-probability conditions before featurization.

    ``C_SEQ`` is authoritative: canonical sequence-conditioned residues receive a
    fully specified one-hot distribution over the 28 probability Conditions. For a
    sequence-conditioned token outside that vocabulary, all 28 probabilities are
    explicitly zero. Existing specified values must agree with that distribution.
    """

    def forward(self, data: dict) -> dict:
        aap = as_atom_array_plus(data["atom_array"])
        starts = get_residue_starts(aap)
        seq_mask = C_SEQ.mask(aap, default="generate")
        residue_seq = np.maximum.reduceat(seq_mask, starts).astype(bool)
        seq_atoms = spread_residue_wise(aap, residue_seq)
        sequence = C_SEQ.annotation(aap, default="generate")

        resolved: dict[type, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        conflicts: list[str] = []
        for condition in SEQUENCE_PROBABILITY_CONDITIONS:
            probability = condition.annotation(aap, default="generate").astype(np.float32)
            mask = condition.mask(aap, default="generate").copy()
            expected = (sequence == condition.token_name).astype(np.float32)
            conflict = seq_atoms & mask & ~np.isclose(probability, expected)
            if conflict.any():
                n_conflicting_residues = int(np.maximum.reduceat(conflict, starts).sum())
                conflicts.append(f"{condition.token_name}: {n_conflicting_residues} residue(s)")
            resolved[condition] = probability, mask, expected

        if conflicts:
            raise ValueError(
                "Sequence-probability conditioning conflicts with C_SEQ exact identities: " + ", ".join(conflicts)
            )

        for condition, (probability, mask, expected) in resolved.items():
            probability[seq_atoms] = expected[seq_atoms]
            mask[seq_atoms] = True
            probability[~mask] = np.nan
            # Remove stored columns before writing so Biotite preserves the normalized float32 dtype.
            for name in (condition.full_name, *condition.aliases):
                if name in aap.get_annotation_categories():
                    aap.del_annotation(name)
            if mask.any():
                condition.set_annotation(aap, probability)

        data["atom_array"] = aap
        return data
