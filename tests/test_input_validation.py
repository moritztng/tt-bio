"""Regression tests for input validation / normalization on the fold path.

Two silent-corruption cases these guard against:
  * an empty (or whitespace-only) polymer sequence flowing unchecked into featurization and
    crashing deep in the model with a non-actionable message;
  * embedded whitespace in a Protenix sequence tokenizing to extra UNK residues, silently
    lengthening the chain (the Boltz and ESMFold parsers already strip; Protenix did not).

Host-only — no device, no checkpoints.
"""
from __future__ import annotations

import pytest
import torch

from tt_bio.data.parse import parse_boltz_schema
from tt_bio.protenix_data import aatype_from_sequence, seq_to_restype


@pytest.mark.parametrize("seq", ["", "   ", "\n\t"])
def test_empty_polymer_sequence_rejected(seq):
    schema = {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq}}]}
    with pytest.raises(ValueError, match="Empty protein sequence for chain 'A'"):
        parse_boltz_schema("t", schema, ccd={})


def test_protenix_strips_whitespace_in_sequence():
    # spaces / tabs / newlines must not become extra residues
    assert torch.equal(seq_to_restype("A R N", "protein"), seq_to_restype("ARN", "protein"))
    assert torch.equal(seq_to_restype("MK\n V\tL", "protein"), seq_to_restype("MKVL", "protein"))
    assert seq_to_restype("A C G", "rna").shape[0] == 3
    # a whitespace-free sequence is unchanged (does not perturb the validated path)
    seq = "MKVLINSTQ"
    assert torch.equal(seq_to_restype(seq, "protein"), aatype_from_sequence(seq))
