"""Regression: FASTA input must validate the sequence alphabet, like YAML does.

The /v1 API's `sequence`/`input` convenience wraps a bare sequence into a FASTA
string (">A|protein\\n<seq>"). Before the fix, the FASTA branch of
limits.inspect() counted residues but never checked the alphabet, so invalid
characters (emoji, digits, punctuation) slipped past submit-time validation and
failed deep in the worker with an opaque codec error instead of a clean 400.
"""
from __future__ import annotations

from tt_bio.platform import limits


def test_fasta_emoji_protein_flagged():
    info = limits.inspect(">A|protein\nMK\U0001f9ecΩ字")  # MK🧬Ω字
    assert info["bad_seq"] and "invalid characters" in info["bad_seq"]


def test_fasta_valid_protein_passes():
    info = limits.inspect(">A|protein\nMKTAYIAKQRQISFVKSHFSRQLEE")
    assert info["bad_seq"] is None
    assert info["residues"] == 25


def test_fasta_type_from_header_is_respected():
    assert limits.inspect(">D|dna\nACGTACGT")["bad_seq"] is None
    assert limits.inspect(">D|dna\nACGTXQZ")["bad_seq"]          # non-nucleotide letters
    # default (no |type) is protein, which allows any letters
    assert limits.inspect(">D\nACGTACGT")["bad_seq"] is None


def test_yaml_path_still_validates():
    # the pre-existing YAML guard must be unaffected
    assert limits.inspect("sequences:\n  - protein: {id: A, sequence: 'MK123'}")["bad_seq"]
