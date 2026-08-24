"""``contig_residue_count`` must mirror the featurizer's token plan term for term.

A caller that wants to size an RFD3 submission before a structure is parsed or a
device is opened has to count residues from the contig alone. The count lives
beside the grammar it sizes (``rfd3.input``) so the two cannot drift; this pins
the arithmetic against ``featurize._plan_tokens_from_contig``.
"""
import pytest

from tt_bio.rfd3.input import contig_residue_count, parse_contig


def _count(contig):
    return contig_residue_count(parse_contig(contig))


@pytest.mark.parametrize("contig,expected", [
    ("A1-10", 10),                  # Indexed: end - start + 1
    ("A5", 1),                      # Indexed, single residue
    ("60", 60),                     # Designed: exact length
    ("60-80", 70),                  # DesignedRange: midpoint (60+80)//2
    ("61-80", 70),                  # midpoint floors
    ("A1-10,60-80,A31-40", 90),     # motif + designed + motif
    ("A1-10,/0,60", 70),            # a chain break costs nothing
])
def test_count(contig, expected):
    assert _count(contig) == expected


def test_designed_range_uses_the_midpoint_the_featurizer_resolves():
    # Not `hi`: this port pins DesignedRange to (lo+hi)//2, so sizing on `hi`
    # would reject submissions that in fact fit.
    assert _count("1-2000") == 1000


def test_empty_contig_costs_nothing():
    assert contig_residue_count([]) == 0
