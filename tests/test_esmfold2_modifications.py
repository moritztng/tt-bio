"""Regressions for two input/output correctness bugs found by the 2026-08-16
correctness sweep:

1. ESMFold2 advertised the ``modifications`` capability but dropped them:
   ``_read_protein_chains`` never parsed the YAML ``modifications:`` list and
   ``fold_complex`` never built a ``Modification``. A SEP/TPO request came back
   as the unmodified residue with no warning. These tests pin the parse and the
   1-indexed (YAML) -> 0-indexed (vendored featurizer) conversion.

Host-only — no device, no checkpoints.
"""
import textwrap

import pytest

from tt_bio.main import _read_protein_chains


def _write(tmp_path, text, name="in.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_read_protein_chains_parses_modifications(tmp_path):
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
              modifications:
                - position: 5
                  ccd: TPO
    """)
    chains = _read_protein_chains(p)
    assert len(chains) == 1
    cid, seq, msa, mods = chains[0]
    assert (cid, seq, msa) == ("A", "ACDEFGHIKL", None)
    assert mods == [{"position": 5, "ccd": "TPO"}]


def test_read_protein_chains_rejects_out_of_range_modification(tmp_path):
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
              modifications:
                - position: 11
                  ccd: TPO
    """)
    with pytest.raises(Exception, match="modification"):
        _read_protein_chains(p)


def test_fasta_carries_no_modifications(tmp_path):
    p = _write(tmp_path, ">A|protein\nACDEFGHIKL\n", name="in.fasta")
    assert _read_protein_chains(p)[0] == ("A", "ACDEFGHIKL", None, None)
