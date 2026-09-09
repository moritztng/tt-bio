"""ESMFold2 input plumbing: one reader, one SPI builder, every molecule type.

Two correctness bugs are pinned here.

1. ESMFold2 advertised the ``modifications`` capability but dropped them (2026-08-16
   sweep): the reader never parsed the YAML ``modifications:`` list and ``fold_complex``
   never built a ``Modification``, so a SEP/TPO request came back as the unmodified
   residue with no warning.
2. ESMFold2 had its own protein-only reader, so a ligand / DNA / RNA entry in an input
   file was silently dropped and the job reported success on a bare-protein structure
   (2026-09-09). It now shares ``_read_bio_chains`` with Protenix/OF3/OpenDDE, and
   ``build_spi`` turns those chains into the vendored ``StructurePredictionInput``.

Host-only — no device, no checkpoints. The CCD-conformer paths need ``ccd.pkl``, so the
tests that would touch it assert on the dataclasses instead of running the featurizer.
"""
import textwrap

import pytest

from tt_bio.esmfold2_runtime import build_spi
from tt_bio.main import _read_bio_chains


def _write(tmp_path, text, name="in.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_reader_parses_modifications(tmp_path):
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
              modifications:
                - position: 5
                  ccd: TPO
    """)
    chains = _read_bio_chains(p)
    assert len(chains) == 1
    cid, seq, msa, mt, mods = chains[0]
    assert (cid, seq, msa, mt) == ("A", "ACDEFGHIKL", None, "protein")
    assert mods == [{"position": 5, "ccd": "TPO"}]
    # 1-indexed YAML -> 0-indexed vendored Modification.
    prot = build_spi(chains).sequences[0]
    assert prot.modifications[0].position == 4
    assert prot.modifications[0].ccd == "TPO"


def test_reader_rejects_out_of_range_modification(tmp_path):
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
        _read_bio_chains(p)


def test_fasta_carries_no_modifications(tmp_path):
    p = _write(tmp_path, ">A|protein\nACDEFGHIKL\n", name="in.fasta")
    assert _read_bio_chains(p)[0] == ("A", "ACDEFGHIKL", None, "protein", None)


def test_cocrystal_yaml_keeps_the_ligand(tmp_path):
    """The 2026-09-09 bug: a ligand entry used to be dropped and the fold "succeeded"."""
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
          - ligand:
              id: L
              ccd: BTN
    """)
    chains = _read_bio_chains(p)
    assert [(c[0], c[3]) for c in chains] == [("A", "protein"), ("L", "ligand")]
    assert chains[1][1] == "CCD_BTN"
    entries = build_spi(chains).sequences
    assert entries[1].ccd == ["BTN"] and entries[1].smiles is None


def test_smiles_ligand_and_nucleic_chains(tmp_path):
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
          - dna:
              id: B
              sequence: ATGCATGC
          - rna:
              id: C
              sequence: AUGCAUGC
          - ligand:
              id: L
              smiles: "CC(=O)O"
    """)
    entries = build_spi(_read_bio_chains(p)).sequences
    assert [type(e).__name__ for e in entries] == [
        "ProteinInput", "DNAInput", "RNAInput", "LigandInput"]
    assert entries[3].smiles == "CC(=O)O" and entries[3].ccd is None
    assert entries[1].sequence == "ATGCATGC"


def test_multi_residue_ccd_ligand_chain(tmp_path):
    """A ``ccd:`` list is one ligand chain of several components (a glycan), not one code."""
    p = _write(tmp_path, """\
        sequences:
          - protein:
              id: A
              sequence: ACDEFGHIKL
          - ligand:
              id: L
              ccd: [NAG, NAG, BMA]
    """)
    chains = _read_bio_chains(p)
    assert chains[1][1] == "CCD_NAG,NAG,BMA"
    assert build_spi(chains).sequences[1].ccd == ["NAG", "NAG", "BMA"]


def test_build_spi_accepts_the_legacy_protein_tuples():
    """Older callers pass (id, seq) / (id, seq, msa); they must keep working unchanged."""
    entries = build_spi([("A", "acdefghikl"), ("B", "ACDE", None)]).sequences
    assert [type(e).__name__ for e in entries] == ["ProteinInput", "ProteinInput"]
    assert entries[0].sequence == "ACDEFGHIKL"   # normalised, as before


def test_build_spi_refuses_an_unknown_molecule_type():
    with pytest.raises(ValueError, match="unknown molecule type"):
        build_spi([("A", "ACDE", None, "peptoid", None)])


def test_ligand_only_input_has_no_protein_chain(tmp_path):
    """ESMFold2 is a protein folder; a ligand-only file must fail, not fold nothing."""
    p = _write(tmp_path, """\
        sequences:
          - ligand:
              id: L
              ccd: BTN
    """)
    chains = _read_bio_chains(p)
    assert not any(c[3] == "protein" for c in chains)
