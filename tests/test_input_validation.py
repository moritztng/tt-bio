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


def _fasta(tmp_path, txt):
    p = tmp_path / "in.fasta"
    p.write_text(txt)
    return p


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


def test_blank_chain_id_is_auto_assigned_not_dropped(tmp_path):
    """A record with a blank leading id (``>|protein``) must yield a chain with an auto-assigned
    id, not be silently dropped (which surfaces later as a misleading 'no sequences' error)."""
    from tt_bio.main import _read_bio_chains, _read_protein_chains

    p = _fasta(tmp_path, ">|protein\nMKVL\n")
    prot = _read_protein_chains(p)
    assert len(prot) == 1 and prot[0][0] == "A" and prot[0][1] == "MKVL"
    bio = _read_bio_chains(p)
    assert len(bio) == 1 and bio[0][0] == "A"


def test_msa_id_on_non_protein_rejected(tmp_path):
    """The MSA-id-only-on-proteins check was an ``assert`` (silently disabled under ``python -O``);
    it must be a real raise regardless of optimization level."""
    from tt_bio.data.parse import parse_fasta

    p = _fasta(tmp_path, ">A|dna|somemsa\nACGT\n")
    with pytest.raises(ValueError, match="MSA_ID is only allowed for proteins"):
        parse_fasta(p, ccd={}, mol_dir=tmp_path)


def test_chain_label_bijective_base26():
    """Chain ids must stay unique past 26 chains (the old %26 / chr() schemes collided or ran
    past 'Z', silently corrupting multi-chain output)."""
    from tt_bio.main import _chain_label

    assert [_chain_label(n) for n in range(26)] == [chr(65 + n) for n in range(26)]  # parity <26
    assert _chain_label(26) == "AA" and _chain_label(27) == "AB" and _chain_label(52) == "BA"
    labels = [_chain_label(n) for n in range(60)]
    assert len(set(labels)) == 60  # all unique


def test_load_sequences_accepts_fasta_yaml_dir_and_bare_string(tmp_path):
    """``tt-bio embed``'s DATA argument accepts every documented input shape."""
    from tt_bio.esmc import load_sequences

    fa = tmp_path / "in.fasta"
    fa.write_text(">a\nMKT\n>b\nGVS\n")
    assert load_sequences(str(fa)) == {"a": "MKT", "b": "GVS"}

    d = tmp_path / "fastas"
    d.mkdir()
    (d / "x.fasta").write_text(">c\nACD\n")
    (d / "y.fa").write_text(">d\nEFG\n")
    assert load_sequences(str(d)) == {"c": "ACD", "d": "EFG"}

    yml = tmp_path / "in.yaml"
    yml.write_text("seq1: mkt\nseq2: gvs\n")
    assert load_sequences(str(yml)) == {"seq1": "MKT", "seq2": "GVS"}  # uppercased

    assert load_sequences("mktayiakqr") == {"seq0": "MKTAYIAKQR"}


def test_load_sequences_rejects_bad_input(tmp_path):
    """Bad input raises ValueError with an actionable message (the CLI turns this into a
    ClickException instead of a stack trace) rather than silently returning nothing."""
    from tt_bio.esmc import load_sequences

    with pytest.raises(ValueError, match="not an existing file/directory"):
        load_sequences("not/a/real/path.fasta")

    unsupported = tmp_path / "in.txt"
    unsupported.write_text("MKT")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_sequences(str(unsupported))

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_sequences(str(bad_yaml))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="no FASTA files"):
        load_sequences(str(empty_dir))


def test_write_manifest_documents_shapes_and_files(tmp_path):
    """manifest.json is the one place a downstream consumer looks to learn each
    output file's shape/dtype/pooling without reading the code."""
    import json

    import numpy as np

    from tt_bio.esmc import ESMCEmbedding, write_manifest

    embs = [
        ESMCEmbedding("p1", "MKT", np.zeros((3, 8), np.float32), np.zeros(8, np.float32), None),
        ESMCEmbedding("p2", "GVSE", np.zeros((4, 8), np.float32), np.zeros(8, np.float32), None),
    ]
    out = tmp_path / "manifest.json"
    write_manifest(embs, out, model="esmc-300m", pool="mean", fast=False,
                   out_format="npz", return_logits=False)
    manifest = json.loads(out.read_text())
    assert manifest["model"] == "esmc-300m" and manifest["pool"] == "mean"
    assert manifest["d_model"] == 8 and manifest["dtype"] == "float32"
    assert manifest["sequences"] == [
        {"id": "p1", "length": 3, "file": "p1.npz"},
        {"id": "p2", "length": 4, "file": "p2.npz"},
    ]
