"""A user-supplied `msa:` path must reach the parser under a name it accepts.

`parse_msas_direct` keeps only files whose STEM is a key of
`MSASettings.max_seq_counts` and drops the rest silently; `parse_msas` then indexes the
empty result and raises `IndexError: list index out of range`. So `msa: ./my.a3m` used to
crash the fold for both `--model openfold3` and `--model openbind`, and two committed
examples (`examples/ligand.yaml` stem `seq1`, `examples/prot_custom_msa.yaml` stem `seq2`)
hit it. Card-free: this is all host-side path handling.
"""
from pathlib import Path

import pytest

from tt_bio.openfold3_data import (
    CANONICAL_MAIN_MSA_STEM,
    inference_msa_settings,
    normalize_openfold3_msa_paths,
)

A3M = ">query\nMQIFVKT\n>hit1\nMQIFVKS\n"


class _Chain:
    def __init__(self, paths):
        self.main_msa_file_paths = paths


class _Query:
    def __init__(self, *chains):
        self.chains = list(chains)


def test_non_canonical_stem_is_relinked_and_bytes_are_identical(tmp_path):
    src = tmp_path / "seq1.a3m"
    src.write_text(A3M)
    q = normalize_openfold3_msa_paths(_Query(_Chain([src])), tmp_path / "msa")
    out = Path(q.chains[0].main_msa_file_paths[0])
    assert out != src
    assert out.stem == CANONICAL_MAIN_MSA_STEM
    assert out.suffix == ".a3m"
    assert out.read_text() == A3M


def test_a_canonical_stem_is_left_exactly_alone(tmp_path):
    src = tmp_path / f"{CANONICAL_MAIN_MSA_STEM}.a3m"
    src.write_text(A3M)
    q = normalize_openfold3_msa_paths(_Query(_Chain([src])), tmp_path / "msa")
    assert q.chains[0].main_msa_file_paths == [src]


@pytest.mark.parametrize("stem", ["uniref90_hits", "cfdb_hits", "mmseqs_colabfold"])
def test_every_canonical_source_name_survives(tmp_path, stem):
    src = tmp_path / f"{stem}.a3m"
    src.write_text(A3M)
    assert stem in inference_msa_settings().max_seq_counts
    q = normalize_openfold3_msa_paths(_Query(_Chain([src])), tmp_path / "msa")
    assert q.chains[0].main_msa_file_paths == [src]


def test_stockholm_keeps_its_suffix_so_the_right_parser_runs(tmp_path):
    src = tmp_path / "mine.sto"
    src.write_text("# STOCKHOLM 1.0\nquery MQIFVKT\n//\n")
    q = normalize_openfold3_msa_paths(_Query(_Chain([src])), tmp_path / "msa")
    out = Path(q.chains[0].main_msa_file_paths[0])
    assert out.name == f"{CANONICAL_MAIN_MSA_STEM}.sto"


def test_directories_and_npz_are_untouched(tmp_path):
    d = tmp_path / "alignments"
    (d).mkdir()
    (d / "cfdb_hits.a3m").write_text(A3M)
    npz = tmp_path / "prep.npz"
    npz.write_bytes(b"\x00")
    q = normalize_openfold3_msa_paths(_Query(_Chain([d]), _Chain([npz])), tmp_path / "msa")
    assert q.chains[0].main_msa_file_paths == [d]
    assert q.chains[1].main_msa_file_paths == [npz]


def test_same_stem_from_two_directories_does_not_collide(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "seq1.a3m").write_text(A3M)
    (b / "seq1.a3m").write_text(A3M + ">hit2\nMQIFVKA\n")
    q = normalize_openfold3_msa_paths(
        _Query(_Chain([a / "seq1.a3m"]), _Chain([b / "seq1.a3m"])), tmp_path / "msa")
    p0 = Path(q.chains[0].main_msa_file_paths[0])
    p1 = Path(q.chains[1].main_msa_file_paths[0])
    assert p0 != p1
    assert p0.read_text() != p1.read_text()


def test_chains_without_an_msa_are_left_alone(tmp_path):
    q = normalize_openfold3_msa_paths(_Query(_Chain(None), _Chain([])), tmp_path / "msa")
    assert q.chains[0].main_msa_file_paths is None
    assert q.chains[1].main_msa_file_paths == []


def test_the_committed_examples_that_crashed_now_have_canonical_names(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    for rel in ("examples/msa/seq1.a3m", "examples/msa/seq2.a3m"):
        src = repo / rel
        if not src.exists():
            pytest.skip(f"{rel} not in this checkout")
        q = normalize_openfold3_msa_paths(_Query(_Chain([src])), tmp_path / "msa")
        out = Path(q.chains[0].main_msa_file_paths[0])
        assert out.stem == CANONICAL_MAIN_MSA_STEM, rel
        assert out.read_bytes() == src.read_bytes(), rel
