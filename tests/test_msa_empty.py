"""`msa: empty` means fold this chain single-sequence, on every model that reads the shared input.

The reader used to turn `empty` into None, which is also what a chain with no `msa:` key reads as,
and every MSA-folding model searches for a None chain. Measured on whglx 2026-09-23: protenix-v1
folded a `msa: empty` ubiquitin at msa_depth 9654 after sending it to the online server. And even
with the search skipped, the resolver fell back to the hash cache, so a sequence folded once with
an alignment brought it into every later `msa: empty` fold of the same sequence.
"""
from pathlib import Path

from tt_bio.cache import EMPTY_MSA, msa_pinned, seq_hash
from tt_bio.main import _read_bio_chains, _resolve_a3m_path

SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _yaml(tmp_path, msa_line):
    p = tmp_path / "q.yaml"
    p.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n"
                 + (f"      msa: {msa_line}\n" if msa_line is not None else ""))
    return p


def _cache_hit(msa_dir: Path) -> Path:
    a3m = msa_dir / f"{seq_hash(SEQ)}.a3m"
    a3m.parent.mkdir(parents=True, exist_ok=True)
    a3m.write_text(f">query\n{SEQ}\n>hit\n{SEQ}\n")
    return a3m


def test_the_reader_keeps_empty_apart_from_unset(tmp_path):
    for spelled in ("empty", "EMPTY", "Empty"):
        assert _read_bio_chains(_yaml(tmp_path, spelled))[0][2] == EMPTY_MSA
    assert _read_bio_chains(_yaml(tmp_path, None))[0][2] is None
    fa = tmp_path / "q.fasta"
    fa.write_text(f">A|protein|empty\n{SEQ}\n>B|protein\n{SEQ}\n")
    assert [c[2] for c in _read_bio_chains(fa)] == [EMPTY_MSA, None]


def test_an_empty_chain_never_picks_up_the_hash_cache(tmp_path):
    msa_dir = tmp_path / "msa"
    hit = _cache_hit(msa_dir)
    assert _resolve_a3m_path(EMPTY_MSA, SEQ, msa_dir) is None
    # Control: the same cache does serve a chain with nothing pinned, so the assertion above
    # is about `empty` and not about a cache the resolver cannot see.
    assert _resolve_a3m_path(None, SEQ, msa_dir) == hit


def test_esmfold2_resolver_agrees(tmp_path):
    from tt_bio.esmfold2_runtime import resolve_msa

    msa_dir = tmp_path / "msa"
    _cache_hit(msa_dir)
    assert resolve_msa(EMPTY_MSA, SEQ, msa_dir) is None
    assert resolve_msa(None, SEQ, msa_dir) is not None


def test_empty_is_pinned_so_no_model_searches_it(tmp_path):
    a3m = _cache_hit(tmp_path / "msa")
    assert msa_pinned(EMPTY_MSA)
    assert msa_pinned(str(a3m))
    assert not msa_pinned(None)
    assert not msa_pinned(str(tmp_path / "missing.a3m"))


def test_of3_gets_upstreams_one_row_alignment(tmp_path):
    from tt_bio.openfold3_data import query_only_msa

    p = query_only_msa(tmp_path / "msa", SEQ)
    assert p.read_text() == f">query\n{SEQ}"
    assert p.name == "colabfold_main.a3m"
    assert query_only_msa(tmp_path / "msa", SEQ) == p


def test_an_empty_chain_takes_no_part_in_pairing(tmp_path, monkeypatch):
    """Pairing is a search too. Two searched chains plus one `msa: empty` chain pair as the two,
    and the empty chain gets no paired rows; one searched chain left over does not pair at all."""
    import tt_bio.main as main
    from tt_bio.worker import _paired_a3ms, _paired_msa

    other = SEQ[::-1]
    third = "ACDEFGHIKLMNPQRSTVWY" * 3
    asked = []

    def fake(seqs, *a, **k):
        asked.append(sorted(seqs.values()))
        return {h: f">query\n{s}\n" for h, s in seqs.items()}

    monkeypatch.setattr(main, "_generate_paired_a3m", fake)
    cfg = {"use_msa_server": True}
    chains = [("A", SEQ, None, "protein", None), ("B", other, None, "protein", None),
              ("C", third, EMPTY_MSA, "protein", None)]
    out = _paired_a3ms(Path("q.yaml"), chains, tmp_path, cfg)
    assert asked == [sorted([SEQ, other])]
    assert out[0] and out[1] and out[2] is None
    assert _paired_msa(Path("q.yaml"), chains[1:], tmp_path, cfg) is None
