"""Upstream template and RNA MSA searches on a JSON input (tt_bio.protenix_json.search, Mol-1's own
template_search / rna_msa_search vendored). CPU only: the hmmer binaries are stubbed."""
import json
from pathlib import Path

import pytest

from tt_bio import protenix_json as pj
from tt_bio._vendor.mol1 import template_search


def _job(tmp_path, msa=True):
    d = tmp_path / "msa" / "A"
    d.mkdir(parents=True)
    (d / "non_pairing.a3m").write_text(">query\nMKV\n")
    prot = {"sequence": "MKV", "count": 1}
    if msa:
        prot["unpairedMsaPath"] = str(d / "non_pairing.a3m")
    src = tmp_path / "jobs-update-msa.json"
    src.write_text(json.dumps([{"name": "j", "sequences": [{"proteinChain": prot},
                                                           {"rnaSequence": {"sequence": "GGAU", "count": 1}}]}]))
    return src, d


def test_nothing_asked_is_the_input(tmp_path):
    src, _ = _job(tmp_path)
    assert pj.search(src, tmp_path / "s") == src


def test_template_search_fills_templates_path(tmp_path, monkeypatch):
    src, d = _job(tmp_path)
    calls = []

    def fake(msa_for_template_search_dir, msa_for_template_search_name, **tools):
        calls.append((msa_for_template_search_dir, msa_for_template_search_name, tools))
        Path(msa_for_template_search_dir, "hmmsearch.a3m").write_text(">hit\nMKV\n")

    monkeypatch.setattr(template_search, "run_template_search", fake)
    out = pj.search(src, tmp_path / "s", use_template=True, seqres_database_path="/db.fasta")
    assert out == tmp_path / "s" / "jobs-final-updated.json"          # upstream's name for the prepared JSON
    assert calls == [(str(d), "non_pairing", {"hmmsearch_binary_path": None, "hmmbuild_binary_path": None,
                                             "seqres_database_path": "/db.fasta"})]
    assert pj.read_jobs(out)[0].templates == {"A": str(d / "hmmsearch.a3m")}
    calls.clear()
    pj.search(src, tmp_path / "s", use_template=True)                  # an existing hmmsearch.a3m is reused
    assert calls == []


def test_template_search_needs_a_chain_msa(tmp_path, monkeypatch):
    """As upstream: a chain with no MSA on disk gets no template search."""
    src, _ = _job(tmp_path, msa=False)
    monkeypatch.setattr(template_search, "run_template_search", lambda **_k: pytest.fail("searched"))
    assert pj.search(src, tmp_path / "s", use_template=True) == src


def test_missing_hmmsearch_is_upstreams_error(tmp_path, monkeypatch):
    src, _ = _job(tmp_path)
    monkeypatch.setattr(template_search.shutil, "which", lambda _n: None)
    with pytest.raises(AssertionError, match="apt install hmmer"):
        pj.search(src, tmp_path / "s", use_template=True)


def test_missing_rna_databases_are_named(tmp_path, monkeypatch):
    src, _ = _job(tmp_path)
    monkeypatch.setenv("MOL_ROOT_DIR", str(tmp_path / "root"))
    with pytest.raises(FileNotFoundError, match="RNA MSA search needs databases"):
        pj.search(src, tmp_path / "s", use_rna_msa=True, nhmmer_binary_path="/bin/true",
                  hmmalign_binary_path="/bin/true", hmmbuild_binary_path="/bin/true")


def test_unknown_option_refused(tmp_path):
    src, _ = _job(tmp_path)
    with pytest.raises(TypeError):
        pj.search(src, tmp_path / "s", use_template=True, hmmsearch="x")


def test_predict_takes_upstreams_search_options():
    from tt_bio import main as m
    names = {o for p in m.predict.params for o in p.opts}
    assert {f"--{k}" for k in pj.SEARCH_TOOLS} <= names
