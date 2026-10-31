"""`tt-bio prep`: upstream `mol msa` for a job JSON or a FASTA, with the search stubbed (no network, no device)."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import tt_bio.main as m
from tt_bio.main import cli
from tt_bio.protenix_json import read_jobs
from tt_bio.protenix_prep import msa_dir_names

A, B = "MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPVDAFQLGKLL"


@pytest.fixture
def searches(monkeypatch):
    """The unpaired search writes `<hash>.a3m`; the paired one returns one block per sequence."""
    calls = {"unpaired": [], "paired": []}

    def unpaired(seqs, target_id, msa_dir, *a, **k):
        calls["unpaired"].append(sorted(seqs.values()))
        for h, s in seqs.items():
            (msa_dir / f"{h}.a3m").write_text(f">query\n{s}\n>hit\n{s.lower()}\n")

    def paired(seqs, *a, **k):
        calls["paired"].append(sorted(seqs.values()))
        return {h: f">query\n{s}\n>paired\n{s}\n" for h, s in seqs.items()}

    monkeypatch.setattr(m, "_generate_esmfold2_a3m", unpaired)
    monkeypatch.setattr(m, "_generate_paired_a3m", paired)
    return calls


def _jobs(tmp_path, jobs):
    p = tmp_path / "in" / "jobs.json"
    p.parent.mkdir()
    p.write_text(json.dumps(jobs))
    return p


def test_a_heterodimer_gets_paired_and_unpaired_msas_and_folds_from_them(searches, tmp_path):
    src = _jobs(tmp_path, [{"name": "hd", "sequences": [{"proteinChain": {"sequence": A, "count": 2}},
                                                         {"proteinChain": {"sequence": B, "count": 1}},
                                                         {"ligand": {"ligand": "CCD_ATP", "count": 1}}]}])
    r = CliRunner().invoke(cli, ["prep", str(src), "--out_dir", str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
    dest = tmp_path / "in" / "jobs-update-msa.json"   # beside the input, as upstream writes it
    assert str(dest) in r.output
    chains = [e["proteinChain"] for e in json.loads(dest.read_text())[0]["sequences"] if "proteinChain" in e]
    for c in chains:
        assert c["unpairedMsaPath"].endswith("non_pairing.a3m") and c["pairedMsaPath"].endswith("pairing.a3m")
    assert searches["unpaired"] == [sorted([A, B])] and searches["paired"] == [sorted([A, B])]
    job = read_jobs(dest)[0]
    assert set(job.unpaired_msa) == {"A", "B", "C"} and job.paired_msa["C"].endswith(f"{B}\n")
    assert job.unpaired_msa["A"] == f">query\n{A}\n>hit\n{A.lower()}\n"


def test_a_monomer_does_not_pair(searches, tmp_path):
    src = _jobs(tmp_path, [{"name": "mono", "sequences": [{"proteinChain": {"sequence": A, "count": 1}}]}])
    assert CliRunner().invoke(cli, ["prep", str(src), "--out_dir", str(tmp_path / "out")]).exit_code == 0
    c = json.loads((tmp_path / "in" / "jobs-update-msa.json").read_text())[0]["sequences"][0]["proteinChain"]
    assert "pairedMsaPath" not in c and searches["paired"] == []
    assert (tmp_path / "out" / "mono" / "msa" / "0" / "non_pairing.a3m").exists()


def test_a_job_with_its_msas_is_left_alone(searches, tmp_path):
    src = _jobs(tmp_path, [{"name": "done", "sequences": [{"proteinChain": {"sequence": A, "count": 1,
                                                                            "unpairedMsaPath": "/x.a3m"}}]}])
    r = CliRunner().invoke(cli, ["prep", str(src), "--out_dir", str(tmp_path / "out")])
    assert r.exit_code == 0 and "nothing to search" in r.output
    assert searches["unpaired"] == [] and not (tmp_path / "in" / "jobs-update-msa.json").exists()


def test_fasta_records_get_their_own_directories_and_an_index(searches, tmp_path):
    fa = tmp_path / "seqs.fasta"
    fa.write_text(f">sp|P1|one\n{A}\n>two/b\n{B}\n")
    r = CliRunner().invoke(cli, ["prep", str(fa), "--out_dir", str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
    index = json.loads((tmp_path / "out" / "msa_index.json").read_text())
    assert [(e["id"], e["sequence"]) for e in index] == [("sp|P1|one", A), ("two/b", B)]
    for e, name in zip(index, ["sp_P1_one", "two_b"]):
        assert e["dir"].endswith(name)
        assert {p.name for p in (tmp_path / "out" / name).iterdir()} == {"non_pairing.a3m", "pairing.a3m"}


def test_dir_names_match_upstreams():
    assert msa_dir_names(["a b", "a b", "msa_index.json", "..", "x"]) == ["a_b", "a_b_2", "msa_index.json_2",
                                                                           "seq_3", "x"]


def test_a_failed_search_is_an_error_not_an_empty_msa(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_generate_esmfold2_a3m", lambda *a, **k: None)
    src = _jobs(tmp_path, [{"name": "x", "sequences": [{"proteinChain": {"sequence": A, "count": 1}}]}])
    r = CliRunner().invoke(cli, ["prep", str(src), "--out_dir", str(tmp_path / "out")])
    assert r.exit_code == 1 and "returned nothing" in r.output


def test_use_template_searches_after_the_msas_and_writes_final_updated(searches, tmp_path, monkeypatch):
    """`tt-bio prep --use_template` is upstream's `mol prep`: MSAs first, then hmmsearch templates from them."""
    from pathlib import Path

    from tt_bio._vendor.mol1 import template_search
    calls = []

    def fake(msa_for_template_search_dir, msa_for_template_search_name, **tools):
        calls.append((msa_for_template_search_name, tools["seqres_database_path"]))
        Path(msa_for_template_search_dir, "hmmsearch.a3m").write_text(f">hit\n{A}\n")

    monkeypatch.setattr(template_search, "run_template_search", fake)
    src = _jobs(tmp_path, [{"name": "m", "sequences": [{"proteinChain": {"sequence": A, "count": 1}}]}])
    r = CliRunner().invoke(cli, ["prep", str(src), "--out_dir", str(tmp_path / "out"), "--use_template",
                                 "--seqres_database_path", "/db/seqres.fasta"])
    assert r.exit_code == 0, r.output
    final = tmp_path / "in" / "jobs-final-updated.json"   # beside the input, upstream's name
    assert str(final) in r.output and (tmp_path / "in" / "jobs-update-msa.json").exists()
    assert calls == [("non_pairing", "/db/seqres.fasta")]
    job = read_jobs(final)[0]
    assert job.templates["A"].endswith("hmmsearch.a3m") and job.unpaired_msa["A"].startswith(">query")


def test_template_search_on_a_fasta_is_refused(searches, tmp_path):
    fa = tmp_path / "s.fasta"
    fa.write_text(f">a\n{A}\n")
    r = CliRunner().invoke(cli, ["prep", str(fa), "--use_template"])
    assert r.exit_code != 0 and "json job list" in r.output and searches["unpaired"] == []
