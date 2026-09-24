"""One pairing rule for every model whose upstream pairs.

A complex pairs when it has more than one unique protein sequence, which is what Protenix,
OpenDDE, OpenFold3 and Boltz-2 do upstream. The paired MSA is cached per complex, because
row j of chain A lines up with row j of the partner it was searched with and no other.
"""
import json
from pathlib import Path

import pytest
import torch

from tt_bio import main as tmain
from tt_bio.cache import paired_msa_dir, seq_hash
from tt_bio.worker import _paired_a3ms, _paired_msa

A, B, C = "MKTAYIAKQR", "GSHMLEDPVA", "PEPTIDEKLW"


def _chains(*seqs, rna=False):
    out = [(chr(65 + i), s, None, "protein", None) for i, s in enumerate(seqs)]
    if rna:
        out.insert(1, ("R", "ACGU", None, "rna", None))
    return out


def _a3m(query, rows):
    return "".join(f">{i}\n{s}\n" for i, s in enumerate([query] + rows))


def test_the_paired_dir_is_a_property_of_the_complex(tmp_path):
    assert paired_msa_dir(tmp_path, [A]) is None
    assert paired_msa_dir(tmp_path, [A, A]) is None               # a homomer does not pair
    assert paired_msa_dir(tmp_path, [A, B]) == paired_msa_dir(tmp_path, [B, A, B])
    assert paired_msa_dir(tmp_path, [A, B]) != paired_msa_dir(tmp_path, [A, C])


def _stub_search(monkeypatch, calls):
    def fake(seqs, *_a, **kw):
        calls.append((dict(seqs), kw.get("search")))
        return {h: _a3m(s, ["-" * len(s)]) for h, s in seqs.items()}
    monkeypatch.setattr(tmain, "_generate_paired_a3m", fake)


@pytest.mark.parametrize("seqs", [(A,), (A, A), (A, A, A)])
def test_monomers_and_homomers_never_reach_the_search(tmp_path, monkeypatch, seqs):
    calls = []
    _stub_search(monkeypatch, calls)
    assert _paired_msa(Path("t.yaml"), _chains(*seqs), tmp_path, {"use_msa_server": True}) is None
    assert calls == []


def test_a_heteromer_pairs_once_over_its_unique_sequences(tmp_path, monkeypatch):
    calls = []
    _stub_search(monkeypatch, calls)
    out = _paired_a3ms(Path("t.yaml"), _chains(A, A, B, rna=True), tmp_path,
                       {"use_msa_server": True})
    assert [set(c[0].values()) for c in calls] == [{A, B}]
    assert calls[0][1] is True
    assert out[1] is None and out[0] == out[2] and out[3] is not None   # one entry per chain


@pytest.mark.parametrize("cfg,search", [({"use_msa_server": True}, True),
                                        ({"msa_db_path": "/db"}, True),
                                        ({"msa_endpoint": "http://x"}, False),
                                        ({}, False)])
def test_only_a_real_search_source_searches(tmp_path, monkeypatch, cfg, search):
    """An --msa_endpoint run is not sent to the public ColabFold server behind the user's
    back, and a cache-only run reads the cache."""
    calls = []
    _stub_search(monkeypatch, calls)
    _paired_msa(Path("t.yaml"), _chains(A, B), tmp_path, cfg)
    assert calls[0][1] is search


def test_single_sequence_does_not_pair(tmp_path, monkeypatch):
    calls = []
    _stub_search(monkeypatch, calls)
    assert _paired_msa(Path("t.yaml"), _chains(A, B), tmp_path,
                       {"use_msa_server": True, "single_sequence": True}) is None
    assert calls == []


def _stub_mmseqs(monkeypatch, calls):
    def fake(seqs, prefix, use_pairing=False, **_kw):
        calls.append((list(seqs), use_pairing))
        tag = "P" if use_pairing else "U"
        return [_a3m(s, [s[:-1] + tag, "-" * len(s)]) for s in seqs]
    monkeypatch.setattr(tmain, "run_mmseqs2", fake)


def test_the_paired_search_is_cached_per_complex(tmp_path, monkeypatch):
    calls = []
    _stub_mmseqs(monkeypatch, calls)
    ab = {seq_hash(A): A, seq_hash(B): B}
    got = tmain._generate_paired_a3m(ab, "t", tmp_path, "u", "greedy", None, None, None)
    assert calls == [(sorted([A, B]), True)]
    pdir = paired_msa_dir(tmp_path, [A, B])
    assert {p.name for p in pdir.glob("*.a3m")} == {f"{seq_hash(A)}.a3m", f"{seq_hash(B)}.a3m"}
    # the second call reads the cache, and a cache-only call never searches
    assert tmain._generate_paired_a3m(ab, "t", tmp_path, "u", "greedy", None, None, None) == got
    assert tmain._generate_paired_a3m(ab, "t", tmp_path, "u", "greedy", None, None, None,
                                      search=False) == got
    assert len(calls) == 1
    # A paired against a new partner is a new search, not A's rows from the first one
    ac = {seq_hash(A): A, seq_hash(C): C}
    assert tmain._generate_paired_a3m(ac, "t", tmp_path, "u", "greedy", None, None, None,
                                      search=False) is None
    tmain._generate_paired_a3m(ac, "t", tmp_path, "u", "greedy", None, None, None)
    assert len(calls) == 2


def test_boltz2_heteromer_csvs_live_with_their_complex(tmp_path, monkeypatch):
    """compute_msa writes a heteromer's keyed CSVs to the complex's directory and a monomer's
    to the shared per-chain cache, with upstream's rows: paired first (keys 0..), then the
    unpaired rows without the duplicate query."""
    calls = []
    _stub_mmseqs(monkeypatch, calls)
    tmain.compute_msa({seq_hash(A): A, seq_hash(B): B}, "t", tmp_path, "u", "greedy")
    pdir = paired_msa_dir(tmp_path, [A, B])
    rows = (pdir / f"{seq_hash(A)}.csv").read_text().splitlines()
    assert rows == ["key,sequence", f"0,{A}", f"1,{A[:-1]}P", f"-1,{A[:-1]}U", f"-1,{'-' * len(A)}"]
    assert not (tmp_path / f"{seq_hash(A)}.csv").exists()
    tmain.compute_msa({seq_hash(C): C}, "t", tmp_path, "u", "greedy")
    rows = (tmp_path / f"{seq_hash(C)}.csv").read_text().splitlines()
    assert rows[1] == f"-1,{C}" and all(r.startswith("-1,") for r in rows[1:])


def test_esmfold2_pairs_through_upstreams_key_header():
    """The paired rows reach upstream's construct_paired_msa as paired: row j of both chains
    in one MSA row, the all-gap genome dropped, the unpaired rows block-diagonal below."""
    import numpy as np

    from tt_bio._vendor.esm.models.esmfold2.paired_msa import (
        construct_paired_msa, protein_letter_to_res_type)
    from tt_bio._vendor.esm.utils.msa.msa import MSA
    from tt_bio.esmfold2_runtime import pair_keyed_msa

    a = pair_keyed_msa(MSA.from_sequences([A, "MKTAYIAKQW"]),
                       _a3m(A, ["MKTAYIAKQF", "-" * len(A)]))
    b = pair_keyed_msa(MSA.from_sequences([B, "GSHMLEDPVW"]),
                       _a3m(B, ["GSHMLEDPVF", "GSHMLEDPVY"]))
    assert [e.header for e in a.entries] == ["", "key=1", ""]
    assert [e.header for e in b.entries] == ["", "key=1", "key=2", ""]
    l2r = protein_letter_to_res_type()
    q = {i: np.array([l2r[c] for c in s]) for i, s in enumerate((A, B))}
    asym = np.array([0] * len(A) + [1] * len(B))
    res = np.concatenate([np.arange(len(A)), np.arange(len(B))])
    msa, _d, paired = construct_paired_msa({0: a, 1: b}, q, asym, res, l2r)
    assert paired[1].tolist() == [1.0] * (len(A) + len(B))   # row 1 is paired on both chains
    assert msa[1, len(A) - 1] == l2r["F"] and msa[1, -1] == l2r["F"]
    assert int(paired[1:].sum()) == len(A) + len(B)          # nothing else pairs
    # capped: the query and the paired rows come first
    assert [e.header for e in pair_keyed_msa(None, _a3m(B, ["GSHMLEDPVF"] * 3), 2).entries] \
        == ["0", "key=1"]


def _of3_query(tmp_path, seqs):
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    chains = []
    for i, s in enumerate(seqs):
        main = tmp_path / "main" / seq_hash(s) / "colabfold_main.a3m"
        main.parent.mkdir(parents=True, exist_ok=True)
        main.write_text(_a3m(s, [s[:-1] + "W"]))
        chains.append({"molecule_type": "PROTEIN", "chain_ids": [chr(65 + i)], "sequence": s,
                       "main_msa_file_paths": [str(main)]})
    iqs = InferenceQuerySet.model_validate({"queries": {"q": {
        "query_name": "q", "use_msas": True, "use_paired_msas": False,
        "use_main_msas": True, "covalent_bonds": None, "chains": chains}}})
    return next(iter(iqs.queries.values()))


def test_openfold3_reads_the_complex_paired_msa_as_colabfold_paired(tmp_path):
    from tt_bio.openfold3_data import attach_openfold3_paired_msas, build_openfold3_features

    homo = attach_openfold3_paired_msas(_of3_query(tmp_path, [A, A]), tmp_path)
    assert not homo.use_paired_msas
    q = _of3_query(tmp_path, [A, B])
    base = build_openfold3_features(q.model_copy(deep=True))
    assert attach_openfold3_paired_msas(q, tmp_path).use_paired_msas is False  # nothing cached
    pdir = paired_msa_dir(tmp_path, [A, B])
    pdir.mkdir(parents=True)
    for s in (A, B):
        (pdir / f"{seq_hash(s)}.a3m").write_text(_a3m(s, [s[:-1] + "F", s[:-1] + "Y"]))
    q = attach_openfold3_paired_msas(q, tmp_path)
    assert q.use_paired_msas
    assert [c.paired_msa_file_paths[0].name for c in q.chains] == ["colabfold_paired.a3m"] * 2
    f = build_openfold3_features(q)
    nb, na = int(base["msa_mask"].sum(-1).gt(0).sum()), int(f["msa_mask"].sum(-1).gt(0).sum())
    assert na > nb                                   # the paired rows reached the MSA stack


def test_protenix_and_opendde_share_the_rule():
    """Neither predict path carries its own copy of the search or the trigger."""
    import inspect

    from tt_bio.worker import _WorkerState

    for name in ("_protenix_inputs", "_predict_opendde_one"):
        src = inspect.getsource(getattr(_WorkerState, name))
        assert "_paired_a3ms(path, chains, msa_dir, cfg)" in src, name
        assert "n_prot" not in src and "_generate_paired_a3m" not in src, name
    for name in ("_predict_openfold3_one", "_predict_esmfold2_one"):
        assert "_paired_msa(path, chains, msa_dir, cfg)" in inspect.getsource(
            getattr(_WorkerState, name)), name
    # preview2 does not pair (upstream 0.4.x drops the rows); OpenBind does.
    assert 'model == "openbind" and (paired := _paired_msa(' in inspect.getsource(
        _WorkerState._predict_openfold3_one)
