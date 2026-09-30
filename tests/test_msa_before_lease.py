"""The MSA search runs before a chip is leased, and the fold then searches nothing.

``tt-bio predict`` calls ``prefetch_msas`` in the submitting process, which holds no chip. The
worker runs each model's own MSA stage after it leases the target, so the property that
matters is: once the prefetch has run, that stage finds every alignment cached and issues no
search. Checked per model family against the worker's real MSA code, with the database
search stubbed.
"""
import os
from pathlib import Path

import pytest

from tt_bio import main as tmain
from tt_bio.worker import _paired_msa, prefetch_msas, search_msas

A, B = "MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPVAAIKEELKRRG"


@pytest.fixture
def searches(monkeypatch):
    """Every call the local-DB search receives, as (target, sequences, pair)."""
    calls = []

    def fake(seqs, target_id, msa_dir, db_path, use_env=False, pairing_strategy="greedy",
             pair=True):
        calls.append((target_id, dict(seqs), pair))
        Path(msa_dir).mkdir(parents=True, exist_ok=True)
        for h, s in seqs.items():
            (Path(msa_dir) / f"{h}.a3m").write_text(f">q\n{s}\n>hit\nA{s[1:]}\n")

    monkeypatch.setattr(tmain, "compute_msa_offline", fake)
    return calls


def _cfg(tmp_path, model):
    return {"model": model, "msa_dir": str(tmp_path / "msa"), "msa_db_path": "/db",
            "use_envdb": False, "msa_pairing_strategy": "greedy"}


def _target(tmp_path, name, *seqs):
    p = tmp_path / f"{name}.yaml"
    p.write_text("version: 1\nsequences:\n" + "".join(
        f"  - protein:\n      id: {chr(65 + i)}\n      sequence: {s}\n"
        for i, s in enumerate(seqs)))
    return p


def _fold_side(model, path, cfg):
    """The worker's MSA stage for ``model``, as the fold runs it after the lease."""
    chains = tmain._read_bio_chains(path, what=model)
    search_msas(path, chains, cfg)
    if model != "rf3":
        _paired_msa(path, chains, Path(cfg["msa_dir"]), cfg)


@pytest.mark.parametrize("model", ["protenix-v2", "opendde", "esmfold2", "rf3", "openbind"])
def test_after_the_prefetch_the_fold_searches_nothing(tmp_path, searches, model):
    cfg = _cfg(tmp_path, model)
    paths = [_target(tmp_path, "het", A, B), _target(tmp_path, "mono", A)]
    prefetch_msas(model, paths, cfg)
    assert searches, "the prefetch searched nothing"
    # Unpaired alignments for the whole run in one batched call, never paired across targets.
    unpaired = [c for c in searches if c[2] is False]
    assert len(unpaired) == 1 and len(unpaired[0][1]) == 2
    before = len(searches)
    for p in paths:
        _fold_side(model, p, cfg)
    assert searches[before:] == []


def test_without_the_prefetch_the_fold_searches(tmp_path, searches):
    """The control: the stub does see the fold's own search, so the test above can fail."""
    cfg = _cfg(tmp_path, "protenix-v2")
    _fold_side("protenix-v2", _target(tmp_path, "het", A, B), cfg)
    assert searches



@pytest.mark.parametrize("model", ["protenix-v1", "protenix-v2", "opendde", "rf3"])
def test_a_boltz2_csv_does_not_stand_in_for_the_a3m(tmp_path, searches, model):
    """Boltz-2 caches ``<hash>.csv``; these models read only ``<hash>.a3m``. With the CSV
    counted as cached, a fold after a Boltz-2 fold in the same msa_dir searched nothing and
    ran single-sequence (protenix-v1 on 7ROA: 6.07 A instead of 1.78 A)."""
    cfg = _cfg(tmp_path, model)
    msa_dir = Path(cfg["msa_dir"])
    msa_dir.mkdir(parents=True)
    (msa_dir / f"{tmain.seq_hash(A)}.csv").write_text(f"key,sequence\n-1,{A}\n")
    path = _target(tmp_path, "mono", A)
    search_msas(path, tmain._read_bio_chains(path, what=model), cfg)
    assert [c[1] for c in searches] == [{tmain.seq_hash(A): A}]
    assert tmain._resolve_a3m_text(None, A, msa_dir) is not None

@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.boltz/mols")),
                    reason="needs the bundled CCD mol library (~/.boltz/mols)")
def test_boltz2_featurises_from_the_prefetched_cache(tmp_path, searches):
    from tt_bio.data.featurizer import Boltz2Featurizer
    from tt_bio.data.mol import load_canonicals
    from tt_bio.data.tokenize import Boltz2Tokenizer

    cfg = _cfg(tmp_path, "boltz2")
    path = _target(tmp_path, "het", A, B)
    prefetch_msas("boltz2", [path], cfg)
    assert len(searches) == 1 and len(searches[0][1]) == 2   # the complex searched whole
    mol_dir = Path("~/.boltz/mols").expanduser()
    feats, _ = tmain.prepare_features(
        path, load_canonicals(mol_dir), mol_dir, Path(cfg["msa_dir"]), Boltz2Tokenizer(),
        Boltz2Featurizer(), False, None, "greedy", None, None, None, 8192,
        msa_db_path="/db")
    assert len(searches) == 1
    assert int(feats["msa"].shape[-2]) >= 2     # the prefetched rows reached the features


def test_a_failed_prefetch_leaves_the_search_to_the_worker(tmp_path, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RuntimeError("colabfold_search died")

    monkeypatch.setattr(tmain, "compute_msa_offline", boom)
    prefetch_msas("protenix-v2", [_target(tmp_path, "het", A, B)], _cfg(tmp_path, "protenix-v2"))
    assert "searches on its chip instead" in capsys.readouterr().err


@pytest.mark.parametrize("model,extra", [("esmfold2-fast", {}), ("af2ig", {}),
                                         ("protenix-v2", {"single_sequence": True}),
                                         ("protenix-v2", {"msa_db_path": None})])
def test_nothing_to_search_searches_nothing(tmp_path, searches, model, extra):
    prefetch_msas(model, [_target(tmp_path, "het", A, B)], {**_cfg(tmp_path, model), **extra})
    assert searches == []
