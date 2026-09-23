"""Regression: every ColabFold search must key its working/cache dir by the
sequence set, not by target_id.

target_id is the input filename stem (e.g. "target_1") and repeats across
inputs. When the cache dir was keyed by target_id, a single-chain run cached one
query under target_1_unpaired_tmp/, and a later multi-chain run with the same
target_id reused that stale cache while expecting N queries — dying with
``KeyError`` on the missing query index (observed as per-target error "102").

These tests mock run_mmseqs2 (no network) and assert the prefix contract that
makes the collision impossible.
"""
from __future__ import annotations

from pathlib import Path

from tt_bio import main as tt_main


def _fake_a3m(seqs):
    """A minimal valid a3m-text result: one query block per input sequence."""
    return [f">query\n{s}\n" for s in seqs]


def _capture_prefixes(monkeypatch):
    """Patch run_mmseqs2 to record the cache-dir prefix of every call. Like the real one, a
    prefix it has already answered returns that answer again (its cached out.tar.gz)."""
    seen: list[Path] = []
    done: dict[Path, list[str]] = {}

    def fake(x, prefix, *args, **kwargs):
        seen.append(Path(prefix))
        return done.setdefault(Path(prefix), _fake_a3m(x))

    monkeypatch.setattr(tt_main, "run_mmseqs2", fake)
    return seen


A = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAP"
B = "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVK"


def test_prefix_excludes_target_id(monkeypatch, tmp_path):
    """The cache prefix must not embed target_id — that is what caused reuse
    across unrelated runs sharing the stem."""
    seen = _capture_prefixes(monkeypatch)
    tt_main.compute_msa({"a": A, "b": B}, "target_1", tmp_path, "http://x", "greedy")
    assert seen, "run_mmseqs2 was not called"
    assert all("target_1" not in p.name for p in seen)


def test_single_then_multi_chain_no_collision(monkeypatch, tmp_path):
    """The exact bug: a single-chain run then a multi-chain run with the SAME
    target_id must use different cache dirs (different sequence sets)."""
    seen = _capture_prefixes(monkeypatch)
    tt_main.compute_msa({"a": A}, "target_1", tmp_path, "http://x", "greedy")
    tt_main.compute_msa({"a": A, "b": B}, "target_1", tmp_path, "http://x", "greedy")
    unpaired = [p.name for p in seen if "unpaired" in p.name]
    assert len(unpaired) == 2
    assert unpaired[0] != unpaired[1], "single- and multi-chain runs collided on the same cache dir"


def test_same_sequences_reuse_prefix(monkeypatch, tmp_path):
    """Two runs of the identical sequence set SHOULD land on the same cache dir
    (so the search is reused, not redone)."""
    seen = _capture_prefixes(monkeypatch)
    tt_main.compute_msa({"a": A, "b": B}, "target_1", tmp_path, "http://x", "greedy")
    tt_main.compute_msa({"x": A, "y": B}, "different_name", tmp_path, "http://x", "greedy")
    unpaired = [p.name for p in seen if "unpaired" in p.name]
    assert len(unpaired) == 2
    assert unpaired[0] == unpaired[1], "identical sequence sets should reuse one cache dir"


def test_unpaired_a3m_same_stem_new_sequences(monkeypatch, tmp_path):
    """The silent case, on the unpaired search esmfold2/protenix/opendde/openfold3 share: two
    runs under one stem with the same NUMBER of different sequences. Keyed by stem, the second
    reused the first's out.tar.gz and cached each new sequence with the old protein's MSA."""
    seen = _capture_prefixes(monkeypatch)
    tt_main._generate_esmfold2_a3m({"a": A}, "target_1", tmp_path, None, False,
                                   "http://x", "greedy", None, None, None)
    tt_main._generate_esmfold2_a3m({"b": B}, "target_1", tmp_path, None, False,
                                   "http://x", "greedy", None, None, None)
    assert seen[0] != seen[1] and all("target_1" not in p.name for p in seen)
    assert (tmp_path / "b.a3m").read_text().splitlines()[1] == B
