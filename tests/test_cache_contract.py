"""Regression: an artifact cache publishes by rename and only counts a non-empty file.

The weight registry fixed exactly this bug class at seven download sites. The MSA
path had it too: two of five producers wrote straight to the final name, and six of
seven readers gated on bare ``Path.exists()``. A killed search then left a truncated
or zero-byte ``{hash}.a3m`` that every later fold of that sequence accepted forever.

These tests pin the contract in ``tt_bio.cache`` and assert the live readers in
``worker.py`` and ``main.py`` route through it, so a new call site cannot quietly
reintroduce the bare-exists gate.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from tt_bio import cache as artifact_cache


def test_seq_hash_is_the_documented_key():
    assert artifact_cache.seq_hash("MKTVR") == hashlib.sha256(b"MKTVR").hexdigest()[:16]


def test_empty_file_is_not_a_cache_hit(tmp_path: Path):
    p = tmp_path / "abc.a3m"
    assert not artifact_cache.cached(p)          # missing
    p.write_text("")
    assert p.exists() and not artifact_cache.cached(p)   # present but empty
    p.write_text(">query\nMKT\n")
    assert artifact_cache.cached(p)


def test_publish_text_is_atomic_and_leaves_no_tmp(tmp_path: Path):
    dst = tmp_path / "sub" / "abc.a3m"
    artifact_cache.publish_text(dst, ">query\nMKT\n")
    assert dst.read_text() == ">query\nMKT\n"
    assert not list(tmp_path.rglob(".*.tmp"))


def test_publish_text_failure_leaves_no_partial_under_the_final_name(tmp_path: Path,
                                                                    monkeypatch):
    """A search that dies mid-write must not publish. The tmp file is the casualty."""
    dst = tmp_path / "abc.a3m"

    real = Path.write_text

    def boom(self, *a, **k):
        real(self, ">query\nPARTI")     # a short write
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        artifact_cache.publish_text(dst, ">query\nPARTIAL\n")
    monkeypatch.undo()
    assert not dst.exists()


def test_staged_leaves_nothing_under_the_final_name_when_the_producer_raises(tmp_path: Path):
    """The OpenFold3 template fetch shape: a download that dies mid-transfer."""
    dst = tmp_path / "1abc.cif"
    with pytest.raises(ConnectionError):
        with artifact_cache.staged(dst) as tmp:
            tmp.write_text("data_1ABC\n_partial")
            raise ConnectionError("connection reset")
    assert not dst.exists()
    assert not list(tmp_path.rglob(".*.tmp"))


def test_publish_file_is_atomic(tmp_path: Path):
    src = tmp_path / "src.a3m"
    src.write_text(">query\nMKT\n")
    artifact_cache.publish_file(src, tmp_path / "out" / "dst.a3m")
    assert (tmp_path / "out" / "dst.a3m").read_text() == ">query\nMKT\n"
    assert not list(tmp_path.rglob(".*.tmp"))


def test_template_fetch_publishes_by_rename():
    """tt_bio/worker.py fetched RCSB template CIFs straight to the final name, gated on
    bare exists() -- the 8th site of the bug class the weight registry fixed at seven."""
    import tt_bio.worker
    src = Path(tt_bio.worker.__file__).read_text()
    assert 'urlretrieve(url, struct_dir' not in src
    assert 'with staged(struct_dir / f"{p}.cif") as tmp:' in src


@pytest.mark.parametrize("module", ["tt_bio.worker", "tt_bio.main",
                                    "tt_bio.msa_server", "tt_bio.openfold3_data",
                                    "tt_bio.esmfold2_runtime"])
def test_live_readers_do_not_gate_on_bare_exists(module):
    """No MSA cache-hit gate reads bare ``Path.exists()``; they all go through ``cached``."""
    import importlib
    src = Path(importlib.import_module(module).__file__).read_text()
    bad = [f"{i}: {ln.strip()}" for i, ln in enumerate(src.splitlines(), 1)
           if (".a3m\").exists()" in ln or ".csv\").exists()" in ln
               or ".a3m\").write_text(" in ln or ".csv\").write_text(" in ln)]
    assert not bad, f"{module} bypasses the MSA cache contract at:\n" + "\n".join(bad)
