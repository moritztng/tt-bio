"""Regression: an artifact cache publishes by rename and only counts a non-empty file.

The weight registry fixed exactly this bug class at seven download sites. The MSA
path had it too: two of five producers wrote straight to the final name, and six of
seven readers gated on bare ``Path.exists()``. A killed search then left a truncated
or zero-byte ``{hash}.a3m`` that every later fold of that sequence accepted forever.

These tests pin the contract in ``tt_bio.cache`` and scan every non-vendored module
under ``tt_bio/`` for a call site that bypasses it. The scan used to name five modules
and match two suffix literals, which is how Nesso-1's host pipeline shipped three
publish sites straight to their final names and kept the check green.
"""
from __future__ import annotations

import hashlib
import re
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
    assert not list(tmp_path.rglob(".*.tmp*"))


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
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_staged_tmp_keeps_the_destination_suffix(tmp_path: Path):
    """A suffix-sensitive producer must see the suffix it expects.

    ``np.savez_compressed`` appends ``.npz`` unless the path already ends in it, so a
    tmp named ``.x.npz.<pid>.tmp`` would be written as ``...tmp.npz`` and the publish
    rename would fail on a path that does not exist. Nesso-1's structure cache
    (``nesso1_input.py:_parse_one``) is that producer.
    """
    import numpy as np

    dst = tmp_path / "tyr48.npz"
    with artifact_cache.staged(dst) as tmp:
        assert tmp.suffix == ".npz", tmp
        np.savez_compressed(str(tmp), a=np.arange(3))
    assert dst.exists() and not list(tmp_path.rglob(".*.tmp*"))
    assert list(np.load(dst)["a"]) == [0, 1, 2]


def test_publish_file_is_atomic(tmp_path: Path):
    src = tmp_path / "src.a3m"
    src.write_text(">query\nMKT\n")
    artifact_cache.publish_file(src, tmp_path / "out" / "dst.a3m")
    assert (tmp_path / "out" / "dst.a3m").read_text() == ">query\nMKT\n"
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_template_fetch_publishes_by_rename():
    """tt_bio/worker.py fetched RCSB template CIFs straight to the final name, gated on
    bare exists() -- the 8th site of the bug class the weight registry fixed at seven."""
    import tt_bio.worker
    src = Path(tt_bio.worker.__file__).read_text()
    assert 'urlretrieve(url, struct_dir' not in src
    assert 'with staged(struct_dir / f"{p}.cif") as tmp:' in src


def _tt_bio_sources():
    """Every non-vendored module under tt_bio/, discovered.

    The scoped version of this scan named five modules and matched two suffix literals.
    Nesso-1's host pipeline was neither, so its three publish sites stayed green -- the
    same hardcoded-scope failure the weights registry and token-axis guards had.
    """
    import tt_bio

    root = Path(tt_bio.__file__).parent
    return [f for f in sorted(root.rglob("*.py")) if "_vendor" not in f.parts]


def test_no_module_gates_the_msa_cache_on_bare_exists():
    """No MSA cache-hit gate reads bare ``Path.exists()``; they all go through ``cached``,
    so a zero-byte a3m from a failed search is redone instead of accepted forever."""
    bad = []
    for f in _tt_bio_sources():
        for i, ln in enumerate(f.read_text().splitlines(), 1):
            if ('.a3m").exists()' in ln or '.csv").exists()' in ln
                    or '.a3m").write_text(' in ln or '.csv").write_text('  in ln):
                bad.append(f"{f.name}:{i}: {ln.strip()}")
    assert not bad, "bypasses the MSA cache contract at:\n" + "\n".join(bad)


_FSTR = re.compile(r'f"([^"]*)"')
_GATE = re.compile(r'\.exists\(\)|(?<![\w.])cached\(')
_WRITE = re.compile(r'(?<![\w.])(?:save_file|np\.savez\w*|torch\.save|json\.dump)\('
                    r'|\.(?:write_text|write_bytes|dump)\(')


def _basename_patterns(line: str) -> set[str]:
    """f-string basenames on a line, placeholders normalised: f"{mid}.safetensors" -> "{}.safetensors"."""
    out = set()
    for lit in _FSTR.findall(line):
        norm = re.sub(r"\{[^}]*\}", "{}", lit)
        if "." in norm and "/" not in norm:
            out.add(norm)
    return out


def test_nothing_a_later_run_skips_on_is_published_to_its_final_name():
    """The poisoning shape, stated exactly: a file a later run SKIPS work because of,
    written straight to its final name.

    A truncated prediction output is visible to the user and overwritten next run. A
    truncated file something GATES on is accepted forever -- that is the whole bug class,
    and it is what makes this scan precise where a scan over cache-looking paths is not.
    On db56e207 it found one site (nesso1_input.py:178, the ESM-2 650M embedding, gated at
    :170) and no false positives anywhere in the tree.

    No module list, no directory list, no suffix list: for each file, the basename patterns
    used in a skip gate must not also appear on a write call outside ``staged``.
    """
    bad = []
    for f in _tt_bio_sources():
        lines = f.read_text().splitlines()
        gated: set[str] = set()
        for ln in lines:
            if _GATE.search(ln):
                gated |= _basename_patterns(ln)
        if not gated:
            continue
        for i, ln in enumerate(lines, 1):
            if not _WRITE.search(ln) or "staged(" in ln:
                continue
            if hit := _basename_patterns(ln) & gated:
                bad.append(f"{f.name}:{i}: gated on {sorted(hit)} -- {ln.strip()}")
    assert not bad, ("published to a final name that something else skips on; wrap it in "
                     "tt_bio.cache.staged:\n" + "\n".join(bad))
