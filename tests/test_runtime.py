"""Regression: detect_tenstorrent_devices must validate an explicit --device_ids against the
cards actually present, raising a clear error instead of passing a bad id straight through to a
deep, opaque ttnn device-open crash. Host-only; device discovery is monkeypatched (no hardware).
"""
from __future__ import annotations

import pytest

from tt_bio import runtime


@pytest.fixture
def two_cards(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob",
                        lambda pat: ["/dev/tenstorrent/0", "/dev/tenstorrent/1"])
    return None


def test_explicit_ids_selected_when_present(two_cards):
    assert runtime.detect_tenstorrent_devices("0,1", 0, max_workers=10) == [0, 1]
    assert runtime.detect_tenstorrent_devices("1", 0, max_workers=10) == [1]


def test_missing_id_raises_clear_error(two_cards):
    with pytest.raises(ValueError) as ei:
        runtime.detect_tenstorrent_devices("7", 0, max_workers=10)
    msg = str(ei.value)
    assert "7" in msg and "0, 1" in msg  # names the bad id and the available ones


def test_missing_id_when_no_cards(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob", lambda pat: [])
    with pytest.raises(ValueError) as ei:
        runtime.detect_tenstorrent_devices("0", 0, max_workers=10)
    assert "none detected" in str(ei.value)


def test_num_devices_and_default(two_cards):
    assert runtime.detect_tenstorrent_devices(None, 1, max_workers=10) == [0]
    assert runtime.detect_tenstorrent_devices(None, 0, max_workers=10) == [0, 1]
    assert runtime.detect_tenstorrent_devices(None, 0, max_workers=1) == [0]  # max_workers cap honored


def test_duplicate_stem_inputs_rejected(tmp_path):
    (tmp_path / "target.fasta").write_text(">A|protein\nMK\n")
    (tmp_path / "target.yaml").write_text("sequences: []\n")
    struct = tmp_path / "structures"
    struct.mkdir()
    with pytest.raises(ValueError, match="share a name stem"):
        runtime.discover_jobs(tmp_path, struct, "cif", override=True)


def test_unique_stems_discovered(tmp_path):
    (tmp_path / "a.fasta").write_text(">A|protein\nMK\n")
    (tmp_path / "b.yaml").write_text("sequences: []\n")
    struct = tmp_path / "structures"
    struct.mkdir()
    jobs = runtime.discover_jobs(tmp_path, struct, "cif", override=True)
    assert sorted(j.id for j in jobs) == ["a", "b"]
