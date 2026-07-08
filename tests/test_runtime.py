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
