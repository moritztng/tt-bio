"""`tt-bio preflight` answers "can this machine run it", so it has to mention the chips."""
import tt_bio.runtime as runtime


def test_no_chips_says_cpu_only(monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: {})
    assert "no Tenstorrent chip visible" in runtime.chip_summary()


def test_healthy_chips_are_listed_by_umd_index(monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index",
                        lambda: {"0000:01:00.0": 0, "0000:61:00.0": 1})
    monkeypatch.setattr(runtime, "quarantined_chips", lambda: {})
    monkeypatch.setattr(runtime, "missing_device_nodes", lambda indices: [])
    assert runtime.chip_summary() == "2 Tenstorrent chip(s): 0, 1"


def test_a_listed_but_unusable_chip_is_named(monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index",
                        lambda: {"0000:01:00.0": 0, "0000:61:00.0": 1})
    monkeypatch.setattr(runtime, "quarantined_chips", lambda: {1: "flips DRAM bits"})
    monkeypatch.setattr(runtime, "missing_device_nodes", lambda indices: [(0, 3)])
    out = runtime.chip_summary()
    assert "2 unusable" in out
    assert "chip 1 quarantined (flips DRAM bits)" in out
    assert "chip 0 has no /dev/tenstorrent/3" in out


def test_a_quarantined_chip_of_another_host_is_ignored(monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: {"0000:01:00.0": 0})
    monkeypatch.setattr(runtime, "quarantined_chips", lambda: {7: "not on this host"})
    monkeypatch.setattr(runtime, "missing_device_nodes", lambda indices: [])
    assert runtime.chip_summary() == "1 Tenstorrent chip(s): 0"
