"""`program_cache_for` clears ttnn's program cache when the input shape signature changes, and only then."""
import pytest
import torch

T = pytest.importorskip("tt_bio.tenstorrent")


class FakeDevice:
    def __init__(self):
        self.calls = []

    def disable_and_clear_program_cache(self):
        self.calls.append("clear")

    def enable_program_cache(self):
        self.calls.append("enable")


def test_clears_on_new_shapes_only(monkeypatch):
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda dev: dev.calls.append("sync"))
    monkeypatch.setattr(T, "_PROGRAM_CACHE_SHAPES", [None])
    dev = FakeDevice()
    a = T.input_shapes({"ref_pos": torch.zeros(5941, 3), "restype": torch.zeros(772, 32), "seed": 105})
    b = T.input_shapes({"ref_pos": torch.zeros(6202, 3), "restype": torch.zeros(791, 32)})
    T.program_cache_for(a, dev)
    assert dev.calls == ["sync", "clear", "enable"]
    T.program_cache_for(a, dev)                      # warm repeat of the same input keeps its programs
    assert len(dev.calls) == 3
    T.program_cache_for(b, dev)
    assert dev.calls[3:] == ["sync", "clear", "enable"]
    T.program_cache_for(None, dev)                   # None clears every time
    T.program_cache_for(None, dev)
    assert dev.calls.count("clear") == 4
