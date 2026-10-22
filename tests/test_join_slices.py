"""`autograd._join_slices`: a lone slice gradient is one pad, disjoint slices one concat.

Run: python3 -m pytest tests/test_join_slices.py            (card-free)
"""
import ttnn

from tt_bio import autograd as ag


class _G:
    def __init__(self, shape, dtype=ttnn.bfloat16):
        self.shape, self.dtype = list(shape), dtype

    def device(self):
        return None


def _record(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "pad", lambda g, pads, v: calls.append(("pad", pads, v)) or g)
    monkeypatch.setattr(ttnn, "concat", lambda bs, dim: calls.append(("concat", len(bs), dim)) or bs[0])
    monkeypatch.setattr(ag, "grad_zeros", lambda s, d, dev: _G(s, d))
    return calls


def test_one_slice_from_zero_is_a_pad(monkeypatch):
    calls = _record(monkeypatch)
    ag._join_slices([([0, 0, 0, 0], [224, 4, 224, 16], _G([224, 4, 224, 16]))], [224, 4, 224, 32])
    assert calls == [("pad", [(0, 0), (0, 0), (0, 0), (0, 16)], 0.0)]


def test_one_offset_slice_keeps_its_zeros(monkeypatch):
    calls = _record(monkeypatch)
    ag._join_slices([([0, 0, 16], [2, 224, 32], _G([2, 224, 16]))], [2, 224, 32])
    assert calls == [("concat", 2, 2)]


def test_disjoint_slices_are_one_concat(monkeypatch):
    calls = _record(monkeypatch)
    parts = [([0, 0, 0], [2, 224, 256], _G([2, 224, 256])),
             ([0, 0, 256], [2, 224, 512], _G([2, 224, 256]))]
    ag._join_slices(parts, [2, 224, 768])
    assert calls == [("concat", 3, 2)]
