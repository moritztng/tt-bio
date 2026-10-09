"""`pair_mm`: the class it serves and the calls it leaves alone, card-free."""
import types

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import ops, pair_mm  # noqa: E402


class _Shape:
    """Indexable and iterable but NOT sliceable, like ttnn.Shape."""
    def __init__(self, dims):
        self._d = list(dims)

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise TypeError("ttnn.Shape does not support slicing")
        return self._d[i]


def _t(shape, dtype=None, layout=None):
    return types.SimpleNamespace(shape=_Shape(shape), dtype=dtype or ttnn.bfloat16,
                                 layout=layout or ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("x,w,tb", [
    ((1, 288, 288, 128), (128, 512), False),   # pair-transition fc1
    ((1, 288, 288, 512), (512, 128), False),   # fc2
    ((82944, 512), (128, 512), True),          # fc1's dX
    ((1, 288, 288, 1024), (1024, 128), False),  # OPM out-projection
])
def test_serves_the_swept_class(x, w, tb):
    assert pair_mm.config(_t(x), _t(w), tb) is not None


@pytest.mark.parametrize("x,w,tb", [
    ((1, 1, 288, 128), (128, 512), False),            # MSA/single track: too few rows
    ((1, 288, 288, 256), (256, 256), False),          # unswept key
    ((1, 288, 288, 128), (512, 128), False),          # K mismatch
    ((1, 288, 288, 128), (1, 128, 128), False),       # batched right operand
])
def test_declines_outside_it(x, w, tb):
    assert pair_mm.config(_t(x), _t(w), tb) is None


def test_declines_float32_and_row_major():
    assert pair_mm.config(_t((1, 288, 288, 128), ttnn.float32), _t((128, 128))) is None
    assert pair_mm.config(_t((1, 288, 288, 128), layout=ttnn.ROW_MAJOR_LAYOUT),
                          _t((128, 128))) is None


def test_off_by_default_and_inert(monkeypatch):
    assert pair_mm.PAIR_MM_FUSED is False
    assert pair_mm.matmul(_t((1, 288, 288, 128)), _t((128, 128))) is None


def test_l1_output_is_declined(monkeypatch):
    monkeypatch.setattr(pair_mm, "PAIR_MM_FUSED", True)
    before = list(pair_mm.STATS)
    out = pair_mm.matmul(_t((1, 288, 288, 128)), _t((128, 128)),
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out is None and pair_mm.STATS[1] == before[1] + 1


def test_transposed_weight_is_cached_per_object(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "transpose", lambda w, a, b: calls.append(w) or ("T", w))
    pair_mm._WT.clear()
    w1, w2 = object(), object()
    assert pair_mm._transposed(w1) == ("T", w1)
    assert pair_mm._transposed(w1) == ("T", w1)
    assert pair_mm._transposed(w2) == ("T", w2)
    assert calls == [w1, w2]


def test_armed_by_fast_round():
    """Blackhole arms it; Wormhole keeps it off (`bindcraft2._BLACKHOLE_ONLY`, graded on BH only)."""
    from tt_bio import bindcraft2, tenstorrent
    armed_here = not tenstorrent.is_wormhole()
    with bindcraft2.fast_round() as armed:
        assert armed["PAIR_MM_FUSED"] is armed_here and pair_mm.PAIR_MM_FUSED is armed_here
    assert pair_mm.PAIR_MM_FUSED is False
    assert ops._pair_mm is pair_mm


def test_relu_rides_the_pack_and_other_activations_decline(monkeypatch):
    monkeypatch.setattr(pair_mm, "PAIR_MM_FUSED", True)
    seen = {}
    monkeypatch.setattr(ttnn.experimental, "minimal_matmul",
                        lambda **kw: seen.update(kw) or "out")
    x, w = _t((1, 288, 288, 128)), _t((128, 512))
    assert pair_mm.matmul(x, w, activation="relu") == "out"
    assert seen["fused_activation"].op_type == ttnn.UnaryOpType.RELU
    assert pair_mm.matmul(x, w) == "out" and seen["fused_activation"] is None
    before = pair_mm.STATS[1]
    assert pair_mm.matmul(x, w, activation="gelu") is None
    assert pair_mm.STATS[1] == before + 1
