"""The widen_add gate and the page count it shares with rne_add.

`bcx-p10-widenadd`. The kernel's arithmetic is graded against float64 on the card
(`perf/bcx_p10_widenadd/grade.py`); what can go wrong without a card is the gate serving a call
the kernel does not compute, and the page count disagreeing with ttnn's TILE padding, which is
how the first build of this kernel read 0.29-0.67 rel L2 on every non-aligned shape.
"""
import types

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import rne_add  # noqa: E402


class FakeTensor:
    def __init__(self, shape, dtype=None, layout=None, memory_layout=None):
        self.shape = shape
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self.layout = ttnn.TILE_LAYOUT if layout is None else layout
        self._mc = types.SimpleNamespace(
            buffer_type=ttnn.BufferType.DRAM,
            memory_layout=(ttnn.TensorMemoryLayout.INTERLEAVED if memory_layout is None
                           else memory_layout))

    def memory_config(self):
        return self._mc


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(rne_add, "WIDEN_ADD", True)
    rne_add.WIDEN_REACH.clear()
    return rne_add


@pytest.mark.parametrize("shape, pages", [
    ([1, 288, 288, 128], 288 * 9 * 4),                       # aligned: rneker's 10368
    ([3, 50, 70], 3 * 2 * 3),                               # H padded per leading index
    ([1, 17, 33, 100], 17 * 2 * 4),
    ([7, 1, 45, 31], 7 * 2 * 1),
    ([100], 1 * 4),
])
def test_page_count_is_ttnns_tile_padding(shape, pages):
    assert rne_add._tile_count(FakeTensor(shape)) == pages


def test_off_by_default_records_nothing(monkeypatch):
    monkeypatch.setattr(rne_add, "WIDEN_ADD", False)
    rne_add.WIDEN_REACH.clear()
    a = FakeTensor([1, 288, 288, 128])
    assert not rne_add.widen_eligible(a, a)
    assert dict(rne_add.WIDEN_REACH) == {}


@pytest.mark.parametrize("da, db", [
    (ttnn.bfloat16, ttnn.bfloat16), (ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32)])
def test_the_fan_ins_dtypes_are_served(gate, da, db):
    assert gate.widen_eligible(FakeTensor([288, 1, 288, 384], da), FakeTensor([288, 1, 288, 384], db))


@pytest.mark.parametrize("a, b, reason", [
    (FakeTensor([4, 64], ttnn.float32), FakeTensor([4, 64], ttnn.float32), "both float32"),
    (FakeTensor([4, 64], ttnn.bfloat8_b), FakeTensor([4, 64]), "dtype"),
    (FakeTensor([4, 64], layout=ttnn.ROW_MAJOR_LAYOUT), FakeTensor([4, 64]), "not TILE"),
    (FakeTensor([1, 4, 64]), FakeTensor([4, 64]), "shape"),
    (FakeTensor([4, 64], memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
     FakeTensor([4, 64]), "sharded"),
])
def test_every_refusal_is_named(gate, a, b, reason):
    assert not gate.widen_eligible(a, b)
    assert dict(gate.WIDEN_REACH) == {"declined: " + reason: 1}
