"""The rne_add_masked gate: which (residual, update, mask) triples the kernel may serve.

The arithmetic is graded exactly on a card (`perf/bcx_p10_rneker/grade_masked.py`); what can go
wrong without one is the gate serving a mask the kernel does not index right. It indexes the mask
by `page % slab tiles`, which is only the broadcast when the mask is the update's last two dims
shared over its leading ones: AF2's per-row draw. A per-column draw broadcasts inside a tile.
"""
import types

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import rne_add  # noqa: E402


class FakeTensor:
    def __init__(self, shape, dtype=None, layout=None):
        self.shape = shape
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self.layout = ttnn.TILE_LAYOUT if layout is None else layout
        self._mc = types.SimpleNamespace(buffer_type=ttnn.BufferType.DRAM,
                                         memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED)

    def memory_config(self):
        return self._mc


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(rne_add, "MASKED", True)
    monkeypatch.setattr(rne_add, "_ENABLED", True)
    monkeypatch.setattr(rne_add._ops, "declines_under_tape", lambda name, *operands: False)
    return rne_add


Z = FakeTensor([1, 224, 224, 128])


def test_on_by_default():
    assert rne_add.MASKED is True or "TT_BIO_RNE_MASK" in __import__("os").environ


def test_a_per_row_mask_is_served(gate):
    assert gate.masked_eligible(Z, Z, FakeTensor([1, 1, 224, 128]), None)
    assert gate.masked_eligible(FakeTensor([1, 195, 195, 128]), FakeTensor([1, 195, 195, 128]),
                                FakeTensor([1, 1, 195, 128]), None)


@pytest.mark.parametrize("mask", [
    FakeTensor([1, 224, 1, 128]),                    # per-column: broadcast inside a tile
    FakeTensor([1, 224, 224, 128]),                  # full-size: not a shared slab
    FakeTensor([1, 1, 224, 64]),                     # wrong channel count
    FakeTensor([1, 1, 224, 128], dtype=ttnn.float32),
    FakeTensor([1, 1, 224, 128], layout=ttnn.ROW_MAJOR_LAYOUT),
])
def test_anything_else_is_declined(gate, mask):
    assert not gate.masked_eligible(Z, Z, mask, None)


def test_a_slab_too_large_for_l1_is_declined(gate):
    big = FakeTensor([1, 512, 512, 256])
    assert not gate.masked_eligible(big, big, FakeTensor([1, 1, 512, 256]), None)


def test_no_mask_and_an_ineligible_sum_are_declined(gate):
    assert not gate.masked_eligible(Z, Z, None, None)
    assert not gate.masked_eligible(Z, FakeTensor([1, 224, 224, 128], dtype=ttnn.float32),
                                    FakeTensor([1, 1, 224, 128]), None)
