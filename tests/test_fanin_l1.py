"""The fan-in L1 gate refuses by name, and the budget is priced on the tensor's own bytes.

`bcx-p10-l1fuse` leg 3. The gate decides where the backward's promoted cotangent lives, and the
two things that can go wrong with it are silent: admitting a tensor that does not fit (which the
allocator then refuses mid-backward), and declining everything (which reads as a lever that did
nothing). Both are checked here, against a budget the test sets itself rather than against
whatever card happens to run it.
"""
import types

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import fanin_l1  # noqa: E402


class FakeTensor:
    def __init__(self, shape, layout=None, buffer_type=None, memory_layout=None):
        self.padded_shape = shape
        self.layout = ttnn.TILE_LAYOUT if layout is None else layout
        self._mc = types.SimpleNamespace(
            buffer_type=ttnn.BufferType.DRAM if buffer_type is None else buffer_type,
            memory_layout=(ttnn.TensorMemoryLayout.INTERLEAVED if memory_layout is None
                           else memory_layout))

    def memory_config(self):
        return self._mc


#: qb2 card 0, read off the allocator by perf/bcx_p10_l1fuse/budget.py: 11x10 cores at
#: 1,461,760 B a bank is 153.34 MiB of aggregate L1.
QB2_GRID_L1 = 1461760 * 11 * 10


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(fanin_l1, "FANIN_L1", True)
    monkeypatch.setattr(fanin_l1, "FANIN_L1_SHARE", 0.35)
    monkeypatch.setattr(fanin_l1, "_grid_l1_bytes", lambda: QB2_GRID_L1)
    fanin_l1.REACH.clear()
    return fanin_l1


def test_off_by_default_and_costs_one_boolean(monkeypatch):
    monkeypatch.setattr(fanin_l1, "FANIN_L1", False)
    fanin_l1.REACH.clear()
    assert fanin_l1.config(FakeTensor([1, 288, 288, 128]), ttnn.float32) is None
    assert dict(fanin_l1.REACH) == {}


def test_the_rounds_own_cotangent_is_served(gate):
    # 1x288x288x128 float32 is 40.50 MiB, 26.4 % of the grid, under the 35 % share.
    assert gate.config(FakeTensor([1, 288, 288, 128]), ttnn.float32) == ttnn.L1_MEMORY_CONFIG
    assert gate.REACH["served"] == 1


def test_the_triangle_attention_bias_is_refused_by_the_budget(gate):
    # 288x1x288x384 float32 is 121.50 MiB, 79.2 % of the grid. This is the chain leg 2 said
    # does not fit, and the budget has to be what says so.
    assert gate.config(FakeTensor([288, 1, 288, 384]), ttnn.float32) is None
    assert gate.REACH["declined: over the L1 budget"] == 1
    assert gate.REACH["served"] == 0


def test_the_budget_is_bytes_and_not_a_shape(gate):
    # Same element count, half the bytes: bfloat16 passes where float32 fails. At the 35 %%
    # share qb2 gives a 53.67 MiB budget, and 1x288x288x192 is 60.75 MiB as float32 and
    # 30.38 as bfloat16, so the same shape lands on both sides of it.
    big = FakeTensor([1, 288, 288, 192])
    assert gate.config(big, ttnn.float32) is None
    assert gate.config(big, ttnn.bfloat16) == ttnn.L1_MEMORY_CONFIG


@pytest.mark.parametrize("kwargs,reason", [
    ({"layout": ttnn.ROW_MAJOR_LAYOUT}, "declined: not TILE"),
    ({"memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED}, "declined: not interleaved"),
    ({"buffer_type": ttnn.BufferType.L1}, "declined: already off DRAM"),
])
def test_every_refusal_is_named(gate, kwargs, reason):
    assert gate.config(FakeTensor([1, 288, 288, 128], **kwargs), ttnn.float32) is None
    assert gate.REACH[reason] == 1
    assert gate.REACH["served"] == 0


def test_an_unreadable_operand_declines_rather_than_raising(gate):
    class Broken:
        padded_shape = [1, 288, 288, 128]
        layout = ttnn.TILE_LAYOUT

        def memory_config(self):
            raise RuntimeError("no")

    assert gate.config(Broken(), ttnn.float32) is None
    assert gate.REACH["declined: could not read the operand"] == 1


def test_an_unknown_dtype_does_not_fit(gate):
    assert not gate.fits(FakeTensor([1, 32, 32, 32]), object())


def test_the_allocator_refusing_l1_falls_back_and_is_counted(gate, monkeypatch):
    calls = []

    def fake_typecast(t, dtype, memory_config=None):
        calls.append(memory_config)
        if memory_config is not None:
            raise RuntimeError("out of L1")
        return "dram result"

    monkeypatch.setattr(fanin_l1.ttnn, "typecast", fake_typecast)
    assert gate.typecast(FakeTensor([1, 288, 288, 128]), ttnn.float32) == "dram result"
    assert calls == [ttnn.L1_MEMORY_CONFIG, None]
    assert gate.REACH["spilled: the allocator refused L1"] == 1
    assert gate.REACH["served"] == 0
