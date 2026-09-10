"""Above the pair-FFN parity window, a DRAM refusal must fall back to the row block.

ESMFold2 at 1536 tokens died on `Not enough space to allocate 4831838208 B DRAM buffer` with
26.67 of 31.875 GiB resident on a p150a: room enough, no single free block big enough. That
4831838208 B is [1,1536,1536,1024] bf16, fc1's whole 2*d_ff activation, allocated in one piece
because both blocking levers were off above 1024 tokens -- `PAIR_FFN_ROW_BLOCK_SEQ` is (320, 1024)
and `pair_row_tile` returned 0 on a big grid at every L.

`pair_row_tile` now has its own Blackhole budget (`BH_PAIR_SINGLE_PASS_MAX`), so 1536 itself is
tiled directly and never reaches the unblocked path this file is testing -- the fallback below
still matters for the window that remains, at or under that budget (1024), where the tensor is
still small enough that the code tries the unblocked call first.

The window is there to stop a lever changing the last bf16 bit of a fold that already works, so
the fallback fires only AFTER a refusal. Inside the window nothing changes: the unblocked path is
still tried first and still succeeds, so every size whose byte-identical output was measured
keeps running the path it was measured on.

Host-only. No device, no ttnn tensors: the fake input carries the two attributes `__call__` reads
off a real one, and the four collaborators are stubs, so what is under test is the control flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import esmc as E  # noqa: E402

# The refusal verbatim, from perf/bh1536/runs/esmfold2_1536/fold.log.
REFUSAL = ("Not enough space to allocate 4831838208 B DRAM buffer across 8 banks, where each "
           "bank needs to store 603979776 B, but bank size is 4278190016 B (allocated: "
           "3579670528 B, free: 698519488 B, largest free block: 504088512 B)")


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.padded_shape = tuple(shape)


class _Probe:
    """A SwiGLUFFN with every collaborator stubbed, so only the routing runs."""

    def __init__(self, ffn_raises=None):
        self._ffn_raises = ffn_raises
        self.calls: list[str] = []
        self.row_block_rows: int | None = None

    _rows_after_refusal = E.SwiGLUFFN._rows_after_refusal
    __call__ = E.SwiGLUFFN.__call__
    residual = E.SwiGLUFFN.residual

    def _split_plan(self, x):
        return True, 0                      # split on, row block off: above the window

    def _ffn(self, x, split=False, **kw):
        self.calls.append("unblocked")
        if self._ffn_raises is not None:
            raise self._ffn_raises
        return "unblocked-out"

    def _row_blocked(self, x, rows, residual):
        self.calls.append("row_blocked")
        self.row_block_rows = rows
        return "row-blocked-out"


@pytest.fixture(autouse=True)
def _clean_cache():
    E._UNBLOCKED_REFUSED.clear()
    E.WINDOW_FALLBACK_STATS[:] = [0, 0]
    yield
    E._UNBLOCKED_REFUSED.clear()


def test_the_recorded_refusal_is_recognised_as_one():
    assert E._is_alloc_refusal(RuntimeError(REFUSAL)) is True
    # The negative controls on the predicate: it must not treat every failure as an OOM, or the
    # fallback would paper over real bugs by re-running them row by row.
    assert E._is_alloc_refusal(RuntimeError("TT_THROW @ program.cpp:1052")) is False
    assert E._is_alloc_refusal(ValueError("shape mismatch")) is False
    assert E._is_alloc_refusal(RuntimeError("out of memory")) is False


def test_a_refused_pair_ffn_falls_back_to_the_row_block():
    probe = _Probe(ffn_raises=RuntimeError(REFUSAL))
    out = probe(_FakeTensor((1, 1024, 1024, 256)))
    assert out == "row-blocked-out", "the refusal was not caught; the fold dies as it did"
    assert probe.calls == ["unblocked", "row_blocked"], probe.calls
    assert probe.row_block_rows == E._PAIR_FFN_ROW_BLOCK
    assert E.WINDOW_FALLBACK_STATS == [1, 0], (
        f"census must say the fallback served this call: {E.WINDOW_FALLBACK_STATS}")


def test_the_unblocked_path_is_still_tried_first():
    """The whole point of firing only after a refusal: a size inside the window keeps running
    the path its byte-identical output was measured on. A fallback that pre-empted the unblocked
    call would silently move every fold at every size onto the row block."""
    probe = _Probe()
    assert probe(_FakeTensor((1, 1024, 1024, 256))) == "unblocked-out"
    assert probe.calls == ["unblocked"], probe.calls
    assert E.WINDOW_FALLBACK_STATS == [0, 1], E.WINDOW_FALLBACK_STATS
    assert not E._UNBLOCKED_REFUSED, "nothing was refused, so nothing may be cached"


def test_a_refused_shape_takes_the_row_block_without_a_second_refusal():
    """ESMFold2 runs 48 pair transitions per recycle. Paying one failed 4.5 GiB allocation each
    is minutes of wasted DRAM traffic, so the shape is remembered."""
    shape = (1, 1024, 1024, 256)
    first = _Probe(ffn_raises=RuntimeError(REFUSAL))
    first(_FakeTensor(shape))
    second = _Probe(ffn_raises=RuntimeError(REFUSAL))
    assert second(_FakeTensor(shape)) == "row-blocked-out"
    assert second.calls == ["row_blocked"], (
        f"the second call refused again instead of reading the cache: {second.calls}")
    # A different length is a different shape and gets its own first attempt.
    other = _Probe()
    other(_FakeTensor((1, 768, 768, 256)))
    assert other.calls == ["unblocked"], other.calls


def test_anything_that_is_not_an_allocator_refusal_still_propagates():
    boom = RuntimeError("TT_THROW @ program.cpp:1052: circular buffer clash")
    probe = _Probe(ffn_raises=boom)
    with pytest.raises(RuntimeError) as info:
        probe(_FakeTensor((1, 1024, 1024, 256)))
    assert info.value is boom
    assert probe.calls == ["unblocked"], "a real bug must not be retried row by row"


def test_a_three_d_input_is_not_row_blocked():
    """The row block partitions dim 1 of a 4-D pair tensor. ESMC's LM FFN is [B,L,d] and has its
    own tiling, so a refusal there must not be handed to a path that does not apply to it."""
    probe = _Probe(ffn_raises=RuntimeError(REFUSAL))
    with pytest.raises(RuntimeError):
        probe(_FakeTensor((1, 1536, 2560)))
    assert probe.calls == ["unblocked"], probe.calls


def test_residual_takes_the_row_block_once_the_shape_is_known_refused():
    """`residual` folds the add into the block (lever F). Once the shape is known to be refused
    it must go straight there rather than through `self(x)`, or the fold loses that lever for
    every one of the 48 transitions."""
    shape = (1, 1536, 1536, 256)
    E._UNBLOCKED_REFUSED.add(shape)
    probe = _Probe()
    assert probe.residual(_FakeTensor(shape)) == "row-blocked-out"
    assert probe.calls == ["row_blocked"], probe.calls
