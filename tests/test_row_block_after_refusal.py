"""One fallback mechanism for every op that DRAM refuses in a single pass.

Three ops had grown their own copy of "try the single pass, catch an allocator refusal, re-run
it in row blocks, halve the block if that is refused too, remember what the shape settled at":
`OuterProductMean`, `SwiGLUFFN`'s pair FFN and, once esmfold2 at 1536 got past the first two,
`TransitionLayer`. They are now one function, so a fourth op inherits it instead of getting a
fourth near-copy with its own off-by-one.

Host-only: `single` and `blocked` are plain callables here, so what runs is the control flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import tenstorrent as T  # noqa: E402

# Verbatim from perf/bh1536/runs/esmfold2_1536_splitfix/fold.log, the transition's own refusal.
REFUSAL = ("Not enough space to allocate 2415919104 B DRAM buffer across 8 banks, where each "
           "bank needs to store 301989888 B, but bank size is 4278190016 B (allocated: "
           "3689691136 B, free: 588498880 B, largest free block: 277086144 B)")


class _Op:
    """`single` refuses `refuse_single` times; `blocked` refuses while rows > `fits_at`."""

    def __init__(self, refuse_single=0, fits_at=10 ** 9):
        self.refuse_single, self.fits_at = refuse_single, fits_at
        self.singles = 0
        self.blocks: list[int] = []

    def single(self):
        self.singles += 1
        if self.refuse_single > 0:
            self.refuse_single -= 1
            raise RuntimeError(REFUSAL)
        return "single-out"

    def blocked(self, rows):
        self.blocks.append(rows)
        if rows > self.fits_at:
            raise RuntimeError(REFUSAL)
        return f"blocked:{rows}"


def _run(op, memo, key=(1, 1536, 1536, 512), rows=384):
    return T.row_block_after_refusal(memo, key, op.single, op.blocked, rows=rows, tag="test")


def test_a_single_pass_that_fits_is_left_alone():
    """The whole point of firing only after a refusal: a size that fits keeps running the path
    its byte-identical output was measured on."""
    op, memo = _Op(), {}
    assert _run(op, memo) == "single-out"
    assert op.blocks == [], op.blocks
    assert memo == {}, "nothing was refused, so nothing may be remembered"


def test_a_refused_single_pass_is_re_run_in_blocks():
    op, memo = _Op(refuse_single=1), {}
    assert _run(op, memo) == "blocked:384"
    assert op.singles == 1 and op.blocks == [384], (op.singles, op.blocks)
    assert memo[(1, 1536, 1536, 512)] == 384


def test_the_block_halves_until_it_fits_and_stays_tile_aligned():
    op, memo = _Op(refuse_single=1, fits_at=100), {}
    assert _run(op, memo) == "blocked:96"
    assert op.blocks == [384, 192, 96], op.blocks
    assert all(r % 32 == 0 for r in op.blocks), op.blocks
    assert memo[(1, 1536, 1536, 512)] == 96


def test_a_refusal_that_never_clears_is_raised_at_one_tile():
    op, memo = _Op(refuse_single=1, fits_at=0), {}
    with pytest.raises(RuntimeError) as info:
        _run(op, memo)
    assert "2415919104" in str(info.value)
    assert op.blocks[-1] == 32, op.blocks
    assert memo[(1, 1536, 1536, 512)] == 32, "it must stop shrinking at one tile"


def test_a_known_refused_shape_skips_the_single_pass():
    """esmfold2 runs its pair transitions once per recycle and its pair FFN 48 times. Paying a
    failed multi-GiB allocation each time is minutes of DRAM traffic for a known answer."""
    op, memo = _Op(), {(1, 1536, 1536, 512): 192}
    assert _run(op, memo) == "blocked:192"
    assert op.singles == 0, "the single pass ran again for a shape already known refused"


def test_anything_that_is_not_an_allocator_refusal_propagates():
    """The negative control: a real bug must not be retried row by row until it reads as a
    capacity problem."""
    boom = RuntimeError("TT_THROW @ program.cpp:1052: circular buffer clash")

    class _Boom(_Op):
        def single(self):
            raise boom

    op, memo = _Boom(), {}
    with pytest.raises(RuntimeError) as info:
        _run(op, memo)
    assert info.value is boom
    assert op.blocks == [], "a non-allocator failure was row-blocked"


def test_a_refusal_from_the_blocked_path_that_is_not_an_allocator_refusal_propagates():
    boom = RuntimeError("shape mismatch")

    class _Boom(_Op):
        def blocked(self, rows):
            raise boom

    op, memo = _Boom(refuse_single=1), {}
    with pytest.raises(RuntimeError) as info:
        _run(op, memo)
    assert info.value is boom
