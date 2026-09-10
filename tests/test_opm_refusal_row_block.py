"""esmfold2's outer product must row-block itself when DRAM refuses the single pass.

The 1536-token refusal was attributed by `worker._origin_frame` to
`esmfold2.py:1111 in _rows` -- the permute of `OuterProductMean._rows`' matmul product, NOT the
pair FFN. `_rows` builds [B, Bl*32, L*32], quadratic in L with a 32x blow-up on both axes: at
L=1536 that is 49152 x 49152 in bf16, 4831838208 B, and the permute needs a second one live at
the same time. `tenstorrent.pair_row_tile` returns 0 on a big grid, so Blackhole runs it in one
pass, which was right at every size it was measured at and is not at 1536.

Row-blocking is bit-exact here by the op's own argument, already in the source comment: output
row i depends only on a[i] and b, so tiling i partitions independent rows and reassociates
nothing. The fallback still fires only AFTER a refusal, so every size that fits keeps its
single pass.

Host-only: no device, no weights. `_rows` and the ttnn slicing are stubbed, so what runs is the
routing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import esmfold2 as E  # noqa: E402

REFUSAL = ("Not enough space to allocate 4831838208 B DRAM buffer across 8 banks, where each "
           "bank needs to store 603979776 B, but bank size is 4278190016 B (allocated: "
           "3579670528 B, free: 698519488 B, largest free block: 504088512 B)")


class _Slice:
    """Stands in for a ttnn tensor: only [:, s:e, :, :] is ever taken of it."""

    def __init__(self, L):
        self.shape = (1, L, 8192, 32)

    def __getitem__(self, key):
        return self


class _OPM:
    """OuterProductMean with `_rows` stubbed. `refuse_until` many calls raise the refusal."""

    _row_blocked = E.OuterProductMean._row_blocked

    def __init__(self, refuse_first: int = 0):
        self.refuse_first = refuse_first
        self.row_calls: list[int] = []      # Bl of each _rows call

    def _rows(self, a_blk, b2, recip_blk, L, M):
        self.row_calls.append(a_blk.shape[1] if hasattr(a_blk, "shape") else -1)
        if self.refuse_first > 0:
            self.refuse_first -= 1
            raise RuntimeError(REFUSAL)
        return "block"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    E._OPM_ROWS_REFUSED.clear()
    monkeypatch.setattr(E.ttnn, "concat", lambda parts, dim=1: f"concat:{len(parts)}")
    yield
    E._OPM_ROWS_REFUSED.clear()


def test_the_block_scales_with_L_and_is_tile_aligned():
    for L in (512, 1024, 1536, 2048):
        rows = E._opm_fallback_rows(L)
        assert rows % 32 == 0, f"L={L} gives a non-tile-aligned {rows}"
        assert 32 <= rows <= L // 4, f"L={L} gives {rows}"
    # The transient is rows*32 x L*32 and TWO are live across the permute. At the measured
    # L=1536 the pair has to fit the 504088512 B largest free block, per bank, with room left.
    rows = E._opm_fallback_rows(1536)
    per_bank_pair = 2 * (rows * 32) * (1536 * 32) * 2 // 8
    assert per_bank_pair < 504088512, f"{per_bank_pair} B per bank does not fit the free block"


def test_a_refused_single_pass_is_re_run_in_row_blocks():
    """The path __call__ takes after the refusal: one block per chunk of L, each asking `_rows`
    for the full L on the second axis, concatenated back."""
    L = 1536
    opm = _OPM()
    rows = E._opm_fallback_rows(L)
    result = opm._row_blocked(_Slice(L), object(), _Slice(L), L, 8192, rows)
    nblk = -(-L // rows)
    assert result == f"concat:{nblk}", result
    assert len(opm.row_calls) == nblk, opm.row_calls


def _through_the_helper(opm, L, rows=None):
    """What `__call__` does after `pair_row_tile` returns 0: hand the single pass and the
    row-blocked one to the shared fallback. The shrink loop lives there, not in `_row_blocked`,
    so this is where the shrink is exercised."""
    from tt_bio import tenstorrent

    return tenstorrent.row_block_after_refusal(
        E._OPM_ROWS_REFUSED, L,
        lambda: opm._rows(_Slice(L), object(), _Slice(L), L, 8192),
        lambda r: opm._row_blocked(_Slice(L), object(), _Slice(L), L, 8192, r),
        rows=rows if rows is not None else E._opm_fallback_rows(L), tag="opm")


def test_the_block_halves_on_a_further_refusal_and_stops_at_one_tile():
    L = 1024
    rows = E._opm_fallback_rows(L)              # 256
    opm = _OPM(refuse_first=2)                  # single pass refuses, first block refuses
    result = _through_the_helper(opm, L)
    assert E._OPM_ROWS_REFUSED[L] == rows // 2, E._OPM_ROWS_REFUSED
    assert result == f"concat:{-(-L // (rows // 2))}", result


def test_a_refusal_that_never_clears_is_raised_and_not_looped_forever():
    L = 1024
    opm = _OPM(refuse_first=10_000)
    with pytest.raises(RuntimeError) as info:
        _through_the_helper(opm, L)
    assert "4831838208" in str(info.value)
    assert E._OPM_ROWS_REFUSED.get(L, 32) == 32, "it must stop shrinking at one tile"


def test_a_size_that_already_refused_skips_straight_to_the_block():
    """48 pair ops per recycle: paying a failed multi-GiB allocation on each one is minutes of
    wasted DRAM traffic, so the size is remembered."""
    L = 1024
    E._OPM_ROWS_REFUSED[L] = 128
    opm = _OPM()
    assert _through_the_helper(opm, L) == f"concat:{-(-L // 128)}"
    assert len(opm.row_calls) == -(-L // 128), opm.row_calls


def test_anything_that_is_not_an_allocator_refusal_propagates_untouched():
    """The negative control: a real bug must not be re-run row by row until it looks like
    a capacity problem."""
    L = 1024

    class _Boom(_OPM):
        def _rows(self, a_blk, b2, recip_blk, L, M):
            raise RuntimeError("TT_THROW @ program.cpp:1052: circular buffer clash")

    opm = _Boom()
    with pytest.raises(RuntimeError) as info:
        opm._row_blocked(_Slice(L), object(), _Slice(L), L, 8192, E._opm_fallback_rows(L))
    assert "circular buffer clash" in str(info.value)
    assert L not in E._OPM_ROWS_REFUSED, "a non-refusal must not be remembered as one"
