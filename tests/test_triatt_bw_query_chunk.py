"""`triatt_bw`'s query chunk is priced but not implemented, and must never reach a kernel.

`cb_table` sizes the three score-sized buffers at `Qt * Nt` tiles while
`compute/triatt_bw.cpp` works in `score_tiles = Nt * Nt` and takes no chunk argument. A plan
with `Qt < Nt` therefore prices a config that would read past those buffers rather than refuse,
which is why `build` stops it. Card-free: all of this is the host's own arithmetic.
"""
import re
from pathlib import Path

import pytest

from tt_bio import triatt_bw as T

SHAPE = (64, 4, 352, 32)   # 352 is where the whole-query form first stops fitting
GRID = (8, 9)


def test_whole_query_is_the_default():
    p = T.plan(*SHAPE, GRID)
    assert p["Qt"] == p["Nt"]


def test_build_refuses_a_chunked_plan():
    p = T.plan(*SHAPE, GRID, q_chunk_tiles=1)
    assert p["Qt"] == 1 and p["Nt"] == 11
    assert T.fits_l1(p), "the point of the guard is that this one PRICES as fitting"
    with pytest.raises(ValueError, match="priced but not implemented"):
        T.build(None, *([None] * 9), p, (None,), 0.176)


def test_the_kernel_has_no_query_loop():
    """The guard's premise, asserted against the kernel rather than trusted."""
    src = (Path(T.__file__).parent / "kernels" / "triatt_bw" / "compute" / "triatt_bw.cpp").read_text()
    assert "score_tiles = Nt * Nt" in src
    assert not re.search(r"\bq_chunk|\bQt\b", src)


def test_a_query_loop_would_reach_384_and_no_further():
    """What the unwritten loop would be worth, so the next reader does not re-derive it."""
    served = [n for n in range(192, 641, 32)
              if T.largest_fitting_q_chunk(64, 4, n, 32, GRID) is not None]
    whole = [n for n in range(192, 641, 32) if T.fits_l1(T.plan(64, 4, n, 32, GRID))]
    assert max(whole) == 288 and max(served) == 384
    # bias + dbias alone, which no query chunk shrinks, is what closes it above 384.
    Nt = 512 // 32
    assert Nt * Nt * (2048 + 4096) > T.L1_PER_CORE - T.PROGRAM_RESERVE
