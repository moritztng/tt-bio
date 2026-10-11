"""`triatt_bw`'s query loop: what it serves and the premises it rests on, card-free.

The compute kernel runs query chunk outside and leading axis inside, so the fronted bias and the
float32 dbias accumulator are `Qt * Nt` tiles, not `Nt * Nt`. Until this loop existed the
whole-query form stopped fitting at 320 tokens and a chunked plan was refused at build, because
the kernel would have read past buffers sized for a chunk. These hold the new contract, priced
per board: the two boards' L1 differ, and only Blackhole has run the loop.
"""
import re
from pathlib import Path

from tt_bio import triatt_bw as T

GRID = (8, 9)
KDIR = Path(T.__file__).parent / "kernels" / "triatt_bw"


def test_whole_query_is_kept_where_it_fits():
    """288 is the shipped BindCraft 2 axis: it must take the unchanged single-chunk kernel."""
    for wh in (False, True):
        p = T.serving_plan(64, 4, 288, 32, GRID, wormhole=wh)
        assert p["Qt"] == p["Nt"] == 9


def test_every_bucket_up_to_1024_is_served_on_blackhole():
    for grid in (GRID, (13, 10), (8, 8)):
        for n in range(288, 1025, 32):
            p = T.serving_plan(64, 4, n, 32, grid, wormhole=False)
            assert p is not None and T.fits_l1(p, False), (grid, n)
            assert p["Nt"] % p["Qt"] == 0, n


def test_each_board_is_priced_by_its_own_l1():
    """Wormhole has 1464 KiB of L1 to Blackhole's 1536. At 480 the chunk Blackhole takes would be
    refused by tt-metal on a Wormhole chip, inside a backward, so the price must not carry over."""
    assert T.cb_budget(False) == 1572864 - 109056
    assert T.cb_budget(True) == 1395424
    p = T.largest_fitting_q_chunk(480, 4, 480, 32, (8, 8), wormhole=False)
    assert p["Qt"] == 5
    assert not T.fits_l1(p, True)
    assert T.largest_fitting_q_chunk(480, 4, 480, 32, (8, 8), wormhole=True)["Qt"] == 3
    # every bucket fits on Wormhole's own budget
    for n in range(288, 1025, 32):
        assert T.largest_fitting_q_chunk(n, 4, n, 32, (8, 8), wormhole=True) is not None, n


def test_wormhole_serves_the_chunk_exactly_where_the_whole_query_does_not_fit():
    """Graded on a Wormhole chip at 288-576 (`perf/spd/whchunk_grade.py`), so both boards chunk.

    Below the whole-query limit the plan must stay the unchunked one.
    """
    for n in range(288, 1025, 32):
        p = T.serving_plan(n, 4, n, 32, (8, 8), wormhole=True)
        assert p is not None and T.fits_l1(p, True), n
        assert p["Nt"] % p["Qt"] == 0, n
    assert T.serving_plan(64, 4, 288, 32, GRID, wormhole=True)["Qt"] == 9  # still whole-query
    assert T.serving_plan(512, 4, 512, 32, (8, 8), wormhole=True)["Qt"] == 4
    assert T.serving_plan(512, 4, 512, 32, (8, 8), wormhole=False)["Qt"] == 4


def test_bias_and_dbias_shrink_with_the_chunk():
    p = T.plan(64, 4, 768, 32, GRID, q_chunk_tiles=2)
    sizes = {idx: n * page for idx, n, page, _f in T.cb_table(p)}
    assert sizes[T.CB_BIAS] == 2 * 24 * 2048
    assert sizes[T.CB_DBIAS] == 2 * 24 * 4096


def test_chunked_dk_dv_are_float32_and_whole_query_is_not():
    import ttnn
    fmt = lambda p: {idx: f for idx, _n, _pg, f in T.cb_table(p)}
    assert fmt(T.plan(64, 4, 768, 32, GRID, q_chunk_tiles=2))[T.CB_DK] == ttnn.float32
    assert fmt(T.plan(64, 4, 288, 32, GRID))[T.CB_DK] == ttnn.bfloat16


def test_the_kernels_loop_over_query_chunks():
    """The contract above, asserted against the kernel sources rather than trusted."""
    for f in ("compute/triatt_bw.cpp", "dataflow/reader.cpp", "dataflow/writer.cpp"):
        src = (KDIR / f).read_text()
        assert re.search(r"chunks = Nt / Qt", src), f
        assert re.search(r"for \(uint32_t c = 0; c < chunks; \+\+c\)", src), f
    # the read-after-write guard on the float32 partials
    assert "noc_semaphore_wait_min" in (KDIR / "dataflow/reader.cpp").read_text()


def test_cb_table_covers_the_kernels():
    for qt in (None, 1, 2):
        T.check_cb_coverage(T.plan(64, 4, 768, 32, GRID, q_chunk_tiles=qt))
