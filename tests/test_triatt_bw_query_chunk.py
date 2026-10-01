"""`triatt_bw`'s query loop: what it serves and the premises it rests on, card-free.

The compute kernel runs query chunk outside and leading axis inside, so the fronted bias and the
float32 dbias accumulator are `Qt * Nt` tiles, not `Nt * Nt`. Until this loop existed the
whole-query form stopped fitting at 320 tokens and a chunked plan was refused at build, because
the kernel would have read past buffers sized for a chunk. These hold the new contract.
"""
import re
from pathlib import Path

from tt_bio import triatt_bw as T

GRID = (8, 9)
KDIR = Path(T.__file__).parent / "kernels" / "triatt_bw"


def test_whole_query_is_kept_where_it_fits():
    """288 is the shipped BindCraft 2 axis: it must take the unchanged single-chunk kernel."""
    p = T.serving_plan(64, 4, 288, 32, GRID)
    assert p["Qt"] == p["Nt"] == 9


def test_every_bucket_up_to_1024_is_served():
    for n in range(288, 1025, 32):
        p = T.serving_plan(64, 4, n, 32, GRID)
        assert p is not None and T.fits_l1(p), n
        assert p["Nt"] % p["Qt"] == 0, n


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
