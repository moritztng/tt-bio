"""`genq.compact_plan` must refuse anything it cannot reproduce exactly.

The plan replaces per-core runtime arguments with arithmetic the core runs itself, so a plan that
is wrong by one core is a kernel reading or writing someone else's tiles. Every test here is a
shape the arithmetic must NOT accept, plus the two it must.
"""
import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import genq  # noqa: E402


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                              for x, y in cores])


def _row_major(w, h, n, p1, n1=None, p2=None):
    """`n` cores of a `w` x `h` grid, row-major, first `n1` taking `p1` units and the rest `p2`."""
    n1 = n if n1 is None else n1
    p2 = 0 if p2 is None else p2
    cores = [(i % w, i // w) for i in range(n)]
    assign, first = {}, 0
    for i, c in enumerate(cores):
        per = p1 if i < n1 else p2
        assign[c] = (first, per)
        first += per
    return assign, _crs(cores)


def test_uniform_row_major_split_is_accepted():
    assign, crs = _row_major(10, 13, 130, 80)
    plan = genq.compact_plan(assign, crs)
    assert plan is not None
    order, x0, y0, span, n1, p1, p2 = plan
    assert (order, x0, y0, span, n1, p1) == (genq.ROW_MAJOR, 0, 0, 10, 130, 80)


def test_two_group_split_is_accepted_and_recomputes_every_core():
    assign, crs = _row_major(10, 13, 130, 80, n1=48, p2=79)
    plan = genq.compact_plan(assign, crs)
    assert plan is not None
    order, x0, y0, span, n1, p1, p2 = plan
    for (x, y), (first, num) in assign.items():
        i = (y - y0) * span + (x - x0) if order == genq.ROW_MAJOR else (x - x0) * span + (y - y0)
        got = (i * p1, p1) if i < n1 else (n1 * p1 + (i - n1) * p2, p2)
        assert got == (first, num), ((x, y), got, (first, num))


def test_a_placed_core_with_no_slice_is_refused():
    """The bottom-of-DRAM write in bcx-TABWD.md: a core in the program with no arguments."""
    assign, crs = _row_major(10, 13, 130, 80)
    assign.pop((0, 0))
    assert genq.compact_plan(assign, crs) is None


def test_three_group_sizes_are_refused():
    assign, crs = _row_major(10, 13, 6, 8)
    assign[(0, 0)] = (0, 7)
    assign[(1, 0)] = (7, 9)
    assert genq.compact_plan(assign, crs) is None


def test_noncontiguous_units_are_refused():
    """Same two group sizes, but a core's first unit does not follow the one before it."""
    assign, crs = _row_major(10, 13, 8, 8)
    assign[(4, 0)] = (999, 8)
    assert genq.compact_plan(assign, crs) is None


def test_group_sizes_interleaved_rather_than_contiguous_are_refused():
    assign, crs = _row_major(10, 13, 4, 8)
    assign[(0, 0)], assign[(1, 0)] = (0, 8), (8, 4)
    assign[(2, 0)], assign[(3, 0)] = (12, 8), (20, 4)
    assert genq.compact_plan(assign, crs) is None


def test_empty_assignment_is_refused():
    assert genq.compact_plan({}, _crs([(0, 0)])) is None


def test_refusals_are_counted_by_reason():
    genq.REFUSED.clear()
    genq.compact_plan({}, _crs([(0, 0)]))
    assert genq.REFUSED == {"empty": 1}
