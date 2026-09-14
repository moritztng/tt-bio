"""Every group walk visits every group exactly once, whatever the split.

`reblock_permute._walk` decides which groups a core touches and in what order; the kernel just
follows it. The three walks exist to move which groups are in flight together, so the one thing
that must never change is the SET each core covers, and the union over cores. A walk that dropped
or doubled a group would still produce a plausible tensor -- the forward leg writes every output
page from exactly one group -- so this is checked here rather than left to a device run to notice.

Host only, no card. The device-side proof that the walks agree on the value is the `torch.equal`
column in `perf/ttx_splitwork/*.json`.
"""

import pytest

from tt_bio import core_split
from tt_bio.reblock_permute import _NO_WRAP, _walk

# The 11x10 and 13x10 grids the fleet runs, against unit counts that cover the whole rule: even
# splits, the group-2 remainder, the `units < cores` edge, one unit, and the two counts a 512 aa
# fold actually asks for (1024) plus the hole band `ttnn.split_work_to_cores` used to refuse (400,
# 900), where `units % cores` is a non-zero multiple of the grid height.
CORES = [110, 130, 100, 80, 64, 8, 1]
UNITS = [1, 2, 7, 63, 64, 100, 110, 111, 256, 400, 512, 900, 1024, 1600, 4095]
WALKS = ["block", "stride", "rotate"]


def _visited(units, cores, walk):
    """The groups each core walks, exactly as the kernel steps them."""
    n, n1, w1, w2 = core_split.units_per_core(units, cores)
    out, block = [], 0
    for i in range(n):
        per_core = w1 if i < n1 else w2
        first, stride, hi, lo = _walk(walk, i, block, per_core, n)
        g, seq = first, []
        for _ in range(per_core):
            seq.append(g)
            g += stride
            if g == hi:
                g = lo
        out.append(seq)
        block += per_core
    return out


@pytest.mark.parametrize("walk", WALKS)
@pytest.mark.parametrize("cores", CORES)
@pytest.mark.parametrize("units", UNITS)
def test_walk_is_a_partition(units, cores, walk):
    seqs = _visited(units, cores, walk)
    flat = [g for s in seqs for g in s]
    assert sorted(flat) == list(range(units)), (units, cores, walk)
    for s in seqs:
        assert len(set(s)) == len(s), (units, cores, walk, s)


@pytest.mark.parametrize("cores", CORES)
@pytest.mark.parametrize("units", UNITS)
def test_walks_agree_on_what_each_core_owns(units, cores):
    """The walks reorder a core's groups; they never move a group to another core.

    `rotate` must hold this because it is the same block; `stride` must NOT, and does not -- it is
    listed as the exception so that a future walk cannot quietly join it.
    """
    block = [set(s) for s in _visited(units, cores, "block")]
    assert [set(s) for s in _visited(units, cores, "rotate")] == block
    strided = [set(s) for s in _visited(units, cores, "stride")]
    assert [len(s) for s in strided] == [len(s) for s in block]


@pytest.mark.parametrize("cores", CORES)
@pytest.mark.parametrize("units", UNITS)
def test_only_rotate_wraps(units, cores):
    """`block` and `stride` must hand the kernel a wrap that cannot fire.

    The sentinel is what keeps one kernel loop serving all three walks: if a non-rotating walk ever
    emitted a reachable `group_wrap_hi`, the loop would fold back mid-block and silently redo a
    group instead of failing.
    """
    n, n1, w1, w2 = core_split.units_per_core(units, cores)
    for i in range(n):
        per_core = w1 if i < n1 else w2
        for walk in ("block", "stride"):
            assert _walk(walk, i, 0, per_core, n)[2] == _NO_WRAP
