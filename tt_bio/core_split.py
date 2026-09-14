"""The column-wise work split over a core grid, in Python.

``ttnn.split_work_to_cores(CoreRangeSet, units, row_wise=False)`` on the pinned wheel raises
``TT_FATAL @ work_split.cpp:305: remaining == 0`` for a whole family of unit counts: it anchors core
group 2 at the BOTTOM of the next column instead of the top (``work_split.cpp:474``,
``{last_core_group_1.x + 1, range_containing_last_core_group_1.end_coord.y}``), so group 2 takes one
core per column, runs out of columns, and fails to place its cores. Upstream fixed the one word in
``f0e55f63fd6`` (#45501) for v0.78.0; we are pinned below that and are not bumping tt-metal, so the
split lives here instead.

It fires exactly when ``units > cores`` and ``units % cores`` is a non-zero multiple of the grid
height -- on an 11x10 grid that is 358 of the first 4000 unit counts. Two tt-bio call sites exist
only because of it, and both now come through this module: ``reblock_permute._split_plan``, which
used to search for the largest rectangular sub-grid the wheel would accept and lost the rest of the
grid on every affected band, and ``rfd3_bias._even_split``.

The distribution rule is the wheel's own, and is verified against it: ``num_cores = min(units,
cores)``, group 1 is the first ``units % num_cores`` cores in column-wise order and takes ``ceil(units
/ num_cores)`` each, group 2 is the rest and takes one less. When the division is even, group 1 is
every core and group 2 is empty. ``scripts/verify_core_split.py`` checks the CoreRangeSets
range-for-range against the wheel on every unit count the wheel can serve, and checks this module's
own invariants on the ones it cannot.
"""

from __future__ import annotations

import ttnn


def core_count(grid) -> int:
    return int(grid.x) * int(grid.y)


def units_per_core(units: int, cores: int) -> tuple[int, int, int, int]:
    """``(num_cores, n_group_1, units_group_1, units_group_2)`` for ``units`` over ``cores``.

    The whole distribution rule, shared by every view of it below. Group 2 carries one unit less
    than group 1 and is empty when the split is even.
    """
    num_cores = min(units, cores)
    if num_cores <= 0:
        return 0, 0, 0, 0
    per, rem = divmod(units, num_cores)
    if rem == 0:
        return num_cores, num_cores, per, 0
    return num_cores, rem, per + 1, per


def column_ranges(start: int, count: int, height: int) -> list:
    """``count`` cores in column-wise order from linear index ``start``, as merged CoreRanges.

    The same shape the wheel emits: the tail of the column the span starts in, then a rectangle of
    whole columns, then the head of the column it ends in, each part omitted when empty.
    """
    out = []
    x, y = divmod(start, height)
    if count > 0 and y != 0:
        take = min(count, height - y)
        out.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y + take - 1)))
        count -= take
        x += 1
    if count > 0:
        full, count = divmod(count, height)
        if full:
            out.append(
                ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x + full - 1, height - 1))
            )
            x += full
        if count:
            out.append(ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, count - 1)))
    return out


def split_work_to_cores(grid, units: int):
    """``ttnn.split_work_to_cores(<grid as one CoreRangeSet>, units)`` without the group-2 bug.

    Returns the wheel's own 6-tuple ``(num_cores, all_cores, core_group_1, core_group_2, units_1,
    units_2)``, so a caller can swap one for the other. ``grid`` is a grid size, as
    ``device.compute_with_storage_grid_size()`` returns it.
    """
    height = int(grid.y)
    num_cores, n1, w1, w2 = units_per_core(units, core_count(grid))
    crs = ttnn.CoreRangeSet
    return (
        num_cores,
        crs(column_ranges(0, num_cores, height)),
        crs(column_ranges(0, n1, height)),
        crs(column_ranges(n1, num_cores - n1, height)),
        w1,
        w2,
    )


def even_split(units: int, cores) -> list[int]:
    """``units`` over ``len(cores)`` cores as evenly as possible, remainder to the front.

    The per-core view of the same rule, for a caller that gives every core its own runtime args and
    needs no core-range grouping at all.
    """
    n = len(cores)
    _, n1, w1, w2 = units_per_core(units, n)
    return [w1 if i < n1 else w2 for i in range(n)]
