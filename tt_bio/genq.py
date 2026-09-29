"""A `ttnn.generic_op` dispatch that costs what a stock ttnn op costs.

**What the tax actually is.** `state/perf10/bcx-GENQ.md` leg 1 priced every stage of a
`generic_op` call at `[1, 288, 288, 128]` on pc card 0 against a drained queue. The two
`buffer_address()` reads the brief blamed are 0.4 % of the call and the runtime-arg mutation is
1.9 %. Half the call is inside `ttnn.generic_op` itself, and a kernel that touches nothing
(`perf/bcx_p10_genq/nullk.py`) prices that half exactly:

    130 cores x 3 kernels, 2 per-core runtime-arg words   0.0230 ms
    130 cores x 3 kernels, NO per-core runtime args       0.0079 ms
    130 cores x 3 kernels, 0 per-core + 341 common words  0.0053 ms

The dispatch charges **per (core, kernel) that carries per-core runtime args**, and is flat in
how many words each one carries: 2 words and 32 words cost the same. So the fix is not to make
the arguments smaller. It is to stop having per-core arguments at all.

**Why the table cannot simply move into `common_runtime_args`.** Common args are broadcast, so
one list would serve every core -- but the list has to be reassigned whenever an address changes,
and that assignment is O(words): 2 words cost 0.00023 ms, 262 words 0.00371, and across three
kernels that is the whole of a stock op. The per-call cost has to stay at a handful of words.

**So the row is recomputed on the core.** For any kernel whose work split is
`split_work_to_cores` -- the first `n1` cores in placement order take `p1` units each and the
rest take `p2` -- a core's `(first_unit, num_units)` is a function of its own index, and its index
is a function of its own logical coordinates. Seven compile-time constants carry that, the
per-core args go away entirely, and the per-call cost is the two or three address words it always
was.

**It is fail-closed.** `compact_plan` is handed the host's OWN per-core assignment and returns a
plan only if the recomputation reproduces every placed core's row exactly, in both scan orders.
A kernel whose split does not fit the shape gets `None` and keeps its per-core args, which is
also what a caller that never opts in gets. A core that is placed but not covered is the defect
in `bcx-TABWD.md` -- a writer with a zero base address writing into the bottom of DRAM -- so a
placed core missing from the assignment refuses the plan rather than guessing a row for it.
"""

from __future__ import annotations

from .envflags import env_flag

# Scan orders. The value is a compile-time argument, so the kernel header switches on it.
ROW_MAJOR, COL_MAJOR = 0, 1

# tt-metal refuses a kernel whose unique + common runtime args exceed this on any core
# (`tt_metal/impl/kernels/kernel.cpp:453`). Recorded because it is what rules out shipping the
# per-core table as one broadcast list, and because a caller adding common args needs the budget.
MAX_RUNTIME_ARG_WORDS = 341

_COMPACT = env_flag("TT_BIO_GENQ_COMPACT", False)

# Why a plan was refused, by reason. A kernel that quietly stops taking the cheap path is a
# silent perf regression, so the counts are readable from a probe.
REFUSED: dict = {}


def compact() -> bool:
    """Is the cheap dispatch path armed? Default OFF, release-gated."""
    return _COMPACT


def set_compact(on: bool) -> bool:
    """Arm or disarm the cheap path. Returns the previous value.

    Every descriptor cache in the engine keys on this, so a caller that flips it mid-process gets
    a rebuild rather than a stale descriptor.
    """
    global _COMPACT
    prev, _COMPACT = _COMPACT, bool(on)
    return prev


def _refuse(reason):
    REFUSED[reason] = REFUSED.get(reason, 0) + 1
    return None


def compact_plan(assign, core_ranges):
    """Seven constants that let a kernel recompute `assign[(x, y)]` from its own coordinates.

    `assign` maps `(x, y) -> (first_unit, num_units)`, built by the caller's own placement loop.
    `core_ranges` is the `CoreRangeSet` the kernels are placed on. Returns
    `[order, x0, y0, span, n1, p1, p2]`, ready to append to a kernel's compile-time args, or
    `None` if the assignment is not of that shape -- in which case the caller keeps its per-core
    runtime args and nothing changes.
    """
    if not assign:
        return _refuse("empty")
    bb = core_ranges.bounding_box()
    x0, y0 = int(bb.start.x), int(bb.start.y)
    w = int(bb.end.x) - x0 + 1
    h = int(bb.end.y) - y0 + 1

    # Every core the kernels are placed on must have a row. A placed core without one reads its
    # arguments as zero and writes from base address 0 (`state/perf10/bcx-TABWD.md`).
    placed = {(int(x), int(y))
              for cr in core_ranges.ranges()
              for x in range(cr.start.x, cr.end.x + 1)
              for y in range(cr.start.y, cr.end.y + 1)}
    if placed != set(assign):
        return _refuse("placement_mismatch")

    # The split is `n1` cores of `p1` units then the rest of `p2`, in index order.
    sizes = {n for _, n in assign.values()}
    if len(sizes) > 2:
        return _refuse("more_than_two_group_sizes")

    for order, span, index in ((ROW_MAJOR, w, lambda x, y: (y - y0) * w + (x - x0)),
                               (COL_MAJOR, h, lambda x, y: (x - x0) * h + (y - y0))):
        rows = sorted((index(x, y), fn) for (x, y), fn in assign.items())
        if [i for i, _ in rows] != list(range(len(rows))):
            continue                      # this order does not index the placed set densely
        p1 = rows[0][1][1]
        n1 = next((i for i, (_, n) in rows if n != p1), len(rows))
        p2 = rows[n1][1][1] if n1 < len(rows) else 0
        if any(n != (p1 if i < n1 else p2) for i, (_, n) in rows):
            continue                      # the two group sizes are not contiguous in this order
        if any(f != (i * p1 if i < n1 else n1 * p1 + (i - n1) * p2) for i, (f, _) in rows):
            continue                      # units are not laid out contiguously in this order
        return [order, x0, y0, span, n1, p1, p2]
    return _refuse("no_order_reproduces_assignment")
