# Pass-2 predictions, written BEFORE the runs that test them

Committed before `perf/esm4pd/moves_c0.json` and `callcount_c0.json` exist. Pass 1's predictions
are in `PREDICTIONS.md`.

## What pass 2 already measured (context, not a prediction)

`perf/esm4pd/trimul_ops2_c0.json`, qb1 card 0, under benchlock, batched (8 back-to-back calls, one
`synchronize_device`), median of 5:

* whole TriangleMultiplication wall **13.960 ms/call** (start 13.968, end 13.952), against the
  14.755 ms carried in from qb2.
* MEASURED DRAM roof this session **429.7 GB/s** (`ttnn.add`, 2 reads + 1 write of a 128 MiB pair).
* Sum of the nine ops timed batched in isolation is 15.056 ms, **1.0785x** the whole-op wall.
* The op furthest off the DRAM roof, by ratio, is `_channel_move_back`: **1.434 ms for 256 MiB =
  187.2 GB/s**, against `ttnn.clone` moving the identical 256 MiB at **401.8 GB/s (0.668 ms)** on
  this card in the same session. That is 2.15x, and it is the one number in the table that got
  *worse* than qb2's (268.6 GB/s there).

## P-A. The back move: is 187.2 GB/s this card's permute rate, or this kernel's bug?

The floor in §4 prices both permutes at "their own demonstrated rate" rather than at the DRAM roof,
because a (0,3,1,2)-class index move is NOC-transaction bound and not byte bound. That pricing is
only legal if the demonstrated rate is measured **on this card**; §4 imported 185.5 and 268.6 GB/s
from qb2, and this card is a 13x10 grid, not 11x10.

`perf/esm4pd/moves.py` times, batched and synced once per batch, at the production shapes: the
plain forward `reblock_permute`, the E6 gated move, the back kernel, the two stock spellings of the
back move (`ttnn.permute(0,2,3,1)` and `transpose(1,2)+transpose(2,3)`), and `ttnn.clone` at the
same bytes as a same-traffic control. Every alternative is checked with `torch.equal` against the
shipped one.

**PREDICTED, before the run:**

1. No bit-exact alternative spelling of the back move beats the 1.434 ms kernel by more than 10 %.
   The kernel exists because both stock spellings lost; the two-transpose moves 512 MiB (two full
   round trips) so it cannot go below 1.25 ms at the measured roof even in principle, and the
   single `ttnn.permute` is a 3-way axis rotation that this code already measured at ~3x the
   decomposed cost. **KILL GATE: below 1.20x on the isolated move is NO-GO** (1.20x is
   0.239 ms/call = 0.257 s/fold at 1076 calls, under the 0.45 s gate this lineage uses).
2. This card's demonstrated plain-permute rate lands in **240-270 GB/s**. The E6 gated move already
   reaches 237.5 GB/s here with a slice, a sigmoid and a multiply riding inside it, so a plain move
   with no arithmetic cannot be slower.
3. Therefore the back move's honest floor is **~0.95-1.07 ms**, not the 0.625 ms the DRAM roof
   prices it at, and the trimul's honest per-call headroom lands at **2.2-2.6 ms**, not 4.277 ms.

**How this dies:** if (1) fails -- some spelling is >1.20x -- then the back move is a lever and the
floor drops. That is the outcome that would make this a CONTINUING pass rather than a FLOOR one, so
it is named in advance as the thing that changes the verdict.

## P-B. The call count

The floor arithmetic multiplies a per-call headroom by a call count, and §2 and §4.1 use two
different ones (1084 and 1076). `perf/esm4pd/callcount.py` counts the real
`TriangleMultiplication.__call__` invocations of one real 512 aa fold of the page fixture.

**PREDICTED: 1076, and the trimul body is 1076 x 13.960 ms = 15.02 s = 41.3 % of the 36.343 s qb1
wall**, which has to land close to the 42.9 % share the instrumented decomposition measured
independently. If it does not, the per-call wall and the fold decomposition disagree and neither
can be used.
