# PREDICTIONS_QB1 — committed before the device is opened

p3-permute-qb1, Phase 3. Host **qb1 card 0**, ttnn **0.67.4**. This leg re-takes X6's qb2 / 0.68.0
figures as a campaign absolute (charter §4.8). X6's numbers are the thing being tested, so they are
named as X6's everywhere below and never as this card's.

Ten predictions with the losing outcome stated for each. Verdicts go in
`~/.coworker/state/protenix-trunk--p3-permute-qb1.md`; this file is not edited after the device
opens.

- **Q1 — the kernel JIT-compiles at 0.67.4.** `KernelDescriptor(kernel_source=...)` builds the three
  `.cpp` files against the 0.67.4 wheel's own tt-metal headers. The kernels need
  `api/dataflow/dataflow_api.h`, `ttnn/operations/data_movement/common/kernels/common.hpp`,
  `get_common_arg_val`, `TensorAccessor`/`TensorAccessorArgs`, and
  `tt::data_movement::common::tt_memmove`. X6 proved the *Python* symbols import at 0.67.4 and
  explicitly did not count that; this is the compile. **WRONG if any include path, template or
  free function is missing and the JIT fails** — in which case 188.8 is a 0.68.0-only number and
  that is the answer.

- **Q2 — bit-exact, 20 of 20 checks.** `torch.equal` against `x.permute(0,3,1,2)` True at
  `[1, 298, 298, 32]` and `[1, 320, 320, 32]`, on L1 and on DRAM (4 of 4); `torch.equal` against
  stock `ttnn.permute` **including the output tile-padding rows** True at 298 on both buffer types
  (2 of 2); 298/320 alternating through one cache 12 of 12; two source buffers through one cached
  descriptor 4 of 4. **WRONG on a single False.** A permute is a pure index move, so anything but
  20/20 is a kernel bug on this wheel, not a precision trade.

- **Q3 — the cached host cost survives the wheel change: 3.0-8.0 us per call, and the rebuild path
  120-200 us.** X6 measured 4.54 us cached against 151.29 us rebuilt on 0.68.0. This is pure Python
  over nanobind, so it is wheel-dependent and re-measured rather than inherited. **WRONG if cached
  is above 15 us** (the op has ~100 us of device time, so 15 us of host is where the lever starts
  losing its margin), **or if the rebuild path is under 60 us**, which would mean X6's blocker was
  wheel-specific and the cache was never needed here.

- **Q4 — qb1's 13x10 grid engages 100 of 130 cores at N=298, and the op's wall does not move much
  for it.** `num_groups = Nt*Nt = 100` at both 298 and 320, one group per core, so 30 of qb1's 130
  cores get nothing (against 10 of 110 idle on qb2). Latency is set by the slowest single core, which
  owns one group either way. Predict the wired op at N=298 L1 lands **80-130 us** against X6's
  101.29 us on qb2. **WRONG outside 70-160 us** — a figure outside that says the per-core cost, not
  the group count, moved with the wheel or the card.

- **Q5 — the fold A/A noise floor on this card is 60-250 ms of per-arm spread, and the apparent A/A
  delta on identical code is under 150 ms.** X6 measured 41.8 ms apparent delta with per-arm spreads
  of 69.9 and 55.5 ms on a quiet qb2 board. qb1's host is **not** quiet this pass: card 3 is running
  `perfwar-qb1-rebaseline-and-land`'s parity gate and cards 1 and 2 carry sibling trunk legs that
  open and close devices, so the floor should be at or above X6's. **WRONG if the apparent A/A delta
  exceeds 250 ms**, in which case the fold wall cannot resolve a 188.8-class delta on this host and
  the trimul block wall is the only instrument this leg has.

- **Q6 — the fold A/B delta is positive and lands 100-280 ms/fold, with every round positive.** Same
  kernel, same gate, one wheel back. **WRONG if the paired mean is negative, or under 60 ms, or if
  any single round is negative** — any of those makes X6's 188.8 a 0.68.0-only number and the merge
  recommendation changes to "0.68.0-only", which is a result and not a failure. Fold wall itself
  predicted **28-34 s** (X6: 30.19 s on qb2).

- **Q7 — the two instruments agree within 2x.** Trimul block wall ratio **1.010-1.035x** per call
  (X6: 1.0199x, 0.1617 ms/call), and the ms/fold it projects through charter §4.9's **x524**
  conversion lands within 2x of the fold-wall figure. **WRONG if they disagree by more than 2x, or
  disagree in sign** — and if they do, that disagreement is this leg's finding rather than a number
  to average away.

- **Q8 — `_transpose_memory_config` returns `BufferType.L1` at `[298, 320, 256]`, before and after,
  and it has more headroom on qb1 than on qb2, not less.** The fit test is
  `2.5 * padded_volume * 2 bytes <= per_core * gx * gy`: 122.06 MB of demand against
  `per_core x 130` on qb1's rebound **13x10** grid. Read `COMPUTE_GRID_MAIN` **after** device open,
  because it is 11x10 at module scope and a probe that reads it at import builds for a grid the fold
  does not run. **WRONG if either call returns DRAM**, which would mean this change silently switched
  off W6's landed 1010.9 ms/fold lever.

- **Q9 — the `(0,3,2,1)` inversion holds at 0.67.4 too.** X6 found stock `ttnn.permute(0,3,2,1)`
  *cheaper* than stock `ttnn.permute(0,3,1,2)` on 0.68.0, which is the opposite of what the
  decomposition argument assumes, and measured the decomposition at 0.869x. Predict `(0,3,2,1)` is
  again the cheaper of the two stock calls at N=298 L1, by **0-25 %**. **WRONG if `(0,3,2,1)` is the
  more expensive of the two on this wheel** — that would reopen the decomposition as a lever on
  0.67.4 specifically. One-line check, not a project.

- **Q10 — there is real instruction-count headroom on the ragged path, worth 3-15 us on a ~100 us
  op, and it is in the READER, not in the ragged groups.** X6 measured one `if` in the writer's
  gather loop at 10.7 us on a 97 us op. The ragged-N fix put a **per-iteration conditional in the
  reader's 32-push loop** — `page = (row < D1 ? row : 0) * Nt + jt` — which every group pays, not
  just the 10 ragged ones, and it carries a multiply the loop does not need. Splitting the loop at
  `rows_valid` and strength-reducing `page` to an `+= Nt` removes both. Predict the same for the
  writer's gather loop, whose `src_elem` is invariant in `il` and whose `dst_elem` is a div/mod plus
  a multiply per transaction. Combined, predict **1.03-1.18x** on the op at N=298 L1.
  **WRONG under 2 us combined**, which would mean the compiler already strength-reduced all of it
  and X6's 10.7 us was specific to a branch the optimiser could not remove.
  Secondary, same prediction: **the ragged groups are not on the critical path.** At N=298 only the
  10 groups with `it=9` are ragged and each runs 10 gather rows instead of 32, so a ragged core does
  **less** work than a full one and the op's latency is set by a full group. **WRONG if masking the
  ragged handling out entirely (at the cost of correctness, as a timing-only probe) recovers more
  than 3 us.**

**Fence, restated so it stays beside the claim:** Q10 is an *instructions per transaction* lever
only. **64 NOC transactions per source tile is a proven floor over all kernel structures** (P4's
bf16 -> fp32 test left the time unchanged while the byte-bound control rose 1.36x), so no prediction
here is about a better kernel structure and none will be chased.

**Retracted figures that will not be reintroduced as targets:** X4's **273.8 ms/fold** and P4's
**610.1 ms/fold** are projections taken at `[1, 320, 320, 32]` against an N=320 baseline, a shape the
fold never constructs; they overstate X6's delivered figure by 1.45x and 3.2x.
