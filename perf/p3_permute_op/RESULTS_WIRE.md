# p3-permute-wire — the descriptor is cacheable, the shape was wrong, and the fold gains 188.8 ms

X6, Phase 3, OPTIMISE. protenix-v2, the trunk, 298 aa. Host qb2, **card 2** (board 005 chip 2),
board mate chip 3 idle for the whole pass. Production ttnn **0.68.0** wheel. Branch
`wk/protenix-trunk--p3-permute-wire`, branched from X4's `wk/protenix-trunk--p3-permute-op` at
`92e460b9` and pushed. **Nothing merges; the recommendation is at the bottom.**

Every figure here is a **qb2 ratio, never a campaign absolute** (charter §4.8) and owes a qb1
re-measurement at 0.67.4. Q16 is answered: that re-take is possible.

| deliverable | verdict |
|---|---|
| 1 — cache the `generic_op` descriptor, kill X4's 155 us blocker | **DONE. 4.54 us of Python per call against 151.29 us to rebuild**, bit-exact through the cache |
| 2 — wire it behind the corrected gate and A/B it | **DONE, after the first wiring served ZERO calls.** The fold runs `[1, 298, 298, 32]`, not the `[1, 320, 320, 32]` this org priced. Fixed. **188.8 ms/fold, whole-fold A/B** |
| 3 — parity at the fold's own shape | **Bit-exact.** `torch.equal` True at 298 and 320; 10 folds, coordinates identical at max abs 0.0 Å |
| 4 — merge recommendation | **Hold on the branch, propose for merge after a qb1 re-take.** Default-OFF today |

## Predictions (before measuring)

Committed as `perf/p3_permute_op/PREDICTIONS_WIRE.md` in `957810ea` and pushed **before the device
was opened**. Verdicts, two of ten wrong.

- **W1 — the descriptor is mutable and the program cache honours it.** `KernelDescriptor.common_runtime_args`
  is assignable from Python at 0.68.0 and `generic_op` picks up a mutated value on a program-cache
  hit. WRONG if the attribute is read-only or a second call at a different address returns the first
  call's data. **CONFIRMED.**
- **W2 — host cost after caching under 10 us**, against X4's 154.78 us. WRONG above 25 us.
  **CONFIRMED at 4.54 us.**
- **W3 — per-call net at N=320 L1, host included, >= 1.15x.** WRONG under 1.05x. **CONFIRMED at
  1.178x.**
- **W4 — the cache key is complete.** WRONG if two different N through one cached path disagree with
  torch. **CONFIRMED.**
- **W5 — 8384 eligible calls, half of T2's 16768, so the 273.8 ms/fold prize is at most ~137.**
  WRONG if the count is 16768 or the memory config is DRAM. **WRONG: the count is 8704**, and the
  reason the prize moved has nothing to do with the count.
- **W6 — the `(0,3,2,1)` half is reachable by decomposition.** WRONG if the pair loses to the single
  permute. **WRONG: 0.869x.**
- **W7 — the fold wall resolves it: 100-320 ms gain, paired noise under 100 ms.** WRONG if the noise
  exceeds the delta. **CONFIRMED: 172.8-210.1 ms per round against a measured ~70 ms noise band.**
- **W8 — `_transpose_memory_config` returns L1 before and after.** WRONG if it flips to DRAM.
  **CONFIRMED, and it cannot flip.**
- **W9 — parity: `torch.equal` True through the cache at two different N, coordinates identical at
  max abs 0.0 Å.** **CONFIRMED.**
- **W10 — `generic_op` / `ProgramDescriptor` / `KernelDescriptor(kernel_source=)` exist at 0.67.4.**
  WRONG if any symbol or the keyword is missing. **CONFIRMED.**

## Roofs, measured on this card

**Measured on qb2 card 2 in this pass, in the same process as the arm each one scores, and not
inherited** (charter §4.1). X4's floors were taken at the 48.82 MB pair-tensor size for a different
op; this op's tensor is the trimul chunk, so they are re-measured at its size.
`ttnn.synchronize_device` on both sides of every timed region, median of 15. `[1, 320, 320, 32]`
bf16 = **6.5536 MB one way**.

| floor, this op's own shape | us | GB/s one way |
|---|---:|---:|
| `ttnn.clone` L1 -> L1 | **31.50** | **208.0** |
| `ttnn.clone` DRAM -> L1 | 33.90 | 193.3 |
| `ttnn.clone` L1 -> DRAM | 42.10 | 155.7 |
| `ttnn.clone` DRAM -> DRAM | 50.61 | 129.5 |
| the wired op, DRAM/L1 -> L1 | 99.47 | 65.9 (**3.16x the L1 clone floor**) |
| `ttnn.permute(0,3,1,2)` baseline | 117.19 | 55.9 (3.72x the floor) |

**Arithmetic intensity 0.0 FLOP/byte.** A permute computes nothing, so it is on the **memory side**
of the charter's 338 FLOP/byte machine balance — and it is not **bandwidth**-bound either, at 32 % of
a copy of the same bytes. The **binding roof is NOC transaction issue rate on the dataflow RISCs**.
P4 settled that with the bf16 -> fp32 test; this pass produced an independent confirmation by
accident. Adding a single `if (row_valid)` inside the writer's 64-iteration gather loop cost
**10.7 us on a 97 us op** (1.178x -> 1.066x at N=320), and hoisting the same logic out of the loop
gave all of it back. An instruction in that loop is not free, which is what issue-rate-bound means.

**Core utilisation: 100 of the 110 cores** on this card's 11x10 grid, one tile-group each, read out
of the `CoreRangeSet` the work split returned rather than assumed. Same count at 298 and at 320
(`Nt = ceil(N/32) = 10` for both), which is why the gate's L1 window has an upper bound: at N=384
`Nt=12` gives 144 groups over 110 cores and eleven cores carry two.

## The descriptor cache

X4's blocker: `generic_op` takes the whole descriptor per call, the runtime args carry the buffer
addresses, tt-bio's allocator changes them every call, and rebuilding the descriptor in Python costs
**154.78 us** at N=320 against **91.52 us** of device time — **0.47x per call**, which is why X4
correctly did not wire it.

**The fix is the six lines the brief named.** `src_addr` and `dst_addr` move out of the per-core
runtime args into **`common_runtime_args`** (`get_arg_val` -> `get_common_arg_val` in the reader and
the writer; the other per-core args shift down one index).

**What the cache is keyed on, and why the key is complete.**

```
(device.id(), N, str(dtype), str(layout), str(in memory_config), str(out memory_config),
 grid.x, grid.y, tuple(reader compile-time args), tuple(writer compile-time args))
```

Both `TensorAccessorArgs` compile-time vectors are in the key **verbatim**, so whatever the accessor
bakes into the kernel — buffer type, page size, shape, shard spec — is covered whether or not the
key's author knows what it means. What remains is a pure function of `(N, grid)`: the CB sizes and
depths, the core ranges, the work split and the per-core `start` / `per_core` / `Nt` / `N` args. The
**only** per-call values are the two buffer addresses, and they are written on every call, on the
hit path and the miss path alike. That is the completeness argument, and an incomplete key is a
failure this codebase has already had: a trace-cache key that missed conditioning identity returned
a stale program that computed the wrong thing silently
(`perfwar-ttnn-trace-cache-conditioning-reuse-bug`).

**Proved, not argued** (`perf/p3_permute_op/wire_probe298.py`):

1. **Two different N alternating through one cache.** N=298 and N=320 interleaved 12 times in one
   process, `torch.equal` against the torch golden every call: **12 of 12 True**, two cache entries.
   A key that collapsed the two would have run the wrong `Nt` and failed on the first swap.
2. **Two different buffers through one cached descriptor.** Two source tensors at one N, different
   contents, different `buffer_address()`, alternating: **4 of 4 True**. A descriptor holding the
   first call's address would have returned the first tensor's data for the second.

| arm at the fold's own shape, N=298 L1 | us |
|---|---:|
| Python per call, descriptor rebuilt (X4's blocker reproduced on this build) | **151.29** |
| Python per call, cached (allocate + key + hit + two address writes) | **4.54** |
| whole op, host + device, synced | 101.29 |
| `ttnn.permute(x, (0,3,1,2))`, same session, same process | 109.58 |

**33x off the host cost**, which is what turns 0.47x per call into a positive ratio. The addresses
reach the cached descriptor by direct mutation (`pd.kernels[i].common_runtime_args = [addr]`): the
binding hands back a live reference, so the `ProgramDescriptor` is never rebuilt. A fallback that
rebuilds only the `ProgramDescriptor` from cached kernel objects is in the module and is unused on
this wheel.

## What changed, and the A/B that measured it

### The first wiring served zero calls, and that is the leg's most important finding

Wired behind the corrected gate, the first whole-fold A/B served **0 eligible calls in a 298 aa
fold**. Not a small gain — none. The tensor `_transform_chunk` moves at 298 aa is

```
[1, 298, 298, 32]     L1, interleaved, bf16, TILE
```

and the ported kernel required `N % 32 == 0`. **Every per-call figure this org carries for this op —
X4's 273.8 ms/fold, P4's 610.1, the basis of the ledger's 1533.4 ms/fold row — was measured at
`[1, 320, 320, 32]`, a shape the fold never constructs.** 298 aa is padded to 320 only in tensors
whose sequence axis is one of the last two dims; here the permuted axis is dim 1 and stays at its
logical 298, so the tile grid is `ceil(298/32) = 10` with a **ragged last row-group**: 10 real rows
and 22 rows of tile padding.

This is `tt-bio-l1-residency-guard-dead-in-real-folds` in its original form — a probe-shape win the
production shape never reaches — caught by an eligible-call counter before anything merged rather
than after.

**The fix is not six lines.** The reader keeps every group at a fixed 32 pushes (the CB accounting,
the compute kernel and the writer's 32-tile L1 window all depend on it) and points the padding rows'
page reads at a valid page whose value is never used. The writer zeroes those output rows — they
must be **zero**, not a copy: this tensor is an operand of the triangle matmul and the padding sits
on the contracted axis, so a non-zero there changes the product. They are the same rows for all 32
channels of a group and the DRAM write only reads the staging slot, so the fill is hoisted to **once
per group per staging slot**: 704 scalar stores per ragged group instead of 11,264.

### The A/B protocol

One process, model loaded once, `hoist=True` so the timed region is `model.fold` only. The arms
differ by one module flag (`reblock_permute.set_enabled`), so there is no worktree edit between them
and **the baseline is restored in-session after every round**. A cold fold under each arm first, so
no timed arm pays JIT compilation. Arms alternate BASE, WIRE, BASE, WIRE.

**The instrument's own resolution, measured.** The first A/B ran with the gate silently refusing
every call, which makes it an unintended **A/A control**: identical work in both arms, ten folds.

| A/A control, 5 rounds, identical code path | |
|---|---:|
| "BASE" median | 30.1196 s |
| "WIRE" median | 30.0778 s |
| apparent delta on identical work | **41.8 ms** |
| BASE spread over 5 folds | 69.9 ms |
| WIRE spread over 5 folds | 55.5 ms |

**A whole-fold A/B on this harness cannot see anything under about 70 ms.** Any fold-wall delta this
org quotes below that is noise.

### The whole-fold A/B, after the ragged-N fix

Four paired rounds, same session, arms alternating, baseline restored each round.

| round | BASE s | WIRE s | delta ms | eligible calls served |
|---:|---:|---:|---:|---:|
| 0 | 30.2002 | 30.0274 | **172.8** | 8704 |
| 1 | 30.1927 | 29.9825 | **210.1** | 8704 |
| 2 | 30.1649 | 29.9790 | **185.9** | 8704 |
| 3 | 30.1327 | 29.9462 | **186.5** | 8704 |
| **median arm** | **30.1927** | **29.9825** | **210.2** | — |

**Paired mean 188.8 ms/fold; every round positive; the smallest round (172.8 ms) is 2.5x the A/A
control's apparent delta and above both arms' spread.** Ratio on the medians **1.00701**. BASE spread
67.5 ms, WIRE spread 81.2 ms, in line with the A/A control, so nothing about the wired arm made the
instrument noisier.

### The block wall, which resolves where the delta comes from

The production `TriangleMultiplication` captured from the fold at its own pair shape
`[1, 298, 298, 256]`, replayed synced, arms alternating three times each:

| trimul block wall | ms/call |
|---|---:|
| BASE | 8.2750 |
| WIRE | 8.1133 |
| delta | **0.1617** |
| ratio | **1.0199x** |

`0.1617 ms x 2 trimuls x 524 c_z=256 `PairformerLayer` executions` (charter §4.9, the x524
conversion, and I counted the calls rather than assuming the constant) = **169.4 ms/fold**, plus the
template stack's 320 further eligible calls. That accounts for the fold wall's 188.8 ms from an
independent instrument.

### Call count, counted not assumed

**8704 eligible `(0,3,1,2)` calls per warm 298 aa fold**, identical in all four rounds. 8384 of them
are the c_z=256 pair track (8 chunks x 2 trimuls x 524 layer executions) and 320 are the template
stack at c=64 (2 chunks x 2 trimuls x 80 executions). T2's 16768 counts both channel moves per chunk
iteration; only the `(0,3,1,2)` one is this op.

### The probe figure and the production figure, side by side, and the gap named

| | us/call saved | ms/fold |
|---|---:|---:|
| **probe**, N=320 L1, isolated tensors (the shape the org priced) | 17.56 | — |
| **probe**, N=298 L1, the fold's own shape | 8.29 | **72.2, projected** |
| **production**, trimul block wall | 20.2 (0.1617 ms over 8 calls) | 169.4, projected from the block |
| **production**, whole-fold wall A/B | — | **188.8, measured** |

**Two gaps, in opposite directions, and both matter.** The first is the shape: moving from the priced
N=320 to the real N=298 halves the per-call saving before a fold is run, because the op's cost is set
by the group count (100 either way) while the stock baseline's cost falls with the smaller tensor.
The second is that **the probe understates production by 2.6x** — 8.29 us/call isolated against
20.2 us/call inside the block. In the fold the stock `ttnn.permute` runs under real L1 occupancy and
back-to-back dispatch; the probe hands it an empty card. That direction of gap is the rarer one and
it is why the fold wall, not the probe, is the number reported.

### Clause 6 — does this move somebody else's landed lever?

**No, and it cannot.** `_transpose_memory_config` (W6, `cbf070de`, a landed 1010.9 ms/fold lever)
decides L1 vs DRAM by a **2.5x-headroom fit test** whose only inputs are the tensor's shape, its
dtype, `get_max_worker_l1_unreserved_size()` and `COMPUTE_GRID_MAIN`. Live L1 occupancy is not one of
them, so no residency change can flip it. Measured anyway, by calling the production helper at the
fold's own pair shape `[298, 320, 256]` in the same process before and after the wired arm folded:
**`BufferType.L1` both times.** It is also reached only from `TriangleAttention`, and this change is
in `TriangleMultiplication`'s chunk loop.

### The corrected gate, re-measured at the shape that matters

X4's gate `(DRAM and N >= 256) or (L1 and 288 <= N <= 352)` ships as written, minus the
tile-alignment requirement the fold fell foul of. Wall-clock, host cost included, this card, this
wheel:

| N | buffer | wired us | `ttnn.permute` us | ratio |
|---:|---|---:|---:|---:|
| **298** | **L1 — what the fold runs** | **101.29** | **109.58** | **1.082x** |
| 298 | DRAM | 107.66 | 162.63 | 1.511x |
| 320 | L1 | 98.87 | 116.43 | 1.178x |
| 320 | DRAM | 107.70 | 164.75 | 1.530x |
| 256 | L1 | 94.92 | 80.36 | 0.847x |
| 256 | DRAM | 98.86 | 121.35 | 1.228x |

### The other half of the calls, killed

Each chunk-loop iteration issues one `(0,3,1,2)` and one `(0,3,2,1)`; only the first is this op.
Decomposing the second as custom `(0,3,1,2)` + `ttnn.transpose(-2,-1)` — bit-exact, and already
documented as such in `_transform_chunk` — measures **0.869x** at the fold's shape on L1 (96.79 us
single against 111.36 us decomposed). On this wheel the stock `permute(0,3,2,1)` is *cheaper* than
`permute(0,3,1,2)`. W6 KILLED; that half is not available.

### What was not re-fought

A better kernel structure is **settled and closed** and this pass did not go looking for one, because
64 NOC transactions per source tile is a floor over all structures, proved by P4's bf16 -> fp32 test.
The two kernel edits here are correctness (ragged N) and instruction count (hoisting a branch), not
structure.

### Overlap: `max(read, write)`, measured this pass

The op's total is nearer `max(compute, comm)` than `compute + comm`, and the test is the source
buffer type. At N=298 the same move costs **101.29 us from L1** and **107.66 us from DRAM** — 6.3 %
apart on a leg whose DRAM traffic is the entire 5.68 MB tensor. A `ttnn.clone` of those bytes costs
33.90 us from DRAM against 31.50 us from L1, so if the read leg and the write leg were additive the
DRAM-source arm would land ~30 us above the L1-source arm rather than 6.4. The read hides inside the
write. The compute kernel is one `transpose_wh` per tile and is never the pole: it runs concurrently
with both dataflow RISCs and the arm-to-arm difference tracks the destination, not the arithmetic.

## Delivered ms/fold

**188.8 ms/fold, bit-exact, measured on a whole-fold A/B with the arms alternating in one session.**
Paired mean of four rounds; per-round range 172.8-210.1 ms; ratio 1.00701 on the medians;
corroborated at 169.4 ms/fold by an independent trimul block wall (1.0199x per trimul call). That is
0.63 % of a 30.19 s fold, and it is the largest delivered number this org has produced (X2 31.5 ms,
X3 22.0 ms).

**Deliverable 1's own number is host-side: 151.29 us -> 4.54 us of Python per call, 33x.** That is
the piece that was actually blocked, and without it the wired arm is a 0.47x-per-call regression.

**The org's carried figures for this item were wrong in both directions and the corrections are
large.** X4's **273.8 ms/fold** and P4's **610.1** are priced at N=320 against an N=320 baseline, a
shape the fold never runs — a **projection**, not a delivered number, and a 1.45x and 3.2x
overstatement of the 188.8 that a real fold gives. The ledger's **1533.4 ms/fold** row for the
in-move counts both channel moves; only half of those calls are this op.

## Parity

**Bit-exact everywhere, measured with `torch.equal`, not argued.**

- **At the fold's own shape.** `torch.equal(reblock_permute(x), x.permute(0,3,1,2))` **True** for
  `[1, 298, 298, 32]` on L1 and DRAM, and `[1, 320, 320, 32]` on both. 4 of 4.
- **Through the cache.** N=298 and N=320 alternating 12 times through one cache: **12 of 12 True**.
  Two source buffers through one cached descriptor: **4 of 4 True**.
- **Against the stock op including tile padding.** `torch.equal(ours, ttnn.permute(...))` at N=298
  **True** — the ragged group's padding rows match what production puts there, which is the check
  that matters, because those rows sit on the contracted axis of the triangle matmul.
- **At the fold level.** 10 folds in one session across both arms, same seed: coordinate arrays
  identical, **max abs delta 0.0 Å** over all pairs, `torch.equal` True, pLDDT identical to every
  digit (0.8594889044761658 in all ten).

A permute is a pure index move, so a correct kernel carries no accuracy trade. The expected result,
and it is measured rather than assumed.

## Merge recommendation

**Hold on `wk/protenix-trunk--p3-permute-wire`. I recommend Moritz take it, after a qb1 re-take.**
Nothing merges without his explicit OK and nothing has been merged.

- **Parity class: bit-exact**, `torch.equal` at the op and 0.0 Å at the fold. The strongest class
  available, and the reason this is worth proposing at 0.63 %.
- **Release-gated and default-OFF today** (`TT_BIO_REBLOCK_PERMUTE`, `_ENABLED = False`), so the
  branch is a no-op for production as it stands. It touches a kernel that produces a matmul operand's
  padding, which is exactly the class of change the charter holds on a branch.
- **A qb1 re-take at 0.67.4 is required before merge** (charter §4.8) and is **possible**:
  `ttnn.generic_op`, `ttnn.ProgramDescriptor`, `ttnn.KernelDescriptor(kernel_source=...)`,
  `common_runtime_args`, `CBDescriptor`, `RuntimeArgs`, `split_work_to_cores` and
  `TensorAccessorArgs` **all exist on the 0.67.4 wheel qb1 runs** — checked by import on qb1 with no
  device opened. **Q16 closed: the route is not 0.68-only and the win does not have to wait for a
  wheel bump.**
- **Only the L1 leg of the gate earns anything here.** The DRAM leg is a real 1.5x but protenix at
  298 aa runs L1. Worth recording for whichever model runs these moves on DRAM; **recorded, not
  chased** (charter §1).
- The `generic_op` no-build route is the larger prize and it is now proven twice: a hand-written
  kernel reaches the shipped wheel with no tt-metal source build, on **both** 0.68.0 and 0.67.4.

## Corrections to the inherited record

1. **The fold does not run `[1, 320, 320, 32]`. It runs `[1, 298, 298, 32]`.** Measured by wiring the
   op and counting: **0 eligible calls in a whole 298 aa fold** under a gate requiring `N % 32 == 0`.
   Every per-call figure for this op in this org is taken at a shape production never constructs.
2. **The delivered prize is 188.8 ms/fold, not 273.8 and not 610.1.** Same kernel, same card, same
   wheel. X4's and P4's figures are projections from an N=320 probe against an N=320 baseline; the
   fold wall is 1.45x below the first and 3.2x below the second.
3. **A probe can understate as well as overstate, and here it does both.** The N=298 probe says
   8.29 us/call saved; the production block says 20.2. The isolated baseline gets an empty card that
   the fold never gives it. The org's standing worry is probe-optimism; this is the mirror case and
   the rule is the same — take the production number.
4. **X4's 155 us blocker is real, reproduced at 151.29 us, and it is fixed.** Two addresses into
   `common_runtime_args` makes the descriptor cacheable at 4.54 us. The route was sound; only its
   pricing was wrong.
5. **`ttnn.generic_op` with `KernelDescriptor(kernel_source=)` is not 0.68-only — it is on 0.67.4
   too.** Q16 closed. Any ledger item marked "needs a tt-metal source build" can be reopened on
   either wheel.
6. **The `(0,3,2,1)` decomposition loses, 0.869x**, and on this wheel the stock `permute(0,3,2,1)`
   is *cheaper* than `permute(0,3,1,2)` — the opposite of what the decomposition argument assumes.
   X4's "staging fixes a bad source, not a bad destination" rule survives unchanged.
7. **A whole-fold A/B on this harness resolves ~70 ms and no better.** Measured as an A/A: ten folds,
   identical code in both arms, apparent delta 41.8 ms, per-arm spread 69.9 and 55.5 ms. Any leg
   quoting a fold-wall delta under that is quoting noise.
8. **One `if` inside the writer's gather loop costs 10.7 us on a 97 us op** — 1.178x -> 1.066x at
   N=320, recovered in full by hoisting it out. Independent corroboration of P4's
   transaction-issue-rate mechanism: instruction count on the dataflow RISC is the binding resource,
   not bytes.

## Scope

protenix-v2, the trunk, 298 aa only. The ragged-tile-group kernel pattern, the `common_runtime_args`
descriptor-caching pattern, the A/A resolution measurement and the probe-understates-production
direction all generalise well beyond this org; all four are **recorded here and not chased**
(charter §1).
