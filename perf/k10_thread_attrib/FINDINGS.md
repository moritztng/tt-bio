# The CB stall counters are TRISC0 and TRISC2, and the census divided them by TRISC1

`perf/k10_thread_attrib/thread_attrib.py`, host only, opens no device. Run it on any
`--enable-sum-profiling` ops CSV.

## What was wrong

`b2z-kernel-cycle-census` is the measurement this whole optimization campaign is sized from. It
reports the Boltz-2 512 aa pairformer block as *"the math thread resident 88.4 % and stalled on a
circular buffer for two thirds of that"*, with input stalls beating output, and it records an error
bar: *"the per-core normalisation is violated on 50 of 232 ops, so that split is directional, not
exact."*

Both the headline and the error bar come from `perf/b2z2_profiler/cb_split.py`, which computes

    compute_ms = TRISC1_duration - WAIT_FRONT/cores - RESERVE_BACK/cores

and divides every term by `DEVICE TRISC1 KERNEL DURATION`. Those are three different threads. In
tt-metal v0.68.0 the two accumulators are declared on the unpack and pack threads, on both
architectures:

    tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h:18   CB-COMPUTE-WAIT-FRONT
    tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h:20     CB-COMPUTE-RESERVE-BACK
    tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h:67 CB-COMPUTE-WAIT-FRONT
    tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h:20   CB-COMPUTE-RESERVE-BACK

So `WAIT FRONT` is TRISC0 (unpack blocked on input tiles) and `RESERVE BACK` is TRISC2 (pack blocked
on room to write output). Neither is a TRISC1 counter and neither says anything about the math thread.

## The falsifier, which needs no source access

A sum accumulator measures time inside its own thread's kernel, so it can never exceed that thread's
duration. That makes the attribution checkable from the data alone. Run over the two committed
captures — `b2z-kernel-cycle-census`'s Blackhole block (qb2 card 0) and
`b2z2-whglx-profiler-build`'s Wormhole block (whglx card 1):

| rows where... | Blackhole | Wormhole |
|---|---|---|
| `wait + reserve > TRISC1` — what the census counted as a normalisation defect | **821 / 4441** | **1010 / 4441** |
| `wait_front > TRISC0` | **0 / 4441** | **0 / 4441** |
| `reserve_back > TRISC2` | **0 / 4441** | **0 / 4441** |

**Zero out of 8882 rows across two architectures.** Each counter is bounded by exactly one thread's
duration, and it is not TRISC1. The per-core normalisation was never the problem; the divisor was.

## What this changes, in both directions

**The magnitudes survive and get better.** TRISC0 runs 99.19 % of TRISC1's duration on both parts, so
picking the right denominator barely moves the number:

| | over TRISC1 (as published) | over its own thread |
|---|---|---|
| BH wait_front | 64.1 % | **64.6 %** |
| BH reserve_back | 13.3 % | **13.3 %** |
| WH wait_front | 65.2 % | **65.7 %** |
| WH reserve_back | 13.3 % | **13.3 %** |

(These are whole-capture sums over all 4441 compute rows. The census's published 57.0 % / 9.7 % /
5.9:1 are a median over a fenced single-block region, a different and narrower population, so these
figures are not a restatement of its numbers and must not be quoted as a correction of them. The
structural result — 0 anomalies against the right thread — is per-row and does not depend on the
population at all.)

So the stall split is real, it is large, and after this correction it carries **no error bar from
this source**. That is a strengthening.

**The interpretation does not survive.** What is measured is that the **unpack** thread spends ~65 %
of the compute kernel's duration blocked waiting for input tiles. That is not the same claim as *"the
math thread is starved"*, and the difference matters because the unpack and math threads are
decoupled through DST: TRISC0 can sit in `cb_wait_front` while TRISC1 works through tiles already
unpacked. **TRISC1's own stall was never measured**, by this instrument or any other, so the
derived figure *"useful math is ~29 % of wall"* and the 3.4x envelope built on it are not
established. They are not refuted either. They are unmeasured.

Measuring TRISC1 directly is what settles it, and that needs counters that do not exist in stock
tt-metal. `k10-instrument` is building the data-movement side of exactly this.

## Rule worth keeping

A profiler column named for a stage is not a column named for a thread. Before dividing counter A by
duration B, check that A is bounded by B — on every row, not on average. Here that check is one line
and it separates a real instrument defect from a real finding.

## Per-op, with the correct divisor — the block average hides most of the structure

`thread_attrib.py --by-op`. Whole capture, both committed captures. **The two ratios are intensive
per-op properties and barely depend on which region of the capture you take; the "% tot" column is
extensive and does, so it is orientation only — for a block-fenced ranking use the census's table.**

| op code | BH in/TRISC0 | BH in:out | WH in/TRISC0 | WH in:out |
|---|---|---|---|---|
| `Matmul` | 64.5 % | 6.73 | 61.3 % | 11.40 |
| `GenericOp` (fused trimul+TriAtt) | 56.6 % | 15.41 | 58.0 % | 20.73 |
| `BinaryNg` | **76.4 %** | **2.42** | **85.2 %** | **2.43** |
| `LayerNorm` | 60.1 % | 18.65 | 70.9 % | 14.89 |
| `Transpose` | 68.2 % | **1.72** | 82.9 % | **2.40** |
| `Permute` | 85.0 % | 10.02 | 78.1 % | **1.42** |
| `Untilize` | 29.0 % | **0.55** | 25.6 % | **0.40** |
| `Tilize` | 60.2 % | 5.02 | 68.3 % | 1.10 |
| `Softmax` | 20.6 % | 13.71 | 27.1 % | 9.59 |

The block-wide input:output figure is about 5:1. **Per op it ranges from 18.65:1 to 0.40:1**, so the
block average is the one number that describes none of these ops. Four things fall out that the
aggregate cannot show, and the first three reproduce on both parts:

- **`BinaryNg` and `Transpose` are blocked at both ends**, in:out 2.4 on both architectures, with
  `BinaryNg` carrying the **highest input stall of any significant op** (76.4 % BH, 85.2 % WH) *and*
  31.6 / 35.1 % output stall. An op blocked proportionally at both its input and its output is not
  starved by a slow producer; it is moving data and waiting on the memory system at both ends. That
  is the bandwidth-bound hypothesis `k10-p1-binaryng-why` was written to test, and this is evidence
  for it from data that already existed.
- **`GenericOp` and `LayerNorm` are overwhelmingly input-dominated**, 15-21:1. These are the ops where
  "starved" is the right word, and `GenericOp` is the campaign's largest single offender.
- **`Untilize` is the only output-dominated op**, in:out 0.55 BH / 0.40 WH. Its writer is the
  constraint, and nothing else in the model looks like it.
- **`Permute` is where the two architectures disagree most**: in:out **10.02 on Blackhole against
  1.42 on Wormhole**, a 7x difference in character on the same op, while every other significant row
  keeps its shape across the parts. Permute is pure layout, and the parts differ in DRAM banking
  (BH 8 x 3.984 GiB, WH 12 x ~1 GiB). This is independent support for the hypothesis
  `k10-transfer-function` exists to test — that arithmetic levers carry from Wormhole to Blackhole
  and layout levers do not — arrived at from stall structure rather than from lever ratios.

None of this measures TRISC1, and none of it prices a lever. It says where to point Phase 1.

## The producer-side envelope, bounded from the discriminator's own numbers: 1.15-1.29x, not 3.4x

`dataflow_bound.py`, host only, the measured numbers quoted in the source so the arithmetic is
checkable without rerunning anything. Inputs are `k10-instrument`'s closing five-zone capture.

The compute cluster's input stall and the reader's own time bound each other: producer-side work
removes at most what the producer is actually spending, and relieves at most the stall that exists.
Two bounds, because "producer-side" has a tight and a loose reading:

* **TIGHT** = `min(compute stall, reader DRAM)` — bandwidth and latency work only.
* **LOOSE** = `min(compute stall, reader DRAM + reader ISSUING)` — also credits driving address
  generation and NoC issue to **zero**, which no real change achieves. Coalescing reduces issue
  work; it does not abolish it. This end is physically unreachable and is here to be conservative.

| op class | s/fold | compute stall | reader DRAM | reader issuing | tight s | loose s |
|---|---|---|---|---|---|---|
| GenericOp (fused trimul+TriAtt) | 3.889 | 57.9 % | 22.7 % | 25.6 % | 0.883 | 1.878 |
| Matmul | 2.361 | 29.4 % | 11.2 % | 4.3 % | 0.264 | 0.366 |
| BinaryNg | 1.719 | 79.1 % | 52.7 % | 20.2 % | 0.906 | 1.253 |
| LayerNorm | 1.053 | 59.5 % | 18.8 % | 22.4 % | 0.198 | 0.434 |
| Transpose | 0.638 | 22.1 % | 24.8 % | 13.5 % | 0.141 | 0.141 |
| **top five, total** | | | | | **2.392** | **4.073** |

**Remove every second of it at zero cost: 17.989 s -> 15.597 s (1.1534x) at the tight end, 13.916 s
(1.2926x) at the unreachable loose end.** The campaign was sized on 3.4x. Reaching 10 s needs
1.7989x, i.e. 7.989 s out, so producer-side work on the entire pairformer track supplies **29.9 % to
51.0 %** of it. **Neither end of the range reaches the target, so the conclusion does not depend on
where in the range the truth sits.**

**Three independent derivations of the tight figure agree.** This script gives 2.392 s from the
closing capture and 2.399 s from the earlier one; `k10-instrument` computed **2.51 s** by its own
route from the same data. Within 5 %.

**And one end-to-end check says all of them are loose in practice.** `util-op-deletes` attacked
`BinaryNg` and `Transpose`, bounded here at 1.047 s tight (1.062x), and actually banked **1.015x**.

**What this does NOT bound**, and it is where anything left has to come from:

1. **Consumer-side and kernel-internal work.** On the fused op the reader spends **38.5 %** of its
   time idled *by the compute cluster* (16.1 % on a CB it has already filled, 22.4 % waiting for
   compute output) against the starved control's **3.7 %** — ten times. That is the compute side
   being the constraint, and no amount of dataflow work touches it.
2. **The diffusion step, 32.5 % of the fold**, which appears nowhere in this table because nobody has
   ever measured it.

## The campaign's own 2x ceiling misses, once its refuted first rung is removed

`ceiling_recheck.py`, host only, every input quoted from a concluded row.

`b2z2-sampler-ceiling-map` concluded that **"2x is arithmetically reachable and it has 1.2 % of
margin"**, landing at **10.158 s (1.9767x)**. That stack already grants two large concessions: the
trunk is held at a **"movement-free floor nobody knows how to reach" (4.689 s)** and `rest` at its
measured 2.1805 s. Its **first rung is `--diffusion_trace`**, "built, default off", credited with
**26.400 -> 23.378 ms/step**.

**That lever measures 0.9948x on Blackhole** (`b2z2-diffusion-loop-attack`, qb2 card 1, the cell's own
fixture: eager 19.937 s against traced 20.041 s, ranges overlapping, identical CIF; the loop is 93.8 %
device with a 0.33 s/fold host side; qb2 dispatches a one-tile op in 9.56 us against qb1's 22.92 us,
so eager issue cost hides behind a 22.02 ms device step). So on the part that matters that rung is
worth **zero**, and the three levers above it start from 26.400 ms/step, not 23.378.

| | ms/step | sampler | fold | ratio | vs 2x |
|---|---|---|---|---|---|
| as published, all four rungs | 16.442 | 3.288 s | **10.158 s** | **1.9800x** | +0.101 s |
| trace rung removed, other three land in full | 19.464 | 3.893 s | **10.762 s** | **1.8688x** | **+0.706 s (+7.0 %)** |

**2x misses by 7.0 %, not 1.2 %** — and that is still the optimistic case, because it grants a trunk
floor nobody knows how to reach and three levers that are projected rather than built. The map's own
sentence was *"any one lever under-delivering by 15 % kills it"*; one of them under-delivered by
100 %.

**Reconstruction check, stated rather than buried:** this script reproduces the map's fold seconds
**exactly** at every rung (12.149 / 11.545 / 11.111 / 10.560 / 10.158 s), so the ladder is rebuilt
faithfully. Its ratios differ in the third decimal (1.9800 against the map's 1.9767) because the map's
implied denominator is ~20.079 s rather than the 20.113 s cell it names. That 0.17 % does not touch a
conclusion whose margin is 0.706 s.

**This is denominated in the 20.113 s cell, which is the tree the map was built against.** It is not
restated for today's 17.989 s cell: the trunk and `rest` brackets belong to that tree. The structure
carries; the absolutes do not.

---

## Red-team corrections to this document, 2026-09-13

`k10-p1-redteam` (concluded, **VERDICT: PARTIAL — "the campaign's STOP survives, three of the
arguments holding it up do not"**) audited this work by independent re-derivation. What it confirmed,
what it broke, and what I got wrong.

**CONFIRMED.** It reproduced the census stall split from a different capture on a different build,
dividing by each counter's own thread: **57.7 % / 10.2 % = 5.66:1** against the census's 57.0 / 9.7 /
5.9:1 — **1.2 % apart on the headline**. And it attacked the falsifier above by running the control
I did not: if TRISC0 and TRISC1 durations were within 1 % of each other, 0/4441 would be vacuous.
They are not — TRISC0 is the longer on 1266 rows and `wf > TRISC1` fires **51** times where
`wf > TRISC0` never does. **Its 1010 is exactly my 1010.** The wait-front attribution is confirmed by
a route that tried to break it.

**CORRECTION 1 — my reserve-back row is VACUOUS and this document overstated it.** `rb` exceeds *no*
thread's duration — not TRISC0, TRISC1 or TRISC2 — so **0/4441 against TRISC2 carries no
information** and must not be quoted as proof that TRISC2 owns it. That attribution rests on the
source declaration in `llk_io_pack.h` alone, which is sound, but this document should have said so.
Only the wait-front half of the falsifier is load-bearing.

**CORRECTION 2 — "the divisor was the problem, not the normalisation" is right but not the whole
cure.** Switching TRISC1 -> TRISC0 while still *summing* the two counters leaves **975** anomalies
against 1010. The cure is per-thread attribution, not a better denominator.

**CORRECTION 3 — the error bar I implied was zero is not zero.** Saying the split "carries no error
bar from this source" reads as though it has none. Moving the normalisation moves the ratio
**1.0 %**; moving the *population* (fenced block 5.88 vs whole capture 4.86) moves it **17 %**. The
defensible band on 5.9:1 is roughly **4.9 to 5.9**, set by which region of the capture is fenced. At
the worst end input still beats output 4.9:1, so the conclusion survives the whole range.

**CORRECTION 4 — this document and `k10-instrument` normalise differently and neither said so.** Its
table is normalised to **op span**; the `--by-op` table here is normalised to **TRISC0 duration**.
They agree to under 2 % on any op whose TRISC0 residency is near 1 and diverge badly where it is not.
**Transpose is the case: 22.1 % published against 68.2 % here, because Transpose's TRISC0 is resident
only 33.1 % of its span.** The red team's own reduction gives 66.8 %, agreeing with this document.
**Transpose is not a low-input-stall op; it is the second-highest in the block**, and any row that
de-prioritised it on the 22.1 % figure did so on a normalisation artifact. Matmul has the same shape
and an **unreconciled 34.0 % vs 64.5 %** between the two, which is 23 % of the block and is not
closed — the likely cause is the differing populations (fenced block vs whole capture).

**CORRECTION 5 — the 39 % conversion bound is measurably too tight; it is 44 %.** The 1:1 assumption
is calibratable from the starved control itself, which ran at 98.9 % of the DRAM roof so essentially
all of its compute stall is memory-caused: it reads compute wait-front **89.25 %** against reader DRAM
**80.65 %**, i.e. **1.107 ns of compute stall per ns of reader DRAM wait**. Applying the measured
conversion takes the bound to **44.4 %**. That is 5 pp and does not move the verdict.

**CORRECTION 6 — the "loose by ~4x in practice" claim above is wrong and I withdraw it.** It compared
`util-op-deletes`'s 1.015x against a bound taken over the *whole* BinaryNg and Transpose classes,
which is not what that row deleted — D1 removed 560 `ttnn.unsqueeze` calls and D4 folded one trimul
operand transpose into its consuming matmul. **On the one deletion where a prediction and a
measurement both exist, D4's op bench predicted 0.1545 s/fold and the paired A/B banked 0.1130 s/fold:
73 %, not 12 %.** Prediction-to-measurement realisation on this stack is high, not low. (Note also
that the 1.015x is `1.00894 x 1.00613` composed from a two-commit chain; the direct stack A/B was
refused for host contention and never ran.)

**Net effect on the envelope: it survives, and the reasons for it are narrower than stated.** The
producer-side verdict is unchanged and independently reproduced — the target sits at **23.2 %** on
the NoC read barrier against a starved control's **80.7 %** and a compute-bound control's **3.6 %**,
a 22x separation with the target near the compute-bound end. But see CORRECTION 7 below for the
argument that does **not** survive.

**CORRECTION 7 — "the target's consumer-side figure is ten times the starved control's" does not
survive, and I had repeated it.** Both figures were NCRISC-only. Reduced over **both** DM threads the
starved control is **72.5 %** and the target **77.2 %** — a ratio of **1.06x, not 10x**. The reason is
structural: in `ttnn.add` NCRISC reads and BRISC writes, so that kernel's NCRISC never calls
`cb_wait_front` at all and its writer-side blocking sits on BRISC at **68.8 %**, uncounted. A writer
blocked in `cb_wait_front` is what a *starved* pipeline looks like too, because the block propagates.
**`DM-CB-WAIT-FRONT` on a writer thread cannot discriminate, and the starved control proves it.**
The zone that *does* discriminate is reserve-back: starved **3.7 %**, target **20.4 %**, compute-bound
**89.9 %** — the target sits about a fifth of the way from starved to compute-bound. Still
(b)-leaning, still not producer-late, but the "ten times" framing must not be repeated.

**CORRECTION 8 — "issuing" is a residual, not a measurement, so the LOOSE end of the envelope rests
on unmeasured time.** `dm_report.py` defines issuing as whatever remains of residency after the five
zones, so the budget closes by construction. The residual is **82.9 %** of Transpose's BRISC, 73.3 %
of Permute's, 66.5 % of ReshapeView's — those cannot all be address generation, and the known blind
spots (`noc_async_read_barrier_with_trid`, `noc_async_writes_flushed`) land in exactly this bucket.
It cuts both ways: the LOOSE bound credits unmeasured time to the producer side, **and the TIGHT
bound may be an under-estimate if trid-barrier reads are hiding there.** The target op is the least
affected (25.6 % NCRISC, 28.2 % BRISC), which is luck rather than something established.

## The fold budget: where would 7.989 s come from?

`fold_budget.py`, host only, every input a measured number from a concluded row.

| region | s/fold | % fold | what is known about it |
|---|---|---|---|
| pairformer track | 10.220 | 56.8 % | producer-side bounded at **2.392-4.073 s** |
| diffusion step, device | 5.484 | 30.5 % | **never measured — the open question** |
| diffusion step, host | 0.362 | 2.0 % | dead: `--diffusion_trace` 0.9948x on Blackhole |
| everything else | 1.923 | 10.7 % | host residual, repeatedly attacked |

**Grant every bound in full, simultaneously, at zero cost:**

- trunk producer-side at its **tight** bound removes 2.392 s, leaving **5.597 s** still to find —
  which is **102.1 %** of the diffusion step's entire device time.
- trunk producer-side at its **loose** (physically unreachable) bound removes 4.073 s, leaving
  **3.916 s** — **71.4 %** of it.

**So 10 s requires the whole trunk producer-side bound AND 71-102 % of the diffusion step's device
time to vanish.** Removing the diffusion step's *work* is not available — fewer steps is a cheat, not
a speedup, and the 200-step floor is a hard bound this campaign already established. So the honest
question for the one unmeasured region is not *"is there a prize"* but **"is over half of 5.5 s of
device time recoverable without doing less of the model's own work"**.

**For scale, what Phase 1 actually found:**

| lever | fold ratio |
|---|---|
| bank permutation B1+B2+B3 | 1.0254x |
| BinaryNg class, 3 sites | 1.0094x |
| matmul L1 residency (predicted BH) | 1.0300x |
| `M_block` 4 -> 8 (predicted BH) | 1.0100x |
| **composed, optimistic** | **1.0768x -> 16.71 s** |

That is **9.6 % of the margin** 10 s requires, and it is the optimistic end because sub-additivity is
the rule on this stack, not the exception.
