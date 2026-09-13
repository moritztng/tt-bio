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
