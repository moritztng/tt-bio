# The Blackhole tile census: what the math thread waits for is bytes, not tiles

`b2z2-bh-tile-census`, 2026-09-12, revised after four orchestrator directives landed mid-pass.
No device was opened on any host, and **pc's banned card was not touched** — the inputs are
committed captures.

Inputs, all committed:

* BH — `perf/b2z_kernel_census/ops_perf_blocksum_qb2c0.csv.gz` + `ops_perf_step_qb2c0.csv.gz`
  (`b2z-kernel-cycle-census`, qb2 card 0, one Blackhole of a p300c, 11x10, tt-metal v0.68.0).
* WH — `perf/b2z2_profiler/block_prof/.../ops_perf_results_2026_09_12_15_16_25.csv.gz`
  (`b2z2-whglx-profiler-build`, whglx card 1, 8x9, profiler-enabled source build).

Reproduce: `census_tiles.py` (BH) and `wh_crosscheck.py` (WH control); both headers say how to
fetch their inputs. Both regenerate their committed outputs byte-identically.

## How to read this document

Per the orchestrator directive of 2026-09-12 ("you hold no card, so you may produce claims, not
ranked levers"), every quantity below is tagged:

* **MEASURED** — read directly out of a committed device capture.
* **DERIVED** — arithmetic over MEASURED inputs and the model's own shapes. No silicon needed and
  none assumed; a DERIVED number is as good as its inputs and its accounting rules, both of which
  are stated.
* **CLAIM FOR A CARDED ROW TO TEST** — a projection. Not a result. Each one carries the experiment
  that would settle it. **No lever is ranked in this document and no projected second is stated as
  an outcome.**

## 1. The count (DERIVED)

**236,118 CB tile arrivals per core-averaged PairformerLayer at 512 aa on Blackhole.** From the
operand shapes, the matmul program configs read verbatim out of the profiler's `ATTRIBUTES`
(`per_core_M`, `per_core_N`, `in0_block_w` — read, not modelled) and the grid split, over all 272
programs. Also 2,146,037 tile-pair MACs and 151,862 pack events per core.

| | per core-averaged block |
|---|---|
| CB tile arrivals (DERIVED) | **236,118** |
| of those, crossing the DRAM interface | 20,911 (8.9 %) |
| of those, read out of L1-interleaved | 7,025 (3.0 %) |
| multicast, daisy-chain or CB replications of tiles the grid read once | 208,182 (88.2 %) |

Wormhole, same model, same 272 programs, 72 cores: **251,933** arrivals per core.
Diffusion step (BH): **84,607** per core, 10,390 (12.3 %) from DRAM.

Break-even for a datum-rate-bound wait was 880,720 arrivals. The count is 3.73x under it.

## 2. The identity (MEASURED wait, DERIVED count)

    236,118 arrivals  x  20.82 ns/tile  =  4.9160 ms  =  26.8 %  of the ~18.34 ms wait

The falsifier's "near 30 %" branch fired. Stated the other way round, which is the more useful
form: the trunk's **effective cost per tile arrival is 77.66 ns** against a hardware tile pass of
20.82 ns — **3.73x**. On Wormhole it is 188.62 ns against 71.3 — **2.65x**. The tile is not what
costs the time on either architecture.

**26.8 % is an upper bound on the datum rate's share, not an estimate of it**, because those tiles
must be delivered before they can be unpacked.

**Error bar, and it is not small.** `b2z2-whglx-profiler-build` found that
`--enable-sum-profiling`'s stall accumulators are summed over the cores that ran each op while
TRISC1 residency is a duration. This census divides **each op by its own `CORE COUNT`**, which is
the correct rule, and that is why its window reproduces both published figures (BH 18.3696 against
18.3366; WH 47.5204 against 47.4499). But 50 of 232 BH ops and 51 of 272 WH ops still show a
per-core stall longer than their own residency, which is impossible. **18.3366 ms is not exact to
four decimals and nothing here should be read as if it were.** Direction and ordering are safe;
the last several percent are not. Everything in this document is stated as a fraction for that
reason.

## 3. What the remainder is — and what this row cannot tell you

Fitted against the measured per-site wait, over the 40 sites that run a compute kernel:

| model | R2 on BH | R2 on WH |
|---|---|---|
| tile arrivals only | **-0.5559** | **-0.5684** |
| tile-pair MACs only | -0.5744 | -0.5977 |
| DRAM bytes only | 0.7011 | 0.4997 |
| DRAM + L1 bytes | **0.7872** | **0.5465** |
| DRAM + L1 + a per-program constant | 0.7903 | 0.5560 |
| DRAM + L1 + tile arrivals | 0.7879 | — |

Spearman against the measured per-site wait: bytes delivered **+0.909** (BH) / **+0.911** (WH);
tile arrivals +0.767 / +0.776; programs in the site +0.445 / +0.463.

**A tile-count model fits worse than predicting the mean, on both architectures, with almost the
same negative R2.** Adding the tile term on top of bytes buys 0.0007 of R2 on BH and lands a
coefficient of 0.50 ns/tile, 40x under the measured 20.82 — the fit puts no weight on it. A
per-program constant is worth 3.0 us on BH, about 3.8 % of the wait across 232 compute programs.

**So, established (DERIVED, two architectures):** the wait is **proportional to bytes delivered
into the cores**. It is not the tile datum rate, not a per-program fixed cost, not dispatch. Dest
capacity (`b2z2-tile-shape-and-format`) and CB ring topology (`b2z2-cb-depth-prefetch`) were
already closed by measurement.

**Not established, and this row cannot settle it.** *Which* byte-proportional mechanism it is.
There are at least two and they are indistinguishable in this data because both scale with bytes:

1. DRAM interface bandwidth;
2. **serial NOC forwarding.** `b2z2-tile-arrival-latency` read the trunk's dominant matmul
   dataflow kernels (`tt_bio/kernels/mm_split/dm_in0_sender.cpp`) and found the operands are
   **not multicast** — one injector core per axis reads from DRAM and every other core gets the
   block by a semaphore-gated unicast hop from its predecessor, 10-11 hops on the cell. Forwarding
   B bytes down a chain is byte-proportional too.

An earlier draft of this document named DRAM bandwidth as the cause and said the interface sits
"at 79 % of roof while the reads are happening". **That is retracted.** The fitted 352.2 GB/s (BH)
and 154.4 GB/s (WH) are *effective delivery rates for DRAM-resident operands*, whose composition
is unknown; reading them as interface utilisation assumes the answer. `b2z2-tile-arrival-latency`
is measuring exactly this separation on whglx with a card and owns it. This row hands it a
target and does not duplicate it.

Supporting observation, weak, offered as context only: the fitted delivery rate is 79 % of the
DRAM roof on BH (10-11 chain hops) and 64 % on WH (8-9 hops). If chain length dominated, the
shorter chain should do better, and it does worse. whglx was under load 12-40 during that capture,
so this is not evidence, only a thing to check.

## 4. Reconciliation with `b2z2-pairformer-megakernel-build` (assigned by the orchestrator)

That row **measured** 5,145,516 op-boundary tile passes per block by graph capture, and the
orchestrator asked whether it is the same population as this count. **It is not, and the two
reconcile cleanly.**

| | device-wide per block | |
|---|---|---|
| CB tile arrivals (this row, DERIVED from shapes) | 25,972,980 | |
| op-boundary tile passes (that row, MEASURED by graph capture) | 5,145,516 | **19.8 %** of the above |
| never cross an op boundary — intra-op operand re-streaming | 20,827,464 | **80.2 %** |

That row reached "81 % never crosses an op boundary" by dividing the wait by the **Wormhole**
71.3 ns constant. This row reaches **80.2 %** from the model's shapes, with no per-tile constant in
it at all. **Two unrelated methods, one answer, and this one no longer depends on a constant from
the wrong architecture.** Their corollary — that matmul operand re-streaming, not op boundaries,
is where the arrivals are — stands and is now independently derived.

**The orchestrator's arithmetic check.** 5,145,516 / 110 x 20.82 ns = **0.9739 ms = 5.3 %** of the
wait. The arithmetic is right. **The pricing is wrong, and that row's own measurement is what
proves it**: it deleted 524,288 L1 op-boundary passes and the fusion returned +0.35 to +0.56 ms a
call. Translated to the cell's 110 cores that is **73.4-117.5 ns per deleted boundary tile, 3.5x
to 5.6x a tile pass** (86.8-138.9 ns, 4.2-6.7x, at the 130 cores it was actually measured on). So
an op-boundary tile costs several times a tile pass, measured on silicon, independently of this
row — because a boundary tile has to be written out and read back and an intra-op re-stream does
not. **Nobody should price an op boundary at 20.82 ns, including this row.**

**A tension worth naming rather than smoothing over.** This row's global L1 fit prices an
L1-interleaved arrival at 345.6 ns; their measurement says 73-139 ns for the L1 boundary tiles
they deleted. Both cannot be right for the same tiles. The likely reason is that the global
two-parameter fit is an average with real per-site dispersion (sum of absolute residuals is 35 %
of the wait), so **per-stage pricing off this fit is not safe and should not be done.** Their
number is a measurement of one stage; this row's is an average over forty sites. Theirs wins for
that stage. Caveat on theirs, stated in their own doc: it was taken on pc's banned 130-core card.

## 5. Claims, each with the experiment that settles it

**None of these is a result. None is ranked against the others.**

**C1. If every DRAM-interleaved read in the block became L1-interleaved, the DRAM term of the wait
falls 12.5130 -> 6.7604 ms and the block goes 36.3438 -> 30.5912 ms, a 1.188x block ratio.**
CLAIM FOR A CARDED ROW TO TEST. It is the ratio of two *fitted* delivery rates, one of which
(651.9 GB/s L1) has no independent measurement behind it at all, and §3 says the mechanism behind
both is unidentified. **Settled by:** `SILICON-EXPERIMENT.md` E2 — a reader-only sweep, 110 cores,
identical tensor in DRAM-interleaved and in L1-interleaved, rate as a slope over tile count.
Predicted DRAM 330-380 GB/s, L1 600-900 GB/s. **If L1 comes back under 450 GB/s, C1 is wrong.**
Do not multiply C1 into a fold number under any circumstance.

**C2. A fusion that stops an intermediate reaching memory is worth several times a fusion that
only deletes a tile pass.** CLAIM FOR A CARDED ROW TO TEST — though it is the claim with the most
independent support already: `b2z2-pairformer-megakernel-build` measured 3.5-6.7x for its own
deleted boundary tiles (§4). **Settled by:** sorting any candidate fusion chain by whether the
intermediate it eliminates currently crosses DRAM, and measuring two chains that differ on that
axis and match on passes deleted.

**C3. bfp8_b is worth its bytes on the delivery axis.** `b2z2-datum-rate-floor` MEASURED that it
is worth nothing on the movement axis (BH's packer issues 16 PACR whatever the format). A bfp8_b
tile is 1088 B against bf16's 2048, so on any operand that crosses the interface it is 0.531x of
that operand's byte contribution. CLAIM FOR A CARDED ROW TO TEST; unchanged from wave 1, the 512 aa
accuracy kill still applies and already retired one of the nine sites. **Settled by:** an A/B on
one large DRAM-resident pair operand, scoring the 512 aa fold, not only the 298 aa control.

**C4. The sampler's input-tile wait is materially below the trunk's 57 %.** `DEVICE COMPUTE CB WAIT
FRONT` is blank in all 1066 rows of every committed diffusion-step capture, so **the sampler's
inner split does not exist on any architecture.** What is MEASURED: the step is 26.400 ms wall =
22.015 ms device kernel + 4.385 ms exposed dispatch (`b2z2-sampler-ceiling-map`), its math thread
is resident 57.6 % of the step wall against the block's 88.4 %, and 15.3 % of its kernel time is
programs whose math thread is never resident. DERIVED here: it moves 2,340.6 MB of DRAM reads per
step against the trunk's 4,710.8 MB per block. **Settled by:** `SILICON-EXPERIMENT.md` E1,
`--enable-sum-profiling` pointed at the step. Predicted 25-40 % against the block's 57 %.

## 6. What is settled, in one paragraph

The trunk's input-tile wait is **not** a tile-count term. That is DERIVED from the model's own
shapes on two architectures and does not depend on any per-tile constant. It **is** proportional
to bytes delivered. **Which** byte-proportional mechanism — interface bandwidth or serial
operand forwarding — is open, is the difference between two quite different lever classes, and
belongs to `b2z2-tile-arrival-latency`, which holds a card and is attacking exactly that.
