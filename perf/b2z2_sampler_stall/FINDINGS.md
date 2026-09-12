# The diffusion step's inner stall split, measured

whglx card 2, one Wormhole_B0, 8x9 grid, tt-metal v0.68.0 built with Tracy
(`/home/mthuening/work/b2z2-profiler`), tt-bio at `0f3f9f67`, `cdk2x2_512.yaml` + its fixed
35-row a3m, the `Diffusion.__call__` grabbed on call 2 and replayed 3 times between fences.
**Wormhole. The published cell is Blackhole. Fractions transfer, seconds do not.**

`b2z2-sampler-ceiling-map` established that `DEVICE COMPUTE CB WAIT FRONT` and
`DEVICE COMPUTE CB RESERVE BACK` are blank in all 1066 rows of every committed diffusion-step
capture on both architectures. This is the capture that fills them in.

## 1. The split

Of the math thread's **25.9760 ms** of residency in the step:

| term | ms | % of residency |
|---|---|---|
| blocked on input tiles (`cb_wait_front`) | **14.4910** | **55.8** |
| blocked on output room (`cb_reserve_back`) | **3.1492** | **12.1** |
| not stalled on a CB | **8.3287** | **32.1** |

The step accounts for itself: 14.4910 + 3.1492 + 8.3287 + 132.2822 non-resident = 158.2512 ms
against a 158.2582 ms span, 0.004 % apart.

The same numbers for the Pairformer block on the same box, so the two can be read side by side:
62.0 / 10.8 / 27.1 %. **The fractions are close and the step is slightly less input-stalled.**
That is the part of the campaign's floor statement that does transfer.

Per op code, per core, ms/step, ordered by device kernel time:

| op code | n | kernel | TRISC1 | input wait | in % | output wait | compute |
|---|---|---|---|---|---|---|---|
| Matmul | 387 | 18.543 | 11.688 | 6.115 | 52.3 | 0.945 | 4.628 |
| BinaryNg | 339 | 6.972 | 6.460 | 4.936 | **76.4** | 0.918 | 0.606 |
| SDPA | 30 | 3.315 | 3.251 | 0.807 | **24.8** | 0.106 | 2.338 |
| LayerNorm | 114 | 3.296 | 3.031 | 1.373 | 45.3 | 0.168 | 1.489 |
| ReshapeView | 36 | 2.629 | 0 | — | — | — | — |
| NlpCreateHeads | 30 | 2.446 | 0 | — | — | — | — |
| Permute | 12 | 1.600 | 1.233 | 0.968 | 78.5 | 0.861 | -0.596 |
| Transpose | 51 | 0.498 | 0.335 | 0.294 | 87.7 | 0.151 | -0.109 |

BinaryNg is again almost entirely stall (76.4 % here, 85.7 % on the trunk). **SDPA is the
exception on this step: 24.8 % stall against the trunk's op-code range of 50-90 %.** The fused
SDPA kernel is the one op in the sampler whose math thread is actually busy.

The negative `compute` entries on Permute and Transpose are the instrument's own error bar, not
a result: `CORE COUNT` is the only divisor the report offers and it is exact only if every core
ran the compute kernel for the whole op. **208 of 1066 ops (19.5 %)** come out with a per-core
stall longer than the per-core residency it is part of, against **51 of 272 (18.8 %)** on the
block. Same defect, same rate. Direction and ordering are safe; the last percent is not.

## 2. What the instrument cost, measured rather than assumed

| | WH, whglx card 2 |
|---|---|
| step, profiler off, `--reps 3` | 41.5326 ms/call |
| step, profiler off, `--reps 10` | **41.4820 ms/call** |
| step, profiler armed, span of the replayed region | **158.2582 ms** |
| **wall perturbation** | **3.81x** |

3.81x, against the **1.0079x** the same build costs on a 272-op Pairformer block. 1066 tiny
programs per call is exactly the regime the block measurement does not cover.

**The perturbation is entirely in the gaps between programs and the kernels are untouched**,
which is why the split above is still usable. Two independent checks:

* On this capture the device kernel durations sum to **40.4144 ms**. The same step with the
  profiler off runs in **41.4820 ms** of synced wall. The difference is 1.07 ms over 1066
  programs = **1.0 us/program** of device program-to-program latency, which is the right order
  for a replay loop where the host is fully ahead. The profiler adds **117.84 ms** of gap on top
  of that and nothing to the kernels.
* On Blackhole the same harness's committed capture sums to **22.025 ms** of kernel time against
  the fold's own **22.015 ms**, while its span is 45.50 ms against a 26.400 ms production wall.
  Same signature: gaps inflate, kernels do not.

**So: read fractions of residency off this capture, never its span.**

## 3. The wait is NOT byte-shaped. It is program-shaped, and that is the opposite of the trunk

Over the 43 compute-kernel sites of the step, against the same three models the trunk was fitted
with (`CONTEXT §2-CORRECTION-B`), on the same box and through the same code:

| model | R2, diffusion step | R2, Pairformer block |
|---|---|---|
| tile arrivals only | **+0.0295** | -0.5696 |
| tile-pair MACs only | +0.0308 | -0.5988 |
| DRAM bytes only | 0.6219 | 0.5018 |
| DRAM + L1 bytes | 0.6375 | 0.5487 |
| **DRAM + L1 + per-program** | **0.8638** | 0.5585 |

Spearman against the measured per-site wait, step: bytes **+0.960**, DRAM bytes +0.933, tile
arrivals +0.907, programs +0.879. Every candidate ranks well on the step because its 43 sites are
strongly collinear, so the ranks do not separate the models here and the fits do.

**Adding a per-program constant lifts R2 from 0.6375 to 0.8638 on the step and from 0.5487 to
0.5585 on the block.** Decomposing the fitted input wait:

| term | diffusion step | Pairformer block |
|---|---|---|
| DRAM bytes | 3.465 ms, **23.9 %** | 31.376 ms, **66.1 %** |
| L1 bytes | 0.431 ms, 3.0 % | 2.107 ms, 4.4 % |
| per-program constant | **9.114 ms, 62.9 %** | 3.327 ms, 7.0 % |
| the constant itself | **9.76 us/program** | 14.34 us/program |
| fitted total | 89.8 % of 14.491 ms | 77.6 % of 47.450 ms |

The constant is the *smaller* of the two in absolute terms. What differs is how often it is paid:
the step runs **934 compute programs of mean 20.0 us** where the block runs **232 of mean
143.4 us** (Blackhole figures, same reader). A fixed cost at the head of every program is 7 % of
the block's wait and **63 % of the step's**.

This is a per-program cost that sits **inside** `cb_wait_front`, i.e. the math thread waiting for
its first input tiles after the kernel starts: reader startup, the NOC round trip, the semaphore
handshake. It is not host dispatch. `b2z2-diffusion-loop-attack` measured `--diffusion_trace` at
**0.9948x on a single chip**, and this is why: a trace removes host dispatch and the step was
already 93.8 % device-bound, so it never touched the term that dominates.

**For `b2z2-sharded-sampler`: a token-axis shard that halves the bytes does not halve this wait.**
It halves the 23.9 % byte term and leaves the 62.9 % per-program term alone, because a shard keeps
the same 1066 programs on each chip. On the measured coefficients that is 14.491 -> 12.758 ms,
**1.136x on the wait term, not ~2x.**

## 4. The 15.3 % that never computes

132 programs, identical on both architectures, 15.3 % of the step's device kernel time on both:

| op code | n | BH ms | BH GB/s | WH ms | WH GB/s | MB moved |
|---|---|---|---|---|---|---|
| NlpCreateHeads | 30 | 1.470 | 71.7 | 2.446 | 43.1 | 105.4 |
| ReshapeView | 36 | 1.359 | 88.5 | 2.629 | 45.8 | 120.3 |
| Slice | 30 | 0.227 | 281.8 | 0.397 | 161.6 | 64.1 |
| Pad | 6 | 0.145 | 237.9 | 0.416 | 82.8 | 34.4 |
| Copy | 24 | 0.106 | 517.3 | 0.229 | 240.3 | 55.1 |
| NLPConcatHeads | 6 | 0.064 | 213.6 | 0.085 | 161.7 | 13.8 |
| **total** | **132** | **3.370** | **116.5** | **6.202** | **63.4** | **393.0** |

`b2z2-sampler-ceiling-map` reports 153 programs here. Its own six op codes are
30 + 36 + 30 + 6 + 24 + 6 = **132**, which is what the reader counts; the 3.369 ms and the 15.3 %
are unaffected.

**They are not bandwidth-bound.** The two that carry 84 % of the time run at 71.7 and 88.5 GB/s
against the 444.9 GB/s this Blackhole processor was measured at, 16 % and 20 % of the roof. So:

* **a shard**: halves their bytes and, since they are element-addressing-bound rather than
  bandwidth-bound, roughly halves their time too. Worth ~0.34 s/fold on BH.
* **a fusion**: deletes them outright, because they produce no arithmetic — they exist only to put
  bytes in the layout the next op wants. Worth ~0.67 s/fold on BH, and it also removes 132 of the
  program-startup constants priced in §3.
* **nothing else touches them.** They have no math thread to make faster.

## 5. TRANSFER-VERDICT: no

`b2z2-final-ceiling`'s top rung takes the Pairformer block's movement-free multiplier — 0.41, i.e.
14.900 ms of a 36.344 ms block survives if every CB stall is deleted — and applies it verbatim to
the sampler. Substituting the measured fractions and changing nothing else:

    CB stall in the BH step = (0.558 + 0.121) x 15.211 ms resident = 10.330 ms, 39.1 % of the wall
    movement-free step      = 26.400 - 10.330 = 16.070 ms
    the sampler's own multiplier = 1.6428x, against the trunk's 2.4390x

Cross-arch control, the same quantity built entirely on Wormhole from its own bare wall:
41.4820 - 17.6402 = 23.8418 ms, **1.7399x**, **5.9 % from the Blackhole projection**. Same sign,
same magnitude, the way the trunk's byte-vs-tile result came out on both. That is the precedent
for believing the fraction across the architecture boundary, and it holds here.

**The multiplier does not transfer because a multiplier is a fraction of the wall, and the two
blocks hold their math thread resident for very different fractions of it** — 88.4 % for the
block, 57.6 % for the step. The stall *fractions* are nearly the same; the *wall* they are a
fraction of is not.

| rung, against the 18.594 s levered arm | as published | with the measured sampler |
|---|---|---|
| trunk movement-free, host held | 1.5336x | 1.5336x |
| + sampler, conservative | 1.8095x | **1.7062x** (-5.7 %) |
| + sampler, optimistic | **1.9769x** | **1.8015x** (-8.9 %) |

The sampler's own floor moves from 2.5362 s to **3.4518 s**, so **0.9156 s/fold comes off the
books as recoverable.** The single-processor bracket is now **1.53x - 1.80x**, and 2x is further
outside it than the campaign has been told.

## 6. The lever the split names, and why it was built on the wrong block

The term worth building against is the **per-program constant: 9.114 ms of the WH step's 14.491 ms
of input wait, 6.50 ms of the BH step's 26.400 ms wall, 24.6 % of the step.** Well over the
brief's 10 % bar. The lever against it is fusion — fewer, longer programs, intermediates held in
DST or an L1 CB across a chain — and neither a shard nor a byte lever touches it.

`b2z2-pairformer-megakernel-build` built exactly this and it **lost**: 10.2 % of the block's
op-boundary tile passes deleted, block 2.9 % slower. On the block that is the expected outcome —
the per-program term there is 7.0 % of the wait and 232 programs of 143 us each have nothing to
gain from being merged. **The megakernel was pointed at the block whose wait is 66 % bytes,
instead of the one whose wait is 63 % program starts.** Same build, other target.

Two named places to start, both already scoped by `b2z2-sampler-ceiling-map`:
the SDPA head plumbing (153 programs of pure layout, §4) and BinaryNg (339 programs, 76.4 % stall,
6.972 ms of WH kernel time, mean 20.6 us each — the largest population of short programs in the
step). This pass does not build either: it measured the number the wave's upper bound rests on and
that number is now on the branch.

## 7. Reproduce

    python3 perf/b2z2_sampler_stall/stall_split.py \
        --csv perf/b2z2_sampler_stall/ops_perf_step_whglx_c2.csv.gz \
        --meta perf/b2z2_sampler_stall/step_prof_whglx_c2.json \
        --out /tmp/x.json --label "WH step" --arch WH
    python3 perf/b2z2_sampler_stall/transfer.py --split perf/b2z2_sampler_stall/split_step_wh_c2.json

The same reader on the three captures that were already committed returns their published numbers:
the BH block 18.3539 / 3.1069 / 10.7129 ms against 18.3366 / 3.1066 / 10.7010 and a movement-free
2.4420x against 2.439x; the WH block 47.4499 / 8.2977 / 20.7621 ms exactly, bytes R2 0.5487 against
0.5465, tile R2 -0.5696 against -0.5684, 251,023 arrivals against 251,933; the BH step 1066
programs, 22.0252 ms kernel and 15.2132 ms residency against 22.015 and 15.211, with Matmul 41.2 %,
BinaryNg 16.3 % and SDPA 11.4 % of it. Four rows, three captures, two architectures, one pipeline.
