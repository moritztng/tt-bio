# The diffusion step has no dispatch prize, and a trace makes it slower

whglx (`j10glx02`) card 16, one Wormhole_B0, 8x9 grid, `tt-galaxy-wh-l`. tt-bio `50542ade8`
(`wk/k10-p1-diffusion-measure`, main merged), tt-metal v0.68.0 `1452925b` built with Tracy at
`/home/mthuening/work/b2z2-profiler/tt-metal` -- the same build for every arm, profiler off
except where it says armed. `cdk2x2_512.yaml` + its 35-row a3m, 200 sampling steps, one sample.
**Wormhole. The published cell is Blackhole. Fractions transfer, seconds do not.**

Every run on this branch ran under `runtime.host_thread_cap_env(8)` ->
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8` on a 64-core host, loadavg 3.7 - 8.6 throughout.
The eager arm's fold-to-fold spread is **1.0009x** and the traced arm's is **1.0013x**, against
the 1.10x that says the cap is working and the 2.32x an uncapped box produces.

Core coverage is not re-opened here. `util-grid-coverage` measured the step at 83.3 of 110 cores
time-weighted and priced the recoverable time at 0.000 s; that stands.

## 0. What was predicted, and the miss

`perf/k10_diffusion/PREDICTED.md`, committed before the first number.

| quantity | predicted | measured | |
|---|---|---|---|
| gap fraction, eager, fold context | 6 - 14 % | **5.66 %** | just under |
| gap fraction, traced | 2 - 5 % | **7.74 %** | **miss, and the wrong side** |
| eager minus traced | +4 to +10 points | **-2.08 points** | **miss, wrong sign** |
| leading op code | Matmul, 40 - 48 % | **Matmul, 48.2 %** | hit |
| second | BinaryNg, 14 - 19 % | **BinaryNg, 18.8 %** | hit |
| unpack blocked on input tiles, over its own thread | 55 - 70 % | **60.6 %** | hit |
| input : output stall ratio | 4:1 - 7:1 | **6.07 : 1** | hit |
| programs doing no arithmetic | 12 - 18 % | **15.4 %** | hit |

**The miss is the whole result.** I predicted a trace would delete host dispatch worth 4-10
points of the step. It deletes about 3 points of host cost and adds about 5 points of its own,
so the traced step is **0.89 ms slower per step than the eager one**. The hunch line in
`PREDICTED.md` -- that the two arms would land within ~2 points -- got the magnitude right and
the sign wrong.

## 1. INSTRUMENT-CONTROL

**The reader reproduces both committed step captures before it is used on a new one.**
`perf/k10_diffusion/step_measure.py` on `b2z2-sampler-stall-split`'s Wormhole capture returns
1066 programs, 40.4296 ms of device kernel time, Matmul 18.544 ms / BinaryNg 6.972 ms /
SDPA 3.315 ms / LayerNorm 3.296 ms, and 132 no-arithmetic programs at 15.3 %, against that
branch's published 40.4144 / 18.543 / 6.972 / 3.315 / 3.296 / 132 / 15.3 %. On
`b2z-kernel-cycle-census`'s Blackhole capture it returns 1066 programs and 22.0256 ms against
its published 22.025 ms, with Matmul 41.2 %, BinaryNg 16.3 %, SDPA 11.4 % -- its three published
figures to the digit.

**The armed span is not a wall and this capture says by how much.** Summed over the fenced
region, `OP TO OP LATENCY` on the armed run is **320.66 ms per call** against the **2.23 ms** of
real gap the same call carries with the profiler off. That is **143x**, on a call whose kernels
move by nothing measurable. Anyone reading a gap fraction off an armed capture on this workload
gets 89.6 % instead of 5.7 %. The device kernel durations are the only thing in an armed capture
that can be put next to a wall, and the wall has to come from a separate profiler-off run.
Wall cost of arming, on this card: precursor 7.9 s -> 126.0 s, replayed call 38.26 -> 363.29 ms.

**Per-core normalisation, error stated.** Each stall counter is summed over the cores that ran
its op and is divided by that op's own `CORE COUNT`, never the step's. Against the thread each
counter is actually declared on, **0 of 2784 compute rows** have a per-core stall exceeding its
own thread's duration -- no residual error from this source. Against TRISC1, the divisor
`cb_split.py` and `stall_split.py` use, **633 of those same 2784 rows** are anomalous. The anomaly is the
divisor. Nothing below is a "useful math" or "recoverable seconds" figure: TRISC1's own stall is
not measured by any instrument in this campaign.

All numbers below come from the **stock** whglx profiler build. No DM-side counter is quoted;
the `k10-instrument` patch that provides them lives on qb2 and was not needed for these questions.

## 2. GAP-FRACTION

Device kernel time per step comes from the armed capture. Each wall comes from a separate
profiler-off run on the same card, same commit, same build.

| | ms/step | gap | us/program |
|---|---|---|---|
| device kernel time, 1096 programs | **37.2531** | -- | -- |
| tight replay loop, host fully ahead | 38.2572 | **2.62 %** | 0.92 |
| **the fold, eager** | **39.4867** | **5.66 %** | 2.04 |
| **the fold, `--diffusion_trace`** | **40.3779** | **7.74 %** | 2.85 |

Decomposed, per step:

* **37.2531 ms, 94.34 %** of the eager step is inside a device kernel.
* **1.0041 ms, 2.54 %** is the device's own program-to-program latency, measured where the host
  cannot be the cause. 0.92 us per program, 1096 times.
* **1.2295 ms, 3.11 %** is what the fold pays on top of a tight replay loop: staging `r` and
  `times` onto the card, `ttnn.to_torch` on the way back, and a sampler loop that does host
  torch work between steps instead of running the dispatch queue ahead.

**The eager step has no dispatch prize.** 5.66 % is the entire budget for every lever that
removes host work from the step, and more than half of that 5.66 % is device latency that no
host-side change reaches. The block is 1.1 % gap at 232 programs of 143 us; the step is 5.66 %
at 1096 programs of 34 us. **The step's gap fraction is 5x the block's** -- and it is still
small. Per program the two are close, 2.04 us here against roughly 2 us implied for the block,
so what makes the step's fraction worse is that it pays the same cost 4.7x as often over a
kernel a quarter as long, not that any one gap is worse. (The block figure is Blackhole and this
one is Wormhole; take the ordering, not the difference.)

**`--diffusion_trace` is a regression on this path.** 0.9779x, -0.89 ms/step, -0.18 s/fold at
200 steps, measured with the arms alternating inside one process on one device open and one
weight load. `b2z2-diffusion-loop-attack` reported 0.9948x; on the current trunk it is worse than
that, and worse than neutral. A trace removes the 1.2295 ms of per-step host cost and replaces it
with more than that: `forward_traced` pads `r`, stages two tensors through
`copy_host_to_device_tensor`, launches `execute_trace` non-blocking and then blocks in
`to_torch`, so the staging is serialized in front of the device instead of overlapping with it.
That mechanism is a hypothesis -- the split of the 0.89 ms between staging, launch and readback
was not measured -- but the sign and the size are not.

**The census's "51.6 % eager / <=32 % traced" is a share-of-fold bound, not a gap fraction, and
the two are unrelated.** On this box the step is **7.897 s of a 37.888 s fold, 20.8 %**, eager.

## 3. TOP-KERNELS

Fenced step region, per call, ranked by device kernel time. `in/T0` is `cb_wait_front` over the
unpack thread's own duration; `out/T2` is `cb_reserve_back` over the pack thread's.

| op code | n | ms | % | mean us | mean cores | in/T0 | out/T2 | in:out |
|---|---|---|---|---|---|---|---|---|
| `Matmul` | 381 | 17.956 | **48.2 %** | 47.1 | 64.6 | 55.7 % | 8.0 % | 6.45 |
| `BinaryNg` | 339 | 6.986 | **18.8 %** | 20.6 | 72.0 | **75.6 %** | 14.4 % | 5.39 |
| `LayerNorm` | 114 | 3.293 | 8.8 % | 28.9 | **22.3** | 46.2 % | 5.5 % | 8.13 |
| `SDPA` | 30 | 2.582 | 6.9 % | 86.1 | 72.0 | 57.8 % | 5.4 % | 10.58 |
| `NlpCreateHeads` | 30 | 2.474 | 6.6 % | 82.5 | 27.2 | -- | -- | -- |
| `Pad` | 18 | 1.041 | 2.8 % | 57.8 | 72.0 | -- | -- | -- |
| `ReshapeView` | 24 | 0.964 | 2.6 % | 40.1 | 72.0 | -- | -- | -- |
| `Slice` | 60 | 0.713 | 1.9 % | 11.9 | 72.0 | -- | -- | -- |
| `Transpose` | 51 | 0.497 | 1.3 % | 9.8 | 72.0 | 70.4 % | 32.7 % | 1.99 |
| `Concat` | 6 | 0.250 | 0.7 % | 41.6 | 72.0 | -- | -- | -- |
| `Copy` | 24 | 0.229 | 0.6 % | 9.6 | 72.0 | -- | -- | -- |
| `Untilize` | 6 | 0.092 | 0.2 % | 15.3 | 65.0 | 66.2 % | 0.2 % | 262 |
| `Tilize` | 6 | 0.089 | 0.2 % | 14.9 | 72.0 | 75.1 % | 35.7 % | 1.41 |
| `NLPConcatHeads` | 6 | 0.085 | 0.2 % | 14.1 | 72.0 | -- | -- | -- |

**The step is a different animal from the block.** The block leads with
`GenericOpDeviceOperation` at 36.7 %, the fused trimul + triangle attention. The step has no
trimul at all: it is **67.0 % Matmul and BinaryNg**, two op codes and 720 of 1096 programs, and
the next three together are 22.4 %.

**168 of the 1096 programs, 15.45 % of the device kernel time, never touch the compute cluster.**
`NlpCreateHeads`, `Pad`, `ReshapeView`, `Slice`, `Concat`, `Copy`, `NLPConcatHeads` -- all
TRISC0/1/2-idle, all pure layout. That fraction is unchanged from the 15.3 % both earlier
captures measured, on both architectures, across three commits and two levers landing in
between. The population inside it has moved (`Permute`'s 12 programs are gone, `Slice` doubled to
60, `Pad` tripled to 18, `Concat` is new), the total has not.

## 4. STALL-SPLIT

Per call, per core, each counter over its own thread.

| thread | ms/step | residency over the 37.2531 ms of kernel time |
|---|---|---|
| BRISC | 37.0398 | **99.4 %** |
| NCRISC | 24.5311 | 65.8 % |
| TRISC0, unpack | 23.3523 | 62.7 % |
| TRISC1, math | 23.8218 | 63.9 % |
| TRISC2, pack | 24.1759 | 64.9 % |

| | ms/step | over its own thread |
|---|---|---|
| `cb_wait_front`, unpack blocked on input tiles | **14.1551** | **60.6 % of TRISC0** |
| `cb_reserve_back`, pack blocked on output room | 2.3307 | 9.6 % of TRISC2 |
| input : output | | **6.07 : 1** |

The compute cluster is resident for only **63 %** of the step's device kernel time, and for
**60.6 %** of the time it is resident the unpack thread is blocked waiting for input tiles. So
**14.16 ms of the 37.25 ms step, 38.0 %, is one thread sitting in `cb_wait_front`.** BRISC is
resident 99.4 % of the time, which by itself says nothing about whether it is working -- the
three-way reader split that would separate `noc_async_read_barrier` from `cb_reserve_back` needs
the `k10-instrument` DM counters, which exist only on qb2.

Per op the ratio runs from `SDPA` at 10.58:1 to `Transpose` at 1.99:1. **`BinaryNg` is the most
input-starved op in the step at 75.6 %**, and it is 339 programs of 20.6 us each -- the largest
population of short programs in the model. `Matmul` carries the most absolute wait, 5.99 ms.
`LayerNorm` is the outlier the other way: 46.2 % stalled but running on **22.3 cores of 72**,
which is the shape-driven core count `util-grid-coverage` already priced at zero recoverable.

Two numbers moved against the older card-2 capture at `0f3f9f67`, and both moved the way the
landed levers predict: input stall 56.8 % -> 60.6 % and in:out 4.60 -> 6.07:1. The levers took
2.28 ms of TRISC2 time out (26.46 -> 24.18 ms) while leaving the input wait where it was
(14.49 -> 14.16 ms). **Deleting output-side work makes the step more input-bound, not less.**

## 5. PREDICTS -- what Phase 2 should and should not build

| lever | predicted value at 512 aa | basis |
|---|---|---|
| anything that removes host dispatch from the step (trace, graph capture, dispatch batching) | **0 to -0.9 ms/step, i.e. nothing or a loss** | measured. The eager step is 94.34 % in-kernel; a trace was measured at 0.9779x here. **Do not build this.** |
| fusing the 168 no-arithmetic programs away | **up to 5.755 ms/step, 15.45 %** | they produce no arithmetic and hold no compute thread; they exist to put bytes where the next op wants them. Upper bound: they also stop paying the 0.92 us/program device latency, worth a further 0.15 ms. |
| fusing short compute programs into longer ones (`BinaryNg`, 339 x 20.6 us) | **bounded by 4.95 ms/step, 13.3 %**, and realistically well under it | that is `BinaryNg`'s entire input wait. `b2z2-pairformer-megakernel-build` lost 2.9 % pointing this at the block, whose programs are 143 us; the step's are 34 us, which is the regime where a per-program head cost can matter. |
| feeding `Matmul` faster (layout, CB depth, L1 residency of its inputs) | **bounded by 5.99 ms/step, 16.1 %** | largest absolute input wait in the step, 381 programs. |
| a token-axis shard of the step | **sub-linear on the wait, as `b2z2-sampler-stall-split` priced it** | a shard halves bytes and keeps all 1096 programs, so it does not touch the per-program term. Not re-measured here. |

**The single strongest statement this pass can make for Phase 2: the step's entire non-kernel
budget is 2.23 ms of 39.49, and 1.00 ms of that is device latency no host lever reaches. Every
remaining millisecond has to come out of the 37.25 ms inside the kernels, and 14.16 ms of that
is one thread waiting for input tiles.**

## 6. VERDICT

The step is **94.34 % in-kernel eager** and there is no dispatch prize in it. The lever the
campaign has not built is the one aimed at `cb_wait_front`: 38.0 % of the step's device kernel
time, concentrated in `Matmul` (5.99 ms) and `BinaryNg` (4.95 ms), plus 15.45 % of pure layout
programs that could be deleted outright. `--diffusion_trace` should be left off by default on
this path; it is measured at 0.9779x.

## 7. Reproduce

    # fold-context walls, both arms alternating in one process (opens the card)
    perf/k10_diffusion/step_gap.py --out gap.json --rounds 2 --workers 8 --open-lock LOCK

    # bare replay wall, then the armed capture, same card and commit
    perf/k10_diffusion/kernel_census.py --out step_bare.json --phase step --reps 10
    python -m tracy -r --no-op-info-cache --enable-sum-profiling -o step_prof \
        --op-support-count 30000 -- perf/k10_diffusion/kernel_census.py \
        --out step_prof.json --phase step --reps 3

    # host only, opens no device
    perf/k10_diffusion/step_measure.py --csv perf/k10_diffusion/ops_perf_step_whglx_c16.csv.gz \
        --meta perf/k10_diffusion/step_prof_whglx_c16.json --bare-eager-ms 39.4867 \
        --bare-traced-ms 40.3779 --bare-replay-ms 38.2572 --label "WH step c16" --arch WH \
        --out perf/k10_diffusion/step_wh_c16.json

`prior_step_wh_c2.json` and `prior_step_bh_c0.json` are the same reader on the two committed
captures, kept so the control in §1 stays checkable.
