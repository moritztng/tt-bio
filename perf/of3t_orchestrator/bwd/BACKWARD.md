# The backward pass gets the same treatment the forward got

**ENGINE-LEVEL, not OF3T's.** It lives under `state/of3t/` for historical reasons; the kernels it
describes serve Boltz-2, OF3T, BC2 and RFD3. `bcx-orchestrator` is briefed on it as of 2026-09-25
16:2x. For BC2 it matters MORE, not less: the gradient phase is 79.8 % of a BC2 cycle.

> **[of3t-tapedfwd, 2026-09-25] READ BEFORE YOU QUOTE SECTION 1. The step those numbers come
> from is from a tree two days behind the tape.** `of3t-stepfloor`'s 466.702 s step ran at
> commit `451ed56f4` (2026-09-21 18:07Z, recorded in the artifact's own `env.commit`). Exact
> softmax and layer norm went default-ON inside every `tape()` in `502ed112e` (2026-09-23
> 03:59Z), which is NOT an ancestor of that commit but is an ancestor of main -- so a taped
> forward now runs `ttnn.softmax` and `ttnn.layer_norm` on the HOST in float64.
> Measured consequence at crop 384: a taped trunk cycle reads **251.66 s** where the banked
> step records **3.415 s**, on two independent uncontended runs. `autograd.backward` opens the
> same scope for itself, so the 456.668 s backward moved the same way.
> **And the cause is now bounded without a card.** The residual over the identical-route arm is
> 247.8 s, and the host float64 softmax alone is a **43.08 s floor of CPU arithmetic plus
> 173.95 GB of host<->device traffic** per trunk cycle -- 96 softmaxes over `[384,4,384,384]`,
> 21.74 G elements, at a contention-surviving 1.9813 ns/element measured on pc with no device
> open (`perf/of3t_tapedfwd/exactprice.py`). The layer norm is not in that and its per-element
> rate is 2.2x. So it is the exactness, not the tape's node recording, and **it is a knob rather
> than a kernel.** Two prices already banked for the same lever and worth not re-deriving:
> `of3t-f64softmax` has **166.8x on the op** and **1.47x on a whole diffusion gradient arm** --
> neither transfers to the trunk, because the gap between them is the softmax's share of the
> scope and a 48-block pairformer cycle is almost nothing but softmax.
> **So the 58-67x, the 98.6 % / 1.4 % partition and every forward/backward share below are
> stale until `fullstep.py` is re-run on main.** The hardware roofs, the tt-train inventory and
> the throughput argument are unaffected. Detail and the git checks: `state/of3t-tapedfwd.md`.

Established with Moritz in conversation 2026-09-25 14:00-16:00 CEST. Every row below reads this
before it starts. Moritz: *"doing the same great job we did for forward also for backward ... using
what already exists in the tenstorrent repositories. adapting. taking inspiration from gpu
implementations ... full speed. with multiple agents."*

## 1. The gap is software, and how much of it

| | |
|---|---|
| OF3T training, measured | **58-67x**, whole step vs whole step, crop 384, 466.70 s vs 7-8 s. A FLOOR: the TT side does strictly less work (`of3t-gpugap`) |
| Partition | **98.6 % ported / 1.4 % unported**, timed per part with `ttnn.synchronize_device`, six parts summing to the recorded step exactly |
| Blackhole compute roof | **115.685 TFLOP/s** measured, dense 4096 cube |
| Machine balance | **247-338 FLOP/byte** depending on the roof pair, so a bandwidth roof near **440 GB/s** |
| DRAM-bound reality | the 16-head pair matmuls (8,448 calls each) carry AI ~101 against balance 247 -> **genuinely DRAM-bound, capped near 43 TFLOP/s** |
| H200 | ~989 TFLOP/s BF16 dense, 4.8 TB/s, balance ~206 FLOP/byte |

So the hardware ratio is **~8.5x compute / ~11x bandwidth**, and this workload sits on the memory
side. **58-67x total over ~9x hardware leaves ~6-7x that is software.** Moritz's read is right: not
a fundamental limit. But 5x chip-to-chip is BELOW the silicon floor — do not promise it.

## 2. Why the GPU side is fast, and it is not because they wrote more kernels

They inherited their backward:

* **cuBLAS/cuDNN** — a matmul's backward is two more matmuls on the same tuned GEMM, so a
  matmul-heavy backward is automatically as fast as its forward. Nobody writes anything.
* **FlashAttention ships a hand-written fused backward.** One hard kernel, written once, inherited
  by everything downstream.
* **XLA / torch.compile fuse the GRADIENT GRAPH automatically.** This is the big one for BindCraft 2,
  which is JAX: fusion is a compiler pass over a graph and the gradient is just another graph, so
  `jit` on the loss yields a fused backward for free.
* **OpenFold** additionally has community fused Evoformer attention (DeepSpeed
  `DS4Sci_EvoformerAttention`) with a backward.

**The asymmetry is structural: on GPU fusion is a COMPILER property and the backward is DERIVED; on
ttnn fusion is a HAND-WRITTEN KERNEL property and the backward does not come with it.** That is
precisely why Boltz-2 (forward only) is close and OF3T training is 58-67x.

## 3. What we already have, and what is missing

`tt_bio/kernels/` has ~11 fused kernels and **every one is forward-only** — `grep backward` across
the directory returns nothing: `triatt`, `triatt_sdpa`, `trimul_tail`, `swiglu_fused`,
`reblock_permute`, `reblock_permute_gated`, `rfd3_softmax`, `rfd3_bias`, `mm_split`, `page_copy`.

**But tt-metal's own training library already has eight backward kernels** —
`/home/moritz/tt-metal/tt-train/sources/ttml/metal/ops/`:

    sdpa_bw   ring_sdpa_bw   layernorm_bw   rmsnorm_bw
    polynorm_bw   cross_entropy_bw   silu_bw   swiglu_elemwise_bw
    (plus SDPA_OPTIMIZATION_PROPOSALS.md)

`sdpa_bw` is a real FlashAttention-style fused backward: it takes `intermediates` (max, 1/sum_exp)
from the forward, which is the right design. Two things stop it being a drop-in, both checkable:

1. **It returns `[grad_Q, grad_K, grad_V]` only.** It accepts an arbitrary `attn_mask` but treats it
   as a CONSTANT. Triangle attention's bias is not a mask — it is a learned projection of the pair
   representation whose gradient is how signal reaches the pair track at all. **`grad_bias` is the
   missing term**, and it is missing because a language model masks with constants and never needed it.
   It may be nearly free to emit: the kernel already holds the softmax intermediates.
2. **Not exposed to Python.** No `sdpa` in `sources/ttml/nanobind/nb_ops.cpp`; it is C++-internal to
   `ttml::autograd`, while tt-bio has its own Python tape in `tt_bio/autograd.py`.

`ring_sdpa_bw` is a multi-chip ring-attention backward — that is the data-parallel story, prebuilt.

## 4. The forward gives up a lot proportionally and almost nothing absolutely: 1.00508x

**PRICED AND CLOSED by `of3t-tapedfwd`, GO, 2026-09-25. Section rewritten rather than appended to,
because the earlier version led with a mechanism that turned out to be worth half a percent.**

**The mechanism is real and bigger than first described.** `ttnn.generic_op` has no tape entry
(`tt_bio/ops.py:72`), so every fused kernel declines under taping by design. Counted at runtime,
not read: **zero fused forwards execute under a tape**, and ten counters that are busy without one
go to exactly zero with one. Thirteen distinct call sites ask the gate **1,528 times per taped
trunk cycle**; only five are `generic_op` kernels, the other eight are L1 residency, an in-place
pair add and a deallocation the tape's backward forbids
(`perf/of3t_tapedfwd/fires.py`, which stubs `tt_bio.ops.taping` to record its caller's frame, so
it sees every gate and not only the ones keeping a counter).

**And it is worth 1.00508x.** Untaped with every lever, 1.2054 s; forced onto the taped route set,
3.9032 s — **3.2381x on the forward**, medians of 3 at 0.21 %/0.24 % spread, pc card 0, 1350 MHz
sampled DURING. But those routes sit in **3.415 s of a 466.702 s step**, so the tape gives up
**2.3604 s**. The ceiling needs no card to compute: the whole taped forward including the
diffusion in the same tape is **3.702 s**, so a forward that cost *nothing* would be **1.0080x**.
**A big ratio on a small slice.**

**Three consequences that bind the rest of the sprint:**

1. **Kernel authoring does not wait behind this.** The taped forward plus its diffusion is
   **0.79 %** of the step; the backward is **97.85 %**.
2. **The taping guards never touch the backward at all.** Every `with ag.tape():` closes before
   `backward()` is called, so the backward already runs with the full lever set. Any hope that
   fused forwards were silently costing the backward seconds is dead.
3. **The exact host float64 softmax and layer norm are the largest single line item this sprint
   has found, and they are a knob.** Bounded in the block at the top of this file. `backward()`
   opens the same scope, so this reaches the 97.85 % too. What is owed is one
   `exact_training(False)` arm on an uncontended card 0 to turn the floor into a split, and then
   an accuracy/perf decision: `of3t-stackexact` bought 0.9823x of the gradient bar with the
   exact stack against 1.4512x without. That decision is `of3t-orchestrator`'s, not a kernel
   row's.

`of3t-tapedfwd` landed one free fix on its branch (`ae7bb5b68`): the fused-HiFi arm was walking
`_sdpa_masked`'s ragged device pad and then the whole config ladder only to reach a `None` that
`triatt_sdpa.sdpa` returns on its first line under a tape. It declines at the arm now. Bit-exact,
no inference call moves.

**The lesson, and it is the one worth carrying: a located mechanism is not a win until it is
priced.** This section previously led with the mechanism and reordered a sprint around it. The
mechanism was correct, the count was larger than claimed, and the answer is half a percent.

## 4b. THE MEASUREMENT CARD IS THE SPRINT'S SCARCEST RESOURCE, and there is one of it

Established 2026-09-25 by `of3t-orchestrator` pass 448, after `of3t-lnbw` went BLOCKED and named
it. Verified with `tt-smi -ls`, not assumed. Every row reads this before asking for a card.

**This sprint's baseline is BLACKHOLE** — the 466.70 s crop-384 step, the 115.685 TFLOP/s roof and
the 1350 MHz clock rule. A timing on another board class is not comparable to any of them.

| host | board | usable for a sprint timing? |
|---|---|---|
| whglx | **Wormhole**, `tt-galaxy`, 32 chips | **No.** Board-insensitive work only |
| pc card 0 | **Blackhole `p150a`**, custom 130-core firmware | **Yes, and it is the only one** |
| qb1 | — | host does not answer ssh, `state/qb1-offline` set, all four cards blocked |
| qb2 card 0 | Blackhole p300c | **No** — reads 800 MHz against 1350 on the others, so a timing is an artifact; no reset path while card 1 holds a live arm |
| qb2 1/2/3 | Blackhole p300c | held by BCX rows, which have a 28 Sep deadline |

**So the kernel rows serialise on one card**, in value order: `of3t-bwattrib` (J0) →
`of3t-lnbw` (J1) → `of3t-softbw` (J2) → `of3t-wheelbw` (J4). The last two are parked behind it
rather than left to burn passes on a card they cannot get.

**The generalisable half, which is the part worth keeping:** a campaign whose baseline is one
board class must dispatch its device rows PINNED to that class, never `card=-`. `card=-` resolves
to whatever is free, and what is free is usually the big idle Galaxy that cannot produce a
comparable number. Four rows went to Wormhole for exactly that reason. **And the split that keeps
them productive meanwhile: a formula-level check and a negative control are board-insensitive, so
they can run anywhere; the graded VJP number and every timing number are Blackhole-only.** Say
which half you have.

## 4c. WHAT THE SPRINT CAN DELIVER AT ITS CEILING: 3.32x, which is 34 % of the way

Computed 2026-09-25 pass 449 after J3 was retired.
`perf/of3t_orchestrator/bwd/sprint_ceiling.py` — run it, do not quote this prose. It is a
**CEILING**, not a projection: every job is credited its own best published number and J0 is
credited a full solve of the per-verb cost (2.107 ms → 0.24 ms).

| scenario | step | speedup | vs H200 |
|---|---|---|---|
| kernels alone, J0 fails | 372.50 s | **1.25x** | 46.6-53.2x |
| J0 alone, full solve | 151.24 s | **3.09x** | 18.9-21.6x |
| J0 + kernels, repriced per R206 | 140.52 s | **3.32x** | 17.6-20.1x |

Reaching the silicon floor (8.5-11x) means a step of **60-88 s**, a **5.3-7.8x** software win.
**The whole remaining list delivers 3.32x at its ceiling — 34 % of that.**

**And here is where most of the difference sits: the 100.668 s of NON-VERB backward time.** The
partition is 466.702 = 10.034 non-backward + 356.00 verb calls + **100.668 that is not verb
calls**. Every job on the JOBS list attacks the 356.00 s. **Nothing attacks the 100.668 s, and it
is 21.6 % of the step, larger than J1 + J2 + J4 combined (94.2 s).** It is assigned to
`of3t-bwattrib` as a subtraction from the histogram it is already collecting rather than as a new
row, because there is one Blackhole card and it holds it.

**The honest statement of the sprint's scope, then:** it can roughly triple the step and it
cannot reach the hardware floor, and the reason is a fifth of the step that nobody has yet
partitioned. That is not a reason to stop — 3.32x is real and the residual may itself be
tractable once it is named — but no number this sprint publishes should imply 6-7x is in reach
from the job list as it stands.

## 5. Throughput, not latency — and the metric we have never computed

Training time-to-model is samples/sec, not step latency. Splitting one step across chips to cut
latency is the wrong tool; **replicating the step across chips to raise samples/sec is the right
one** (one all-reduce per 466 s step — cheap, and `ring_sdpa_bw` exists).

    1 p300c  = 1 sample / 466.7 s = 0.0021 samples/s
    1 H200   = 1 sample / 7.5 s   = 0.133  samples/s   -> ~62 chips to match one H200 today
                                                       -> ~9 chips with the 6-7x software fix

Scale-out does not beat the silicon; it buys the right to compare BOXES. **Nobody has computed
samples/sec per box, per dollar or per watt, and that is the number that answers "competitive".**

BindCraft 2 design needs none of this: it is embarrassingly parallel and already measured —
`bcx-concurrent`, two arms on two chips, **4.6 % cost per arm for 1.91x**, beating the 1.79x BC2's
own docs report from packing 7 workers on a GH200.

## 6. BC2's own ceiling, measured today

`bcx-gpuref`: **89.7x phase-matched** (replacing the 84x lower bound — worse, not better), H200
gradient step **0.6958 s** (p50, n=1698; the 0.511 s the campaign had been quoting was 1.36x too
small with 45 % stated uncertainty), and **the gradient phase is 79.8 % of a BC2 cycle, so porting
it alone caps end-to-end at about 5x.**

## 7. Rules for this work

* **UNIFIED, NOT PER-MODEL** (standing). A backward kernel is engine-level and serves Boltz-2, OF3T,
  BC2 and RFD3. No `of3t_sdpa_bw`.
* **Adapt before you author.** A row that writes a kernel tt-train already has must say what it
  tried and why it did not fit.
* **Take inspiration from the GPU implementations explicitly** — FlashAttention's backward
  recomputation, DeepSpeed's Evoformer kernel, XLA's fusion over the gradient graph. Read them; say
  what you took.
* **Batch the gates** (standing, TASKS.md). Land several levers, then one strong accuracy gate over
  the batch. The fp32 CPU diffusion reference is not reproducible run-to-run (7.17 A vs 11.04 A on
  the same design), so consulting it every micro-step produces noise, not safety.
* **Never a speedup that skips the model's own work.** Bit-exactness is not required; the accuracy
  bar is.
* **Say the axis on every number.** `of3t-gpugap` struck two of three inherited ratios because their
  axes did not survive contact: one was a different model, one had no GPU on either side.
