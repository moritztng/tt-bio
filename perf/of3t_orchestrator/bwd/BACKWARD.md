# The backward pass gets the same treatment the forward got

**ENGINE-LEVEL, not OF3T's.** It lives under `state/of3t/` for historical reasons; the kernels it
describes serve Boltz-2, OF3T, BC2 and RFD3. `bcx-orchestrator` is briefed on it as of 2026-09-25
16:2x. For BC2 it matters MORE, not less: the gradient phase is 79.8 % of a BC2 cycle.

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

**Two consequences that bind the rest of the sprint:**

1. **Kernel authoring does not wait behind this.** The taped forward plus its diffusion is
   **0.79 %** of the step; the backward is **97.85 %**.
2. **The taping guards never touch the backward at all.** Every `with ag.tape():` closes before
   `backward()` is called, so the backward already runs with the full lever set. Any hope that
   fused forwards were silently costing the backward seconds is dead.

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
