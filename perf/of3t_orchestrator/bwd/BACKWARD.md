# The backward pass gets the same treatment the forward got

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

## 4. The forward is NOT at full perf on the taped path, and it is ONE cause

**Established by `of3t-orchestrator` at pass 445, 2026-09-25, by reading the tree. Do not
re-derive it; do price it.** This section previously said "one taped call switched fused triangle
attention off for the whole process", a cached state-dependent refusal. That framing is wrong and
it understates the problem by an order of magnitude.

**`ttnn.generic_op` has no backward.** Every one of tt-bio's fused kernels is driven through
`ttnn.generic_op` — a Python program descriptor over raw `.cpp` kernel sources, which is why
`tt_bio/kernels/` holds `.cpp` files and no build system. So every one of them declines under
taping, by design, with the comment saying so in each case. Thirteen sites in eight modules:

    tt_bio/triatt_qkv.py       74, 179, 276, 390   -> "the three composed ops run instead"
    tt_bio/reblock_permute.py  373, 597, 877       -> "every caller falls back to the unfused
                                                       transpose/permute, which the tape follows"
    tt_bio/triatt_sdpa.py      340, 486            -> "the stock fused SDPA verb is taped"
    tt_bio/softmax_generic.py  369, 514            -> "the composed path runs instead"
    tt_bio/swiglu_fused.py     96                  -> returns the string "taping"
    tt_bio/trimul_tail.py      232
    tt_bio/mm_dualnoc.py       87                  -> falls back to `minimal_matmul`, a taped verb
    tt_bio/eltwise_fusion.py   80, 104, 115        -> FUSE_MASK_ADD and FUSE_NORM_RESIDUAL both off

Residency degrades under taping too, separately from fusion: `tenstorrent.py:9016` selects
`DRAM_MEMORY_CONFIG` where an untaped run gets `L1_MEMORY_CONFIG`, and `tenstorrent.py:4943` drops
the pair-projection L1 output. On a workload already established as DRAM-bound that is its own
cost line, and it is not a kernel-authoring problem.

**So a taped training step runs a decomposed, DRAM-resident model, not the shipped one.** A slice
of the 6-7x software gap is kernels we already have, switched off — before a single backward
kernel is written. The counter-example in the same tree tells you the shape of the fix:
`triatt_sdpa`'s decline comment says *"the stock fused SDPA verb is taped"*, and PTX's shared SDPA
backward pairs that production fused forward with `autograd.triangle_attention`'s chunked
recompute. **A fused kernel survives taping when it is a taped ttnn VERB with a registered
backward, and not when it is a tt-bio `generic_op`.** That is the architectural question this
sprint actually has to answer, and it reframes the job list: for each of the eight modules, either
register a backward for the `generic_op` pair, or re-drive tt-train's backward source through
`generic_op` and register the pair, or establish that the composed fallback is already at parity
and the guard costs nothing there.

**And the plumbing is NOT the blocker — established pass 445 by reading `tt_bio/autograd.py`.**
The tape is not a verb registry. `autograd._tape(out_value, parents, make_fn, reads=None)` wraps
any forward value with any hand-written VJP closure, so a `generic_op` result can carry a backward
today with no new infrastructure, no nanobind and no C++ build. The pattern is already in
production and its own docstring says so: `autograd.triangle_attention` takes a `value=` argument
that *"hands in a forward that has already been computed, and it is what lets the shipped fused
SDPA share this backward instead of getting a second copy of it... which is how the production
forward and this backward end up in one node."* So the fused kernel runs, the authored backward is
taped beside it, and `generic_op`'s own lack of a backward never comes up.

**But it is a ONE-OFF, and that is the real infrastructure job.** `value=` exists on exactly one
autograd op and is used at exactly one call site (`taped_ttnn.py:862`). Generalising that seam —
a precomputed-forward argument on the autograd ops the other seven modules would need, several of
which have no autograd counterpart at all today — is a small Python job inside `autograd.py`, not
a bridge to `ttml::metal`. Size it as such.

Consequence for the job list: **each of the eight modules is "author or adapt the backward maths,
then wire the fused forward through the `value=` seam", not "expose a C++ op to Python".**
`reblock_permute`'s backward is an inverse permutation and may be nearly free; `swiglu_fused` has
tt-train's `swiglu_elemwise_bw` to adapt; `triatt_qkv` and `softmax_generic` are where the maths
is real. That ordering is `of3t-bwsurvey`'s to establish, with `of3t-tapedfwd`'s seconds as the
weights.

**What is still owed, and it is a measurement, not a code read:** the seconds. `of3t-tapedfwd`
counts which of these fire at runtime in a real crop-384 taped step and prices the decline per
family inside the 466.70 s. A code read says the guard exists; only the run says what it costs,
and the order of the whole sprint depends on that number.

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
