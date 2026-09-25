# of3t-bwsurvey — the backward kernels that already exist, and what each one is worth

TASK TYPE: VERIFY/BENCHMARK (a survey that sizes, so every claim carries its source) |
PLAYBOOKS loaded: VERIFY/BENCHMARK + ACCELERATE | memories read: `state/of3t/BACKWARD.md`,
`a-harness-step-is-not-the-loops-own-round`, `tape-recomputes-what-a-fused-kernel-computed`,
`unified-solution-not-per-model-patches`, `a-ratio-is-not-a-defect-until-the-references-own-ratio-
is-measured`, `no-speedup-by-skipping-the-models-own-work`, `bit-exactness-not-required-accuracy-
bar-is`.

Row scope: reading only. No kernel is written here, no default moves, no device is opened (this
row holds no card lease). Every number below is either quoted from a named artifact or derived
from source with the derivation shown.

**The sizing coefficient, used everywhere below.** `of3t-stepfloor` pass 2, qb2 card 3, probe off,
renorm lever off, AICLK median **1350 MHz** sampled DURING the arm over 69 samples
(`perf/of3t_stepfloor/out/d164_probeoff_renormoff_384.json`):

    forward    4.04 s /  84,996 ttnn verb calls = 0.048 ms per call
    backward 356.00 s / 168,922 ttnn verb calls = 2.107 ms per call    44.3x

and at step scope (`of3t-gpugap` PARTITION, arm B rep 2) the backward is **456.668 s of 466.702 s,
97.85 %**. So a backward verb call costs **2.107 ms**, and that one number prices every job here:
a fused kernel's win is the calls it deletes times 2.107 ms, plus whatever the fused call itself
costs less than the calls it replaced.

INVENTORY:

One row per (tt-train backward, tt-bio forward) pair. Source of truth
`/home/moritz/tt-metal/tt-train/sources/ttml/metal/ops/`; our side `tt_bio/kernels/` (eleven fused
forwards, `grep -rn backward` returns nothing) and `tt_bio/autograd.py` (2,785 lines, every
backward is a Python closure over ttnn verbs).

**The brief says eight. There are nine, and the ninth is the cheapest win on the page.**
`ops/softmax_backward/` exists beside the eight named and is not in `state/of3t/BACKWARD.md`:

    ttnn::Tensor softmax_backward(softmax_output, grad, dim, sub_core_grids)
    "output = y * (grad - sum(y * grad, dim, keepdim=True)). Supports last dimension only."

That is `tt_bio/autograd.py:138 softmax_bw_inner` plus the two verbs its three callers wrap round
it, character for character, in one op instead of four to six.

| tt-train backward | our forward | verdict | why, and the shape |
|---|---|---|---|
| `softmax_backward` | `autograd.softmax`, `triangle_attention`'s inner, `taped_ttnn._v_softmax` | **ADAPTS, small** | Identical expression. Two deltas: it is last-dim only (all three of our callers are `dim=-1`), and it does **not** carry `SOFTMAX_BW_RENORM`, the `divide(inner, sum(y))` correction that is default-on since pass 274. Renorm costs 1.32 s of 357.32 (`of3t-stepfloor`), so the question is whether the gradient needs it, not whether it is affordable. Shapes: `[B,H,n_q,n_k]`, at crop 384 `[227,4,384,384]` bf16 = 267.8 MB per block |
| `layernorm_bw` | `autograd.layer_norm` (forward is `ttnn.layer_norm`, production's own) | **ADAPTS — biggest job on the list** | Returns `{dx, dgamma, dbeta}` and computes the per-row dgamma/dbeta partials IN the kernel; `layernorm_bw.cpp:25-43` then does one `ttnn::sum` at `ComputeKernelConfig::precise()` over rows. Needs `mean` and `rstd` from the forward — **and our backward already recomputes both** (`autograd.py:1238-1243`), so the blocker that ruled out `moreh_layer_norm_backward` (documented in that same docstring: moreh needs mean+rstd and then refuses bfloat8_b) does not bind here. Shapes: pair `[1,384,384,128]`, single `[1,1,384,c_s]` |
| `rmsnorm_bw` | nothing | **DROPS, and nothing calls it** | No bio model in the engine uses RMSNorm. Boltz-2, OF3T, BC2 and RFD3 are all LayerNorm/AdaLN. Keep on the shelf for a future LLM-shaped port; it is not OF3T work |
| `polynorm_bw` | nothing | **no pair** | Same |
| `swiglu_elemwise_bw` | `tt_bio/kernels/swiglu_fused/` | **DOES NOT PAIR — different op** | Ours is a *matmul*-fused SwiGLU (`dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `matmul_dataflow_common.hpp` — the fused-matmul template), theirs is the elementwise tail only: `swiglu_elemwise_bw(linear1, gate, dL_dprod) -> {dL_dlinear1, dL_dgate}`. It pairs with the *tail* of our kernel, not our kernel. Useful, but it leaves both matmul backwards outside, which is where the FLOPs are |
| `silu_bw` | `autograd.silu` (6 verbs) | **ALREADY IN THE WHEEL — do not bind ttml's** | `ttnn.silu_bw` is bound in `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp` and ships in `ttnn==0.68.0`, the version `pyproject.toml:107` pins. Binding tt-train's copy would be a second implementation of an op we can already call |
| `cross_entropy_bw` | `tt_bio/train/losses.py` — **numpy, zero `ttnn`** | **ADAPTS, but it is 0.67 % of the step** | The distogram head is a cross entropy. `of3t-gpugap`: loss heads 3.135 s, of which 3.128 s is host arithmetic. Real, cheap, and it caps at 3.1 s |
| `sdpa_bw` | `tt_bio/kernels/triatt_sdpa/` + `autograd.triangle_attention` | **ADAPTS, hard — see GRADBIAS** | Different lineage on both sides. Ours is a transcription of **ttnn's stock** `sdpa_program_factory.cpp` at the `v0.68.0` tag (`tt_bio/triatt_sdpa.py:1-40`) with a persistent-mask edit worth 2.431x at 512 aa; theirs consumes ttml's own `sdpa_fw` intermediates. Neither forward feeds the other today |
| `ring_sdpa_bw` | nothing | **DROPS IN when there is a ring** | Multi-chip ring-attention backward, prebuilt. It serves the replicate-for-throughput story (`of3t-throughput`), not a single-chip step |

**And the inventory has a second half the brief did not name: ttnn's own library.** Not tt-train —
the wheel we already import. `ttnn/cpp/ttnn/operations/` ships **72 unary `*_bw`**, **21 binary
`*_bw`**, and 17 `moreh_*_backward` ops, all nanobound and callable from Python today with no build
work at all: `mul_bw sub_bw add_bw div_bw concat_bw sigmoid_bw silu_bw relu_bw rsqrt_bw exp_bw
log_bw sqrt_bw gelu_bw`, `moreh_softmax_backward`, `moreh_layer_norm_backward`,
`moreh_matmul_backward`, `moreh_sum_backward`. Nine of `autograd.py`'s taped backwards
(`mul scale add relu sigmoid silu reshape narrow concat`) are hand-composed out of 1-6 ttnn verbs
where a single bound `*_bw` exists. Each is a one-line substitution with a PCC check, not a kernel.

GRADBIAS:

**Derivable, and it is one `pack_tile` from a register the kernel already holds.** Not new maths.

`sdpa_bw_kv_compute_kernel.cpp` states its own algebra in its header comment (lines 34-62):

    dP = dO @ V^T ;  u = rowsum(dO ⊙ O) ;  dS = P ⊙ (dP - u) ;  dV = P^T @ dO ;  dK = dS^T @ Q / √d

`compute_grad_scores` in `sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp` builds exactly
that in DST: `sub_tiles_bcast_cols(dP, u) -> grad_reg`, `mul_binary_tile(grad_reg, P)`, and then as
its **last instruction before packing**, `mul_unary_tile(grad_reg, scaler_bits)`.

The forward applies the scale to QK^T and adds the mask after it (`apply_mask_on_reg`, same file,
lines 59-68: `mul_unary_tile(scores, scaler)` then `add_binary_tile(scores, mask)`). So the
composite is `S = scale·QK^T + bias`, and therefore

    dL/dbias = dL/dS = P ⊙ (dP - u)     — grad_reg BEFORE that final mul_unary_tile
    dL/d(QK^T) = scale · dL/dS          — grad_reg AFTER it, which is what dK already uses

**grad_bias is the value in `grad_reg` one instruction earlier.** Cost in the compute kernel: one
extra `pack_tile` into a new CB. That is the whole of the maths.

**The cost is not the maths, it is where the term has to go.** Triangle attention's bias is
`[1, H, N, N]` broadcast over the leading axis B, and B is the pair tensor's own first index — at
crop 384, B = 384 independent attention problems sharing one bias
(`autograd.py:1887-1900`, `taped_ttnn.py:903`). So

    grad_bias[h, j, k] = Σ_i dS[i, h, j, k]

and dS at full extent is `384 · 4 · 384 · 384 · 2 B` = **452.98 MB per triangle-attention call**,
against the 1.18 MB the reduced answer occupies. Writing dS out and reducing it in ttnn costs
~0.9 GB of round-trip traffic per call — 2.1 s per call at the 440 GB/s bandwidth roof
(`state/of3t/BACKWARD.md §1`), against a whole-op arithmetic cost of 72.5 GFLOP ≈ 1.7 ms at the
43 TFLOP/s DRAM-bound cap. **Materialising dS is three orders of magnitude the wrong answer.**

So the job is the reduction, not the gradient: accumulate grad_bias in L1 across the i axis inside
the kernel and write it once. The current KV work split assigns contiguous `global_row_idx` to a
core, and rows are i-major (`row = i·Ht + k_tile`), so one core sees one i and many k tiles —
precisely the wrong axis. Strided assignment (a core owns one k tile and strides by `Ht` across i)
puts the whole i sum inside one core. The L1 accumulator is `Ht` tiles per head: at crop 384,
12 tiles × 2 KB = **24 KB**, trivial against 1.5 MB per core. dK and dV stay correct under the
re-split because both are per-i anyway; only the loop order changes.

**Two other gaps between their sdpa_bw and triangle attention, both real:**

1. **Their mask is binary, ours is an additive learned bias.** `apply_mask_on_reg` transforms a
   1/0 mask to 0/−inf (`add_unary_tile(mask, minus_one)`, `mul_unary_tile(mask, custom_inf)`).
   Triangle attention's bias is a linear projection of the pair representation, a real number.
   A fourth `AttentionMaskType::AdditiveBias` that skips the transform and just adds is a few
   lines, needed symmetrically in `sdpa_fw` and `sdpa_bw`.
2. **Their forward is not our forward.** `sdpa_bw` needs `intermediates` = the per-row logsumexp
   tile, fp32, from ttml's `sdpa_fw`. Our production forward is the stock ttnn SDPA transcription
   and returns the output only. Flash attention computes the running max and sum internally, so
   emitting them is a writer change to `tt_bio/kernels/triatt_sdpa/`, not new arithmetic — but it
   is a change to the kernel that serves 560 of 560 triangle-attention calls on the Boltz-2 512 aa
   fold (`perf/sizegate/baseline/census_boltz2_512_p300c.json`), so it is release-gated.

   There is also a scale-convention trap already root-caused in `taped_ttnn.py:866-880`: the
   shipped fused kernel computes `softmax((q·kᵀ + mask)·scale)·v`, mask **before** scale, and
   production compensates at `tenstorrent.py:7205`. ttml's kernel is mask **after** scale. Getting
   this backwards is a wrong gradient the forward agrees with, measured: error flat at 3.3e-02
   under the right reading, growing 3.3e-02 → 8.3e-01 with mask magnitude under the wrong one.

**And the honest frame: we are not missing a bias gradient. We already compute one.**
`autograd.triangle_attention`'s backward (`autograd.py:1975-2010`) emits `dbias` today, as
`ttnn.sum(ds, dim=0, keepdim=True)` on the broadcast path and `ttnn.clone(ds)` per trunk otherwise,
with the two cases distinguished because the atom transformer's local windows carry a different
bias per trunk and summing there would average the windows together. It is correct and it is
gradient-checked (`perf/hallgrad/gradcheck.py --cases triatt_chunked`, which differences two
chunkings rather than trusting one). What it is not is fused. **The job is to fuse an existing
correct gradient, not to derive a missing one**, and that reframing is what stops a kernel row
spending its first week on algebra.

EXPOSURE:

A `ttml::metal` op reaches `tt_bio/autograd.py` in one of three ways, and they differ by an order
of magnitude in cost.

**Route A — it is already in the wheel (cost: zero).** Everything under
`ttnn/cpp/ttnn/operations/`: the 93 eltwise `*_bw`, `moreh_softmax_backward`,
`moreh_layer_norm_backward`, bound in `moreh_nanobind.cpp:56,70` and the eltwise nanobind files,
shipped in `ttnn==0.68.0`. `import ttnn; ttnn.silu_bw(...)`. No build, no gate. Nine of our taped
backwards can move here this week.

**Route B — bind the ttml op into the `_ttml` module (cost: a tt-metal source build, and a real
hazard).** `sources/ttml/nanobind/__init__.cpp:30` declares `NB_MODULE(_ttml, m)`;
`sources/ttml/CMakeLists.txt:458` builds it with `nanobind_add_module` against the `ttml` static
library. `nb_ops.cpp` is 525 lines and exposes `layernorm.layernorm`, `rmsnorm.rmsnorm`, `swiglu`,
`cross_entropy_loss` — and **no sdpa at all**, confirming BACKWARD.md §3.2.

Two things make this more than "add ten lines to nb_ops.cpp":

* **What it exposes is the wrong tensor type.** Those entries bind `ttml::ops::*`, which take and
  return `ttml::autograd::TensorPtr` and push nodes onto **ttml's own tape**
  (`ops/layernorm_op.cpp:102-107` calls `tensor->add_grad(res[0])`). Two autograd systems in one
  process, neither aware of the other. The right binding target is the `ttml::metal::*` layer
  underneath, whose signatures are plain `ttnn::Tensor` in and out — `layernorm_bw`,
  `softmax_backward` and `sdpa_bw` all qualify, so this part is genuinely ten lines each.
* **`_ttml` and the wheel's `_ttnn` are two builds of the same C++ runtime.** `pyproject.toml:107`
  pins `ttnn==0.68.0`; `_ttml` links tt-metal built from source at `7a6f782e3f8`. Loading both into
  one interpreter means two copies of the tt-metal device runtime, two device managers, two
  allocators. That is the real cost of Route B and it is not a kernel problem: it is
  "tt-bio builds tt-metal from source", which the `tt-metal-source-build` skill already covers for
  qb2 but which no tt-bio release has ever required.

**Route C — upstream the op into `ttnn::experimental` and take it in a wheel bump (cost: weeks of
calendar, zero of ours).** `ttnn-author-custom-op` is the written path. This is how a kernel that
serves Boltz-2, OF3T, BC2 and RFD3 should eventually land, and it is a dependency major-bump, so
release-gated either way.

**Recommendation: Route A now, Route B behind an env flag for development, Route C for the ship.**
A kernel row that needs `sdpa_bw` on a device this month builds tt-metal from source on qb2 and
imports `_ttml` in a probe script — it must not put a source build on tt-bio's dependency list.

GPUREAD:

What transfers, with the file or paper, and what does not.

**FlashAttention's backward (Dao et al., *FlashAttention-2*, arXiv:2307.08691 §2.2; kernel
`csrc/flash_attn/src/flash_bwd_kernel.h`).** The transferable idea is that the backward stores only
the per-row logsumexp and **recomputes** `P = exp(S − lse)` on chip rather than reading an
`O(N²)` probability tensor from DRAM. tt-train already took it: `sdpa_bw_kv_compute_kernel.cpp`
recomputes `Q@Kᵀ` and applies `apply_softmax_statistics_on_dst` with the stored statistics, and the
comment there — *"scores are still in DST at full FP32 from the matmul, apply exp(S − lse) directly
on DST — no CB roundtrip, no TF32 truncation"* — is the recomputation trick plus a precision
argument.

**So does tt-bio.** `autograd.triangle_attention`'s docstring is explicit that it never holds the
scores and that at 800 aa with 8 heads the retained `[S,H,S,S]` bf16 tensor would be 8.19 GB against
10.24 MB for one leading-axis chunk. It also states the one real difference from FlashAttention,
and it is a genuine simplification, not an oversight: **triangle attention's leading axis is already
B independent attention problems, so chunking B and the query axis while keeping every key gives an
exact complete softmax per chunk and needs no running row statistics at all.** FlashAttention chunks
KEYS because it has one big problem; we have S small ones. What we take from FlashAttention is
recomputation. What we do not need is its rescaling bookkeeping — which is also why the `lse`
intermediates `sdpa_bw` demands are an *interface* requirement of their kernel, not a mathematical
one of ours.

**DeepSpeed `DS4Sci_EvoformerAttention` (`deepspeed/ops/deepspeed4science/`, DS4Sci; OpenFold call
site `openfold/model/primitives.py:693-756`, verified in `/home/moritz/.coworker/ref/openfold`).**
The transferable fact is the *signature*:

    _deepspeed_evo_attn(q, k, v, biases)
        biases: List of biases that broadcast to [*, H, Q, K]

An attention kernel written for structure prediction takes a **list of broadcast additive biases**
and differentiates through them, where one written for a language model (tt-train's) takes a single
0/1 mask and treats it as a constant. That difference is the whole of GRADBIAS above, and DS4Sci is
the existence proof that the fused form is buildable — the OpenFold wrapper also shows the cost of
getting there: lines 720-745 are pure shape gymnastics, reshaping `[*, H, Q, C]` into the kernel's
`[B, N, Q, H, C]` and casting to bf16, because the kernel's layout is fixed. Expect the same tax on
our side, and budget the permute rather than discovering it.

**XLA / `jax.jit` over the gradient graph (BindCraft 2).** The transferable fact is a negative one
and it is the most important line in this section. On GPU, fusion is a **compiler pass over a
graph**, and a gradient is just another graph, so `jit(grad(loss))` fuses the backward for free and
the backward inherits every forward optimisation automatically. On ttnn, fusion is a **hand-written
kernel property**, and a fused forward confers nothing on its backward. This is why we have eleven
fused forwards and zero fused backwards, and it is structural, not neglect.

What that implies for ranking is the opposite of the obvious: it is not that we must hand-write all
eight backwards to catch up. It is that **the GPU's advantage arrives as one property applied
everywhere, so matching it kernel-by-kernel is the expensive route and we should first check how
much of our gap is even shaped like a missing kernel.** Which is the next section.

JOBS:

**Read this first, because it reorders everything.** The 6-7x software gap is, at step scope,
456.668 s of backward that has to become roughly 68 s. `of3t-gpugap` already priced the arithmetic:
at the forward's own measured warm rate the backward's 168,922 verb calls would cost **8.03 s**;
they cost **356.00 s**. Even pricing a backward verb at 5x a forward verb leaves 40.1 s predicted
against 356.00 s measured, still 8.9x short. **~348 s — 97.7 % of the backward — is not the model's
arithmetic**, and this is a warm process with the JIT term already burned off and the D164 allocator
probe off.

Now put a fusion programme against that. To reach 6x by deleting calls alone, at the measured
2.107 ms per call, you must delete **142,382 of 168,922 backward verb calls — 84.3 % of them.**
No set of nine kernels does that: the backward's calls are spread across elementwise adds, slices,
concats, permutes, layout conversions and reductions, not concentrated in nine ops. **The kernel
list below is worth roughly 1.6x. The remaining 4x lives in the 2.107 ms itself, and nothing in the
record explains why a backward verb costs 44.3x a forward verb on operands of the same shape.**

That is not a reason to skip the kernels. It is the reason the first job is not a kernel.

**Method for every size below, stated once.** Verbs-per-backward come from an AST count of the
closures in `tt_bio/autograd.py` including one level of helper inlining (`softmax_bw_inner`,
`_sum_leading`, `_tree_sum`, `add_grad`). Invocations come from `of3t-trunkopclass`'s runtime census
("counted on the scored batch by instrumenting module invocation, not read off the source"):
TRI_MUL 384 fwd / 384 LayerNorm-bw / 0 softmax-bw; TRI_ATT 384 / 192 / 3,840; TRANS 384 / 624 / 0;
APB 192 / 96 / 48. The product is a **model, not a measurement**, and where it brackets rather than
resolves I say so. `of3t-stepfloor` records 2,473 tape nodes on the same configuration, which is the
cross-check that keeps the brackets honest.

**J0 — attribute the 2.107 ms. Not a kernel. Blocks nothing, outranks everything.**
Worth up to ~315 s, which is more than every kernel below combined times four. A backward verb at
0.24 ms (5x a forward verb, already pessimistic) puts the 168,922 calls at 40.1 s, the step near
51 s, and that alone is **9.1x**. The suspects are named and cheap to separate: `Tensor.evict`
moving activations host-ward over PCIe, `add_grad`'s `typecast`-to-fp32 on every second
contribution, `to_layout` conversions enforced at every fan-in (`autograd.py:428-436`), and the
allocation churn of a tape that frees and re-reserves per closure. One profiled backward with a
per-verb timing histogram answers it. **This is `of3t-intensity`'s or a new row's work, not a
kernel row's, and it should be dispatched today.**

**J1 — fused LayerNorm backward, from `ttml::metal::layernorm_bw`. The biggest kernel job.**
LayerNorm backward is **1,296 of the 2,473 tape nodes, 52.4 %** — the largest single class in the
backward by node count, larger than triangle attention by an order of magnitude. Per node we spend
16 direct ttnn verbs plus two `_sum_leading` calls for dgamma and dbeta, and `_sum_leading` is where
the cost actually is: on an fp32 input it runs `_tree_sum`, a halving tree of `ttnn.slice` +
`ttnn.add_` + `ttnn.deallocate` at ~5 verbs per level, ~13 levels on a pair-track `[147456, 128]`
flattening, then a second tree — **~96 verbs per `_sum_leading`, ~208 per pair-track node.** On a
bf16 input it falls through to a single `ttnn.sum`, **~24 verbs per node.**

The bracket is 31,104 to 269,568 verbs, and the upper branch is arithmetically impossible against a
measured 168,922 — which is itself the finding: **most LayerNorm backwards must be taking the bf16
fall-through today, and the fp32 ones are disproportionately expensive.** Even the floor is
**31,104 verbs = 65.5 s, 14.4 % of the backward.** `layernorm_bw` computes the per-row dgamma/dbeta
partials in the kernel and leaves one reduction, so the replacement is ~7 verbs per node ≈ 19.1 s.
**Saves at least 46.4 s; plausibly 150 s+ if the fp32 tree fires more than the floor assumes.**

Three things the row must carry. (a) `mean` and `rstd`: `ttnn.layer_norm` returns neither, which is
what ruled out `moreh_layer_norm_backward` — but our backward already recomputes both, so pass them
in and the blocker evaporates. (b) Precision: `layernorm_bw.cpp:29` reduces with `ttnn::sum` at
`precise()`, and `_tree_sum`'s own docstring measures `ttnn.sum` at **8.3e-4** relative L2 from
float64 against the tree's **7.7e-8** (`perf/bcx_reduce`). 8.3e-4 is 60x under the campaign's
5.0e-02 per-tensor bar, so this is a gate to check, not a blocker — and if it reads outside, keep
our tree on the partials and still bank most of the win. (c) `of3t-bwdaccum` found the four
LayerNorm affine leaves holding 92.68 % of the error mass while their own backward is correct to
1.4e-02 on the operands handed to them; the error is injected upstream by AttentionPairBias. So
this job must not be graded on those tensors' absolute error — it will look guilty for someone
else's reason.

**J2 — fused softmax backward, from `ttml::metal::softmax_backward` (or `ttnn.moreh_softmax_backward`
via Route A first).** 4 to 6 verbs to 1, at every softmax on the tape. The census counts 3,888
softmax backwards (3,840 TRI_ATT + 48 APB), which at ~5 verbs each is **19,440 verbs = 41.0 s**,
falling to ~3,888 calls = 8.2 s. **Saves ~32.8 s.** Two caveats, both cheap: last-dim only (all our
callers qualify), and no `SOFTMAX_BW_RENORM` — decide whether the renorm is load-bearing before
dropping it, since it went default-on at pass 274 for a reason and costs only 1.32 s.
**Try `ttnn.moreh_softmax_backward` from the wheel FIRST**: same expression, zero build cost, and if
its accuracy and speed hold, J2 ships without anyone touching C++.

**J3 — fused triangle-attention backward with grad_bias, from `sdpa_bw` + GRADBIAS above. The hard
one, and its size is the survey's one unresolved number.** Per chunked block the closure spends
~19 verbs (4 to recompute the scores, 6 matmuls, softmax backward, dbias sum, two deallocates) plus
~5 slices. The block count is where the two readings diverge: `_sdpa_chunking` with
`SDPA_SCORE_BUDGET = 256 MiB` gives **2 blocks** per call at crop 384 with H=4 (→ ~5,088 verbs over
96 taped calls = **10.7 s**), while the census's 3,840 TRI_ATT softmax-backwards over 96 calls
implies **40 blocks** (→ ~72,960 verbs = **153.7 s**, 34 % of the backward). J0's histogram settles
it in one run. Rank J3 above J2 if the census reading is right, below it if the budget reading is.
Either way this job is the largest kernel-authoring effort on the page: a new mask type, an lse
output on the shipped forward, a strided KV work split, an L1 bias accumulator, and it touches a
path serving 560 of 560 calls on the Boltz-2 512 aa fold, so it is release-gated end to end.

**J4 — nine one-line substitutions to bound `*_bw` ops from the wheel.** `mul scale add relu sigmoid
silu reshape narrow concat` in `autograd.py` are hand-composed from 1-6 verbs each where
`ttnn.mul_bw`, `ttnn.silu_bw`, `ttnn.sigmoid_bw`, `ttnn.relu_bw`, `ttnn.concat_bw` already exist and
ship in 0.68.0. Small individually, but they are on every node in the tape and the cost is one
afternoon plus a PCC check each. **Estimate 5-15 s. Do this while J0 is measuring.**

**J5 — `swiglu_elemwise_bw` against `swiglu_fused`'s tail.** Honest verdict: it pairs with the
elementwise tail only and leaves both matmul backwards outside, so it is worth a handful of verbs
per transition. Park it until J1-J3 land.

**J6 — `cross_entropy_bw` for the distogram head.** Caps at 3.135 s (`of3t-gpugap`), of which 3.128
is host numpy. Real and unglamorous; take it when someone is already in `tt_bio/train/losses.py`.

**J7 — `ring_sdpa_bw`.** Nothing to build for a single-chip step. It is `of3t-throughput`'s
prebuilt all-reduce path for replicating the step across chips; named here so no kernel row
re-derives it.

**Ranked, with what each is worth:** J0 (~315 s, not a kernel) ≫ J3 (10.7 s or 153.7 s, unresolved)
≈ J1 (≥46.4 s) > J2 (~32.8 s) > J4 (5-15 s) > J6 (≤3.1 s) > J5 > J7. J0 resolves J3's bracket as a
side effect, so it is also the cheapest way to finish ranking this list.

**Unified, not per-model** (standing): every kernel above is engine-level. `layernorm_bw` serves
Boltz-2's trunk, OF3T's pairformer, BC2's gradient step and RFD3. The fused triangle-attention
backward serves the same four. There is no `of3t_sdpa_bw` anywhere in this plan, and a row that
finds itself writing one has taken a wrong turn.

VERDICT: GO

The job list exists, it is sized, and it is ranked, so kernel rows can be briefed from it instead of
surveying again. Nine tt-train backward kernels are inventoried against our eleven forward-only
fused kernels, and the inventory found a tenth source nobody was counting — 93 eltwise `*_bw` plus
17 `moreh_*_backward` already bound and shipping in the `ttnn==0.68.0` wheel we import, which makes
J4 and possibly J2 zero-build work.

Three findings change what the campaign should do next.

**grad_bias is not missing maths.** It is `grad_reg` one instruction before
`compute_grad_scores`'s final `mul_unary_tile`, and we already compute the same term in Python in
`autograd.triangle_attention`. The work is the reduction over the broadcast axis — 452.98 MB of dS
per call at crop 384 against 1.18 MB of answer — which is a 24 KB L1 accumulator and a strided KV
work split, not an algebra problem.

**The biggest kernel job is LayerNorm, not attention.** 1,296 of 2,473 tape nodes, at least 65.5 s,
and `ttml::metal::layernorm_bw` returns exactly the three tensors we build by hand — with the
mean/rstd blocker that killed the moreh route already paid for, because our backward recomputes both
today.

**And the kernel programme is not the 6-7x.** At the measured 2.107 ms per backward verb call,
reaching 6x by deleting calls means deleting 84.3 % of 168,922 of them, which nine kernels cannot
do; the whole list is worth roughly 1.6x. `of3t-gpugap` already showed 97.7 % of the backward is not
the model's arithmetic. So the highest-value job on this page is J0, which is a measurement, and it
is a different row's. The kernel rows should start on J1 immediately — it is worth ≥46.4 s on its
own, it needs no new algebra, and it does not depend on J0's answer — while J0 resolves whether J3
is worth 10.7 s or 153.7 s.

No default moved, no kernel was written, and no number here was measured by this row: every figure
is quoted from `of3t-gpugap`, `of3t-stepfloor`, `of3t-trunkopclass`, `of3t-bwdaccum` or derived from
source at a named line, and the two figures that are models rather than measurements (J1's verb
bracket and J3's block count) say so where they appear.
