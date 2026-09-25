# bcx-predictor — BindCraft 2's own campaign, driven by a tt-bio predictor

CURRENT STATE ONLY. Superseded sections are replaced, not appended to.

Branch `wk/bcx-predictor`, qb2 worktree `/home/ttuser/.coworker/wt/bcx-predictor`, pushed at
`fd2f5eea7`. Base: main's tip, then `origin/wk/bcx-e2e`, then `origin/wk/bcx-ckpt`. `bf4b9f4f0`
was not an ancestor of `origin/wk/bcx-e2e`, so that merge was done here and is verified by SHA;
it is worth 3.17x and the section below is mostly about why. Release-gated, merges nothing.

The six BCX documents are on qb2 after all, in `/home/ttuser/.coworker/state/`, not the
`/home/moritz/.coworker/state/` tree the DONE_CHECK reads. Both exist on this host and only the
first is synced. `CMP-ANSWER.md`, `bcx-ckpt.md` and `bcx-orchestrator.md` are read and
reconciled below.

## MEASURED

MEASURED: one gradient step costs **23.733 s at n=224**, the bucket BindCraft 2's PD-L1 state
rounds to, on qb2 card 3 at AICLK median **1350** sampled during, against **177.14 s** for
BindCraft 2's own JAX predictor on the same box's CPU: a **7.46x** gap. At BindCraft 2's own 125
gradient steps a trajectory that is **0.82 h** of device time against **6.15 h** of CPU. Table
`perf/bcx_predictor/price_table.json`.

| arm | n | step | trunk fwd | trunk bwd | load | 125 steps |
|---|---|---|---|---|---|---|
| CPU, BC2's own predictor, `model_1_ptm` | 211 | **177.14 s** | - | - | 16.8 | 6.15 h |
| device, as BC2 hands it | 211 | 26.578 s | 4.504 | 20.450 | 13.4 | 0.92 h |
| device, rounded to the 32 bucket | 224 | **23.733 s** | 1.369 | 19.638 | 26.3 | 0.82 h |
| device, `bcx-e2e`'s size, this base | 256 | 36.370 s | 1.691 | 31.390 | 20.6 | 1.26 h |
| device, `bcx-e2e`'s size, its base | 256 | 115.33 s | 1.685 | 110.76 | 14.0 | 4.00 h |

Commands: `refprice.py --steps 3` (212.70 s cold, then 174.71 and 177.14 warm; design loss
10.473454 on all three, the determinism check) and `TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3
TT_BIO_LEASE_HOLDER=worker:bcx-predictor devprice.py time --n {211,224,256} --reps 3`.
`devprice.py` registers two states with `bcx-e2e`'s harness and changes nothing else.

**These reconcile with `bcx-ckpt`'s qb1 numbers rather than contradicting them.** `CMP-ANSWER.md`
builds its step from `bcx-ckpt`'s 13.40 s whole trunk gradient at n=256 on qb1 card 1 at loadavg
1.3-1.9, forward 1.68 s and backward 11.71 s. This row measured the same size on qb2 card 3 at
loadavg 20.6: **forward 1.691 s against that 1.68 s**, which is the device-bound half agreeing to
0.7%, and backward 31.39 s against 11.71 s, which is the host-bound half at roughly ten times the
load. So the two rows measure the same trunk and the difference is the box, exactly as
`bcx-e2e`'s 1.822x factor said it would be. **A step time from qb2 is an upper bound and is not
the chip-seconds figure CMP should be given**; the qb1-equivalent of the n=224 number is the one
worth quoting and this row has not been able to take it.

**`bcx-e2e`'s 115.33 s is the pre-fix step and any arithmetic on it is wrong by 3.17x.** Same
size, same host, same card, same harness, same process shape: 115.33 s without `wk/bcx-ckpt`,
36.37 s with it. The cost removed is a `gc.collect()` per recompute and a collect walks the whole
interpreter heap, which this process fills with jax, haiku, BC2 and torch at once. This row's own
earlier 4.00 h projection scaled that number and is retired.

**BindCraft 2 hands the predictor a size the card is bad at, and padding it further is faster.**
`padded_prediction_complex` (`af2.py:55`) buckets a chain carrying the DESIGN flag and pads every
other chain to `target_pad_length`, 0 for a single target. The PD-L1 binder goes 77 -> 96, the
115-residue target does not pad, and the total is **211, not a multiple of 32**. Padding the same
design up to 224 does strictly more arithmetic and is **3.29x faster on the trunk forward**
(4.504 -> 1.369 s), while carrying the highest load of the three device runs. The standing
32-token bucket rule reaches this row by a path nobody had looked at: the design chain is
aligned, the complex is not. Rounding to the bucket is one call in the predictor class.

**The step budget is BindCraft 2's own and is 125**, from `settings.py:604`
`DESIGN_STAGE_DEFAULT_ROUNDS = {screen 50, refine 25, anneal 45, harden 5, mutate 15}` at budget
1, `gradient_stage_rounds` dropping `mutate`. `examples/pdl1.json` overrides no step count.

**BindCraft 2 hands the trunk a masked fold on every trajectory, and the trunk asserts against
it.** `tt_bio/af2.py:24-28` states that the ttnn AF2 trunk serves only all-ones masks, and it is
enforced, not just documented: `af2.py:385` asserts `mask_2d` is all ones and `af2.py:509` asserts
`msa_mask is None`. The reason given there is that AF2 masks BOTH halves of the fused
triangle-multiplication projection where `TriangleMultiplication` masks only the `a` half, so a
masked fold needs that difference resolved before a mask can be honoured. Measured on BindCraft 2's
own padded state (`perf/bcx_predictor/mask_probe.json`): **19 of 211 residues masked, 9.0%**,
`seq_mask_all_ones` false. `padded_prediction_complex` pads the DESIGN chain to
`length_bucket_size` and `real_residue_weights` zeroes `seq_mask` on the pad.

**One campaign setting removes it.** `length_bucket_size` is a compile-reuse device, not part of
the model. At 1 the complex is unpadded: 77 + 115 = **192, seq_mask all ones, zero masked
residues**, and 192 is a multiple of 32 so this draw is tile-aligned too. The fold is the unpadded
one, which is the reference's own semantics; no step, recycle, loss or filter changes. Alignment
does not follow in general -- binder length is drawn 60 to 180, so n runs 175 to 295 and most draws
are not multiples of 32 -- but `af2.py:420` says ttnn masks its own tile padding, so that is the
3.29x performance question measured above and not a correctness one.

**n is drawn per trajectory and 211 is one draw.** The campaign announces `binder length 60 to
180, drawn per trajectory` against the 115-residue target, so n runs **175 to 295** across a set
and trajectory 1 of the live reference run drew 146. A single-n price is therefore indicative,
not the campaign's mean, and the acceptance run is what settles it.

## PROVED

PREDICTOR: `perf/bcx_predictor/ttbio_predictor.py`, class `TTBioAlphaFoldDesignModel`,
subclassing `AlphaFoldDesignModel` so only the Evoformer is replaced. BindCraft 2's own
`campaign.py` drove it UNMODIFIED: `run_arm.py --arm control` rebinds `campaign.AlphaFoldDesignModel`
(`campaign.py:262` is the only construction in the repository) and the campaign printed its own
header, its own seven filters and reached
`=== trajectory 1 | pdl1_denovo_l146_d2bfded440698f32 ===` with the stamp recording
`predictor: TTBioAlphaFoldDesignModel` -- `perf/bcx_predictor/control_smoke.json`. Conformance,
the `campaign.py:59` distogram negotiation, bucket rounding and a real two-chain `predict` with
seven metrics in 104.49 s are in `perf/bcx_predictor/predictor_check.json`. Nothing in
`trajectory.py`, `sequence_optimization.py`, `MPNN_stage.py`, `filters.py` or `rank.py` is edited.

TRUNK: `perf/bcx_predictor/device_trunk.py` makes tt-bio's trunk a differentiable JAX value --
a `jax.custom_vjp` over `jax.pure_callback` wrapping the extra-MSA and Evoformer stacks plus
`single_activations`, so BindCraft 2 can differentiate a non-JAX trunk inside its own jitted
program. The residual is an int32 token into a module-held tape, because JAX residuals must be
JAX types. `jax.grad` through it works: **g_msa cos 0.9925 rel L2 0.1226, g_pair cos 0.9965 rel
L2 0.0831** against a float64 torch reference of the same stack on the same inputs, k=1+2 at
n=64; a zero cotangent gives exactly zero; the production 4+48 shape at n=192 runs and is finite
(`perf/bcx_predictor/device_trunk_check.json`, qb2 card 3).

The test caught a bug rather than confirming a belief: the primal banked a tape nobody freed,
`live_tapes_after` reading 1. `predict` is forward-only and `MPNN_stage.py:125` calls it once per
validation model, so at `bcx-ckpt`'s 5.33 GB an Evoformer block that is an out-of-memory bug.
`_forward_notape` runs the ordinary inference path on raw ttnn tensors -- the `autograd.Tensor`
wrapper is only accepted inside `tape()` -- the counter is asserted now, and the primal went
22.21 s to 1.33 s.

RESOLVED, and it was my instrument: the device sits at **1.35x torch bf16's own distance**,
not the 166x the previous pass flagged. `afgrad.embed` returns activations already in
`model.trunk_dtype`, and that control cast them to float32 before handing them to the bf16 model;
`load_models` says in its own comment that the parameters stay float32 and the Linear casts the
weight to the ACTIVATION dtype per call, so float32 activations make a float32 arm. 0.00074 was
float32 wearing a bf16 label. Re-run with bf16 activations, both arms torch, CPU only, k=1+2 at
n=64 against float64 (`perf/bcx_predictor/bf16_control.json`):

| arm | g_msa rel L2 | cos | g_pair rel L2 | cos |
|---|---|---|---|---|
| torch bf16 | 0.0907 | 0.9959 | 0.0674 | 0.9977 |
| device | 0.1226 | 0.9925 | 0.0831 | 0.9965 |

1.35x on `g_msa`, 1.23x on `g_pair`. That agrees with `bcx-e2e` at the full stack on BindCraft 2's
real loss, 2.638 against 2.642 at n=256. The 166x is withdrawn and the mislabelled control is
deleted from the test rather than left to be rediscovered.

SPLICE: **done for the Evoformer, and BindCraft 2's own `predict` runs with 48 blocks on card 3.**
`perf/bcx_predictor/splice.py` patches `modules.py`'s `layer_stack` factory and swaps the one
call site named `evoformer_fn`, leaving the two template pair stacks and the extra-MSA stack in
JAX. Per-block interception is impossible -- `layer_stack` is a `jax.lax.scan`, so the body is
traced once rather than looped -- and `splice_probe.json` shows the closure names discriminate
the three sites uniquely, surviving `hk.remat`. Result (`splice_check.json`): `stacks_swapped`
[48], two device forwards for `design_recycles` 1, **zero tapes left live**, **25.7 s against
97.9 s** for the same class on BindCraft 2's trunk, AICLK median **1350** from 100 in-window
samples, min 1343.

`single_activations` (`modules.py:1599`) sits outside both stacks so it stays JAX-side and the
cut is `(msa, pair)`, not `bcx-e2e`'s `(single, pair)`. That does not reintroduce that row's
zero-gradient hazard, whose cause was BindCraft 2's TAIL reading only `single` and `pair`; here
JAX differentiates the Linear itself. **Templates are active** in BindCraft 2's monomer design,
which `bcx-e2e`'s `template=False` model did not exercise.

MASKDONE: **`tt_bio/af2.py` now takes a real MSA mask, and the template row is fixed.**
`AF2Attention` asserted `msa_mask is None` and `AF2EvoformerBlock` repeated it. The mask is AF2's
`1e9 * (mask - 1)` on the logits, built once per block in two layouts -- `[rows, 1, 1, n]` for the
row attention, whose keys are residues, `[n, 1, 1, rows]` for the column attention, whose keys are
rows. The row path folds it into the pair bias, which broadcasts over rows and becomes
`[rows, heads, n, n]`, 1.2 MB at n=192; the column path had no bias at all and now adds one after
the scale, unscaled, as AF2 does. **`_fp32_softmax_attention` is untouched** -- `af2.py:410` warns
it is shared with Boltz-2, Protenix-v2, OpenFold3 and ESMFold2.

Measured at the splice boundary against BindCraft 2's own Evoformer on its own captured inputs,
48 blocks, qb2 card 3 (`boundary_check_masked1.json`):

| | before | after | bf16 floor from the CPU ablation |
|---|---|---|---|
| msa row 1 | 0.542 | **0.160** | ~0.13 |
| msa row 0 | 0.164 | **0.133** | |
| msa both | 0.253 | **0.137** | |
| pair | 0.181 | 0.199 | ~0.145 |

**Row 1 improves 3.4x and lands near the predicted floor**, norm 1599.9 against BindCraft 2's
1614.4 where it was 1228.9.

**All three mask sites are now in, and every tensor beats the unmasked baseline.** The third is
`AF2MaskedOuterProductMean`, a subclass rather than a change to `tenstorrent.py`'s
`OuterProductMean`, which is also Boltz's, BoltzGen's, OpenFold3's, Protenix's, OpenDDE's, RF3's
and AF2-IG's. That class differs from AF2 in two ways that only show once a mask is not all ones:
it masks the `a` operand alone -- the same one-sided pattern `AF2PairBlock` documents for the
triangle multiplication -- where AF2 masks both, and it divides by a scalar depth where AF2
divides by `eps + sum_s m_si m_sj`, a matrix once the rows disagree about which residues are real.
`_small_depth` with `n_msa=1` returns exactly AF2's numerator including `proj_o`'s bias, so the
subclass masks both operands, asks for that, and applies the pair-wise divisor itself. **`eps`
stops being cosmetic**: it is dropped at an all-ones mask because bf16 rounds `eps + depth` back
to the depth, but the masked norm is 0 wherever both residues are masked in every row.

At the splice boundary against BindCraft 2's own Evoformer, 48 blocks, qb2 card 3:

| | no mask | attention masks | + OPM mask | bf16 floor predicted |
|---|---|---|---|---|
| pair | 0.181 | 0.199 | **0.151** | ~0.128 |
| msa row 0 | 0.164 | 0.133 | **0.125** | |
| msa row 1 | 0.542 | 0.160 | **0.151** | ~0.13 |
| msa both | 0.253 | 0.137 | **0.129** | |

Norms sit within 0.3% of BindCraft 2's. **`pair` lands at 0.151 against a 0.128 prediction**, so
it is close to but not at the bf16 floor and the residue is unexplained; it is smaller than the
effect just removed and smaller than the step from 0.199.

The middle column is not a regression to revert, and the CPU arms proved it before the fix was
built (`opm_ablation.json`): missing the OPM mask ALONE costs `pair` 0.1530 where missing all
three costs 0.1088, because the attention and OPM mask errors partially cancel. Fixing two of
three exposes the third, which is exactly what the device showed.

WITHDRAWN, the `ttnn.repeat` ask: the masked OPM no longer touches it. `_sum_rows` runs the same
contraction transposed, `[I, c_z, D] x [D, J]`, using only permutes and matmuls the trunk already
tapes. Last pass reverted that form after it read pair 0.606 and blamed ttnn; **the unit test says
the algebra was fine and the bug was mine** -- scored against a float64 contraction of the same
numbers, `_small_depth` reads 0.004036 and the transposed form 0.004037
(`perf/bcx_predictor/opm_unit.json`). The patch had inserted `def _sum_rows` into the middle of
`masked`'s body, so the `eps + mask-mask` divisor became part of the new method and `masked`
returned an undivided numerator. 0.606 was a missing divisor wearing a ttnn costume. With the
method where it belongs the boundary reproduces exactly: pair 0.15100 against 0.15105.

BWDSEAM: **the device backward is correct over all 48 blocks, so the gradient defect is the
splice.** Graded the way the forward was -- both arms fed BindCraft 2's own `(msa, pair)` and
mask, the same random cotangent seeded on both outputs, no BindCraft 2 loss anywhere in the
measurement, against a float64 torch reference of the same stack
(`perf/bcx_predictor/bwd_boundary.json`):

| | cos | rel L2 | norm ratio |
|---|---|---|---|
| `d_msa` | 0.9630 | 0.3165 | **1.129** |
| `d_pair` | 0.9751 | 0.2522 | **1.095** |

Tape reach **98 nodes, 96 of them checkpoint** -- two per block over 48 blocks, so it reaches
every one. **The device gradient is slightly LARGER than float64, not 7x smaller.** Whatever
loses the factor sits between JAX and the device, in the `custom_vjp`/`pure_callback` plumbing or
in the cotangent BindCraft 2 hands in, and not in the trunk. That removes the 48-block backward,
the checkpointing and the masked ops from the search, all of which were live suspects before this
measurement.

Next probe, not yet run: log the norm of what crosses the seam in each direction and compare it
against the all-JAX arm at the same point, which separates "the cotangent arriving is already
wrong" from "the gradient leaving is dropped".

GRADIENT: **the step-0 gradient comparison has no discriminating power. Measured, not argued.**
All-JAX on CPU, against the same base arm (dropout=True, key 0) the device was always compared
with (`perf/bcx_predictor/ceiling.json`):

| arm | cos bind | cos targ | ratio bind | ratio targ |
|---|---|---|---|---|
| dropout_off | 0.2939 | -0.0604 | 0.68 | 0.58 |
| dropout_on_key1 | -0.1659 | 0.2333 | 6.12 | 4.31 |
| kill_pair_cotangent | 0.9341 | 0.9505 | 1.15 | 1.06 |
| kill_msa_cotangent | 0.4908 | 0.3855 | 2.75 | 3.26 |
| **DEVICE (measured)** | **0.2840** | **0.1390** | **7.00** | **5.60** |

**The ceiling.** BindCraft 2 defaults `dropout=True` (`af2.py:209`, threaded through
`modules.py:1278`, `:1317-1352`) and tt-bio's `af2.py` has no dropout at all, so the JAX arm
differentiates one sample of a stochastic function and the device differentiates its mean. That
alone costs cos **0.2939** on the binder and **-0.0604** on the target. The device reads 0.2840
and 0.1390: **it is at the ceiling, and its cosine was never evidence of a defect.**

**And two BindCraft 2 runs differing only by PRNG key read cos -0.1659 / 0.2333 with norm ratios
6.12 / 4.31.** The device's four numbers all sit inside BindCraft 2's own key-to-key spread. At
step 0 this comparison cannot separate the arms.

**Which cotangent: neither.** Killing the `pair` cotangent at the cut barely moves anything (cos
0.93/0.95, ratios 1.15/1.06); killing `msa` gives cos 0.49/0.39 and ratios 2.75/3.26. Neither
reproduces the device signature, and the ceiling result says why: there is nothing to name.

Two framings this row had wrong, both corrected. **"7.0x and 5.6x, roughly uniformly" cannot be a
scaling** -- a scale factor preserves direction, so two factors with two low cosines are not one
mis-scaled cotangent. And **`taped: 2, backward: 1` is not a leak**: `af2.py:139-140`
`stop_gradient`s every recycle but the last, so JAX prunes one backward and the counter records
the model's own policy.

TAPECOST: **the discarded recycle tape is a memory lever, not a time one -- measured.**
`calls.primal` is 0 in every gradient run: `recycled_alphafold_outputs` `stop_gradient`s every
recycle but the last (`af2.py:139-140`), JAX prunes that backward, but the forward still ran the
taped path and banked a tape nobody reads. With `design_recycles` 1 that is half the trunk
forwards per design step. Both arms in one process, qb2 card 3, AICLK median **1350** sampled
inside the timed regions, min 1306 (`perf/bcx_predictor/tape_cost.json`):

| n=224, 48 blocks | median s | peak DRAM |
|---|---|---|
| untaped | 0.856 | 0.260 GB |
| taped + checkpointed | 1.254 | 1.519 GB |
| | **1.465x** | **+1.259 GB** |

**Time is the smaller half**: 0.398 s a step against a ~23.7 s step is 1.7%, about 50 s on a
125-step trajectory, and nobody should build a mechanism for that. **Memory is the half that
matters** -- 1.259 GB wasted, roughly 5x the untaped peak, held on every design step. That is the
axis the large-target capability claim runs on, since what a target can reach is bounded by tape
size.

Measured with an all-ones mask: a cost measurement, and the mask changes what the blocks compute
rather than how much tape they bank. Reported rather than fixed, as briefed: the fix is teaching
the splice that a `stop_gradient`'d call wants the primal, which changes how BindCraft 2's recycle
loop reaches the predictor.

Before this goes near main: `_host_twins` curries the reference column attention as a
one-argument lambda, so the per-op substitution probe needs its signature widened.

MASKCOST: **the missing MSA mask is nearly all of the boundary error, and it is priced.**
`perf/bcx_predictor/mask_ablation.py` runs the torch reference -- which already takes a mask,
`af2_reference.py:378` -- on BindCraft 2's own captured inputs, once with the true mask and once
with all ones. CPU only, fp32, both arms identical code, so precision cancels and only the mask
differs (`mask_ablation.json`).

BindCraft 2's `msa_mask` is [2, 192]: row 0 all ones, row 1 (the template torsion row) **40.1%
zeros**. Its **pair mask IS all ones**, so `af2.py:385`'s `mask_2d` precondition holds at
`length_bucket_size` 1 and only the MSA mask needs writing. That halves the job and corrects this
row's earlier framing, which had the pair mask in play.

| | mask only, CPU | device vs BC2, measured |
|---|---|---|
| pair | 0.109, cos 0.9941 | 0.181, cos 0.983 |
| msa row 0 | 0.105, cos 0.9950 | 0.164, cos 0.988 |
| msa row 1 | **0.527**, cos 0.8553 | 0.542, cos 0.845 |

The mask accounts for essentially all of row 1's error and about 60% of pair's. Treating the two
as independent, closing it should leave **~0.13 on row 1 and ~0.145 on pair**, which is bf16's own
share and is the target the fix gets checked against rather than "better".

Remaining work, all in `tt_bio/af2.py`: the row-attention logits, the column-attention logits and
the outer-product-mean divisor. The mask must be folded into `af2.py`'s own bias tensors, NOT into
`_fp32_softmax_attention`, which `af2.py:410` warns is shared with Boltz-2, Protenix-v2,
OpenFold3 and ESMFold2.

BOUNDARY: **the splice's forward is not right either, and the loss hid it.**
`perf/bcx_predictor/boundary.py` captures BindCraft 2's real `(msa, pair)` at
`modules.py:1594` with `jax.debug.callback` -- under jit they are tracers, so a runtime
callback is the only way to see them -- and runs both stacks on exactly those arrays
(`boundary_check.json`):

| | rel L2 | cos | norm BC2 | norm device |
|---|---|---|---|---|
| pair out | 0.181 | 0.983 | 43593 | 43348 |
| msa out, both rows | 0.253 | 0.968 | 4320 | 4356 |
| msa row 0 | 0.164 | 0.988 | 4007 | 4179 |
| **msa row 1** | **0.542** | **0.845** | 1614 | **1229** |

Row 1 is the template torsion row and it is **three times worse than row 0**, which is what the
missing MSA mask predicts: `modules.py:1576-1578` concatenates `torsion_angles_mask` onto the MSA
mask for that row and tt-bio applies none, so masked positions contribute where AF2 excludes
them. Row 0's own 0.164 after 48 bf16 blocks is looser than this port's usual forward bar and is
**not yet explained**.

**A 0.28% loss agreement was insensitivity, and a forward check would have passed this twice** --
once as `predict` at 0.106 A on the target, once as the loss here. The boundary is the instrument;
the loss is not.

GRADIENT: **runs through the splice and is WRONG, and the cause is identified.**
`sequence_gradients` completes with 48 blocks on card 3, 135.4 s against 251.8 s, AICLK median
1350 from 304 in-window samples. The forward is right -- loss 10.2408 against 10.2691, **0.28%**
-- and the gradient is not: binder cos **0.093**, target cos 0.135, norms 4.9x and 3.3x small,
and after BindCraft 2's own normalisation the update direction is cos **0.133**
(`grad_splice_check.json`). `design_dropout` off does not move it, cos 0.056 / -0.167
(`grad_splice_check_dropout0.json`), so dropout is not the cause.

`shape_probe.json` says what is: **BindCraft 2 hands the Evoformer `msa` of shape [2, 192, 256]**.
Templates are enabled in its monomer design, so `modules.py:1570-1578` appends a template torsion
row to `msa_activations` and concatenates `torsion_angles_mask` onto the MSA mask. tt-bio's trunk
serves ONE row: `afgrad.embed` builds `(1, n, 256)`, and `af2.py:418-420` says `msa_mask` is all
ones for every fold this port serves, which a torsion mask is not.

**That is the third precondition of one family**, after the all-ones `mask_2d` and the zero
extra-MSA row: tt-bio's AF2 was ported for PXDesign's folding regime, and BindCraft 2's design
regime differs in the MSA track. The forward tolerates it because the pair path carries the
structure, which is exactly why `predict` looked right at 0.106 A on the target while the
gradient is uncorrelated. **A forward parity check would have passed this and it is wrong.**

Closing it is `tt_bio/af2.py`, which **this row owns** -- the brief says to extend that file, and
last pass's claim that it wanted routing was wrong. `AF2Attention` asserts `msa_mask is None`
(`af2.py:509`) and `AF2EvoformerBlock` repeats it; the mask has to reach the row-attention logits,
the column-attention logits and the outer-product mean's divisor, which today reads the MSA depth
off the tensor because at an all-ones mask that IS the norm.

TAPE: `recycled_alphafold_outputs` routes its stop-gradient pass through `fwd` as well, so a step
banked two tapes and freed one -- measured taped 2, backward 1, live 1. At `bcx-ckpt`'s 5.33 GB an
Evoformer block that is fatal over 125 steps. `_taped` now drops superseded tapes; the
differentiated pass is the last taped call and `_backward` raises by token if that stops holding.

STRUCTURE: quoted against the seed floor, as the accuracy rule requires
(`perf/bcx_predictor/seed_floor.json`; both floor arms are BindCraft 2's own trunk with only the
PRNG key differing, dropout on as BindCraft 2 has it for design):

| | device vs JAX | seed floor, key 0 vs 1 |
|---|---|---|
| target, 115 residues | **0.106 A** | 0.108 A |
| binder, 77 residues | 24.94 A Kabsch | 10.34 A Kabsch |

**The target is AT the seed floor**, 0.106 against 0.108. The binder is 2.41x it. Two JAX runs at
the same key are bit-identical at 0.0 A, so the floor is real model variation and not harness
noise. That reads the way it should: the target is template-conditioned and determined, while the
binder at step 0 is a random sequence with no template that BindCraft 2's own dropout already
moves 10.3 A, with pLDDT agreeing at 0.733 against 0.735 and pTM at 0.5673 against 0.5642. A
step-0 binder coordinate is not this row's bar; whether the loop converges to equal-quality
binders in equal steps is.

The extra-MSA stack is a legal swap not yet made: BindCraft 2 feeds `extra_msa` as a single zero
row under an all-zero `extra_msa_mask` (`bindcraft/af2.py:134`), exactly what tt-bio's extra-MSA
blocks bake into `opm_constant`.

MASK: no longer satisfied by avoiding the mask. `tt_bio/af2.py` now carries AF2's masked
arithmetic, so the precondition the brief names has been removed rather than worked around, and
every trajectory from `trace_exit_seed0` onward runs at BindCraft 2's own DEFAULT bucket 32 with
real masked residues in the fold. Three pieces do it. `af2_pair_masks` (`af2.py:138`) takes
`mask_2d`, asserts it is the outer product `seq_mask[:, None] * seq_mask[None, :]` that AF2
always builds (`af2.py:163-168`), and returns the two tensors `AF2PairBlock` needs, or
`(None, None)` when it is all ones so every fold PXDesign runs today takes the unchanged path.
The both-halves difference the old text called a blocker is resolved by algebra rather than by a
kernel rewrite: masking both halves of the fused projection gives `s_i s_j sum_k s_k a_ik b_jk`
and masking the `a` half alone gives `s_i sum_k s_k a_ik b_jk`, equal wherever `s_j = 1`, and the
masked rows and columns nothing downstream reads are killed by the row/column biases in
`AF2EvoformerBlock._mask_biases`. `AF2MaskedOuterProductMean` contracts with `_sum_rows` and an
`eps = 1e-3` divisor so a wholly masked pair does not divide by zero.

**How much mask BindCraft 2 actually hands it, measured, and it is two different amounts on two
different code paths.** `predict` pads the TOTAL token axis (`bindcraft/af2.py:298`) and
`sequence_gradients` pads the DESIGN chain instead (`:385`), which do not overlap, so
`masked_residue_count` takes a `path` argument and `test_masked_count.py` pins all four cases
(115-residue target alone 13, 77-residue binder alone 19, the 192-token complex 0, the same
complex on the gradient path 19 — all PASS). On PD-L1's initialisation draw that is **19 of 211
tokens, 9.0%, masked on every gradient step** (`perf/bcx_predictor/mask_probe.json`). On the
binder 71 the running trajectory drew it is 96 - 71 = **25 masked tokens on the gradient path**
and 192 - 186 = **6 on the predict path**.

**And the masked path had its own defect, which is why the old bucket-1 answer looked safe.**
`_mask_biases` reshapes the row and column biases into new tiles, and `ttnn.reshape` writes only
the logical elements, so the tile padding kept whatever the buffer last held — after a backward,
often NaN — and the broadcast add carried it into the column-attention score tile where
`softmax_in_place` masks a finite padding value and not a NaN. `bcx-nan` found it and fixed it at
`e5cf74790` with two `ttnn.fill_implicit_tile_padding` calls; the root cause was in code this row
wrote. Verified here on a clean card, 20 replays of the captured failing backward in one process:
**0 of 20 non-finite against 10 of 20 before, and one distinct `d_msa_absmax` value,
0.2893524169921875** (`perf/bcx_predictor/proof/replay_postfix.json`). The old "clean" phase read
0.363677978515625 and the failing one 3.39617752923046e+38, so the value MOVED and the phase that
looked clean was reading the same uninitialised padding — which is why every masked-attention
number this row took before `e5cf74790` is marked history above.

`refuse_masked_state` is retired from the live path. It did its job — it blocked the first device
campaign while the trunk really could not serve a mask — and then blocked three launches after
the masked ops that made it obsolete had landed. It survives only as a fixture in
`test_masked_count.py` and `test_predictor.py`, where it still pins the 0.534 single-chain fold
that `bcx-mono` used to find the dropped `masks["pair"]`.


PROVED: on real hardware, qb2 card 3, P300c `0000:04:00.0`, commit clean in every stamp, AICLK
sampled during every timed region; and against the pinned BindCraft 2 tree
`7a2dfdb8a285232a6f881899fe135c6dc48679f1`.

**BindCraft 2's own campaign runs here, unmodified, and is running now.**
`perf/bcx_predictor/run_arm.py` drives `campaign.run_campaign`; it prints its own header, its
eight losses, its seven filters and `=== trajectory 1 | pdl1_denovo_l146_d2bfded440698f32 |
accepted 0/10 ===`, and the reference arm is detached under
`perf/bcx_predictor/runs/reference_qb2` rooted in this worktree. Nothing in the loop is edited:
the step budget, stage plan, recycles, filters and ranking are BindCraft 2's. Two deviations,
both deliberate: the model pool is pinned to the monomer checkpoints so both arms run the variant
tt-bio ships, and `max_trajectories` bounds the run because `examples/pdl1.json` caps attempts
nowhere. Every trajectory inside that budget runs in full.

qb2 held 1 of the 7 checkpoints in `CAMPAIGN_MODELS`, which is why `launch_campaign`'s preflight
refused; all seven are now at `/home/ttuser/bcx_e2e/af2_params`.

**The predictor class exists and BindCraft 2 drives it.**
`perf/bcx_predictor/ttbio_predictor.py` `TTBioAlphaFoldDesignModel` subclasses
`AlphaFoldDesignModel`, so padding, canonical names, templates, frozen interfaces,
`alphafold_input_features`, the prediction metrics, the Kabsch alignment and the
`StructurePrediction` assembly all stay BindCraft 2's own code and only the Evoformer is ours.
Checked rather than asserted (`predictor_check.json`): `isinstance` against
`DifferentiableProteinPredictor` True, `campaign.py:59`'s `provides_distogram` negotiation
accepts it, bucket rounding takes 192 to 192 and 211 to 224, and `predict` returns both chains
with seven metrics including the distogram, 77 binder residues and finite atoms, in 104.49 s
through BindCraft 2's own call path.

And end to end (`control_smoke.json`): `run_arm.py --arm control` rebinds
`campaign.AlphaFoldDesignModel` -- `campaign.py:262` is the only construction in the repository,
so one name is the whole injection -- and the campaign prints its own header, its own seven
filters and reaches `=== trajectory 1 | pdl1_denovo_l146_d2bfded440698f32 ===` with the stamp
recording `predictor: TTBioAlphaFoldDesignModel`. **That is the upstream PR's claim demonstrated
rather than argued: a predictor the lab does not ship drives `campaign.py` unmodified.** It was
stopped at that point; the question was whether BindCraft 2 drives the class, and the reference
arm wants the cores.

`refuse_masked_state` turns tt-bio's kernel assert into an error naming its own fix, refusing
bucket 32 and passing bucket 1. `trunk='device'` raises `NotImplementedError` naming the seam
rather than falling back to JAX, because a backend that silently runs the reference is the worst
thing this row could ship.

**The Protocol is two protocols.** `prediction.py:9` `ProteinPredictor` declares `predict`; `:14`
`DifferentiableProteinPredictor` adds `sequence_gradients`. Both `@runtime_checkable`, so
conformance is a method-presence check and a non-JAX class satisfies it; `isinstance` was run
here and reads True. `campaign.py:262` is the only construction of `AlphaFoldDesignModel`, `:265`
the validation lambda, and `refuse_predictor_without_distogram` (`campaign.py:59`) reads a
`provides_distogram` attribute defaulting True when absent.

**Nothing downstream re-traces through the predictor**, re-checked at the pin as the brief asked.
`trajectory.py:132` does `float(graph_design_loss)`; `:141` appends each gradient to a list;
`sequence_optimization.py:82-92` runs `combine_sequence_gradients`, `jnp.where` against the
DESIGN mask, `normalize_sequence_gradient` and `gradient_transform.update`, all on values. No
`grad`, `vjp` or trace re-enters the predictor, so concrete host arrays are a sufficient return.
Also worth carrying into a PR: `af2.py:380` is annotated as a 2-tuple while `:418` returns three;
the Protocol has it right.

## VERDICT

VERDICT: PARTIAL — BindCraft 2's own `campaign.py` drives a tt-bio-backed predictor unmodified on
qb2 card 3, three defects inside that predictor have been found and two are fixed, and the best
device trajectory so far reached BindCraft 2's fourth and last gradient stage and missed
acceptance by **0.01 pLDDT** (0.64 against the 0.65 harden bar) — but **0 designs have been
accepted in 9 completed device trajectories, and the reference arms have completed 1 trajectory
between them in 11 hours, which was itself a rejection**,
so the equal-quality question is still open; what has changed is that on the one seed the device
has run on the fixed tree it is **0.07 pLDDT ahead of BindCraft 2's own predictor on the
identical draw**, where before both fixes it was 0.33 behind.

TRAJECTORIES: device **9 completed** and 2 ended in the DRAM allocator; reference **1 completed
across 5 arms running** -- one on qb2 (seed 0, T=10) and four on qb1 (seeds 100/200/300/400, T=3
each) -- the completed one being seed 400, rejected at screen, and the other four still inside
their first trajectory after 9 to 11 hours. Per trajectory,
with `e5cf74790` (the tile-padding fix) as the pre/post line:

| run | seed | bucket | binder n | backwards | ending | vs `e5cf74790` |
|---|---|---|---|---|---|---|
| `device_qb2_seed0` | 0 | 1 | 146 | 6 | rejected at screen, pLDDT 0.41 i_pTM 0.36 | pre, and pre pair-mask |
| `device_qb2_seed100` | 100 | 1 | 71 | 14 | rejected at screen, 0.38 / 0.40 | pre, and pre pair-mask |
| `device_qb2_seed200` | 200 | 1 | 176 | 7 | rejected at screen, 0.38 / 0.37 | pre, and pre pair-mask |
| `device_qb2_seed300` | 300 | 1 | 138 | 2 | rejected at screen, 0.38 / 0.39 | pre, and pre pair-mask |
| `device_qb2_seed400` | 400 | 1 | 180 | 5 | rejected at screen, 0.38 / 0.33 | pre, and pre pair-mask |
| `trace_exit_seed0` | 0 | 32 | 146 | 6 | rejected at screen, 0.41 / 0.36, loss went non-finite at round 6 | pre, post pair-mask |
| `accept_seed200` | 200 | 32 | 176 | 3 | rejected at screen, 0.44 / 0.41 | pre, post pair-mask |
| `accept_seed400` | 400 | 32 | 180 | 4 | rejected at screen, 0.51 / 0.61 | pre, post pair-mask |
| `proof_seed100` | 100 | 32 | 71 | 34 | **passed screen 0.69, refine 0.64, anneal 0.66; rejected at harden 0.64 / 0.74** | pre, post pair-mask |
| `accept_seed0` | 0 | 32 | 146 | — | `bank_manager.cpp:439`; my own replay shared the card, so confounded | pre, post pair-mask |
| `accept_seed300` | 300 | 32 | 138 | — | `bank_manager.cpp:439` on a clean card, 4.248 of 4.278 GB/bank | pre, post pair-mask |
| `post_seed100` | 100 | 32 | 71 | ~75 through refine | **passed screen 0.78 / 0.80 and refine 0.76 / 0.78; running on into anneal, no allocator failure** | **post** |
| `reference_qb2_b1` | 0 | 1 | 146 | — | passed screen 0.68 / 0.80, refine 0.65 / 0.76; in anneal at 11 h 12 m | reference, BC2 on CPU, qb2 |
| `reference_qb1_seed100` | 100 | 1 | 71 | — | passed screen 0.71 / 0.80, refine 0.69 / 0.79; in anneal at 9 h 44 m | reference, BC2 on CPU, qb1 |
| `reference_qb1_seed200` | 200 | 1 | 176 | — | passed screen 0.61 / 0.76; in refine | reference, BC2 on CPU, qb1 |
| `reference_qb1_seed300` | 300 | 1 | 138 | — | passed screen 0.75 / 0.82; in refine | reference, BC2 on CPU, qb1 |
| `reference_qb1_seed400` | 400 | 1 | 180 | — | **rejected at screen 0.56 / 0.74**, moved on to trajectory 2 (`l60`) | reference, BC2 on CPU, qb1 |

**The NaN was not only corrupting values, it was ending BindCraft 2's stages early, and that is
why the backward counts above are so far under 125.** `trajectory.py:136` reads
`if not math.isfinite(design_loss): break` after a fallback recompute at `:133`. So a non-finite
loss does not raise: the gradient stage returns whatever state it had and the trajectory is
graded on it. `trace_exit_seed0` shows the whole mechanism in one place — rounds 1-4 finite,
round 5 the first bad gradient, round 6 `loss=nan`, stage over, rejected at screen on pLDDT 0.41.
`proof_seed100` got 34 backwards where BindCraft 2's own plan asks for 125 and still reached
harden, so **every pre-`e5cf74790` row above was graded on a fraction of the optimisation
BindCraft 2 intended**, which is the row's own question — same quality in the same number of
steps — failing on the second half without saying so. `post_seed100`, the first trajectory on a tree carrying both fixes, spent about **50
backwards inside the screen stage alone** -- more than `proof_seed100` spent on all four --
and **passed screen at pLDDT 0.78 / i_pTM 0.80** on trajectory
`pdl1_denovo_l71_c019673d64134f64`, against 0.69 / 0.77 for that same trajectory before the
fix and **0.71 / 0.80 for BindCraft 2's own CPU arm on that same trajectory**
(`refsnap/qb1/reference_qb1_seed100`). It then passed refine at **0.76 / 0.78** where the same
reference reads 0.69 / 0.79 and the pre-fix device read 0.64 / 0.71, so the device is 0.07
pLDDT ahead of BindCraft 2's own predictor at both stages it has reached, and both readings
are above the campaign's 0.70 FINAL bar rather than merely above the 0.60 and 0.60 stage
bars. Anneal (45 rounds, bar 0.65) and harden (5 rounds, bar 0.65) are left, then MPNN and
the validation refolds, and it is still running.

QUALITY: acceptance rate is **0 of 9** completed device trajectories against **0 of 1**
completed reference trajectory, point estimate 0.00 on both with a rule-of-three 95% upper bound
of 0.33 on the device and an uninformative 1.0 on the reference, so the two arms AGREE only at a
width that answers nothing -- the pre-registered arithmetic below puts the minimum at T=10 each. What the runs do compare, at matched seeds and
matched draws, is the screen-stage filter metrics, and they move with each fix:

| arm | n | screen pLDDT (mean, values) | screen i_pTM | screen passes |
|---|---|---|---|---|
| device, pre pair-mask, bucket 1 | 5 | **0.386** — 0.41 0.38 0.38 0.38 0.38 | 0.370 | **0 of 5** |
| device, post pair-mask, pre `e5cf74790` | 3 | **0.547** — 0.44 0.51 0.69 | 0.597 | 1 of 3 |
| device, post `e5cf74790` | 1 | **0.78** | 0.80 | 1 of 1 |
| BindCraft 2 on CPU, unmodified | 5 | **0.662** — 0.71 0.61 0.75 0.56 0.68 | 0.784 | 4 of 5 |

**The reference arm rejects at screen too, and that corrects something this row wrote earlier.**
qb1 seed 400 (`pdl1_denovo_l180_ffb9c87873af6d36`) is rejected at screen on pLDDT 0.56 by
BindCraft 2's OWN predictor and the campaign moves to trajectory 2. So a screen rejection is part
of BindCraft 2's normal attrition, and the earlier sentence here that treated every device
rejection as a port defect was too strong. What survives is the MATCHED comparison, which is
stronger than the pooled one:

| seed | trajectory | BindCraft 2 on CPU | device |
|---|---|---|---|
| 100 | `pdl1_denovo_l71_c019673d64134f64` | screen 0.71 / 0.80 pass, refine 0.69 / 0.79 pass | **post-fix screen 0.78 / 0.80 and refine 0.76 / 0.78, both pass**; pre-fix 0.69 then 0.64; pre pair-mask 0.38 reject |
| 0 | `pdl1_denovo_l146_d2bfded440698f32` | screen 0.68 / 0.80 pass, refine 0.65 / 0.76 pass | pre pair-mask 0.41 / 0.36 reject; post pair-mask ended in the allocator |
| 200 | `pdl1_denovo_l176_09b8e7a3d08bc7b6` | screen 0.61 / 0.76 pass | 0.44 / 0.41 reject (pre `e5cf74790`) |
| 300 | `pdl1_denovo_l138_aaf0e3cad8e56a88` | screen 0.75 / 0.82 pass | ended in the allocator (pre `e5cf74790`) |
| 400 | `pdl1_denovo_l180_ffb9c87873af6d36` | screen **0.56 / 0.74 REJECT** | 0.51 / 0.61 reject (pre `e5cf74790`) |

On seed 100, the one seed where the device has run on the fixed tree, **the device is 0.07 pLDDT
ABOVE BindCraft 2's own predictor on the identical draw at both stages it has reached** (screen
0.78 against 0.71, refine 0.76 against 0.69) and level on i_pTM. One trajectory
is one trajectory and this is not yet an acceptance, but it is the first reading in this row's
history where the port is not behind. Two differences ride along and are stated rather than
absorbed: the reference arms run at `length_bucket_size` 1 and the device at BindCraft 2's default
32, so the device is folding a padded complex the reference is not, and the reference runs on qb1
at loadavg 82 on 32 cores against the device's qb2 at 67.8 on 16 -- neither of which moves a
pLDDT, but the padding difference could, and a bucket-32 reference arm is the control that would
close it. Snapshot `perf/bcx_predictor/refsnap/qb1/`, refreshed each pass because those runs live
in `/dev/shm`.

CMPNUM: the 8 completed trajectories carrying a recorded wall consumed **7,244 chip-seconds on one Blackhole p300c**
(qb2 card 3, `0000:04:00.0`) and produced **0 accepted designs**, so the only bound the acceptance
runs themselves support is **more than 7,244 chip-seconds per accepted design at 0.00 designs per
hour**, against BoltzGen's 264.3. The clocked per-trajectory cost comes off the running
`post_seed100`, which is the first trajectory on the fixed tree: **AICLK median 1350 MHz, minimum
1325, 121 samples at 5 s on card 3 sampled during the run** (`runs/post_seed100/aiclk.jsonl`),
host load1 median 67.8 on 16 cores. Its screen stage ran 50 rounds in 2,520 s, which is **50.4 chip-seconds per
gradient round**, so BindCraft 2's own 125 rounds are **6,300 chip-seconds of gradient per
trajectory** before the mutate stage, MPNN and the validation refolds. Two caveats travel with
that figure and neither is small: qb2 is 4.2x oversubscribed on CPU for the whole window, which
inflates the host-bound half of every backward; and the 50 is screen's configured round count, corroborated by the
deallocation log at 1,153 lines per backward calibrated on `proof_seed100`'s exact 34. The number CMP is
sized on wants an accepted design under it, and this row has produced none.

**How many trajectories, decided before starting as the brief asks.** Acceptance per trajectory
is roughly Poisson. BindCraft v1 on PD-L1 gives 101 accepted from 91 trajectories, so r is about
1.11, and the relative standard error from T trajectories is 1/sqrt(rT): **30% at T=10, 21% at
T=20, 17% at T=30**. So **T=10 per arm detects a 2x difference and is the minimum worth running;
T=15-20 is what it takes to claim the arms AGREE rather than merely that they are not 2x apart.**
The reference arm is launched at T=10 for that reason.

**The reference is the binding constraint, not the device.** Five reference arms have run 9 to
11 hours and between them have completed one trajectory, a screen-stage rejection, plus nine
gradient stages still in flight; qb1
carries four of them at loadavg 82 on 32 cores and qb2 the fifth at 67.8 on 16, so each arm is
also slowing the others. At that rate T=10 on one arm is on the order of 80 h. The device arm at
roughly 1.8 h a trajectory can produce T=10 in a shift once the allocator stops ending the long
draws. If a quieter host frees up, the REFERENCE is what should move to it -- and the control
this row most wants is not another seed but a reference arm at `length_bucket_size` 32, so the
matched comparison stops carrying a padding difference.

**What is left, in the order it binds.** (1) `post_seed100` has cleared screen and has refine,
anneal and harden left; it is the first trajectory in this row's history running BindCraft 2's
whole step budget on a tree with both fixes, and whether it is ACCEPTED is the open question. (2) `bcx-dram` owns
`bank_manager.cpp:439`; two of eleven trajectories ended there and both were long draws
(binder 146 and 138), so until it lands the device arm can only sample the short draws and its acceptance rate
is measured on a biased slice of BindCraft 2's own length distribution. (3) The boundary grades
(pair 0.151, msa row 0 0.125, row 1 0.151), the 1.35x-of-bf16 gradient reading and the
masked-regime pLDDT control (0.7433 against 0.7349) were all computed on the masked path before
`e5cf74790` and want re-taking on it.

## Where the predictor class attaches, and what blocks it today

**The blocker is the mask, not the wiring.** The trunk on main cannot run BindCraft 2's fold
while BindCraft 2 pads the design chain, and the two ways out are: run the campaign at
`length_bucket_size` 1, which this row has shown gives an all-ones mask at every draw; or teach
`AF2PairBlock` the both-halves masking `af2.py:24-28` describes, which is a kernel change on a
file this row does not own. The first is free and is what the device arm should use; the second
is what a merged upstream backend would eventually want, because the lab's default is 32.

`sequence_gradients` (`af2.py:380-418`) is one jitted program and `_compiled_sequence_gradients`
(`:328-378`) differentiates it w.r.t. the sequences alone, so the trunk is not a drop-in. The cut
to take keeps BindCraft 2's whole program and replaces only the Evoformer: subclass
`AlphaFoldDesignModel` so feature assembly, padding, metric trimming and `StructurePrediction`
building stay the lab's own code, and intercept the vendored `EmbeddingsAndEvoformer`
(`modules.py:1394`, extra-MSA stack at `:1528`, Evoformer stack at `:1583-1597`,
`single_activations` at `:1599`) with a `jax.custom_vjp` primitive backed by tt-bio. Its seam is
`(single, pair)`, which `bcx-e2e` proved; the residual is the tape, held Python-side by token.

Intercepting inside the haiku module rather than reproducing the pipeline matters for a second
reason found this pass: `perf/bcx_afgrad/afgrad.py:95` `embed()` is tt-bio's OWN featurisation --
one MSA row, single chain, no template, no chain break, no mask -- so it does not cover
BindCraft 2's `alphafold_input_features`. That is also why `bcx-e2e` could run a gradient step
without meeting the mask: its state is hand-built and unpadded. Taking the activations after
BindCraft 2's own embeddings avoids re-deriving any of it.

`run_arm.py --arm device` selects it and the class refuses at `_open_device` until the stacks
are wired. `--arm control` is the same class on BindCraft 2's trunk, so a device result is
compared against the same call path rather than a differently shaped program.

## Environment and running jobs

`/home/ttuser/bcx_e2e_venv/bin/python` is `bcx-e2e`'s overlay (jax 0.11.2, haiku, optax,
ml_collections, jmp) layered over the shared tt-bio env by a `.pth`, so ttnn and torch come from
`/home/ttuser/tt-bio-dev/env`. `biotite` stays out of the overlay: tt-bio pins `<1.7` and the
overlay shadows it. BC2 at `/home/ttuser/bcx_e2e/bc2`, all 7 AF2 checkpoints at
`/home/ttuser/bcx_e2e/af2_params`, MPNN weights in the repo at `bindcraft/weights/proteinmpnn`.

BUCKET, CORRECTED: **every arm stamped `length_bucket_size: 1` and every arm ran BindCraft 2's
default 32.** The override was not in `run_arm.py`'s overrides list -- `grep -c` on the committed
file returns 0 -- and the stamp wrote `args.bucket`, the flag, rather than what the campaign
resolved. A later patch rewrote the overrides block and dropped the line; the stamp could not
notice because it never read the settings.

| | chains | n | n % 32 |
|---|---|---|---|
| as the runs actually went, bucket 32 | [96, 115] | **211** | 19 |
| bucket 1, as every stamp claimed | [77, 115] | 192 | 0 |

**The comparison is not damaged**: both arms used the same 32, the seed-0 trajectory hash matched
across arms, and 32 is the lab's own default. Nothing is rerun for this. **Three claims in this
row's writeups are withdrawn**: that the arms run unpadded, that n is whatever BindCraft 2 drew,
and the list n = 186, 253, 261, 291, 295. The padding finding survives and sharpens -- seed 0
lands on 211, 19 short of a tile boundary, exactly the case `_pad_inputs` pads to 224 and the case
the 3.29x was measured on.

Fixed in both halves: the override is back, and the stamp records
`campaign_length_bucket(settings)` with the flag kept beside it as `length_bucket_flag`.

OWNERSHIP, from the orchestrator 2026-09-24 13:0xZ: **`bcx-mask` owns the NaN**, `bcx-mono` owns
the single-chain defect (qb2 card 0, chain-feature path only), and **this row owns the loop**.
No more full trajectories until one of those lands -- five have said everything a rejected
trajectory can. What this row does next: merge their branches when they push and re-run, because
it is the only row that can run the proof. A trajectory reaching the screen's full 50 rounds is
what shows the NaN fix worked; binder-alone pLDDT moving off 0.4039 is what shows `bcx-mono`'s
did.

**Carried correction**: the mask defect is a BACKWARD defect, not a metric one. This row's own
masked control reads 0.7433 against 0.7349 at the metric level, and that distinction is exactly
why five forward controls all came back clean.

`wk/bcx-mask` is live with three commits and already has a candidate: `device_fixed` reads
**3.877e-07 at every pad**, the fp32 floor, where `device_today` grows with the pad. Handed back
to it this pass: the pad sweep cannot be the whole story, because bucket 1 has **no pad at all**
and the loop still NaNs at round 4. A fix validated only on the pad sweep may leave the
trajectory dying anyway. `wk/bcx-mono` has not pushed a branch yet.

PROOF: **the device arm passes screen, refine and anneal, and tracks the reference.** Seed 100,
trajectory `pdl1_denovo_l71_c019673d64134f64` -- the same hash the reference ran -- on qb2 card 3
with `wk/bcx-mask` at 61fa2675e and `wk/bcx-mono` at 838457ba8 merged:

| stage | device NOW | BindCraft 2 reference | device BEFORE the fixes |
|---|---|---|---|
| screen | i_pTM 0.77, pLDDT **0.69** | i_pTM 0.80, pLDDT 0.71 | **REJECTED**, pLDDT 0.38 |
| refine | i_pTM 0.71, pLDDT 0.64 | i_pTM 0.79, pLDDT 0.69 | never reached |
| anneal | i_pTM 0.73, pLDDT 0.66 | not yet reached | never reached |

Three stages cleared against BindCraft 2's own bars (screen 0.60, refine 0.60, anneal 0.65) where
five trajectories were rejected at the first. **Within 0.02 of the reference at screen**, and it
has now run FURTHER than the reference, which is still inside refine after 6.5 h on a loadavg-83
box while this took 35 minutes.

The trajectory finished: **2105.1 s, `device_calls` taped 68 / backward 34, 30 rounds**, and it
was **rejected at harden, i_pTM 0.74 pLDDT 0.64 against a 0.65 bar -- a 0.01 miss**. Before the
fixes the same seed managed **2** backward passes and died at screen. Three of four gradient
stages cleared, and the stop is an ordinary near-miss rather than a defect signature: the five
earlier trajectories died at the FIRST gate against 0.60 reading 0.38.

**The fixes are NOT a uniform cure, and the earlier "the fixes fixed the loop" is withdrawn in
that form.** Both seeds ran with both branches merged:

| seed | binder | n | ending |
|---|---|---|---|
| 100 | l71 | 211 | **34 backwards**, cleared screen/refine/anneal, harden 0.64 by 0.01 |
| 200 | l176 | 307 | **3 backwards**, NaN, rejected at screen 0.44 |
| 300 | l138 | 275 | ~40 min, then **OUT OF MEMORY in the backward**, rc=1 |

**Three trajectories, three different endings, so the campaign has three distinct blockers and
they surface in order of how long a trajectory survives**: the NaN kills short ones, DRAM kills
long ones, and one that dodges both gets judged on its merits. Seed 300's failure, at
`tt_bio/autograd.py:1417` `_recompute` inside the checkpointed backward:

    Out of Memory: 84934656 B DRAM buffer across 8 banks, each bank needs 10616832 B,
    bank size 4278190016 B (allocated 4248022976 B, free 30167040 B)

4.248 GB of 4.278 GB per bank in use. **It connects to a number this row already measured**:
`tape_cost.json` has the taped forward peaking at 1.519 GB against 0.260 GB untaped, +1.259 GB,
and `recycled_alphafold_outputs` routes its stop-gradient recycle through the taped path too, so
half the trunk forwards per design step bank a tape JAX has already discarded. That was reported
as a memory lever worth 5x the untaped peak and explicitly not a time one; **this is that lever
arriving as a hard failure** on trajectories that live long enough to reach it.

What is true is narrower: the fixes are necessary, and one trajectory in two now clears three
stages where five in five previously died at the first. Seed 200 looks exactly like a pre-fix
failure.

The arithmetic worth carrying: if every backward had the replay's ~50% corruption rate, surviving
seed 100's 34 of them would be 2^-34. It did. **So the in-loop NaN rate is far below the replay's
-- the replay is a worst case, not the loop's rate** -- and seed-to-seed variance dominates.
Whether n is the variable (211 survived, 307 did not) is one datapoint each and not a finding;
seeds 300, 400 and 0 are running at n = 275, 307, 275.

**The NaN is not fixed and is not claimed to be** -- the
replay still reads 10/20 non-finite and `bcx-nan` owns it. What changed is that the loop survives
long enough to be judged.

Not to be misread: 34 backward passes is not 34 of the 125 gradient steps. BindCraft 2's schedule
decides when to update, so rounds, updates and gradient calls are three different counts.

ACCEPTANCE ARM RUNNING: four more paired trajectories on seeds 200, 300, 400 and 0, card 3, each
against the reference's own trajectory hash (`launch_accept.sh`). Seed 200 is up as
`pdl1_denovo_l176_09b8e7a3d08bc7b6`, matching the reference's. **Zero accepted so far, so CMPNUM
remains undefined** -- that is what these runs are for.

DISCRIMINATOR: **BindCraft 2 passes the same seed the device failed.** The paired reference
reached a stage judgement on seed 100, trajectory `pdl1_denovo_l71_c019673d64134f64` -- the same
hash in both arms, so the same starting state and the same draws:

| arm | screen | refine |
|---|---|---|
| BindCraft 2 | **PASSED** i_pTM 0.80, pLDDT **0.71** | **PASSED** i_pTM 0.79, pLDDT 0.69 |
| device | **REJECTED**, pLDDT **0.38** | never reached |

BindCraft 2's own bar is `min_plddt_screen` 0.60. The reference clears it at 0.71 and clears
refine too. **That answers the question this row has carried since the first rejection**: if the
reference had also come in near 0.4, the device did nothing wrong and this was BindCraft
rejecting a trajectory, which it does to most. It did not. The five device rejections at
0.41/0.38/0.38/0.38/0.38 were **a device defect, not attrition**, and the loop converges normally
on BindCraft 2's own predictor. i_pTM says it twice as loudly: 0.80 against 0.40.

This adds no new defect. It confirms the two already found and owned are sufficient to explain
the rejections, and it is the campaign's first paired reference datapoint. The other three qb1
seeds are still inside trajectory 1 at 6 h 35 m with 28 to 34 hours of CPU each, on a box at
loadavg 83.

DEFECTS 1 AND 2 ARE SEPARATE, tested at both fix branches' current heads. With
`wk/bcx-mask` at **73cf91c7f** (not the `a1f67bc63` mono carries) and `wk/bcx-mono` at
`838457ba8` both merged, the captured failing backward replays **10/20 non-finite** in one
process -- exactly the 50% alternation measured before the fixes. The missing pair mask was real
and is fixed; the NaN is somewhere else. `bcx-nan` owns it and the `nancap/` reproducer is
unaffected by either fix.

`masked_residue_count` FIXED, the defect `bcx-mono` left here. It counted only what
`pad_design_chains` pads, so it returned **0** for the single-chain fold reading 0.534 pLDDT --
the refusal built to catch that padding would have waved it through. BindCraft 2 pads in two
non-overlapping places: `pad_design_chains` pads each DESIGN chain (the `sequence_gradients`
path, `af2.py:385`), and `predict` pads the TOTAL token axis (`af2.py:298`), padding chains only
when `target_pad_length` is set. The count now takes a `path` argument.
`perf/bcx_predictor/test_masked_count.py`, all passing:

| case | masked |
|---|---|
| target alone, predict path | 13 (115 -> 128) |
| binder alone, predict path | 19 (77 -> 96) |
| two-chain complex, predict path | 0 (192 already aligned) |
| two-chain complex, gradient path | 19 (design chain 77 -> 96) |
| guard refuses the 0.534 fold | raised |

The 13 cross-checks independently against `bcx-mono`: 13 pad rows in 128 gives a pair mask
1 - (115/128)^2 = **19.28%** zero, the figure they measured on that fold.

FIXES MERGED, LOOP STILL BLOCKED. `wk/bcx-mono` merged clean (it carries `wk/bcx-mask`) and
**the root cause was this row's**: `splice.py` handed tt-bio's 48 blocks `masks["msa"]` and
dropped `masks["pair"]`, so the pair track ran unmasked on every fold BindCraft 2 pads. Their
measurements: target alone pLDDT **0.5340 -> 0.9541** against the reference's 0.9501, binder
alone **0.4039 -> 0.6059** which clears BindCraft 2's 0.60 screen bar, pair error against BC2's
JAX **0.24 -> 0.0005**.

The proof trajectory after the merge (`proof_after_fixes.json`), instrumented identically:

| round | loss | finite | gradient |
|---|---|---|---|
| 1 | 8.37551 | yes | ok |
| 2 | 6.01947 | yes | **NaN both chains** |
| 3 | nan | NO | predictions NaN, break |

Seam: taped 1-4 ok, backward 1 ok, **backward 2 finite cotangents in and NaN out**, taped 5-6
poisoned. Identical in shape to the pre-fix run. **The pair mask was a real defect and is not
this one.** The single-chain and accuracy defects are fixed; the loop still exits after two
rounds of fifty, and the acceptance run stays blocked.

NAN FIXED AND VERIFIED, root cause in **this row's own `_mask_biases`**. `bcx-nan` at
`e5cf74790` found the period-2 NaN was never carried by the tape: the reshape to
`[rows,1,1,n]` / `[n,1,1,rows]` writes only the logical elements, so the new tiles keep whatever
the buffer last held -- after a backward, often NaN -- and the broadcast add copies that into the
column-attention score tile, where `softmax_in_place` masks a finite padding value but not a NaN.
Their fix is two `ttnn.fill_implicit_tile_padding` calls where the padding is created.

Verified on a clean card, 20 replays of the captured failing backward in one process
(`proof/replay_postfix.json`):

| | non-finite | distinct `d_msa_absmax` |
|---|---|---|
| before | **10/20** | 2, alternating 0.363677978515625 and 3.39617752923046e+38 |
| after | **0/20** | **1**, 0.2893524169921875 |

**And the value moved.** The old clean phase read 0.3637; the correct answer is 0.2894. That
confirms `bcx-nan`'s own claim from the outside -- the former clean phase was also reading the
garbage, just finitely -- so **every masked-attention number this row produced before e5cf74790
was computed over uninitialised tile padding, including the halves that looked fine.**

Wants re-taking before anyone quotes them: the masked boundary grades (pair 0.151, msa row 0
0.125, row 1 0.151), the 1.35x-of-bf16 gradient reading, and the masked-regime pLDDT control
(0.7433 against 0.7349). The five pre-fix trajectory outcomes stand as history, not as
measurements of the current code.

NONDETERMINISM, and it reframes everything below: **identical inputs alternate clean, NaN,
clean, NaN.** Eight replays of the captured failing backward -- same npz, same depth 48, same
process (`perf/bcx_predictor/replay_nan.json`):

| rep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| | clean | **NaN** | clean | **NaN** | clean | **NaN** | clean | **NaN** |

4/8 non-finite, strict period-2, and both outcomes bit-identical to themselves -- clean always
`d_msa_absmax 0.363677978515625`, NaN always `3.39617752923046e+38`. **Not a race** (a race
scatters) but **state carried from one backward into the next**. 3.4e38 is fp32's ceiling, so
something reads a saturated or stale buffer rather than computing a large number.

**Two earlier readings here are artifacts of this and are withdrawn**: the depth sweep's "first
bad depth 48" and then "first bad depth 47, with 48 clean" on identical inputs was the
alternation, not a depth effect; and the round counts 1/4/5/6/13 across trajectories are a
per-call chance of corruption, not accumulation.

**What it is NOT, tested rather than assumed**: not the shared mask tensor. `splice.py` caches
the uploaded mask by shape and reuses one device tensor for every round -- exactly the
in-place-on-a-caller-tensor shape this fleet has been bitten by. A fresh upload per call leaves
it at 4/8; the phase shifts, the rate does not. So the carried state is the tape teardown,
`release_pins`, or a cached tensor inside the blocks such as the OPM's folded `proj_o` -- all
`af2.py` and `autograd.py`, which this row does not touch.

Reproducer: `perf/bcx_predictor/replay_nan.py nancap/nan_backward_inputs.npz --depths 48,48,48,48`

NAN LOCALISED TO OUR BACKWARD: **BindCraft 2 hands finite cotangents in and our backward returns
NaN.** Seam-level finiteness on every call of a real device trajectory
(`perf/bcx_predictor/trace_seam.json`, reproduce with `BCX_NANLOG=1 trace_exit.py`):

| i | op | call | cotangent msa | cotangent pair | out msa | out pair |
|---|---|---|---|---|---|---|
| 2 | backward | 1 | ok | ok | ok | ok |
| 4 | taped | 4 | ok | ok | ok | ok |
| **5** | **backward** | **2** | **ok** | **ok** | **NaN 15856** | **NaN 233131** |
| 6 | taped | 5 | NaN 37376 | NaN 9680000 | - | - |

Line 5 is the finding: finite in, NaN out. Everything after is downstream -- the optimiser
poisons the logits, so the next forward receives a NaN sequence. **Not BindCraft 2's tail, not
the loss, not the predictions, not the cotangent. Inside `dev.ag.backward` over the 48 masked
Evoformer blocks.**

**MY REFUTATION OF THE ZERO DIVISOR IS ITSELF WITHDRAWN.** This row told `bcx-mask` its pad
sweep "cannot be the whole story" because the loop still NaNs at bucket 1 where nothing is
padded. Wrong: n=192 comes from `initialize_design_trajectory` at the campaign seed, where the
binder is 77, but the **running trajectory draws binder 146**, so at bucket 1 the complex is
261 and `_pad_inputs` pads it to 288. Measured off the captured failing state:

| | |
|---|---|
| msa leaf / pair leaf | (2, 288, 256) / (288, 288, 128), real n **261** |
| mask row 0 zeros | **27** (the pad) |
| mask row 1 zeros | 173 (template torsion) |
| OPM norm minimum | **0.0** |
| pairs with norm exactly 0 | **14823** |

The zero-divisor case is fully present in the run I claimed refuted it. **The eps candidate
stands and `bcx-mask`'s pad sweep is on the right object.** This is the second time this campaign
that confusing the design state's draw with the trajectory's draw produced a wrong claim -- the
first was the n = 186/253/261/291/295 list. `initialize_design_trajectory`'s binder length is not
the length the trajectory runs.

**DETERMINISTIC REPRODUCER, new this pass.** `splice.py` under `BCX_NANCAP` writes the exact
inputs of the first backward returning a non-finite gradient, and `replay_nan.py` replays that
one call. The NaN arrives after 1 to 13 rounds and the round varies run to run, so through
BindCraft 2 it is a 6-minute stochastic trajectory; this makes it seconds long on identical
numbers.

    capture   BCX_BUCKET=1 BCX_NANCAP=nancap BCX_NANLOG=1 perf/bcx_predictor/trace_exit.py
    replay    perf/bcx_predictor/replay_nan.py nancap/nan_backward_inputs.npz

The replay confirms the forward on that state is finite (msa absmax 366, pair absmax 11584) and
the cotangents are finite at 0.072 and 0.046: **the NaN is made inside the backward on inputs
that are all well scaled.**

What survives is the localisation and now it is the whole of it: **finite cotangents in, NaN out
of `dev.ag.backward` over the 48 Evoformer blocks, after 3 to 5 healthy rounds.** It accumulates
-- the round varies between runs of the SAME seed -- which points at growth in the backward
rather than a structural zero. **bf16's 3.4e38 ceiling in a 48-block chain is the next thing to
measure**, and measuring it means logging per-block gradient magnitudes rather than proposing
another mechanism.

Incidental, not to be read into: bucket 32's loss fell monotonically 7.678 -> 6.629 before the
NaN; bucket 1's bounces, 6.998 / 7.791 / 6.721 / 6.935. Both then NaN.

The break round varies between runs of the same seed -- 6 rounds then 3 -- so it accumulates
rather than hitting a fixed edge.

ROOT CAUSE OF THE EXIT: **our gradient goes NaN at round 5; the predictions are the second
domino.** Every round of a real device trajectory instrumented for finiteness -- gradients,
every metric and atom array in the predictions, the design loss, the input logits -- with the
wrapper only observing (`perf/bcx_predictor/trace_exit.json`):

| round | design loss | finite | gradient | predictions | binder logits absmax |
|---|---|---|---|---|---|
| 1 | 7.67801 | yes | clean | clean | 10000.0 |
| 2 | 6.99722 | yes | clean | clean | 10000.0 |
| 3 | 6.93312 | yes | clean | clean | 10000.0 |
| 4 | 6.62854 | yes | clean | clean | 10000.0 |
| **5** | 6.28056 | yes | **NaN/inf** | clean | 10000.0 |
| 6 | nan | NO | NaN/inf | NaN | nan |

**Rounds 1-4 are healthy and the loss FALLS monotonically**, so the optimiser works and the
device gradient carries real signal. At round 5 the gradient goes non-finite while the loss and
predictions are still fine; that poisons the logits through `sequence_optimization.py`'s update;
round 6 folds a NaN sequence, both finiteness checks at `trajectory.py:132-136` fail, and the
loop breaks.

**This refutes the reading that the culprit is in the predictions dict** -- they are clean on the
round the gradient is already bad. The variable round counts across seeds (1, 4, 5, 6, 13) are
the round at which the gradient first goes non-finite, so this accumulates rather than hitting a
fixed edge.

`10000.0` on the binder logits from round 1 is BindCraft 2's own `OMITTED_AMINO_ACID_LOGIT`
sentinel, not a runaway value.

SINGLE-CHAIN DEFECT: **the device is right on two-chain folds and wrong on every single-chain
one, and that is the mechanism for all five rejections.** Folded through BindCraft 2's own
`binder_alone_state`, the exact call `trajectory.py:71` makes for the `Unbound_Binder_pLDDT` the
screen gate reads (`pos_control.json`, `binder_alone.json`):

| case | n | chains | BC2 pLDDT | device pLDDT | ratio |
|---|---|---|---|---|---|
| start complex | 211 | 2 | 0.7349 | 0.7465 | **1.016** |
| target alone | 115 | 1 | 0.9501 | **0.5340** | 0.562 |
| binder alone | 77 | 1 | 0.5932 | **0.4039** | 0.681 |

Agreement to 2% on the two-chain complex -- on both chains separately -- and 32 to 44% off on
both single-chain folds. **Two sizes, 77 and 115, so it is chain count and not n.** The target
alone is a natural protein with a known fold reading 0.534 against 0.950: a failed positive
control, not a precision gap.

**And the number lands where the rejections did**: device 0.4039 on the binder alone against five
trajectories rejected at 0.41, 0.38, 0.38, 0.38, 0.38. The gate reads a single-chain fold and
single-chain folds are the broken case, which is why the rejection barely depended on binder
length.

Against over-reading: BindCraft 2's own binder-alone pLDDT here is **0.5932**, already under the
0.60 bar, because this is an unoptimised random binder at round 0. The reference clears it by
optimising. **The device arm starts 0.19 lower AND exits the loop after 1 to 13 of 50 rounds --
two live defects that compound.**

Not root-caused. What differs between one chain and two is `asym_id`, `entity_id`,
`interface_asym_id` and `monomer_chain_break_indices`, all computed by BindCraft 2 and handed to
the trunk, plus whatever our Evoformer does with a pair track that has no cross-chain block.

EARLY EXIT: **the screen stage is 50 rounds and the device arm is running 1 to 13 of them.**
This supersedes the metric and gradient-direction hypotheses as the leading explanation for the
rejections. From the save-enabled seed-0 run's own stamp (`perf/bcx_predictor/device_save_stamp.json`):

| | |
|---|---|
| `stage_plan.per_stage.screen` | **50** rounds, BindCraft 2's own budget |
| rounds in the losses CSV | **1** |
| `device_calls` | taped 4, **backward 2** |

Two backwards is two gradient steps; with `design_recycles` 1 each step tapes twice, which is why
taped is 4. So `judge_stage` saw a design optimised for ~1 round out of 50, and **pLDDT 0.39 is
the confidence of a design that has barely been optimised**. The five earlier trajectories
recorded 5, 13, 6, 1 and 4 rounds -- all far short of 50 and all different, which is an early
exit, not a budget. It also explains a 5-to-18-minute screen stage where 50 steps at the measured
~24 s would be 20 minutes.

**The candidate, named and not proven**: `trajectory.py:134-136` does
`if not math.isfinite(design_loss): break` after `float(graph_design_loss)` and a
`weighted_design_loss` fallback. A non-finite design loss silently leaves the gradient loop. That
fits everything this row has seen -- flat confidence terms, moving geometry, pLDDT clustered at
0.38, round counts varying 1 to 13.

This does **not** exonerate the gradient: a non-finite loss would most likely come from our own
arithmetic. It relocates the question from "is our metric right" and "is our direction right" to
**"why does the loss go non-finite"**, which is cheaper to instrument than either.

`save_failed_trajectories=true` did NOT produce a sequence or PDB -- only the losses CSV,
`!_Trajectories.csv` and `summary.csv` -- so the same-sequence refold is still blocked and the
setting alone is not what unlocks it.

METRIC, SETTLED FOR THE INITIAL STATE: **our pLDDT is BindCraft 2's pLDDT in the masked regime
the trajectories actually ran.** `splice_check.json`'s 0.7327 vs 0.7349 was taken at bucket 1 with
an all-ones mask and says nothing about runs that went at bucket 32, so the control was repeated
there -- n=211, **19 pad residues masked**, same state, same class, same BindCraft 2 code, only
the Evoformer differing (`perf/bcx_predictor/metric_masked.json`):

| | pLDDT mean | min | max | pTM | i_pTM |
|---|---|---|---|---|---|
| BindCraft 2 | 0.7349 | 0.2605 | 0.9744 | 0.5642 | 0.0884 |
| device | 0.7433 | 0.3123 | 0.9770 | 0.5698 | 0.0870 |
| ratio | **1.0115** | | | | |

**The metric hypothesis is refuted for the initial state at bucket 32.** No `af2.py` mask site
was touched: `bcx-mask` owns those now, and this row stops editing them.

**The reading that matters more than the ratio**: both arms start at pLDDT ~0.74 and the device
trajectories END screen at 0.38-0.41. A static metric offset cannot do that -- it would read low
from round 1. Whatever is wrong is in the optimisation direction, not the measurement, which is
where the loss traces already pointed when the confidence terms sat still while compactness moved.
Caveat against over-reading: 0.74 is the whole 211-residue complex including the
template-conditioned 115-residue target, while the screen bar is bound to the binder role through
`resolve_binder_role`, so 0.74 -> 0.38 is not a like-for-like fall. What IS like-for-like is the
device-against-BindCraft-2 ratio, 1.0115.

`predict` costs **29.7 s on card against 127.2 s** for BindCraft 2's own trunk, AICLK median 1350
from 112 in-window samples.

DEVICE RESULTS: **five trajectories completed on card, zero accepted, all rejected by BindCraft
2's own screen filter** against `min_plddt_screen` 0.60:

| seed | binder | n | stage | i_pTM | pLDDT | rejected on | rounds |
|---|---|---|---|---|---|---|---|
| 0 | 146 | 275 | screen | 0.36 | **0.41** | pLDDT | 5 |
| 100 | 71 | 211 | screen | 0.40 | **0.38** | pLDDT | 13 |
| 200 | 176 | 307 | screen | 0.37 | **0.38** | pLDDT | 6 |
| 300 | 138 | 275 | screen | 0.39 | **0.38** | pLDDT | 1 |
| 400 | 180 | 307 | screen | 0.33 | **0.38** | pLDDT | 4 |

**pLDDT reads 0.38 four times out of five**, across binder lengths 71 to 180, five seeds and three
different n. Five independent random binders do not give a spread that tight. Something is pinning
it; the clustering does not say which of the two candidates -- our confidence head reading a
`single` our trunk shapes differently, or a fold that genuinely does not form -- but it says the
number is not scatter. `perf/bcx_predictor/device_summary.json`.

**No accepted design, so chip-seconds per accepted design is UNDEFINED from this run** and is not
reported. What exists is a per-seed trajectory cost at its own n with the stage it reached.

(superseded three-trajectory note) **three trajectories completed on card, all rejected by
BindCraft 2's own screen filter**, `min_plddt_screen` 0.60 (`settings/core/default.json`, applied by `filters.py:824-830`
through `judge_stage`). Seed 0 pLDDT 0.41 / i_pTM 0.36, seed 100 0.38 / 0.40, seed 200 similar.
That is BC2's quality gate, not a crash, and BindCraft rejects most trajectories.

The confidence terms do not move across the stage while the geometry does:

| seed | rounds | i_pTM first -> last | pTM first -> last | compactness first -> last |
|---|---|---|---|---|
| 0 | 5 | 0.36 -> 0.36 | 0.32 -> 0.33 | 2.3 -> 1.1 |
| 100 | 13 | 0.37 -> 0.39 | 0.34 -> 0.37 | 2.0 -> 1.8 |
| 200 | 6 | 0.39 -> 0.36 | 0.36 -> 0.34 | 0.54 -> 0.34 |

**No discriminator exists yet and none of this is a device verdict.** The paired reference on seed
0 has printed no stage outcome after 4 h 54 m and has written no `*_losses.csv`. Its pLDDT at the
end of screen is the one number that decides whether 0.41 is a device defect or BindCraft
rejecting a trajectory.

Against the metric hypothesis, from this row's own earlier measurement: `splice_check.json` has
**pLDDT 0.7327 device against 0.7349 JAX** on the same state, 0.3% apart. That is the initial
state and the complex-bound screen metric is a different quantity, so it is evidence and not a
settlement.

`save_failed_trajectories` and `save_design_sequences` are now on for both arms: the three
completed device trajectories kept only their losses CSV, so refolding the device's own binder
with BindCraft 2's predictor -- the sharp version of the metric test -- is impossible on the runs
we have.

DEVICE ARM: **running.** BindCraft 2's campaign is driving tt-bio's Evoformer on qb2 card 3,
AICLK **1350** under load against the 800 idle floor. `launch_device5.sh` walks seeds
**0, 100, 200, 300, 400** -- the same five the reference streams run -- one trajectory each,
under `perf/bcx_predictor/runs/device_qb2_seed<N>`. **Pairing verified by hash, not assumed**:
seed 0's trajectory is `pdl1_denovo_l146_d2bfded440698f32`, byte-identical to the qb2 reference
stream's. If any later seed's hash differs from its reference twin, the pairing claim comes out
of the writeup.

Two changes made it run. The unmasked-fold guard is **retired from the device path**: it existed
because the trunk asserted an all-ones mask, and all three mask sites are now in `af2.py`. It was
right to stop the first launch and stale to keep afterwards. And **the backend pads the token axis
itself**: BindCraft 2 samples a binder length per trajectory, so at `length_bucket_size` 1 the
complex is n = **186, 253, 261, 291, 295** across the five live trajectories and **not one is a
multiple of 32**. This row measured 211 padded to 224 running **3.29x faster** on the forward
while doing strictly more arithmetic, so `_pad_inputs` pads up, masks what it added and slices
back. Seed 0 is n=261 padded to 288.

PADDING AND THE TEMPLATE STACK: the orchestrator's correction warns that padding makes the mask
non-all-ones everywhere including `AF2DeviceTemplatePairStack`, which asserts all ones at
`af2.py:387`. **That assert cannot fire in this splice**, and the reason is architectural rather
than lucky: `splice.evoformer_on_device` replaces ONLY the `layer_stack` call whose closure is
named `evoformer_fn`. `splice_probe.json` records all three call sites and the two template pair
stacks (`block`, `num_layers` 2) are handed straight to BindCraft 2's own JAX. tt-bio's
`AF2DeviceTemplatePairStack` is reached from `AF2DeviceModel.__call__`, which this splice never
calls -- it calls `dev.stack`, which runs `device_evoformer[i]` and nothing else. Empirically the
arm has been running with padding on for 35 minutes without touching `:387`.

The correction's cost claim stands and is the reason the padding is in: five live trajectories at
n = 186, 253, 261, 291, 295, not one tile-aligned.

OWED on that padding: the padded-against-unpadded grade is **running on card 2** and has not
finished (`pad_grade.py`, n=211 padded to 224, both arms against a float64 reference of the same
48 blocks). It runs on an idle sibling card so the acceptance arm keeps card 3. Until it lands the
padding is an unverified change on the arm that produces the campaign's answer, and if it grades
badly the arm is rerun without it -- `_pad_inputs` is one call.

LIVE, two hosts, up to 22 trajectories of reference capacity:

- **qb2**, pid **2837200**, `--trajectories 10 --seed 0 --bucket 1`, 8 threads, under
  `perf/bcx_predictor/runs/reference_qb2_b1` in the worktree. On trajectory 1 (`l146`).
- **qb1**, four processes at seeds **100/200/300/400**, three trajectories each, 8 threads each,
  under `/dev/shm/bcx-ref/runs/reference_qb1_seed<N>`. Draws `l71`, `l176`, `l138`, `l180`, so the
  seeds are genuinely independent of each other and of qb2's `l146`, and a pooled acceptance rate
  over the two hosts is legitimate. loadavg 30.8 on 32 cores.

Stop any of them by explicit pid, never by pattern. Run outputs are gitignored.

**qb1 runs entirely in tmpfs and tmpfs is volatile.** Its root disk has 1.2 G free, which is why
the arm was not there before; `/dev/shm` is RAM-backed with 252 G and holds the venv, the
parameter set and the BindCraft 2 clone. Those outputs **do not survive a qb1 reboot** and must be
copied into the worktree before they are quoted. Staging scripts are committed at
`perf/bcx_predictor/qb1/` so the host can be rebuilt in one command.

Three things the staging had to get right, recorded so the next host costs minutes: qb1's python
is 3.10 and jax 0.11.2 needs 3.11+, so `uv` installs a 3.12 into tmpfs with
`UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` redirected there as well, or uv writes the interpreter
to the full disk; the first parameter fetch reported `curl rc=0` and extracted **3 of 16** files,
so the retry verifies the tar size and the file count rather than the exit status; and BindCraft 2
wants `matplotlib`, which its own `selfcheck.missing_modules()` names in one call instead of one
import error per launch. jax on qb1 is **0.11.2**, matching qb2 exactly.

It runs at `length_bucket_size` 1 because the device arm has no choice about it, and an arm
comparison where only one side pads is not the same program twice. An earlier launch at the
default 32 was stopped 8 minutes in for that reason rather than left to spend 60 hours producing
a non-comparable arm.

The `TT_FATAL: Tensor is not allocated` line `bcx-e2e` reported fires under a tape at every size
run here, swallowed and harmless by that row's evidence. Unchanged and unowned.

Variant gap: the device trunk is monomer `model_1_ptm` and BC2 samples design models from the
multimer_v3 pool. Both arms are pinned to monomer so the comparison is like-for-like;
`bcx-multimer` holds the multimer forward at 0.9963789 PCC under `model-merge-approval-gate`.
