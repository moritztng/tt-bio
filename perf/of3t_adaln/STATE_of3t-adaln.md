# of3t-adaln — the AdaLN gate is clean, the attention softmax is the defect, and neither the block nor the transition is broken

VERDICT: GO

Row `of3t-adaln`, 2026-09-20, qb2 card 3. Branch `wk/of3t-adaln` (pushed, verified against
origin). Artifacts in `perf/of3t_adaln/`. Nothing in `tt_bio/` changed: the tracked diff
against the commit this branch was cut from (592d632e1 on `wk/of3t`) touches 0 files under
`tt_bio/`; everything it adds is under `perf/of3t_adaln/`. Every
ablation below is installed from the instrument and removed again. The orchestrator holds the
merge gate; this branch stays on its own. This file lives in three places that must stay
identical: here, on qb2 at the same path, and in the branch at
`perf/of3t_adaln/STATE_of3t-adaln.md`. The row's gate runs on qb2, the dispatch host, and for
four passes this document existed only on the orchestrator host, so the check passed wherever I
ran it and failed where it counted.

The brief was amended six times while this pass was running. Amendment 1 refuted its own premise
from the model's natural A/B: two instances of `tenstorrent.AdaLN` per DiT block, the same class
on the same `(a, s)`, one reading 10.6980 mass-weighted and the other 0.1828. Amendment 2 closed
the AdaLN op by measurement and handed over a five-item list on the DiT attention path. This
document reports what the row measured against the amended scope, and the micro-arm it had
already run, which reaches amendment 1's conclusion independently.

MICROARM: one `tenstorrent.AdaLN` from DiT block 8's real checkpoint weights through
`remap_of3_adaln`, taped on real-shaped `(a, s)` = `[1, 384, 768]` and `[1, 384, 384]`, against a
float64 torch autograd reference built from upstream 0.4.3's own `AdaLN` class
(`normalization.py:88`, `of3pkg043` on qb2) with the same weights, inputs and cotangent.
Instrument `perf/of3t_adaln/adaln_micro.py`. A18's forward discriminator, read off the same taped
forward the backward is taken from, is 1.055e-03 against the 5.0e-02 bar. Then:

  layer_norm_s.weight   rel 2.072079e-03   r 1.00086787   cos 0.99999823   |g_ref| 3.430232e+03
  linear_g.weight       rel 2.001244e-03   r 0.99950548   cos 0.99999812   |g_ref| 2.593827e+01
  linear_g.bias         rel 8.679667e-04   r 1.00005758   cos 0.99999962   |g_ref| 1.284351e+02
  linear_s.weight       rel 2.194329e-03   r 0.99876525   cos 0.99999835   |g_ref| 1.117242e+02

Mass-weighted 2.071014e-03 against a measured zero-model baseline of exactly 1.000000, a 483x
separation, 0 of 4 over bar. The leaf that reads 18.504 in the model reads 2.072079e-03 here with
a norm ratio three parts in ten thousand from unity. The same synthetic inputs through blocks 0,
5, 6, 7, 8 and 12 spread 1.15x (1.861037e-03 to 2.135488e-03) against the 6.8x the model shows
over those six blocks, so the AdaLN weights are not the block-to-block variable either. This
agrees with `of3t-conditioning`'s 18-arm unit gradcheck and was reached from a different
direction.

BLOCKARM: one real DiT block, ours (`_DiTBlock`, taped, fp32 activations) against upstream
0.4.3's own `DiffusionTransformerBlock` in float64, loaded strict — 0 missing and 0 unexpected
keys, both reported as numbers because a reference that silently drops tensors is how D23/R126
survived. Operands are the model's own: `(a, s, z, mask)` from the captured diffusion boundary at
384 tokens, with `a` at block b produced by running THEIR blocks 0..b-1 in float64. The cotangent
is random with a fixed seed, which tests the linear map rather than one vector through it.
Instrument `perf/of3t_adaln/dit_block_gradcheck.py`.

At block 8 the shipped block reproduces the campaign's A/B inside a single block:
`attention_pair_bias.layer_norm_a.layer_norm_s.weight` reads 8.060854e-01 while its sister
`conditioned_transition.layer_norm.layer_norm_s.weight` — same class, same block, same `s` —
reads 3.038103e-02. A 26.5x gap with nothing between them but which branch they sit in.

GRADIENTS: the worst tensor is the one the row was dispatched for,
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`.
In the model it reads rel 1.850397e+01 with `r` 19.2415, `cos` 0.74941, `||g_ref||` 9.0991e-01
and `||g_dev||` 1.750810e+01 — measured here by running
`perf/of3t_diffusion/device_gradient.py --structs all --cap /home/ttuser/of3t_rebase/diffcap043
--dump-per-tensor` unmodified into this row's namespace, which reproduces the published
`worst_rel` 18.503974843073646 to all digits. Over all 547 compared tensors the median cosine is
0.99002 and the median `r` is 1.0208, and 17.00 % sit below cos 0.5. On the isolated block the
same leaf reads 8.060854e-01 under a random cotangent, and under a float64 softmax it reads
1.459256e-02 — 55.2x lower, and below its sister's 2.466644e-02. The attention-side gradient
defect is the softmax and nothing else on that path.

FUSED_VS_UNFUSED: the gate unfused — `ttnn.sigmoid` as its own taped op then a plain
out-of-place `ttnn.multiply`, with the module's own `s_terms` and everything else identical —
against the same float64 reference. It is bit-identical to the shipped
`multiply_(a, s_scale, input_tensor_b_activations=[SIGMOID])` path on `rel_l2`, `r` and `cos` for
all four tensors under exact float equality, in fp32 (2.071014e-03) and in bf16 (5.813832e-03).
Amendment 1 said this arm was answered before it ran, and the measurement agrees with it: the
fused sigmoid rule at `taped_ttnn.py:444-515` produces the same gradient as the unfused form.

SCALES: A22 binds every sweep here; powers of two appear only as arithmetic controls.

  On the AdaLN micro-arm, `s` at 0.0137 / 0.1 / 1.0 / 3.7 gives 1.962329e-03 / 2.006337e-03 /
  2.071014e-03 / 2.082929e-03, a 6.2 % move over a 270x range; `a` at 0.0137 / 1.0 / 3.7 gives
  2.075291e-03 / 2.071014e-03 / 2.048761e-03. The sample axis 1 / 4 / 48, accumulating into the
  same leaves, gives 2.071014e-03 / 2.085571e-03 / 2.116898e-03, and the accumulation probe grows
  3.433209e+03 to 2.364329e+04 over the 48, a factor 6.887 against the 6.928 independent draws
  predict. The arithmetic control is on the cotangent, where the backward is exactly linear: at
  cot x 2^-5 and x 2^+5 the reference norms scale by exactly those factors while `rel_l2`, `r`
  and `cos` are bit-identical to the base arm.

  On the softmax sandwich, score sharpness at 0.0137 / 0.1 / 3.7 / 11.3 with the cotangent held
  at cancel 0.0137 gives 5.413076e-02 / 1.786319e-01 / 1.120163e+00 / 4.938666e+00: a sharper
  softmax is a worse gradient, monotonically, over three decades of scale. The power-of-two
  sharpness arms 0.125 / 2.0 / 8.0 read 7.019920e-01 / 7.917269e-01 / 3.490364e+00 and sit on the
  same curve, which is what says the curve is a property of the computation and not of the
  mantissas.

CONTROL: five, because this pass had to withdraw a reading of its own.

  (i) A16's zero-model baseline, measured on every arm rather than assumed: replacing our
  gradient with zeros reads exactly 1.000000.

  (ii) The negative control on the ablation harness, and it earned its place. The first version
  of the sandwich arm reported the shipped and the precise-reduction rules as identical to all
  digits and I nearly published that as a refutation. `_taped_verb` reads `_VERBS` once and
  `_Ttnn.__getattr__` caches the wrapper in the shim's instance dict, so replacing the registry
  entry alone is a silent no-op on a verb that has already been called, and the arm had measured
  the shipped rule twice. A control rule that drops the `inner` term entirely reads 7.329269e+01
  against the shipped 6.566019e-01 on the same rung, and only once it moved were the ablation
  columns worth reading.

  (iii) The float64 bound. The softmax computed on the host in float64, forward and backward, is
  not a lever but a ceiling: it says what the rest of the attention path owes once the softmax
  owes nothing. At block 8 that is 1.459256e-02, which is where the leaf lands.

  (iv) The selectivity control, which is the block's own sister AdaLN. Across shipped /
  precise_config / `_accurate_softmax` / float64 the transition-side gain moves 3.038103e-02,
  2.901677e-02, 2.498690e-02, 2.466644e-02 — 1.23x total — while the attention-side gain moves
  55.2x. A lever that moved both equally would be measuring the harness.

  (v) The reference is checked, not trusted: strict key load with both sets reported, and on the
  AdaLN arm a hand replica that reproduces upstream's own `AdaLN.forward` at rel exactly 0.0.

MECHANISM: the softmax kernel's own accuracy, amplified by its backward's cancellation.

  Forward, on [1, 16, 384, 384] fp32 against float64 on identical values: `ttnn.softmax` with no
  `compute_kernel_config` reads 2.274755e-02. With `precise_config()` it reads 6.888150e-04.
  `tenstorrent._accurate_softmax` reads 2.780055e-04. Torch float32 on the host reads
  6.053848e-08. `numeric_stable=False` reads 1.746234e-02 and 5.583554e-04, so the stability
  shift is not the term. That 2.3e-02 is on the inference path as well as the training one.

  Backward, `y * (g - sum(g*y))`: `g - inner` is a near-cancellation, so the error in `inner`
  survives at full size into a difference that is small. Measured on the sandwich, with
  K = ||g|| / ||g - inner||: rel 2.405795e-02 at K 1.424, 9.327789e-02 at K 10.2, 6.561025e-01 at
  K 73.58, 5.467103e+00 at K 609.5 and 1.678499e+01 at K 1932, with `r` tracking `rel` and `cos`
  falling to 0.0637. That is the same law and the same signature the model's worst tensor shows.

  The reduction inside that backward is innocent, which was worth checking because it is the one
  `ttnn.sum` on the path that takes neither `precise_config()` nor an fp32 output dtype while
  `_sum_leading` (autograd.py:526) takes the first. Giving it `precise_config()` changes nothing
  at any rung, and `inner` recomputed in float64 on the DEVICE's own `y` reads 8.808894e-03
  against the device sum's 8.954591e-03. `y` is the source.

  The call sites line up with the failures. `openfold3_diffusion_transformer.py:208`,
  `openfold3_atom_transformer.py:183` and `tenstorrent.py:8399` pass no compute kernel config;
  `tenstorrent.py:3797` and `:3808` pass `sm_ckc` and the comment there says "fp32 reduction",
  and `autograd.py:702` defaults to `precise_config()`. Of the campaign's ten worst tensors, six
  are DiT `attention_pair_bias.layer_norm_a.*` and two are `linear_g` inside
  `atom_attn_enc`'s atom transformer: the two no-config attention softmaxes.

LEVERS: two exist and both are priced on the real block at the same leaf, block 8 first, then 0
and 12.

  `compute_kernel_config=precise_config()` on the forward softmax: 8.060854e-01 -> 3.585557e-01
  (2.25x), 2.933287e-01 -> 1.218734e-01 (2.41x), 1.829792e-01 -> 9.183976e-02 (1.99x).
  `tenstorrent._accurate_softmax`, which already exists opt-in: -> 1.837565e-01 (4.39x),
  7.131739e-02 (4.11x), 6.063781e-02 (3.02x). The float64 bound is 1.459256e-02 (55.24x),
  5.217859e-03 (56.22x), 6.438348e-03 (28.42x).

  So `_accurate_softmax` recovers about 4x of a 55x. Both change the forward, so both move
  inference on five models, and both stay on this branch, flagged, per the release gate. The
  `_accurate_softmax` docstring records 4.22x the fused kernel's cost at [1,16,1024,1024], so the
  decision is a real trade and is the orchestrator's, not this row's. One factual correction for
  whoever takes it: that docstring says a compute kernel config changes nothing, and on
  [1,16,384,384] on a p300c it changes 2.274755e-02 into 6.888150e-04.

HANDOFF: of the five items amendment 2 listed, item 1 is closed and it accounts for the whole
attention-side disagreement, so items 2 to 5 have nothing left to explain at this block. The
float64-softmax arm bounds them together: with the softmax exact, the attention-side gain lands
at 1.459256e-02 (block 8), 5.217859e-03 (block 0) and 6.438348e-03 (block 12), at or below the
transition side's own reading on the same arm, and every one of the eight compared tensors is
inside the 5.0e-02 bar. `scale_add`, the `a_ln` fan-out, the padded qkv projection with
`nlp_create_qkv_heads`, and the two sigmoid-gated multiplies are therefore bounded by those
numbers collectively rather than individually, which is a weaker statement than eight separate
gradchecks and a sufficient one for deciding what to do next.

MODELS: 0 of 5 shipped models move, because no `tt_bio/` file changed. The reach of the finding
is wider than the row: 3 of the 3 no-config attention softmax call sites are on shared code —
`openfold3_diffusion_transformer.py:208` (OpenFold3 and OpenBind), `openfold3_atom_transformer.py:183`
(the same two) and `tenstorrent.py:8399` — and `protenix.py:567` and `:662` also call
`ttnn.softmax` without one, so a lever here would touch 4 of 5 models and must be gated as such.
The AdaLN clearance covers 4 of the 6 `tt_bio` modules that construct an AdaLN
(`tenstorrent.py` via `ConditionedTransitionBlock`, `openfold3_atom_transformer.py`,
`openfold3_diffusion_transformer.py`, `protenix.py`); `boltz2.py` and `esmfold2.py` carry their
own class and are outside it.

AMENDMENT3: the softmax backward's own reduction is not the lever, and the control proves the
test could have said otherwise. `taped_ttnn.py:200` is `inner = ttnn.sum(g * y, dim)` with no
`compute_kernel_config`, inside the `g - inner` cancellation. Giving it `precise_config()` is
bit-identical at every rung of the cancellation ladder — 2.405795e-02 at K 1.424 through
1.678499e+01 at K 1932, unchanged to all digits. The control that makes that null readable is a
rule that drops the `inner` term altogether: 7.329269e+01 against the shipped 6.566019e-01 on the
same rung. And `inner` recomputed in float64 on the DEVICE's own `y` reads 8.808894e-03 against
the device sum's 8.954591e-03, so the reduction is faithful to the `y` it is given. `y` is the
source, which is the softmax forward, which is what LEVERS prices.

AMENDMENT4: the summand product is not the lever either, and on the arm every ladder number was
taken at the premise does not hold. `autograd.py:1611` is
`gamma.add_grad(_sum_leading(ttnn.multiply(g, norm), shape))`, and the product passes no dtype.
The patched rule records what it actually builds: on fp32 activations `g` is FLOAT32, `norm` is
FLOAT32 and the product is FLOAT32 already. Forcing it to fp32 is bit-identical at every rung
including the high ones amendment 4 asked to watch — 2.071014e-03 at K 20.4, 5.744067e-01 at
K 2.1e+04, 4.161453e+00 at K 2.1e+05, 2.635289e+01 at K 2.1e+06, all unchanged to all digits.

  The control is the same knob in the other direction, because `ttnn.multiply` takes a dtype and
  no kernel config. Forcing the product to bf16 degrades those four rungs by 1.5653x, 2.6643x,
  2.3640x and 2.1735x. The knob reaches the kernel, so the null is a null and not a patch that
  missed.

  On bf16 activations the premise does hold — `g` and `norm` are both BFLOAT16 — and there the
  lever is worth 0.8834x, 0.9246x, 1.0286x and 0.8711x across the same rungs. About 1.1x, in the
  helpful direction three times out of four. That is not the 793x-to-6,970x device-against-torch-
  fp32 floor, which therefore sits somewhere other than the summands' storage. Amendment 4 is
  right that my MECHANISM paragraph read as if the floor were unfixable and right that nobody had
  tested it; the test says this particular lever is not the one, and the floor's location is now
  a named open question rather than an assumption.

AMENDMENT5: block 8 is not higher-K than block 9, and the within-leaf split does not live in the
block. K is measured on the reference's own float64 arithmetic, as
`sum_i||term_i|| / ||sum_i term_i||` over the 384 per-token summands of the gain sum, with a
self-check that the summands add back to the reference's own gamma gradient at 3e-15 or better:

  block   K (attention side)   K (transition side)   model rel   curve
      0                36.37                 4.507      2.7329   off
      1                17.12                 9.632      0.8331   off
      5                52.82                 6.044      4.5419   off
      6               207.90                12.200      2.8408   off
      7                87.45                 6.773      5.3071   off
      8               172.60                10.540     18.5040   off
      9               175.60                 7.682      0.1237   ON
     12                72.90                 6.429      5.6459   off
     23                14.67                 3.328           —   —

  Blocks 8 and 9 sit 1.7 % apart in K and 150x apart in the model. Under the same controlled
  cotangent they also read the same on the device: the attention-side gain is 8.060854e-01 at
  block 8 and 6.047823e-01 at block 9, 1.33x apart, and the float64-softmax arm puts them at
  1.459256e-02 and 1.984451e-02. Code, shapes, operands and conditioning are matched at the two
  blocks; whatever produces the 150x arrives through the real cotangent, not from the block.

  One correction to amendment 5's reading of the ladder. It says the cosine never goes negative
  across five decades; one arm does, fp32 activations with fp32 weights at K 2.1e+06, cos
  -0.019878. A random direction in 384 dimensions has |cos| about 0.051, so -0.02 is what a
  fully error-dominated result looks like and the mechanism can reach zero and cross it. It
  cannot reach the -0.796 and -0.805 that blocks 0 and 6 read, which are roughly 16 sigma from
  zero. The conclusion holds in its strong form; the evidence for it is the DISTANCE from zero
  rather than the sign.

AMENDMENT6: the conditioned transition is not broken either, and the widened block arm caught a
defect in this instrument before it became a finding.

  The transition branch, run ALONE against upstream 0.4.3's own `ConditionedTransitionBlock` in
  float64 on the real `(a, s)` the block hands it — captured by a pre-hook on THEIR module while
  THEIR block runs, so the operands are the model's — reads mass-weighted 5.468736e-03 at block
  8, 7.278610e-03 at block 9 and 9.595681e-03 at block 0, with 0 of 9 parameters over the
  5.0e-02 per-tensor bar at every block, cosines at or above 0.99997 and norm ratios within 1 %
  of unity. The gain itself reads 5.548868e-03, 7.438482e-03 and 9.684134e-03. Feeding it OUR
  block's `a` rather than the reference's — a 1.912528e-03, 8.487426e-04 and 4.523406e-03
  difference at the three blocks — moves the headline to 7.897092e-03, 8.039326e-03 and
  5.733396e-03, so at most 1.44x and at block 0 slightly better. The module's own K is 17.4 to
  35.8. Extrapolating this arm's slope, the transition would need K near 1.2e+03 to reach the
  0.1828 the model shows. So the 15.5125 % leaf is not made inside the transition; it is carried
  by what arrives at the transition's output, which is the blocks above it, the atom decoder, or
  the accumulation over 48 samples.

  Widening the block arm from 8 parameters to all 19 with a 1:1 checkpoint tensor gives the
  whole block under an exact softmax at 1.489217e-02 mass-weighted, inside the 2.0e-02 mass bar,
  against 7.856040e-01 shipped. Two tensors are still over the per-tensor bar,
  `attention_pair_bias.linear_z.weight` at 1.107297e-01 and `layer_norm_z.weight` at
  5.341521e-02, and between them they hold 0.008 % of the block's gradient mass; the leaf this
  row was dispatched for holds 94.939 % of it and reads 1.459256e-02. Those two are in the pair
  bias, they barely move under any softmax arm (3.2187e-01 and 4.0666e-01 shipped), and they are
  the one thing in this block the row leaves located and unexplained.

  This is one block under one controlled cotangent and it does not overturn amendment 6's
  whole-arm extrapolation, which is taken over 547 tensors under the real cotangent and 48
  samples. It does say the block's own arithmetic is inside the bar once the softmax is exact,
  and the difference between that and 0.1867 is again the cotangent.

  The instrument correction: the first widened run reported
  `attention_pair_bias.mha.linear_o.weight` and `mha.linear_g.weight` at rel 1.411 with
  `r` about 0.996 and `cos` 0.000135 and 0.001519 — the exact r ~ 1, cos ~ 0 signature this row
  spent three passes teaching people to read as a wrong transform. Both are 768x768, and the
  instrument inferred the transpose from the shape, which is silent on a square matrix, so it
  compared the device gradient against the reference's transpose. Reading the flag
  `_DiTBlock._w_tt` already records in its `(key, transpose)` cache key gives 2.492647e-02 with
  cos 0.999740 and 2.469198e-02 with cos 0.999853. Ordinary. A shape-inferred transpose
  manufactures the one signature a reader is primed to believe.

  And the A23 note in this document was wrong where amendment 6 says it was. A cancelled
  component does NOT carry little mass: `blocks.8...layer_norm_a.layer_norm_s.weight` is
  8.05416 % of the model, the fourth-heaviest tensor in OpenFold3, 94.939 % of its own block's
  gradient mass, and the one at the highest cancellation. That sentence is withdrawn. What
  stands is the rest: a per-tensor RELATIVE bar is not a readable instrument on a cancelled
  component, and the 474-of-547 over-bar count must be read with that in mind.

PROVES: the sigmoid-gated AdaLN backward is right — 4 of 4 parameter gradients at 8.679667e-04
to 2.194329e-03 against a float64 reference built from upstream 0.4.3's own class, norm ratios
inside [0.99877, 1.00087], cosines above 0.9999981, zero-model baseline 1.000000, and the fused
gate bit-identical to the unfused one in fp32 and bf16. The attention-side gradient defect is the
softmax: replacing it with float64 on the real operands takes the campaign's worst leaf from
8.060854e-01 to 1.459256e-02 at block 8, and from 2.933287e-01 and 1.829792e-01 to 5.217859e-03
and 6.438348e-03 at blocks 0 and 12, while the sister AdaLN with no softmax above it moves 1.23x.
The kernel's forward accuracy is 2.274755e-02 with no compute kernel config, 6.888150e-04 with
`precise_config()` and 2.780055e-04 through `_accurate_softmax`, against a host float32 floor of
6.053848e-08, and the backward's `g - sum(g*y)` cancellation amplifies whichever of those it is
given, linearly in the cancellation ratio. Three named precision levers are refuted with a
control that moves: `precise_config()` on the softmax backward's reduction, and the gain
gradient's summand product in fp32 on both activation dtypes. And the within-leaf split is not
conditioning: the attention-side gain sum has K 172.6 at block 8 and 175.6 at block 9 while the
model reads 18.504 and 0.1237, and under one controlled cotangent the two blocks read 8.060854e-01
and 6.047823e-01. The conditioned transition, run alone on real operands against upstream's own
float64 module, is inside the mass bar at three blocks with 0 of 9 parameters over the
per-tensor bar, so the campaign's 15.5125 % leaf is not made inside it; and with an exact
softmax the whole block is 1.489217e-02 mass-weighted over 19 of 19 parameters.

DOESNOT: this does not ship a fix. Both levers change the forward, so they move inference on four
of five models and they stay on this branch under the release gate. It does not measure the 48
noise levels or the real downstream cotangent at the block, so the block-arm numbers are the
linear map's, not one particular vector's through it. It does not gradcheck items 2 to 5 of the
handoff individually; it bounds them together. The K values are measured under a controlled
random cotangent, not the model's own: they establish that blocks 8 and 9 are matched in
conditioning, and they do not measure what K each block has in the real backward, which needs a
cotangent at `dit_out` that the boundary capture does not carry. It does not locate the
793x-to-6,970x floor between the device and torch float32; it rules out the summand dtype and
the two reduction configs and leaves the rest open. It does not explain
`attention_pair_bias.linear_z.weight` and `layer_norm_z.weight`, which stay over the per-tensor
bar under an exact softmax at 1.107297e-01 and 5.341521e-02 while holding 0.008 % of the block.
And the whole-block and transition readings are taken under one controlled cotangent on one
sample, so they measure the modules' arithmetic and not the arm's number under the real
cotangent over 48 samples, which is where amendment 6's 0.1867 comes from. It does not re-derive the model-scope number after
a lever, so how 55x at one block composes over 24 blocks and 48 samples is not established here.
And it is a statement about one backward at one step, not about stability: it does not show that
a training run stays on the reference trajectory over 100k steps, it does not bound long-run
drift, and nothing here licenses a claim about a full run.

A23 note, corrected. A per-tensor relative bar is not a readable instrument on a gradient
component whose reference has cancelled: a correct implementation reads rel 26.4 at a
cancellation ratio of 2.1e+06 and 2.1e-03 at 20, so the 474-of-547 over-bar count at diffusion
scope must be read with that in mind. An earlier version of this note added that a cancelled
component carries little mass by construction. It does not, and amendment 6 is right to strike
it: the tensor at the highest cancellation here is 8.05416 % of the model and its
fourth-heaviest. A large result can be a heavily cancelled sum, which is why it was worth
chasing.
