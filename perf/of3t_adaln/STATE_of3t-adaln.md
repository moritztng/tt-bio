# of3t-adaln — the AdaLN gate is clean; the attention softmax is the defect

VERDICT: GO

Row `of3t-adaln`, 2026-09-20, qb2 card 3. Branch `wk/of3t-adaln` (0b7584fc3, pushed, verified
against origin). Artifacts in `perf/of3t_adaln/`. Nothing in `tt_bio/` changed: the tracked diff
against `wk/of3t` touches 0 files under `tt_bio/` and adds 11 under `perf/of3t_adaln/`. Every
ablation below is installed from the instrument and removed again. The orchestrator holds the
merge gate; this branch stays on its own.

The brief was amended twice while this pass was running. Amendment 1 refuted its own premise
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
given, linearly in the cancellation ratio.

DOESNOT: this does not ship a fix. Both levers change the forward, so they move inference on four
of five models and they stay on this branch under the release gate. It does not measure the 48
noise levels or the real downstream cotangent at the block, so the block-arm numbers are the
linear map's, not one particular vector's through it. It does not gradcheck items 2 to 5 of the
handoff individually; it bounds them together. It does not re-derive the model-scope number after
a lever, so how 55x at one block composes over 24 blocks and 48 samples is not established here.
And it is a statement about one backward at one step, not about stability: it does not show that
a training run stays on the reference trajectory over 100k steps, it does not bound long-run
drift, and nothing here licenses a claim about a full run.

A23 note. A per-tensor relative bar is not a readable instrument on a gradient component whose
reference has cancelled: a correct implementation reads rel 26.4 at a cancellation ratio of
2.1e+06 and 2.1e-03 at 20. The mass-weighted headline is not affected the same way, since a
cancelled component carries little mass by construction. The 474-of-547 over-bar count at
diffusion scope should be read with that in mind.
