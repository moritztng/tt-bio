# of3t-adaln — the AdaLN micro-arm, and what the 18.504 actually is

VERDICT: GO

Row `of3t-adaln`, pass 1, 2026-09-20, qb2 card 3. Branch `wk/of3t-adaln` (ab60d845f, pushed,
verified against origin). Artifacts in `perf/of3t_adaln/`. Nothing in `tt_bio/` changed: the
tracked diff against `wk/of3t` touches 0 files under `tt_bio/` and adds 6 under
`perf/of3t_adaln/`. The orchestrator holds the merge gate; this branch stays on its own.

The record had four candidate causes for the campaign's worst gradient disagreement, three
eliminated by source read and one weakened by an order-of-magnitude argument. A source read is
an argument. This row replaces it with a measurement, and the answer is that the sub-module is
correct and the number is real but means something other than what it looks like.

MICROARM: one `tenstorrent.AdaLN`, built from DiT block 8's real checkpoint weights through
`remap_of3_adaln`, run taped on real-shaped `(a, s)` = `[1, 384, 768]` and `[1, 384, 384]`,
seeded with a fixed cotangent, against a float64 torch autograd reference built from upstream
OpenFold3 0.4.3's own `AdaLN` class (`openfold3/core/model/primitives/normalization.py:88`,
`of3pkg043` on qb2) with the same weights, the same inputs and the same cotangent. Instrument
`perf/of3t_adaln/adaln_micro.py`, 6 s per arm on one card.

A18 first: the forward discriminator is read off the SAME taped forward the backward is taken
from, and reads 1.055e-03 against the 5.0e-02 bar. Then the four parameter gradients, each with
`rel_l2`, the norm ratio `r = ||g_dev||/||g_ref||` and the cosine, because rel alone bounds r to
[1-rel, 1+rel] and cannot tell a 19x magnitude error from a wrong direction:

  layer_norm_s.weight   rel 2.072079e-03   r 1.00086787   cos 0.99999823   |g_ref| 3.430232e+03
  linear_g.weight       rel 2.001244e-03   r 0.99950548   cos 0.99999812   |g_ref| 2.593827e+01
  linear_g.bias         rel 8.679667e-04   r 1.00005758   cos 0.99999962   |g_ref| 1.284351e+02
  linear_s.weight       rel 2.194329e-03   r 0.99876525   cos 0.99999835   |g_ref| 1.117242e+02

Mass-weighted over the four: 2.071014e-03, against a measured zero-model baseline of exactly
1.000000, a 483x separation. 0 of 4 over the 5.0e-02 per-tensor bar. The gated branch is the
BEST of the four here, not the worst. `layer_norm_s.weight` is the leaf that reads 18.504 in the
model and it reads 2.072079e-03 here with a norm ratio three parts in ten thousand from unity.

GRADIENTS: the micro-arm's worst is 2.194329e-03 on
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.linear_s.weight`,
the one parameter in the module that never touches the sigmoid gate. The model's worst, the one
this row was dispatched for, is
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at rel 1.850397e+01, and this row measured what was missing from it: `r` 19.2415, `cos` 0.74941,
`||g_ref||` 9.0991e-01, `||g_dev||` 1.750810e+01. The published artifact kept only `(name, rel)`
for its ten worst, so `r` and `cos` did not exist anywhere until now; they were produced by
running `perf/of3t_diffusion/device_gradient.py --structs all --cap
/home/ttuser/of3t_rebase/diffcap043 --dump-per-tensor` unmodified, writing into this row's own
namespace, and it reproduces the published `worst_rel` 18.503974843073646 to all digits. That
`r` tracks `rel` while `cos` falls to 0.75 is the whole finding: a wrong transform reads r ~ 1
with cos ~ 0, a wrong constant reads r ~ 19 with cos ~ 1, and neither is what the tensor says.
Across all 547 compared tensors the median cosine is 0.99002 and the median `r` is 1.0208, so
the bulk of the diffusion module agrees in both scale and direction and the failures are a
minority with a shared shape: 17.00 % of tensors sit below cos 0.5.

FUSED_VS_UNFUSED: the gate unfused — `ttnn.sigmoid` as its own taped op, then a plain
out-of-place `ttnn.multiply`, with the module's own `s_terms` and everything else identical —
against the same float64 reference. It is BIT-IDENTICAL to the shipped
`multiply_(a, s_scale, input_tensor_b_activations=[SIGMOID])` path: all four tensors agree on
`rel_l2`, `r` and `cos` under exact float equality, in fp32 (2.071014e-03) and again in bf16
(5.813832e-03). The fused sigmoid rule in `taped_ttnn.py:444-515` is therefore not merely
source-correct, it is measured to produce the same gradient as the unfused form, and the
elimination the orchestrator made by reading now has a measurement under it.

SCALES: neither of the two variables the defect was expected to scale with moves it.

  Input magnitude, PROTOCOL A22 dirty mantissas. `s` at 0.0137 / 0.1 / 1.0 / 3.7 gives
  1.962329e-03 / 2.006337e-03 / 2.071014e-03 / 2.082929e-03. Over a 270x range the headline
  moves 6.2 %. `a` at 0.0137 / 1.0 / 3.7 gives 2.075291e-03 / 2.071014e-03 / 2.048761e-03, a
  1.3 % spread. The power-of-two `s` arms (0.125, 2.0) read 2.068573e-03 and 2.070098e-03,
  inside the dirty band, and they are reported as arms rather than as a control: `layer_norm`'s
  eps and the sigmoid are not homogeneous in `s`, so a power-of-two `s` arm is nearly invariant
  by construction and A22's objection applies to it.

  The arithmetic control is on the COTANGENT, where the backward IS exactly linear. At
  cot x 2^-5 and cot x 2^+5 the reference norms scale by exactly 2^-5 and 2^+5 while `rel_l2`,
  `r` and `cos` are bit-identical to the base arm on all four tensors. Linearity in the seed is
  established, so the non-power arms above are interpretable.

  The sample axis. 1, 4 and 48 tapes accumulating into the same four leaves give 2.071014e-03,
  2.085571e-03 and 2.116898e-03, a 2.2 % move over 48x. The accumulation probe watches a leaf
  every sample touches and grows 3.433209e+03 to 2.364329e+04, a factor 6.887 against the 6.928
  that independent draws predict, so accumulation across tape contexts is exact through the gate
  as well as around it.

  And the weights are not the block-to-block variable either. The same inputs and the same
  cotangent through blocks 0, 5, 6, 7, 8 and 12 — the six DiT blocks in the campaign's ten worst
  — give 1.861037e-03, 2.135488e-03, 2.087538e-03, 2.008489e-03, 2.071014e-03, 2.105445e-03: a
  1.15x spread against the 6.8x spread the model shows across the same six blocks (2.7329 at
  block 0, 18.5040 at block 8).

CONTROL: four, because the whole value of this row is that a clean number is believed.

  (i) A16's zero-model baseline, measured on this comparison on every arm rather than assumed:
  replacing our gradient with zeros reads exactly 1.000000 mass-weighted and 1.000000 per
  tensor. The base reading is 483x below it.

  (ii) The reference is checked instead of trusted. The hand replica used to compute the
  cancellation ratio reproduces upstream 0.4.3's own `AdaLN.forward` at rel exactly 0.0 on every
  arm, so the conditioning statistic is computed on the same function the gradient is compared
  against.

  (iii) The float32 HOST floor: the same class, the same weights, the same inputs, the same
  cotangent, in torch float32 on the CPU with no device and no tape. It reads 2.971330e-07. Our
  device arm is 6,970x above it at benign conditioning, which is what says the device reading is
  an arithmetic floor and not a rule error. It is also the number that stops "2.071014e-03 is
  small" from being the end of the story.

  (iv) The precision lever, priced rather than argued. `AdaLN.__init__`
  (`tenstorrent.py:9791-9794`) hands `torch_to_tt` the literal `ttnn.bfloat16` instead of routing
  it through `_dtype`, so all four of its weights stay bf16 even under an fp32 activation
  override. Replacing them with fp32 copies from outside `tt_bio/` buys 2.4x at K = 20
  (2.071014e-03 to 8.479500e-04, with `r` 1.0000 and `cos` 1.000000 on the gain), 1.2x at
  K = 2.1e+04 and 1.0x at K = 2.1e+06. It is real and it is small, and at the conditioning that
  would be needed to explain 18.504 it buys nothing. No change to `tt_bio/` is proposed on this
  evidence.

MECHANISM: what does move the number is the conditioning of the LayerNorm-gain reduction itself.
The gain gradient is a sum over tokens, `g_gamma = sum_i (dL/d ln_out)_i * s_hat_i`, and its
relative error is set by the size of the summands against the size of their sum. The ladder
holds `a` and `s` constant across tokens so the cotangent-to-summand map is the same at every
token, then alternates the cotangent's sign so the 384 summands cancel down to a controlled
residual. `K = sum_i||term_i|| / ||sum_i term_i||` is measured on each rung, not assumed:

  K 3.009e+01   device 2.059085e-03   host fp32 4.027367e-07   gain r 1.00053   cos 0.999998
  K 2.112e+02   device 7.351748e-03   host fp32 2.671127e-06   gain r 1.00111   cos 0.999974
  K 2.100e+03   device 5.864179e-02   host fp32 2.747817e-05   gain r 0.99426   cos 0.998285
  K 2.100e+04   device 5.744067e-01   host fp32 2.598909e-04   gain r 1.13783   cos 0.863185
  K 2.100e+05   device 4.161453e+00   host fp32 2.595698e-03   gain r 4.21914   cos 0.171778
  K 2.100e+06   device 2.635289e+01   host fp32 3.324537e-02   gain r 26.52932   cos 0.162413

rel is linear in K to within a factor of two over five decades, and the SIGNATURE at high K is
the one the real tensor shows: `r` rises with `rel` while `cos` collapses, because the
reference's sum cancels and the device's rounding residue does not. rel 18.504 sits between the
K = 2.1e+05 and K = 2.1e+06 rungs; log-log interpolation puts it near K = 1.3e+06. The host
float32 column is the same statement about single precision generally: torch fp32 on a CPU
crosses the 5.0e-02 bar on this same sum somewhere near K = 3e+06.

MODELS: 0 of 5 shipped models move. The branch adds `perf/of3t_adaln/` and nothing else, so no
inference path changes and there is nothing to re-digest. The sub-module this clears is shared:
4 of the 6 `tt_bio` modules that construct an AdaLN use `tenstorrent.AdaLN` — `tenstorrent.py`
itself via `ConditionedTransitionBlock`, `openfold3_atom_transformer.py`,
`openfold3_diffusion_transformer.py` and `protenix.py` — while `boltz2.py` and `esmfold2.py`
carry their own class and are outside this reading. The two atom-transformer `linear_g` tensors
in the campaign's ten worst come through `openfold3_atom_transformer.py`, which is one of the 4,
so the same clearance covers them.

PROVES: the sigmoid-gated AdaLN backward computes the right gradient. Against a float64 autograd
reference built from upstream 0.4.3's own class, at block 8's real weights and at real shapes,
all 4 of 4 parameter gradients read 8.679667e-04 to 2.194329e-03 with norm ratios inside
[0.99877, 1.00087] and cosines above 0.9999981, 0 of 4 over bar, against a measured zero-model
baseline of 1.000000. The fused-sigmoid multiply is bit-identical to the unfused form in fp32
and bf16, so `taped_ttnn`'s fused activation rule is confirmed by measurement and not only by
reading. Input magnitude across 270x, the sample axis across 48x and the block weights across
six blocks together move the headline by less than 7 %, so none of them is the block-to-block
variable. The disagreement class the campaign has been chasing is the amplification of a device
arithmetic floor by an ill-conditioned reduction, and the norm ratio and cosine of the actual
worst tensor — 19.2415 and 0.74941, measured here for the first time — match that signature and
exclude both a wrong transform and a wrong constant.

DOESNOT: this does not measure K at the model's own block-8 inputs. That needs the real
activations and the real upstream cotangent at that AdaLN, which live inside a float64 DiT run
this row deliberately did not do, so "the real K is about 1.3e+06" is an interpolation on this
arm's slope and is labelled as one everywhere it appears. It does not price a fix, because the
one lever found is worth 2.4x at benign conditioning and nothing at the conditioning in
question. It says nothing about `boltz2.py`'s or `esmfold2.py`'s own AdaLN classes, which are
different code. It carries no timing claim and no AICLK, because it makes none. And it is a
statement about one backward rule at one step, not about stability: it does not show that a
training run stays on the reference trajectory over 100k steps, it does not bound long-run
drift, and nothing here licenses a claim about a full run.

A23 note for the protocol. A per-tensor RELATIVE bar is not a readable instrument on a gradient
component whose reference has cancelled. At K = 2.1e+06 a correct implementation reads rel 26.4;
at K = 20 the same implementation reads 2.1e-03. The campaign's mass-weighted headline is not
affected the same way, since a cancelled component carries little mass by construction, which is
the argument for A23 stated from the other side. The 474-of-547 over-bar count at diffusion scope
should be read with that in mind.
