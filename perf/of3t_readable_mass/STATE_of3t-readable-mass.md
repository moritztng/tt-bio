# of3t-readable-mass — the denominator of the reproduction claim, read off the artifacts

TASK TYPE: VERIFY/BENCHMARK. Branch `wk/of3t-readable-mass`, head `758ff7003`, pushed, unmerged.
Artifacts `perf/of3t_readable_mass/`. One device run, card 3 on qb2, Blackhole p300c, a presence
census with no timing claim, so no AICLK is quoted. Everything else is CPU.

FINDING: **92.1568215695272 % of OpenFold3's gradient mass is comparable today**, 907 of the
4,170 reference tensors that receive a gradient, and the campaign's "about 2 % has no reading,
and part of it never can" is right on the share and wrong on the second half. The share with no
per-tensor device gradient anywhere is **2.01501 %**. None of it is permanently unreadable, and
the piece the record calls structural is **less than half the real host-applied share**.

`perf/of3t_orchestrator/COVERAGE_CEILING_IS_NOT_100.json` files 0.74055 % as structural and
names one tensor, `input_embedder.atom_attn_enc.linear_q.0.weight`, applied on the host after a
`ttnn.to_torch`. There is a **second host leg** it does not carry. `tt_bio/worker.py:1555` calls
the host replica `openfold3_host_prep.ref_atom_embed` for
`diffusion_module.atom_attn_enc.ref_atom_feature_embedder`, and `run_input_atom_encoder` calls
the same replica again for the input embedder's copy. Seventeen tensors,
**1.5202384841128946 %** of the model's squared gradient norm, 2.05x the figure on the
record. That is `of3t-hostops`' own
1.5202 % (D127) arrived at from the opposite direction: D127 read it off the discovery walk,
this reads it off the float64 reference gradient plus which function the shipped path calls.
Two independent routes, same number. The ceiling artifact was never updated and the campaign
has been quoting the smaller figure.

And it is not structural even so. `tt_bio/openfold3.py:88` already holds a device
`RefAtomFeatureEmbedder` with its own PCC gate (`tests/test_openfold3_ref_atom_feat.py`), and
nothing outside that test constructs it. The remedy for 0.75304 % of the model's mass is wiring
code that exists, not writing code that does not.

CAUSE: the campaign has been decomposing the uncompared mass **by section** and never by
blocker, so five unrelated obstructions with five different remedies were carried as one bucket
and the largest of them was described by the smallest of its parts. Underneath that, the
"0.74055 % never can" wording outlived its own correction: pass 213 had already settled that the
input embedder leg is blocked by two adjacent lines rather than by nature and that porting the
op closes it, and the prose kept the earlier framing.

FIX: one census, per tensor, in one denominator, every uncompared tensor given the blocker that
has to be removed FIRST, and the two classes that mean nothing is in the way kept apart.
`perf/of3t_readable_mass/readable_mass.py`, output `READABLE_MASS.json`.

    class                       tensors    % of model gradient mass    what has to change
    COMPARED                        907                  92.15682      nothing
    READ_ON_ANOTHER_BOUNDARY       2736                   5.82817      one arm on the model's batch
    HOST_APPLIED                     17                   1.52024      wire two host legs onto the card
    FUSED_NOT_SPLIT                  96                   0.27153      one inverse selection
    NO_ARM                          151                   0.11051      nobody has looked
    NOT_A_LEAF_LAZY                 126                   0.10450      register 84 weights as leaves
    ARM_RAN_DID_NOT_CARRY           137                   0.00822      somebody looked, it did not come back
                                  -----                 ---------
                                   4170                 100.00000

NO_ARM and ARM_RAN_DID_NOT_CARRY are split deliberately, and the split is derived rather than
judged: a section holding at least one compared tensor had an arm run over it. `msa_module` and
`aux_heads` are in the second class, and `msa_module`'s 73 are `of3t-auxheads`' open m-track
presence gap, where the arm ran, read 154 tensors and produced nothing for the nine-weight
`proj_m`/`proj_g`/`proj_o` group and its siblings. That is a different next step from
`template_embedder`'s 96, which nobody has pointed an instrument at.

CONSTRUCTION, because a share without one is the thing this row exists to replace. The
denominator is BUNDLE-MIN-043's **float64** reference gradient `grads_f64_043.pt`, sha256
`1d4ea922…95cc4`, checked by the script before it reads a number. A tensor's mass is its own
squared float64 gradient 2-norm; the model's mass is the sum over the 4,170 tensors that have
one, and it comes out at 10.279642678524981. The 765 checkpoint tensors the reference step gives
no gradient at all (763 of them `sample_diffusion`, 2 `diffusion_module`) carry no mass and are
counted but never given a share: a share of zero reads as "small" where the right word is
"absent". Computed from the file rather than copied, so 92.1568215695272 % is an independent
reproduction of `MODEL_shipped.json`'s `coverage_total.pct_of_model_compared`
(92.15682156952718) and not a restatement of it.

WHAT THE BIJECTION ACHIEVES. It is four instruments, not one, and none of them is a name map.
(1) `of3t-equivalence`'s tracer runs upstream's own remap functions on float64 tracer ids and
decodes each of our tensors back into the ids and dim-0 spans it is built from: 3,275 of their
4,935 map to exactly one (our tensor, span), 0 unmapped in scope, 0 unresolved spans, 0
one-to-many, and **232 of ours are fused from 464 of theirs**. Negative control: blanking one
source changes exactly the one of ours that sources it. (2) `of3t-gradients`' device bijection
matches by VALUE against the model built on a card after a materialising forward at 64 tokens:
3,545 of the 4,147 gradient-bearing tensors carried, 536 device tensors are concatenation
fusions holding 1,515 of theirs. (3) `of3t-hostops/fused.py` matches fusion GROUPS by a
fingerprint invariant to appended zeros and to concatenation order, 105 groups, with a shuffle
control that matches 0 of 200. (4) `of3t-trunkg043/dev_grad.py` accepts a derived placement only
if rebuilding THEIR weight out of the device weight is bit-identical. At model scope what that
buys is stronger than name agreement: a fusion, a transpose, a head-lane pad and a folded scale
are all things a name map would walk straight past and all four of these catch.

THE FUSED CLASS, and the answer is not the one the brief expected. A fused gradient compared
against a combination of upstream's would be weaker, because two errors could cancel inside the
fusion. **That is not what was done, and for this port it is not what is available.** Every
fusion the by-value bijection can name is `concat_axis1` (536 of 536), a disjoint selection of
the sources into a larger zero-padded array, so the same selection applied to the device
GRADIENT is the gradient of the source exactly, with no sum and no scale. `apb_inverse` in
`of3t-trunkg043/dev_grad.py` performs precisely that inverse, and the trunk's dump at the c64
boundary carries all 2,736 of their tensors separately, including the 240 fused qkv leaves and
the 384 fused trimul leaves, with distinct non-zero norms per component. There is no
cancellation channel in a concatenation, so no cancellation test is called for; the test that IS
run is the stronger one in the direction that matters, bit identity on the rebuilt weight, with
a placement that cannot reproduce its weight refused. The one fusion that is not a pure
selection is `linear_z.weight`, scaled by `_bias_scale` on the way to the card, and the
instrument divides by the scale it MEASURED on the card rather than the one the code requested.
The bound on this: 208 device tensors are unmatched by the by-value instrument and their
construction is not established by it, so "all fusions here are concatenations" covers the
fusions that instrument can name.

Where the fused comparison has NOT been done is the diffusion transformer. Its 96 reference
tensors (24 blocks x `mha.linear_{q,k,v}.weight` + `linear_q.bias`, 0.27153 %) go into one padded
`qkv_w`/`qkv_b` at `openfold3_diffusion_transformer.py:122-140`, and no instrument in that scope
applies the inverse. This is an instrument change, not a port change, and the probe below is
what makes that distinction a reading.

PERMANENTLY UNREADABLE VERSUS MERELY UNREAD. **Nothing in the 2.01501 % is permanently
unreadable.** Every class has a named remedy in live code. 1.52024 % needs two host legs moved
onto the card, and the device module for the larger of them already exists and passes its PCC
gate. 0.27153 % needs the inverse selection the trunk instrument already implements for the
identical fusion. 0.10450 % needs 84 weights registered as leaves, which is one call over a
cache the module already keeps. **"We have not compared it yet" is true of 0.11873 %** and of
nothing else: 0.11051 % where no arm has been pointed at the section and 0.00822 % where one
was and came back short. The other **1.89627 %** is "no reading exists until something in the
port or an instrument changes", which is a third claim, distinct from both of the two the
campaign has been making.

EVIDENCE: `perf/of3t_readable_mass/READABLE_MASS.json` for the census and
`perf/of3t_readable_mass/LEAFPROBE.json` for the two classes that would otherwise be source
readings. `leafprobe.py` runs `of3t-diffusion`'s `device_gradient.py` unmodified (it wraps
`of3_coverage._device_weights`, which that script imports by name inside `main()`, so the shared
instrument is not edited) and reads the module's own weight walk before and after the forward:

  * 870 device tensors reachable before the forward, **954 after, 84 materialised inside it**,
    named in the report as `dec.at._wc.('blocks.0…swiglu.linear_a.weight', True)` and the rest.
    `OF3AtomTransformer._w_tt` builds them lazily in `forward`; the walk that registers
    trainable leaves has already run. **0 of 84 are registered leaves and 0 of 84 receive a
    gradient**, while **870 of 870 walked before the forward do**. That is 42 per atom
    transformer, which is exactly the 14 weights the class resolves through `_w_tt` at its
    `_lin` sites over its 3 blocks, against the 37 it builds in `__init__` (`layer_norm_z` plus
    nine `AdaLN` submodules) and which are all compared.
  * the 48 fused `qkv_w`/`qkv_b` leaves, 24 blocks x 2, **48 of 48 carry a gradient**, from 2.89e-05
    (`dit.blocks.20.qkv_b`) to 6.67e-03 (`dit.blocks.10.qkv_w`), none zero. The gradient for
    those 96 reference tensors exists on the card and is not read.

Controls the numbers rest on, all from the artifacts rather than asserted: the reference is
float64 and pinned by digest, never another device arm; A13 determinism on that reference is
4,170 of 4,170 bit-identical across two runs; the campaign's A16 zero model reads 1.0 through
the same scorer and its break control 2.009101e+01; the instrument floor, upstream's own fp32
arm against float64 on the same 907 tensors, is 7.608574e-05.

Two smaller corrections worth keeping. The composed figure on the record, 97.98502156952716 %,
carries the trunk share rounded to 5.8282; the exact share is 5.828171499134286 and the composed
total is 97.98499306866148. And "0.74055 % can never be read" is one tensor of the
seventeen: it is a strict subset of the 1.52024 %, not a second reading of it, so quoting it as
the host-applied share understates that share by 2.05x and calls a wiring gap a ceiling.

VERDICT: PARTIAL. The denominator is now one number with its construction, the exclusions are
classified per tensor rather than per section, and the two classes that were source readings are
measurements. The campaign's headline share (92.1568 %) stands unchanged and was reproduced
independently. What moves is the excluded part: the host-applied share doubles to 1.52024 %, its
"structural" label does not survive, and the blockers behind the 2.01501 % have five
different remedies, of which the three largest are instrument or wiring work on code that
already exists. Nothing here merges; landing is `land-standing`'s.
