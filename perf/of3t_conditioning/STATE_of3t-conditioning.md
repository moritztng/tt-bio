# of3t-conditioning — instrument A at `diffusion_module.diffusion_conditioning`

VERDICT: GO

Row `of3t-conditioning`, pass 1, 2026-09-20, qb2 card 2. Branch `wk/of3t-conditioning`
(5ae68df2d, pushed, verified against origin). Artifacts in `perf/of3t_conditioning/`.
The orchestrator holds the merge gate; this branch stays on its own.

The record said this scope was blocked on moving the boundary `device_gradient.py` walks. It
was not, and had not been since the sub-boundary capture landed: the reference side was
complete and our side did not exist. Our side now exists.

SCOPE: the reference drew `use_conditioning=False` at this step (their coin flip at p=0.8), so
their own forward zeroes `si_trunk` and `zij_trunk` before the concat and the published
gradient is that branch. We feed zeros for the same reason — it is the function the cotangent
and the published gradient were taken at. It is also not the branch a user gets, so a second
arm runs the conditioned branch and is reported separately. Nothing in `tt_bio/` changed.

BOUNDARY: `perf/of3t_conditioning/capture_cond_boundary.py` takes the conditioning's INPUTS,
which `sub_boundary.py` captured in a pre-hook and did not save, by rebuilding their
`DiffusionConditioning` standalone in float64 at the 0.4.3 diffusion boundary. It checks itself
twice rather than asserting. It reproduces `sub_boundary.pt`'s `cond_out` at 0.000e+00 on si
and 0.000e+00 on zij, so it is the function the cotangent was taken at. And seeding that
output with `cond_out_cot` reproduces all 26 published float64 gradients at 0.000e+00 worst,
26 of 26 — which is PROTOCOL A20's precondition as a measurement rather than an argument: a
cotangent at a boundary is a complete gradient only if the parameters reach the loss through
that boundary alone, and `DiffusionConditioning.forward` returning `(si, zij)` is the reason
to expect it, not the evidence for it. D19 is upstream of these inputs, so the floor under
everything below is zero and the 6.735e-03 forward gap is not this row's result.

OURS: `OF3DiffusionConditioning`, `tt_bio/openfold3_diffusion.py:64` — the class
`openfold3_diffusion_module.py:187` names as the separate thing that feeds `OF3DiffusionModule`.
Its device instrument is `perf/of3t_conditioning/device_cond_gradient.py`; the bijection is the
transpose-invariant fingerprint over `ttnn.from_torch`, and it named 26 of 26 device weights
with 0 fingerprint clashes, so no load path was missed.

FORWARD: PROTOCOL A18's discriminator, run before any gradient, on the captured inputs, per
output. At fp32 activations (the arm the gradient is taken at) si reads 2.641455e-03 and zij
reads 2.629445e-03 against the 5.0e-02 bar; the worst of the 48 noise levels is 3.432350e-03.
At bf16 (shipped inference) si reads 2.542290e-03 and zij 2.522024e-03, worst sample
4.979349e-03. On the conditioned arm, fp32 si 2.674416e-03 and zij 3.778416e-03. Four readings,
all inside A19's `< 5.0e-02` branch, so the gradient is taken and the forward is reported
beside it as the floor under it. Per the A18 addendum this clears a NECESSARY condition and
nothing more: D9 measured a 3.2x gradient shift under a 12 % forward shift, so 2.6e-03 on the
forward is not a prediction of 2.6e-03 on the gradient.

GRADIENT: mass-weighted `rel_l2` over the concatenated 26 is 7.865385e-03, against the
2.0e-02 mass-weighted bar and a measured zero-model baseline of 1.0 — a 127.1x separation from
a deleted model. The compared set is all 26 tensors and holds 36.9462 % of the model's squared
gradient norm, which is the full reference mass in scope (A20: 100 % of the section, nothing
unreached, nothing absent). The median over tensors is 7.440910e-03 and is reported here
beside the headline and never instead of it (A23): the median tensor of this model holds
1.305e-04 % of its mass, and 91.85 % of this section sits in four tensors.

GRADIENTS: per-tensor, 26 of 26 compared against the reference's float64 gradients, 0 over the
5.0e-02 per-tensor bar. Worst is 1.204672e-02 on
`transition_s.1.layer_norm.weight` (norm ratio 0.99222, cos 0.999957, 1.02205 % of the model);
best is 4.488700e-03 on `linear_z.weight`. The 22 tensors outside the heavy four hold
3.01075 % of the model between them and their worst is that same 1.204672e-02. Split by
branch: the single branch is 14 tensors and 36.88049 % of the model, worst 1.204672e-02; the
pair branch is 12 tensors and 0.06569 %, worst 9.072400e-03. Both arms land in the same place,
so the disagreement is not localised anywhere — it is the flat fp32-against-float64 floor,
which is the shape A18 says to expect when nothing is mis-wired.

HEAVY: the four tensors holding 33.93543 % of the model over 1,985 scalars, each with its own
`rel_l2`, norm ratio `r = ||g_dev||/||g_ref||` and cosine, because rel alone bounds r to
[1-rel, 1+rel] and says nothing about direction.

  transition_s.0.layer_norm.bias    10.38237 %   rel 7.2277e-03   r 0.996500   cos 0.999980
  transition_s.1.layer_norm.bias     9.29095 %   rel 6.6915e-03   r 0.996320   cos 0.999984
  transition_s.0.layer_norm.weight   8.80266 %   rel 6.5958e-03   r 1.003747   cos 0.999985
  layer_norm_s.weight                5.45946 %   rel 1.1330e-02   r 0.997889   cos 0.999938

All four are inside the per-tensor bar, all four have cos > 0.9999, and the norm ratios
bracket unity (0.9963 to 1.0037) rather than sitting to one side, so there is no systematic
under- or over-estimate to report. Two of them — `transition_s.1.layer_norm.bias` and
`transition_s.0.layer_norm.weight` — are finite-difference validated against the reference at
5.088e-03 and 1.501e-05, so a disagreement on those would have been unambiguously ours. There
is none above the FD set's own worst point.

CONDITIONED-ARM: the branch inference runs. The reference never took it at this step, so its
float64 answer was recomputed from their code, their weights and the same inputs with the
trunk live, seeded with the same cotangent, and is labelled a synthetic-branch reference
everywhere it appears (`capture_cond_boundary.py --force-conditioning`). It is a different
function and the numbers say how different: si moves 1.861e-01 and zij 1.042e+00 between the
branches. Mass-weighted `rel_l2` there is 7.911357e-03, worst 1.035072e-02 on
`transition_s.0.swiglu.linear_a.weight`, 0 of 26 over the per-tensor bar, and the heavy four
read 8.823e-03 / 6.988e-03 / 9.527e-03 / 7.119e-03. The model-share denominators are emitted
as null on this arm rather than as numbers, because the model's squared norm was measured on
the drawn branch and a share of it is not defined against a different function.

CONTROL: two, because a single agreeing number is the easiest thing in this campaign to get
wrong. (i) A16's zero-model baseline, measured on this comparison rather than assumed: with
our gradient replaced by zeros the mass-weighted statistic reads exactly 1.0 and the median
over tensors reads exactly 1.0. It lands on 1.0 by construction — a relative L2 normalised by
the reference's own norm can do nothing else when the numerator is the reference — and the
useful content is the 127.1x separation from 7.865e-03, not the baseline itself. (ii) Because
that control breaks OUR side only, a second one breaks THEIRS: `--negative-control
reverse-cot` seeds sample k with sample 47-k's cotangent, leaving the forward, the weights,
the bijection and the arithmetic untouched. The headline moves to 2.161892e-01, 27.5x the
real reading and 10.8x the bar, 14 of 26 tensors over bar, worst 8.301132e-01 on
`layer_norm_n.weight`, and all four heavy tensors leave the bar. The check reads their seed.

A third guard is the accumulation probe. The 48 noise levels are summed by 48 tapes
accumulating into the same leaves, which is exact only if `backward` adds across tape
contexts. The probe first watched whichever leaf had a gradient first — `layer_norm_z.weight`,
a pair-branch tensor the pair tape fills once and no single tape touches. It read the same
value 48 times and would have read the same value had accumulation been broken, so it could
not fail. Pinned to `transition_s.0.layer_norm.bias` it grows 4.6692e-02 to 1.0295e+00 over
the 48, a factor of 22.0.

A14: 0 of 26 tensors excluded. A14's 1e-12 reference-norm cut removes nothing here — the
smallest reference norm in the section is 2.9e-03 — and per A23 the low-mass tensors are
weighted by their own norm in the headline rather than dropped.

MODELS: 0 of 5 shipped models are touched by this row. The tracked diff against `wk/of3t` is
empty; the branch adds `perf/of3t_conditioning/` and nothing else, so no inference path moves
and there is nothing to re-digest. `OF3DiffusionConditioning` has 6 call sites outside this
row's own instrument and 0 of 6 belong to the other four models — they are
`openfold3_diffusion.py`, `openfold3_diffusion_module.py`, `openfold3_sample_diffusion.py`,
`scripts/of3_sample_diffusion_golden.py` and the two OF3 tests. What the other four do share
is the tape: this row exercised 6 of the 37 verbs registered in `tt_bio/taped_ttnn.py` (concat,
layer_norm, linear with a fused silu, multiply, add, deallocate) and every gradient above is
evidence about that shared surface at 384 tokens. 1 of 5 models has a float64 reference
gradient at a conditioning boundary today; the other four have no captured boundary, so the
arithmetic here is OpenFold3's alone.

The fixture PCC gate `tests/test_openfold3_diffusion_conditioning.py` skips on qb2:
`~/of3_ref_out.pkl` on this host is a partial per-host capture carrying 7 keys and
`diffusion_conditioning_real` is not among them. The A18 forward above is the stronger control
in any case — 2.641e-03 relative L2 against a float64 reference on a real 384-token boundary,
against that gate's PCC > 0.98 on a 76-token fixture — so the row is not held on it.

---

# Amendment 1 (orchestrator pass 151, D53) — the diffusion arm's mass-weighted number

RERUN: the re-run this amendment asks for was already on disk and did not need a card.
`perf/of3t_rebase/device_gradient_043pt.json` carries the per-tensor array for all **547 of
547** compared tensors, with `rel_l2`, `ref_norm`, `device_norm`, `norm_ratio` and `cos`. It is
the same measurement as the `device_gradient_043all.json` the campaign quotes, checked rather
than assumed: 26 of 26 aggregate keys equal, all 48 `forward_rel` entries bit-identical
doubles, the 48-entry accumulation probe bit-identical, `worst10` equal. And the boundary it
read is now pinned directly rather than by inference — a one-structure re-run against
`diffcap043` on card 2 reproduces that run's structure-0 forward rel **1.104135e-02** and probe
norm **1.2509712481726117e-04** to the last digit. Analysis in
`perf/of3t_conditioning/mass_on_the_diffusion_arm.py`, results in `MASS_ON_THE_DIFFUSION_ARM.json`.
No number below was re-measured.

DIFFUSION-ARM: the mass-weighted `rel_l2` over the 547 is **7.569**, against the same run's
median-over-tensors of **0.16588** — the mass-weighted figure is **45.6x worse**, and the
measured zero-model baseline is 1.0, so by mass this arm is **7.6x worse than a deleted
model**. D53 predicted the direction and understated the size. The compared set holds
**51.1358 %** of the model's squared gradient norm; the full reference mass our
`OF3DiffusionModule` is asked to cover (the captured diffusion module minus the 26
conditioning tensors, which are a separate class on our side) is **52.2644 %** of the model, so
reach is **97.84 %** of scope with **1.1286 %** of the model in scope uncompared (A20). The
bijection named 547 of 870 reachable device weights, and the 323 it did not name are
1.13 % of the model between them, so incompleteness of the name map does not explain the
reading. Both statements are true and they are different: by mass one leaf decides the
number, and by count **474 of 547** tensors are over the 5.0e-02 per-tensor bar at a median
of 0.166.

LEAF: one leaf is **99.925 %** of the diffusion arm's squared error.
`attention_pair_bias.layer_norm_a.layer_norm_s.weight`, present in all 24 DiT blocks, holds
25.579 % of the model and reads mass-weighted **10.698**, median 0.839, worst 18.504, minimum
cosine **-0.8046** and maximum norm ratio **19.242**. The next leaf down,
`attention_pair_bias.layer_norm_a.linear_g.bias`, is 0.030 % of the squared error. Grouping by
leaf before reading the worst tensor is what makes this a locus rather than a tail: block 8
looked like the location because block 8 is 9.84 % of the model, but the leaf misbehaves in
every block. The control that makes it specific is one row down the same table —
`conditioned_transition.layer_norm.layer_norm_s.weight` is the same kind of AdaLN gain on the
other sub-block of the same 24 blocks, holds a comparable **15.513 %** of the model, and reads
mass-weighted **0.183** with max norm ratio 1.240 and min cosine 0.800. So this is not AdaLN
gains in general and not the DiT in general; it is the AdaLN gain inside
`attention_pair_bias`. Mass-weighted rel inside the DiT is **8.195** and outside it **0.200**.

HYPOTHESIS: D53's magnitude hypothesis is **REFUTED**. Over the full family of **24 of 24**
blocks, Spearman rank correlation between `rel` and `||g_ref||` is **0.236** and a log-log fit
gives slope 0.331 with **r² = 0.094** — 9 % of the variance. On the six worst points alone,
the ones the hypothesis was formed from, the same correlation reads **0.657**. The trend was
the selected tail, exactly as D53 said it might be. The family's actual signature is not
magnitude but **direction and scale**: across the whole 547, **21 tensors holding 4.8058 % of
the model have a NEGATIVE cosine** against the reference — our gradient points the other way —
and **7 tensors holding 14.1415 % of the model have a norm ratio above 2**, up to 19.2 on
block 8. Neither is visible in `rel`, which bounds r to [1-rel, 1+rel] and says nothing about
c. Both were already in the run's own output and had never been read.

The general lesson the amendment names is now enforced rather than argued:
`perf/of3t_diffusion/device_gradient.py` writes the per-tensor array to a sidecar
`device_gradient<tag>_per_tensor.json` **unconditionally**, not behind `--dump-per-tensor`, and
every report and sidecar now records its `--cap`, its checkpoint and its argv. Tying
`_043pt` back to `_043all` needed 48 bit-identical doubles precisely because no artifact said
which boundary either had read. Validated by execution, not by reading: the one-structure run
above wrote both files with 547 per-tensor entries and the provenance block populated.

PROVES: our device `OF3DiffusionConditioning` computes the same function and the same parameter
gradients as OpenFold3's `DiffusionConditioning` at BUNDLE-MIN-043's r = 0 boundary, to
7.865e-03 mass-weighted over 36.9462 % of the model's squared gradient norm, with every one of
the 26 tensors inside the 5.0e-02 per-tensor bar and the four tensors that carry 33.9354 % of
the model agreeing to 6.6e-03..1.2e-02 with cosines above 0.9999. The reference is float64 and
two of the four heavy tensors are additionally finite-difference validated, so no branch of
"is the reference right" is open on them. The same holds on the conditioned branch inference
runs, at 7.911e-03, against a float64 reference recomputed from their code. The comparison is
gated on a forward that agrees at 2.6e-03, its reach is the full reference mass in scope with
nothing absent, and it is demonstrated to fail when the reference's own seed is permuted.
Separately (Amendment 1): the diffusion arm's mass-weighted rel_l2 at the 0.4.3 boundary is
7.569 against a zero-model 1.0, 99.925 % of its squared error sits in one leaf,
`attention_pair_bias.layer_norm_a.layer_norm_s.weight`, across all 24 DiT blocks, and D53's
magnitude hypothesis for that leaf is refuted on the full family at r² = 0.094.

DOESNOT: Amendment 1's numbers are a re-analysis of a run taken by `of3t-rebase`, not a new
measurement, so they inherit that run's scope exactly and add nothing to it; naming the leaf
locates the disagreement and does not diagnose its cause, and no fix is proposed or made
here. The conditioning arm proper: this is one gradient at one boundary of one step, so it proves the update rule's
input at that point and says nothing about stability over a full run — not over 100k steps,
not over any drift the optimizer accumulates, and not over the long-horizon behaviour a
training reproduction would need. It is not a claim about the diffusion transformer (43.8936 %
of the model) or any other section. An agreeing forward clears a necessary condition only
(A18 addendum, D9). The drawn branch has `use_conditioning=False`, so the published-reference
arm exercises the conditioning with both trunk inputs zeroed; the conditioned arm covers that
gap but its reference is one we computed, not one the campaign published. No timing claim is
made anywhere in this row, so no AICLK is quoted. And nothing here is merged.

GAP: the conditioned arm's reference is a synthetic-branch one by construction — the bundle
would have to draw `use_conditioning=True` at some step for a published equivalent to exist.
