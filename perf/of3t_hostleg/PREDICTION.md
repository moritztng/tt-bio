# of3t-hostleg — pre-registration

Written and committed BEFORE the first device run of this row. Nothing below is adjusted after a
number exists; corrections go in an amendment block at the bottom with the date and what already
existed.

## The question

`READABLE_MASS.json` (`perf/of3t_readable_mass/`, sha256 `19ced3e9...`) classifies 17 of the 4,170
reference tensors as HOST_APPLIED: the shipped forward applies the weight on the host after a
`ttnn.to_torch`, so no cotangent returns through the graph the forward severed and no device
gradient exists. They hold **1.5202384841128946 %** of the model's squared gradient norm. Two
prefixes plus one name resolve them, and the checkpoint agrees at exactly 17:

  * `diffusion_module.atom_attn_enc.ref_atom_feature_embedder.` — 8 weight-only linears,
    0.7530394291090198 % of the model
  * `input_embedder.atom_attn_enc.ref_atom_feature_embedder.` — the same 8 linears in the other
    encoder, 0.0266940520755035 % of the model (the census total for the section, 0.7671990550038752, less linear_q.0.weight)
  * `input_embedder.atom_attn_enc.linear_q.0.weight` — one [384, 128] weight, **0.7405050029283717 %**
    of the model on its own, the single heaviest of the 17 and the tensor the campaign carried for
    sixteen passes as "0.74055 % can never be read" until A32 withdrew that wording

## Where the two legs are in the shipped tree, by line

  * `tt_bio/openfold3_host_prep.py:178 ref_atom_embed` — the host replica of
    `RefAtomFeatureEmbedder`, five single linears summed into `cl` and three pair linears masked
    by `vlm` into `plm`, all in host float32.
  * `tt_bio/openfold3_host_prep.py:210 run_input_atom_encoder` — calls `ref_atom_embed` for the
    input embedder, completes the pair rep on host, runs the device `OF3AtomTransformer`, then
    `ttnn.to_torch` and `F.linear(ql * atom_mask, linear_q.0.weight).relu()` plus the
    atom-to-token mean, both on host.
  * `tt_bio/worker.py:1553` calls `run_input_atom_encoder` and `tt_bio/worker.py:1557` calls
    `ref_atom_embed` again for the diffusion module's encoder. Both results cross into
    `OpenFold3.fold(dm_aux_host=...)` as values, not as graph.

`tt_bio/openfold3.py:88 RefAtomFeatureEmbedder` is a device port of the whole of the first of
those, PCC-gated by `tests/test_openfold3_ref_atom_feat.py` at > 0.98 on cl and plm. Nothing
outside that test constructs it, which is verified by grep over the tree and is the reason this
row is a wiring row.

## THE FINDING THIS ROW REGISTERS BEFORE IT RUNS: the 17 are not one population

`readable_mass.py` files a tensor under the blocker that has to be removed **first**, and
HOST_APPLIED precedes NO_ARM in its precedence order. `sections_with_an_arm` is derived from the
COMPARED set, and `READABLE_MASS.json.classes.COMPARED.by_section` has **no `input_embedder`
entry**: the nine sections with an arm are the six diffusion-module sections, `aux_heads`,
`msa_module` and nothing else. The census writes the consequence down for its NOT_A_LEAF_LAZY
input-embedder tensors (`two_blockers`: "also NO_ARM: no device arm covers input_embedder on the
model's batch") and does **not** write it for the HOST_APPLIED ones, because `both` is only
populated on the `lazy` branch. The fact is true of them all the same.

So the 1.52024 % splits into two halves that cost different things:

  * **0.7530394291090198 %, 8 tensors, one blocker.** `diffusion_module.atom_attn_enc` already has
    an arm on the model's own batch — `perf/of3t_diffusion/device_gradient.py`, 48 of its tensors
    COMPARED, 4.734845387109066 % of the model. That instrument calls
    `HP.ref_atom_embed` at line 321 and hands the result in as `cl0_d`/`plm0_d` values. Wiring the
    device module into the shipped forward is sufficient: the existing arm then carries the 8.
  * **0.7671990550038752 %, 9 tensors, two blockers.** Nothing covers `input_embedder` on the
    model's batch at all. A device leg alone produces no reading here; an arm has to exist, which
    means a cotangent captured at the input embedder's own output boundary from upstream 0.4.3 and
    an instrument that seeds it. That is new code, not wiring.

The row's brief says "the remedy is wiring code that exists, not writing code that does not."
That is true of 0.75304 % of the model and false of 0.76720 % of it, including the 0.74055 %
tensor the brief names. Registered here so the row cannot later present the cheap half as the
whole.

## The arms

  * **OFF** — the shipped host legs, no flag. The inference control and the A/A floor.
  * **ON** — both legs on the card, selected by one flag. Release-gated and default-off, because
    the charter's HARD CONSTRAINT is that nothing this row does may make inference slower or
    change its output, and a device linear in bf16 or fp32 cannot be bit-identical to the host
    float32 one it replaces.
  * **BREAK** — `--permute-cot`, the diffusion instrument's existing control: structure k seeded
    with structure k+1's cotangent. A device leg whose gradient does not move under it is not
    reading the seed.

## Predictions

**P1 — coverage, if both halves land.** `pct_of_model_compared` over 907 + 17 = 924 tensors:
**93.67706005364017 %**, which is 92.15682156952727 + 1.5202384841128946. The row's contribution is
**+1.5202384841128946** points and is fixed by the census, not by the reading: the 17 names are
fixed, the checkpoint resolves exactly 17, and any other contribution means the bijection placed a
different set and the difference is the finding. Composed with `of3t-modelboundary`'s 5.82817 the
total is 99.50523155277445 % against the 99.2594 % bar.

**P1b — coverage, if only the single-blocker half lands this row.** **+0.7530394291090198**,
coverage 92.90986099863629 %, and composed with of3t-modelboundary 98.73803249777057 % — which
does **not** clear the 99.2594 % bar. Registered because it is the outcome the two-blocker finding
above makes likely for the first landing, and because the closure file's claim that these two rows
together clear the bar depends on the input-embedder arm existing.

**P2 — how many of the 17 are over the 5.0e-02 per-tensor bar, against float64.** The model-scope
reading today is 678 over the bar of 900 measurable (74.7 %), and `diffusion_module.atom_attn_enc`
holds 48 COMPARED tensors. Predicted: **12 of 17 over the bar, band 6-17.** These are bias-free
single linears one op deep from the input, so they are the shallowest weights in the model and
should read better than the model median; against that, their cotangent arrives through the whole
atom transformer and the DiT, so the error upstream of them is the model's error. The band is wide
on purpose and the reading is the finding.

**P3 — the worst of the 17.** Predicted **`diffusion_module.atom_attn_enc.ref_atom_feature_embedder.linear_ref_pos.weight`**,
on the grounds that it carries 0.6734034330383528 % of the model's mass, 89.4 % of its own
section's HOST_APPLIED mass, and is the only one of the eight fed by a coordinate rather than a
mask or a one-hot. A [128, 3] weight has 384 scalars, so its rel L2 is an average over very few
numbers and is the least forgiving of the eight.

**P4 — inference with the flag OFF is byte-identical.** The fold digest on OpenFold3 at the
row's cell equals `origin/wk/of3t`'s, and the A/A floor is 0 bytes moved. If it is not, the wiring
touched the default path and the row stops and says so.

**P5 — inference with the flag ON moves the digest.** A host float32 linear and a device bf16 or
fp32 one are not the same arithmetic. The structural question is whether the move is under the
kill bar with the seed floor beside it; the wall-clock question is whether the eight small linears
plus the [384, 128] one are faster on the card than on the host, and at 1UBQ's 602 atoms they are
small enough that the host round trip may well dominate either way. Registered as: **the digest
moves, and the row reports the Angstrom against the kill bar and the seed floor rather than
claiming byte-identity.** Whether that trade ships is Moritz's call, not this row's.

**P6 — the A16 zero baseline** over the 17 reads exactly **1.0** against float64, measured through
`agreement.py`, not asserted.

**P7 — the BREAK control moves the 17.** `--permute-cot` changes the seed, so a gradient that
reads the seed must move. A reading that does not move under it is measuring something other than
the gradient and invalidates P2 and P3.

## Bars, fixed here

Per-tensor **5.0e-02** relative L2 (PROTOCOL 3d). Mass-weighted 2.0e-02. Coverage bar **99.2594 %**
(`COVERAGE_CEILING_IS_NOT_100` pass 175, its REASON withdrawn at A32, the bar kept). A26's sqrt(2)
applies against upstream's bf16 step and **not** against float64 (A26-SCOPE). Denominator
`model_squared_gradient_norm` **10.279642678524981** over 4,170 tensors, from
`grads_f64_043.pt` sha256 `1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`,
verified by digest before loading. Sidecar in the `shipped_vs_FLOAT64.json` schema.

Every timing figure carries its board class and an AICLK sampled DURING the timed work. Every
Angstrom figure carries the kill bar and the measured seed floor beside it.

## What this row may not do

It may not edit `perf/of3t_readable_mass/` or `perf/of3t_wholemodel/`: concluded rows own those
(A33). If a result here supersedes a number the charter reads, it says so by path in its
conclusion and leaves the file alone. It may not merge to main; the shared-path change is
release-gated and stays on `wk/of3t-hostleg`.
