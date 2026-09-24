# bcx-chainbreak — BindCraft 2's optimiser and its filters read two different molecules, and the difference is bit-exactly one missing call

Filed upstream as **https://github.com/PacesaLab/BindCraft2/issues/20**.

PREREGISTERED. The PREDICTION section was committed at `8e75129ff` before any model ran.
`git log --follow state/bcx-chainbreak.md` shows that commit preceding every artifact commit.

## The asymmetry, at a pinned commit

Upstream `PacesaLab/BindCraft2`, clean checkout at `~/bcx_e2e/bc2` on qb1, `7a2dfdb`. Every line
number below was re-read at today's HEAD `301efdd` and is quoted at HEAD.

`af2.py:185-193` defines `monomer_chain_break_indices`: it rewrites `residue_index` so the step
across a chain junction is `MONOMER_CHAIN_GAP + 1` = 50. Its docstring is right about why — a
monomer AlphaFold2 reads `residue_index` alone, and numbering straight through hands it a peptide
bond that is not there. `grep -rn monomer_chain_break_indices bindcraft/` finds one definition and
one call, `af2.py:308`, inside `_predict_complex`.

The gradient path is `predict_complex_arrays` inside `_compiled_sequence_gradients`, `af2.py:338`.
It builds `asym_id`, `entity_id` and `seq_mask` (`:339-341`) with the same three helpers
`_predict_complex` uses at `:304-306`, then passes `residue_index` straight into
`alphafold_input_features` (`:343`). No chain break, no branch that could add one.

For a monomer model that is the whole of chain identity: relpos reads `offset` clipped to +-32
(`af/alphafold/model/modules.py:1469-1485`, `max_relative_feature` 32 at
`af/alphafold/model/config.py:226`) and `asym_id` never reaches it. `model_1_ptm` is monomer-family
(`af2.py:195-198`) and a binder complex has two chains, so `:307-308` is live on every step.

## Which path grades which stage

- `screen`, `refine`, `anneal`, `harden` are optimised through `run_gradient_design_stage`, whose
  predictions come from `design_model.sequence_gradients` (`trajectory.py:131`) — no chain break.
- `judge_stage` grades those same objects (`:310` calling `:252-256`). For a single-target campaign
  nothing replaces them: the pooling that would call `predict` lives in
  `pooled_prediction_operation` (`:182`) and is only installed for a multitarget campaign (`:215`).
- `mutate` optimises through `run_sequence_mutation_stage`, which scores every candidate with
  `design_model.predict` (`:162`) — chain break on — and is graded on `predict` again (`:272-275`).
- `final` (`:321-323`) inherits the mutate predictions, so it is on the predict path too.

Four of six gates in a single-target trajectory read one instrument and two read the other, and the
two optimisers inside one trajectory optimise different objectives.

MEASURED: all on qb1 (`tt-quietbox`), CPU only. **No Tenstorrent device was opened and no card lease
was held** — `JAX_PLATFORMS=cpu`, `jax.devices()` is `[CpuDevice(id=0)]`. jax 0.11.2,
`~/bcx_e2e_venv/bin/python`, `PYTHONPATH=/home/ttuser/bcx_e2e/bc2` at `7a2dfdb`, params
`/home/ttuser/bcx_e2e/af2_params/params_model_1_ptm.npz`, `model_1_ptm`, `num_recycle` 1, `dropout`
False, sequence parameters 1.0 / 1.0 / 0.01 / 2.0 identical on every leg. loadavg 50.6 to 60.6; no
perf claim is made here, so load costs wall-clock and nothing else. One leg per process. 24 leg
JSONs in `perf/bcx_chainbreak/out/`; `perf/bcx_chainbreak/summarise.py` regenerates every table
below from them, so nothing here is transcribed by hand (`perf/bcx_chainbreak/TABLES.md`).

**The two entry points, same complex, binder 32 + target 96:**

| leg | pTM | i_pTM | pLDDT | binder pLDDT |
|---|---|---|---|---|
| `predict` (mutate/final grade) | 0.294875 | 0.141805 | 0.363348 | 0.337788 |
| `sequence_gradients` (screen..harden grade) | 0.352421 | **0.522476** | 0.408438 | 0.445945 |
| `predict` with `monomer_chain_break_indices` made the identity | 0.352421 | **0.522476** | 0.408438 | 0.445945 |

Row 3 equals row 2 **bit for bit** on all four metrics (`0.35242122411727905`,
`0.5224756002426147`, `0.4084381478605792`, `0.44594516418874264`), in separate processes. And the
converse closes too: `sequence_gradients` fed a target whose own numbering already carries the
junction step of 50 reproduces row 1 **bit for bit** on all four. Both directions of the fix land
exactly, so 100.0% of the i_pTM gap of 0.380671 is that one call with nothing left over.
`bcx-unbound` got to within i_pTM 0.001214 and attributed the residual to the gradient path's
binder padding; choosing every chain length a multiple of 32 removes that padding and the residual
with it.

**Distributions, gradient-path numbering minus predict-path numbering, one row per arm:**

| binder | target | target numbered from | draws | in-window pairs | in-window share | i_pTM gap (each draw) | median |
|---|---|---|---|---|---|---|---|
| 32 | 96 | 1 | 4 | 1552 | 50.5% | +0.2317, +0.3433, +0.3807, +0.4749 | +0.3620 |
| 32 | 96 | 4000 | 1 | 0 | 0.0% | +0.0000 | +0.0000 |
| 32 | 160 | 1 | 4 | 1552 | 30.3% | +0.2283, +0.2857, +0.3812, +0.4090 | +0.3335 |
| 32 | 160 | 4000 | 1 | 0 | 0.0% | +0.0000 | +0.0000 |
| 32 | 288 | 1 | 3 | 1552 | 16.8% | +0.3213, +0.3847, +0.4237 | +0.3847 |
| 96 | 128 | 1 | 2 | 5712 | 46.5% | +0.1281, +0.3202 | +0.2242 |

**Same pairs, other metrics, median over draws:**

| binder | target | i_pTM | pTM | pLDDT | binder pLDDT |
|---|---|---|---|---|---|
| 32 | 96 | +0.3620 | +0.0280 | +0.0326 | +0.1061 |
| 32 | 160 | +0.3335 | −0.0256 | −0.0164 | +0.0310 |
| 32 | 288 | +0.3847 | −0.0056 | −0.0046 | −0.0189 |
| 96 | 128 | +0.2242 | +0.0625 | +0.0165 | +0.0346 |

The damage is concentrated in i_pTM. pTM and pLDDT move several times less and change sign between
arms. i_pTM is the metric the stage filters reject on, and it is the metric every reporter on
upstream issue #13 was rejected by.

PREDICTION: written before any model ran, reported either way.

- **H1 (single chain) — CONFIRMED.** With one chain the branch at `:307` cannot fire, so both paths
  get the same `residue_index` and the gap should be zero rather than merely small; refuted if a
  single-chain design showed a gap of the same order as the two-chain gap. On one chain of 128
  residues: pTM `0.2111928015947342` from both entry points, bit for bit; pLDDT
  `0.2989578155102208` against `0.2989578079432249`, apart by **7.6e-9**, which is float32
  reduction order in a mean and not a feature difference. i_pTM 0.0 on both, undefined with one
  assembly. Against the two-chain gap of 0.380671 that residual is smaller by 5e7. Remove the
  second chain and the two paths are the same program.

- **H2 (the carrier is the window, not the numbering) — CONFIRMED, in its sharpest form.** A
  complex engineered to have zero cross-chain pairs inside the +-32 window should show no gap.
  At both target lengths tested, shifting the target's own numbering to start at 4000 makes the two
  numberings **bit-identical on all four metrics** — and identical to the chain-broken answer.
  Two numberings 3918 apart, same answer to the last bit. "The numbering changed" is not the
  mechanism; the relpos window is.

- **H3 (mine, about length) — REFUTED, and so is my own mid-pass replacement for it.** I predicted
  the gap would SHRINK with target length, because the in-window count saturates at `65*Lb` while
  the denominator `Lb*Lt` grows, so the share falls. The census half was exactly right: 1552
  in-window pairs at all three target lengths, share falling 50.5% → 30.3% → 16.8%. The gap did not
  follow the share down — arm medians +0.3620, +0.3335, +0.3847. Mid-pass I replaced H3 with "the
  gap tracks the absolute count, which is flat", and **the binder sweep refutes that too**: going
  from a 32-residue binder to a 96-residue one multiplies the in-window count by 3.7 (1552 → 5712)
  and does not raise the gap (median +0.2242). Neither the count nor the share orders the arms.

  What survives is the binary result, and it is strong: **all 13 draws with in-window pairs show a
  gap of at least +0.128, and both zero-window arms are bit-identical.** The magnitude varies more
  WITHIN an arm (+0.2283 to +0.4090 across four draws at 32/160) than between arms, so four arms
  cannot support a law and this row does not assert one. Three predictions about the size were made
  here — the brief's, mine, and my own correction — and all three failed. The controls held.

CONFOUNDS, listed before the arms were named because `bcx-mono` named a variable that turned out to
be padding. Chain lengths are multiples of 32, so `_predict_complex`'s bucket padding and
`pad_design_chains`'s binder padding both pad zero residues and the two paths see identical shapes.
Both chains are built from sequence with no template, so `target_template_features` contributes
nothing that varies. Dropout off, `self.key` fixed, same sequence parameters on both paths. The only
input that differs is `residue_index` — which the bit-identity in both directions demonstrates
rather than asserts. Between the 32-binder and 96-binder arms, binder length, total length,
in-window count and the sequence draws all move together, which is exactly why no variable is named
as the magnitude's carrier.

ASYMMETRY: `af2.py:308` against `af2.py:338-343` at HEAD `301efdd`, both quoted above; and the
stage/filter split at `trajectory.py:131` and `:310` against `:162` and `:272-275`. Sized on the
lab's own gate metrics at four arms and 14 binder draws in the tables above.

REPRO: `perf/bcx_chainbreak/chainbreak.py`, with `README.md` and `summarise.py` beside it. Pure
upstream BindCraft 2 on CPU: no tt-bio import, no ttnn, no Tenstorrent anything, no GPU, and no
structure file — both chains are built from sequence. From a clean checkout it needs `PYTHONPATH`
pointed at the checkout and one parameter file, `params_model_1_ptm.npz`, the monomer set
`bindcraft fetch-weights` downloads. `python chainbreak.py census` needs neither the model nor the
parameters and finishes in under a second. On 32 loaded CPU cores a `predict` leg takes 62 s at 128
residues including compile, 108 s at 192, 233 s at 320; a `sequence_gradients` leg takes 205 s at
128. The whole table is about 15 minutes of one CPU box. Each leg runs in its own process because
both entry points memoise their traced function and neither cache key mentions the numbering, so a
patch installed after a trace returns the old answer bit-for-bit and reads as a null result — the
trap `bcx-unbound` recorded as its VOID CONTROL.

FIXDIRECTION: two readings exist — the gradient path is missing the call, or `_predict_complex`
adds it deliberately — and the lab decides. The one argument that would favour omitting it does not
survive, and it is now a measurement rather than the orchestrator's reading:
`sequence_design_loss` is differentiated by `jax.value_and_grad(..., argnums=2, has_aux=True)`
(`af2.py:378`), argnum 2 is `sequences`, and `residue_index` arrives inside `state_templates`,
argnum 3, which is not differentiated. Run rather than argued: the gradient path on the
chain-broken numbering compiles and returns a finite design loss (0.47752439975738525) and finite
sequence gradients, and its metrics match `predict` bit for bit.

UPSTREAM: filed as **https://github.com/PacesaLab/BindCraft2/issues/20**, "Gradient path does not
apply `monomer_chain_break_indices`, so the design stages optimise a fused chain", under
`moritztng`, with the numbers above and `chainbreak.py` inline. It states what was measured, says
the fix direction is theirs, and explicitly does not claim to be the cause of their issue #13 —
that one was closed 2026-09-24T17:16:09Z by PR #17 (`301efdd`), an `aa_bias` fix that touches
shipped settings and nothing in the residue numbering, and we have not tested this against the
PDL1-VHH case. Context that made the issue worth filing anyway: #13 is users reporting trajectories
that pass screen and refine and are then rejected `due to [i_pTM]`, with `ErikMaeots` confirming
2026-09-23 "I reproduced this bug in the release version... this has regressed in the current
version." The mechanism here is untouched at HEAD and produces that symptom shape. **Not linked
from our PR #19**, deliberately: #19 is ROCm worker fan-out in `design_workers.py` and shares no
code with this, so cross-posting would be noise.

OURRESULTS: **this does not bias our matched-seed comparison, because both arms inherit it
identically.** The device arm and the reference arm run the same BC2 harness through the same two
entry points, so both optimise the un-chain-broken molecule for four stages and both are judged at
mutate on the chain-broken one. It is common-mode and cancels in any device-against-reference
reading. What it does change is any ABSOLUTE acceptance number from either arm: four of six gates
are graded on an i_pTM that reads +0.13 to +0.47 high at the sizes measured here, so a
designs-per-accepted or chip-seconds-per-accepted figure from this harness describes the harness as
much as the silicon. `bcx-gpuref`'s denominator work and the campaign's competitive axis should
carry that caveat.

VERDICT: GO — the asymmetry is established as a bit-exact causal fact in both directions rather
than an argument, H1 and H2 are confirmed and H3 is refuted along with my own replacement for it,
the reproducer is pure upstream CPU and runs in minutes from a clean checkout, the fix direction is
settled by measurement rather than by reading, and the finding is filed upstream at issue #20.
