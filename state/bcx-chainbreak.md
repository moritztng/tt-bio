# bcx-chainbreak — BindCraft 2's optimiser and its filters read two different molecules, and the difference is bit-exactly one missing call

PREREGISTERED. The PREDICTION section below was committed at `8e75129ff` before any model ran.
`git log --follow state/bcx-chainbreak.md` shows that commit preceding every artifact commit.

## The asymmetry, at a pinned commit

Upstream `PacesaLab/BindCraft2`, clean checkout at `~/bcx_e2e/bc2` on qb1, `7a2dfdb`. Re-read at
today's HEAD `301efdd` through the GitHub API: both findings below are unchanged there.

`bindcraft/af2.py:185-193` defines `monomer_chain_break_indices`. It rewrites `residue_index` so
the step across a chain junction is `MONOMER_CHAIN_GAP + 1` = 50. Its docstring is right about why:
a monomer AlphaFold2 reads `residue_index` alone, and numbering straight through hands it a peptide
bond that is not there. `grep -rn monomer_chain_break_indices bindcraft/` finds one definition and
one call, at `af2.py:305-306` (`:308` at HEAD), inside `_predict_complex`.

The gradient path is `predict_complex_arrays` inside `_compiled_sequence_gradients`,
`af2.py:337-345`. It builds `asym_id` (`:338`), `entity_id` (`:339`) and `seq_mask` (`:340`) with
the same three helpers `_predict_complex` uses at `:302-304`, then passes `residue_index` straight
into `alphafold_input_features` at `:342`. No chain break, and no branch that could add one.

For a monomer model this is the whole of chain identity: monomer relpos reads `offset` clipped to
+-32 (`bindcraft/af/alphafold/model/modules.py:1469-1484`, `max_relative_feature` 32 in
`config.py:226`), and `asym_id` is a multimer feature that never reaches it. `model_1_ptm` is
monomer-family (`af2.py:195-198`) and a binder complex has two chains, so `:305` is live on every
step of this campaign.

## Which path grades which stage — the part that makes it cost something

Read off `bindcraft/trajectory.py`, line numbers at HEAD `301efdd`:

- `screen`, `refine`, `anneal`, `harden` are optimised through `run_gradient_design_stage`, whose
  predictions come from `design_model.sequence_gradients` (`:131`) — no chain break.
- `judge_stage` grades those same prediction objects (`:310` calling `:252-256`). For a
  single-target campaign nothing replaces them: the pooling operation that would call `predict` is
  gated on `len(design_settings.prepared_states) > 1` (`:178-183`).
- `mutate` optimises through `run_sequence_mutation_stage`, which scores every candidate with
  `design_model.predict` (`:162`) — chain break on — and is then graded on
  `design_model.predict` again (`:272-275`).
- The `final` filter (`:321-323`) inherits the mutate predictions, so it is also on the predict path.

So a single-target trajectory runs two optimisers against two different objectives and grades four
of its six gates on one instrument and two on the other. The instrument changes at the
harden/mutate boundary.

MEASURED: all on qb1 (`tt-quietbox`), CPU only, no Tenstorrent device opened and none held.
`JAX_PLATFORMS=cpu`, jax 0.11.2, `~/bcx_e2e_venv/bin/python`, `PYTHONPATH=/home/ttuser/bcx_e2e/bc2`
at `7a2dfdb`, params `/home/ttuser/bcx_e2e/af2_params/params_model_1_ptm.npz`, `model_1_ptm`,
`num_recycle` 1, `dropout` False, sequence parameters 1.0 / 1.0 / 0.01 / 2.0 held identical on
every leg. loadavg 50.6 to 60.6 across the runs; no perf claim is made here, so the load costs
wall-clock and nothing else. One leg per process, because both entry points memoise their traced
function and neither cache key mentions the numbering. Raw JSON in `perf/bcx_chainbreak/out/`.

**The two entry points, same complex, binder 32 + target 96:**

| leg | pTM | i_pTM | pLDDT | binder pLDDT |
|---|---|---|---|---|
| `predict` (mutate/final grade) | 0.294875 | 0.141805 | 0.363348 | 0.337788 |
| `sequence_gradients` (screen..harden grade) | 0.352421 | **0.522476** | 0.408438 | 0.445945 |
| `predict` with `monomer_chain_break_indices` made the identity | 0.352421 | **0.522476** | 0.408438 | 0.445945 |

The third row equals the second **bit for bit** on all four metrics
(`0.35242122411727905`, `0.5224756002426147`, `0.4084381478605792`, `0.44594516418874264`), in two
separate processes. Not to six decimals: every bit. So 100.0% of an i_pTM gap of **0.380671** is
that one call, with nothing left over. `bcx-unbound` measured the same thing to within i_pTM
0.001214 and attributed the residual to the gradient path's binder padding; choosing every chain
length a multiple of 32 removes that padding and the residual with it.

**Dose-response across target length, binder held at 32, one binder sequence per cell:**

| target | in-window cross pairs | in-window share | i_pTM predict | i_pTM gradient-numbering | i_pTM gap | pTM gap | pLDDT gap |
|---|---|---|---|---|---|---|---|
| 96 | 1552 | 50.5% | 0.141805 | 0.522476 | **+0.380671** | +0.057547 | +0.045090 |
| 160 | 1552 | 30.3% | 0.133754 | 0.542787 | **+0.409032** | −0.001248 | −0.005858 |
| 288 | 1552 | 16.8% | 0.138566 | 0.562280 | **+0.423715** | +0.036184 | +0.007557 |

**The zero-window control**, binder 32 + target 96 with the target's own numbering shifted to start
at 4000 so that no cross-chain pair is inside the +-32 window: `predict` and identity-patched
`predict` return `ptm 0.2948746085166931, iptm 0.14180460572242737, plddt 0.36334849160630256,
binder_plddt 0.33778767427429557` — identical to each other bit for bit, and identical to the
unshifted `predict` row. Two residue numberings 3918 apart, same answer to the last bit.

PREDICTION: written before any model ran, with its refutation beside it, and reported either way.

- **H1 (single chain).** With one chain the branch at `af2.py:305` cannot fire, so both paths get
  the same `residue_index` and the gap should be zero rather than merely small. REFUTED IF a
  single-chain design shows a gap of the same order as the two-chain gap.
  **CONFIRMED.** On one chain of 128 residues the two entry points agree: pTM
  `0.2111928015947342` from both, bit for bit; pLDDT `0.2989578155102208` from
  `sequence_gradients` against `0.2989578079432249` from `predict`, a difference of
  **8e-9**, which is float32 reduction order in the mean and not a feature difference. i_pTM is 0.0
  on both, undefined with one assembly, as expected. Against the two-chain i_pTM gap of 0.380671
  that residual is smaller by a factor of 5e7. Remove the second chain and the two paths become the
  same program.

- **H2 (the carrier is the window, not the numbering).** The corruption is the count of cross-chain
  pairs with `|residue_index_t − residue_index_b| <= 32`, computable from the inputs with no model.
  A complex engineered to have zero in-window cross-chain pairs should show no gap. REFUTED IF a
  zero-in-window complex still shows a gap, or a 45%-in-window complex does not.
  **CONFIRMED, and in its sharpest form: the zero-window control is bit-identical between the two
  numberings, and the 50.5%-in-window complex differs by i_pTM 0.380671.** The carrier is the
  relpos window. "The numbering changed" is not sufficient: a numbering shifted by 3918 changes
  nothing at all.

- **H3 (my own prediction about length, which I expected to beat the brief's).** The brief proposed
  the gap grows with target length. I predicted the opposite and wrote the arithmetic down first:
  the in-window count is about `65*Lb − 32*33` once `Lt > Lb`, so it saturates at the SHORTER chain
  while the denominator `Lb*Lt` keeps growing, and since i_pTM is a normalised mean over interface
  pairs I predicted the gap would SHRINK as the target got longer.
  **REFUTED. Mine is wrong and the brief's is closer.** The census confirms the count saturates
  exactly as predicted — 1552 in-window pairs at every one of the three target lengths, share
  falling 50.5% → 30.3% → 16.8% — but the i_pTM gap did not follow the share down. It rose:
  +0.380671 → +0.409032 → +0.423715. The gap tracks the absolute count of corrupted pairs, which is
  flat, and not their share, which falls threefold. The residual +0.043 of drift over a 3x target
  length is small next to the gap itself, so the honest statement is that the effect is **saturated,
  not dose-dependent in length**: it is set by the shorter chain, and the binder is always the
  shorter chain. Neither "grows with the target's length" nor "shrinks with it" survives as a
  mechanism claim; what survives is that a binder of any usual length pulls its full 65*Lb band of
  target residues inside the window no matter how large the target is.

The damage is concentrated in i_pTM. Across the three lengths the pTM gap is +0.058 / −0.001 /
+0.036 and the pLDDT gap is +0.045 / −0.006 / +0.008, both non-monotonic and an order of magnitude
smaller. i_pTM is the metric BindCraft 2's stage filters reject on.

CONFOUNDS, listed before the arms were named because `bcx-mono` named a variable that turned out to
be padding. Chain lengths are multiples of 32, so `_predict_complex:298`'s bucket padding and
`pad_design_chains`'s binder padding both pad zero residues and the two paths see identical shapes.
Both chains are built from sequence with no template, so `target_template_features` contributes
nothing that varies. Dropout off and `self.key` fixed on both paths. Same sequence parameters. The
only input that differs between the legs is `residue_index`, which is what the bit-identity of the
gradient leg against the identity-patched predict leg demonstrates rather than asserts.

ASYMMETRY: `af2.py:305-306` (`:308` at HEAD `301efdd`) against `af2.py:337-345`, both quoted above;
and the stage/filter split at `trajectory.py:131` and `:310` against `:162` and `:272-275`. Sized
on the lab's own gate metrics at three target lengths in the table above; per-length distributions
over further binder draws are accumulating in `out/` under `runner.sh`.

REPRO: `perf/bcx_chainbreak/chainbreak.py` with `perf/bcx_chainbreak/README.md`. Pure upstream
BindCraft 2 on CPU: no tt-bio import, no ttnn, no Tenstorrent anything, no structure file, both
chains built from sequence. From a clean checkout it needs `PYTHONPATH` pointed at the checkout and
one parameter file, `params_model_1_ptm.npz`, the monomer set `bindcraft fetch-weights` downloads.
`python chainbreak.py census` needs neither the model nor the parameters and finishes in under a
second. On 32 loaded CPU cores a `predict` leg at 128 residues takes 62 s including compile, 108 s
at 192 and 233 s at 320; the `sequence_gradients` leg at 128 takes 205 s. The whole three-length
table is about 15 minutes of one CPU box.

UPSTREAM: not filed this pass, and filing it is the next action. Groundwork done: upstream issue
**#13** ("Pdl1-VHH exemplar campaign does not deliver accepted designs", opened by `batch2k`) is the
live report of the symptom — users seeing trajectories pass screen and refine and then be rejected
`due to [i_pTM]`, and `ErikMaeots` confirming on 2026-09-23 "I reproduced this bug in the release
version... this has regressed in the current version. We are investigating." It was closed
2026-09-24T17:16:09Z by PR #17 (`301efdd`), an amino-acid-bias fix that touches shipped settings and
nothing in the residue numbering. So the mechanism here is untouched at HEAD and is a separate
candidate for the same symptom, which is worth saying plainly and without claiming it is the cause
of #13. Our PR #19 is open on the same repo under `moritztng`, so the account is already engaged
there.

Whether this changes the reading of our own matched-seed results: **it does not bias the
comparison, because both arms inherit it identically.** Our device arm and the reference arm run the
same BC2 harness through the same two entry points, so both optimise the un-chain-broken molecule
and both are judged at mutate on the chain-broken one. The asymmetry is common-mode and cancels in
any device-against-reference comparison. What it does change is the reading of any ABSOLUTE
designs-per-accepted number from either arm: a campaign on this harness is graded at screen,
refine, anneal and harden on an i_pTM that reads roughly 0.4 high at the sizes measured here, so
acceptance rates from either arm describe the harness as much as the silicon. The campaign's
chip-seconds-per-accepted-design axis inherits that, and `bcx-gpuref`'s denominator work should
know it.

VERDICT: PARTIAL — the asymmetry is established as a bit-exact causal fact rather than an argument,
H1 and H2 are confirmed and H3 (mine) is refuted with the census that explains why, and the
reproducer is pure upstream CPU and runs in minutes. Outstanding for the next pass: the per-length
distributions over further binder draws, accumulating under `runner.sh` into `out/`, and the
upstream issue, which is written from this file once they land. The issue is held back one pass
deliberately: it is the outward-facing artifact, the mechanism half of it is already bit-exact, and
the effect-size half reads better with more than one draw per cell.
