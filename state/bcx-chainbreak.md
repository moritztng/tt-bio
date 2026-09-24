# bcx-chainbreak — BindCraft 2 grades four of its six stages on an instrument that is missing a call the other one makes

PREREGISTERED. This file's PREDICTION section was committed before any model ran. Its commit is
the evidence; `git log --follow state/bcx-chainbreak.md` shows the prediction commit preceding
every artifact commit.

## The asymmetry, re-read at the pinned commit

Upstream `PacesaLab/BindCraft2` at `7a2dfdb`, clean checkout at `~/bcx_e2e/bc2` on qb1.

`bindcraft/af2.py:185-193` defines `monomer_chain_break_indices`: it rewrites `residue_index` so
that the step across a chain junction is `MONOMER_CHAIN_GAP + 1` = 50, instead of whatever step
the concatenated per-chain numbering happens to produce. Its docstring says why, and the reason is
correct: a monomer AlphaFold2 reads `residue_index` alone and would otherwise be handed a peptide
bond that is not there.

`grep -rn monomer_chain_break_indices bindcraft/` finds one definition and one call:

    af2.py:305-306   (inside _predict_complex)
        if self.model_families[model][0] == 'monomer' and len(chain_names) > 1:
            residue_index = monomer_chain_break_indices(chain_lengths, residue_index)

The gradient path is `predict_complex_arrays` inside `_compiled_sequence_gradients`,
`af2.py:337-345`. It builds `asym_id` (`:338`), `entity_id` (`:339`) and `seq_mask` (`:340`) with
the same three helpers `_predict_complex` uses at `:302-304`, and then passes `residue_index`
straight into `alphafold_input_features` at `:342`. There is no chain break on that path, and no
branch that could add one.

Both paths reach the same `alphafold_input_features` (`af2.py:128`) and the same metric function
`alphafold_prediction_metrics` (`af2.py:145`), so `residue_index` is the only feature that differs
between them for a monomer model on a multi-chain complex. `model_1_ptm` is monomer-family
(`af2.py:195-198`, no 'multimer' in the name) and a binder design complex has two chains, so the
branch at `:305` is live on every step of this campaign.

Why `residue_index` matters to a monomer model: monomer relpos builds its pair feature from
`offset = residue_index[:, None] - residue_index[None, :]` clipped to +-32
(`bindcraft/af/alphafold/model/modules.py:1469-1484`, `max_relative_feature` 32 in
`config.py:226`). `asym_id` is a multimer feature and never reaches it. So for a monomer model,
chain identity is carried by the residue numbering and by nothing else.

## Which path grades which stage

This is the part that turns a feature difference into a lab-facing cost, and it is read off
`bindcraft/trajectory.py` at the same commit:

- `run_trajectory:303` runs each of `screen`, `refine`, `anneal`, `harden` through
  `run_gradient_design_stage`, whose predictions come from `design_model.sequence_gradients`
  (`trajectory.py:131`) -- the path with no chain break.
- `run_trajectory:310` then calls `judge_stage` on exactly those predictions
  (`trajectory.py:252-256`). For a single-target campaign nothing replaces them: the pooling
  operation that would call `predict` is gated on `len(design_settings.prepared_states) > 1`
  (`trajectory.py:178-183`, `multitarget_filter_models`).
- `run_mutation_polish:271` then does `predictions = design_model.predict(protein_states)` -- the
  path WITH the chain break -- and grades the `mutate` filters on that (`:273-275`).
- The `final` filter at `:321-323` inherits the mutate predictions, so it is also on the predict
  path.

So four of the six filter gates in a single-target trajectory read a metric computed without the
chain break, and the last two read a metric computed with it. The instrument changes at the
harden/mutate boundary. A design that is optimised and accepted against the first instrument is
then judged by the second.

PREDICTION: written before any model ran, with its refutation stated beside it.

The mechanism, stated so it can fail: the entire difference between the two paths' metrics on a
monomer model is the `residue_index` feature, and its effect is carried by how many cross-chain
residue pairs the +-32 relpos window admits. Nothing else about the two paths differs in a way
that reaches the model.

- H1 (single chain). With one chain the branch at `af2.py:305` cannot fire, so the two paths
  receive an identical `residue_index`. Predicted: the metric gap between them is zero to the
  precision of the arithmetic, not merely small. REFUTED IF a single-chain design shows a gap of
  the same order as the two-chain gap; that would mean something other than the chain break
  separates the paths, and the campaign's reading of the 0.1999 i_pTM gap would have to be
  reopened.

- H2 (dose-response on the window, not on the length). The corruption is the count of cross-chain
  pairs with `|residue_index_t - residue_index_b| <= 32`. That count is a property of the two
  chains' NUMBERING, computable from the inputs with no model. Predicted: the metric gap tracks
  that count's share of all cross-chain pairs, and in particular a complex engineered to have
  ZERO in-window cross-chain pairs (by shifting the target's own residue numbering far away, which
  a real PDB can do for free) shows NO gap, while a complex with a large in-window share shows the
  full gap. REFUTED IF a zero-in-window complex still shows a gap, or a 45%-in-window complex does
  not.

- H3 (the brief's suggested prediction, which I expect to FAIL as stated). The brief proposes the
  gap should grow with the target chain's length. I predict it does not, and I am writing the
  arithmetic down before the run so this is not a retrofit. With binder 1..Lb and target numbered
  from 1, the in-window cross-chain pair count is about 65*Lb - 32*33 once Lt > Lb: it saturates
  at the SHORTER chain and does not grow. The denominator Lb*Lt keeps growing. So the in-window
  SHARE falls roughly as 1/Lt. i_pTM is a normalised mean over interface pairs, so predicted: at
  fixed binder length the i_pTM gap SHRINKS as the target gets longer. REFUTED IF the gap grows
  with target length, which would mean the metric responds to the absolute count of corrupted
  pairs rather than their share, and H2's normalisation is wrong.

H1 and H3 are the two that can embarrass this row, which is why they are here. A confirmation of
H2 alone, with H3 unstated, would have been a retrodiction: the finding was discovered after the
rejections it is being asked to explain, and this fleet has filed a mechanism as a cause on that
footing before.

CONFOUNDS, listed before the arms are named, because `bcx-mono` named a variable that turned out
to be padding. Every property that differs between my single-chain and two-chain arms: chain
count; total residue count; whether the bucket-32 padding in `_predict_complex:298` pads anything;
whether `interface_asym_id` has two values or one, which decides whether i_pTM is even defined;
the binder-chain padding the gradient path inherits from its template shapes. The arms are built
so that only chain count and residue numbering move, and each of the others is either held or
reported.

MEASURED: pending this pass's runs; the census and the model arms are recorded here as they land.

ASYMMETRY: `af2.py:305-306` against `af2.py:337-345` at `7a2dfdb`, quoted above with both sides,
plus the stage/filter split at `trajectory.py:303`/`:310` against `:271`/`:273-275`.

REPRO: `perf/bcx_chainbreak/` -- pure upstream BindCraft 2 on CPU, described below once it runs.

UPSTREAM: an issue on `PacesaLab/BindCraft2` under `moritztng`, or a reasoned decision not to file.

VERDICT: PARTIAL -- prediction registered, arms building.
