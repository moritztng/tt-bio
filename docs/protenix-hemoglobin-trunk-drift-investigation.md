# Protenix-v2 hemoglobin@10 trunk drift — reference-vs-device measurement

Follow-up to `wk/tt-bio-protenix-confidence-fix` (`docs/protenix-confidence-head-rootcause.md`),
which left one question open: is hemoglobin's +0.61 Å regression at `--recycling_steps 10`
the model's faithful output, or on-device recycling drift specific to larger/tetramer inputs?
That doc's own qualitative evidence leaned "faithful" — this task does the bounded numeric
comparison it named as the decisive test, and finds a real answer: **partly neither** — a
genuine, previously-undocumented on-device omission (not size-specific drift, not pure
model faithfulness).

## What was measured

Full CPU reference fold of hemoglobin (α₂β₂ tetramer, N_token=574, `use_msa=False`,
`use_template=False`, isolates trunk behavior from MSA/template search) is ~650s/cycle —
10 cycles is ~1.8h, too slow for one bounded turn. Captured 3 cycles instead (`N_cycle=3`,
1949s total, `protenix_hemo_ref_cycles.py`/`.pkl` on qb1), hooking `pairformer_stack.forward`
to record the trunk (s,z) after every cycle within one N_cycle=3 forward (causal recycling:
cycle k's state depends only on cycles 1..k, so this is a valid stand-in for "device runs
n_cycles=k" at each k). Fed the same `s_inputs`/`relp`/`token_bonds`/`feat` into on-device
`tt_bio.protenix.Trunk` at `n_cycles=1,2,3` and compared (`scripts/protenix_hemo_device_trunk_pcc.py`,
qb1 card 0):

| cycle | s_PCC | z_PCC | s_absmean ref/dev | z_absmean ref/dev |
|------:|------:|------:|-------------------:|-------------------:|
| 1 | 0.99739 | 0.76792 | 55.89 / 54.58 | 8.900 / 8.635 |
| 2 | 0.99829 | 0.76489 | 57.86 / 56.90 | 9.645 / 9.199 |
| 3 | 0.99779 | 0.67681 | 58.47 / 57.78 | 9.784 / 8.937 |

Single representation (s) matches closely and stays stable (PCC ~0.997–0.998). Pair
representation (z) is far below the only other on-record trunk-parity number — the
38-token porting validation, `docs/porting-protenix-v2.md` "MILESTONE: FULL 10-CYCLE TRUNK
VALIDATED", **s 0.99110 / z 0.98967 at 10 cycles** — and it gets *worse* with more cycles
(0.768→0.765→0.677), not better. Device z is also systematically smaller in magnitude than
reference at every cycle, and the gap widens (dev/ref ratio 0.970→0.954→0.913). That
signature — s fine, z low and declining, device magnitude systematically undershooting — is
the fingerprint of a missing small additive term to z that compounds every recycle, not
generic bf16 rounding noise (which wouldn't produce a monotonic one-directional magnitude
gap) and not tile/reshape corruption (which would break s too, given s and z are coupled
through every pairformer block).

## Root cause: a real architectural omission, not size-dependent numerical drift

Traced it to source. Upstream Protenix-v2's `TemplateEmbedder` is invoked unconditionally
whenever `template_embedder.n_blocks > 0` (`protenix/model/protenix.py:248,268`) — a **fixed
model-config gate**, not the CLI's `use_template` flag. protenix-v2's own config sets
`template_embedder.n_blocks = 2` (`configs/configs_model_type.py:62`, overriding the
architecture base default of `n_blocks=0`). So the real v2 model always runs a 2-block
template pass every recycle, for every fold, MSA/template flags notwithstanding.

Confirmed empirically (`/tmp/protenix_hemo_feat_check.py`, hemoglobin, `use_template=False`):
the reference's own inference dataloader still emits `template_aatype` with shape **(4, 574)**
— `TemplateFeatureAssemblyLine(max_templates=4)` always pads to 4 template slots even when
zero real templates are found. `template_atom_positions` is genuinely all-zero (no real
geometry), but `template_aatype`'s one-hot encoding (`pairformer.py`'s `TemplateEmbedder`
concatenates `template_restype_i/j` un-masked by any position/pseudo-beta mask) is **not**
zero, so it produces a real, non-degenerate contribution to z through `linear_no_bias_a` →
2-block pairformer → `linear_no_bias_u`, every single cycle.

tt-bio's own featurizer (`tt_bio/protenix_data.py:15-17`) documents an incorrect assumption:
> "templates empty), matching the reference's use_msa/use_template=False inference path"

It never emits `template_aatype`/`template_distogram`/etc. at all. `tt_bio/protenix.py:1162`:
```
nt = feat["template_aatype"].shape[0] if "template_aatype" in feat else 0
```
—so on-device `Trunk.__call__` always takes `nt=0` and skips the template pass entirely, in
**every real predict call**, for every target, not just hemoglobin. This is a genuine gap
between tt-bio's port and the true reference, present since the original port — it was masked
because the one existing trunk-parity validation (`scripts/protenix_trunk_assembly.py`, the
38-token case) fed the reference's *own* captured feat dict directly (which does carry the
real `nt=4` dummy-template tensors), so that check happened to exercise — and validate — the
`nt>0` code path correctly. Production's featurizer just never reaches it.

## Answer to the open question

This is **not** primarily a size-dependent numerical-drift bug, and it is **not** the "faithful
model output" the confidence-fix doc leaned toward either — it's a third thing: a real,
uniform (not hemoglobin-specific) omission whose *visible impact scales with cycle count*.
At 3 shipped cycles the missing per-cycle z-nudge barely accumulates; at 10 spec cycles it
compounds over more than 3x as many recycles (this measurement only got to see 3 of the 10 —
the actual 10-cycle divergence is presumably larger still). So some real fraction of
hemoglobin's +0.61 Å regression at `--recycling_steps 10` is attributable to this omitted term,
on top of whatever genuine model-recycling behavior exists — the two are not separated by this
measurement, and doing so needs the fix below, then a controlled before/after comparison.

One correction to the confidence-fix doc's evidence: reason (b) there — "confidence-head pae
PCC 1.0000" (`scripts/protenix_confidence_parity.py`) — does **not** test trunk fidelity at
hemoglobin/7ROA's size; that harness feeds the reference's own *golden* `s_trunk`/`z_trunk`
into the device confidence head, so it validates the confidence head module only, given an
already-correct trunk. It says nothing about whether the on-device trunk itself reproduces the
reference at these input sizes. This measurement is the first direct trunk-vs-reference check
at a size other than 38 tokens.

## Not fixed here (deliberately)

Reproducing upstream's `max_templates=4` placeholder-template padding in
`tt_bio/protenix_data.py` (exact aatype fill value, mask fields, `TemplateFeatureAssemblyLine`
semantics) and re-validating `Trunk`'s `nt>0` path at multiple sizes is real, non-trivial
featurizer work with its own accuracy-parity re-validation burden — out of scope for this
bounded investigation task per the calling instructions. Recommend as a dedicated follow-up:
implement the 4-slot dummy-template padding, re-run this same harness
(`scripts/protenix_hemo_device_trunk_pcc.py`) to confirm z_PCC recovers toward the 38-token
0.98967 benchmark, then re-measure hemoglobin@10's delivered RMSD to see how much of the +0.61 Å
this closes.

## Harnesses (new, this branch)

- `/home/ttuser/protenix_hemo_ref_cycles.py` → `.pkl` (qb1, not committed — 1.2GB artifact):
  CPU reference hemoglobin trunk, per-cycle (s,z) capture, N_cycle=3.
- `scripts/protenix_hemo_device_trunk_pcc.py`: on-device Trunk vs the above, per-cycle PCC.
- `scripts/protenix_template_feat_check.py`: fast (~10s, no model load) check of what
  `template_*` features the reference featurizer actually emits under `use_template=False`.
