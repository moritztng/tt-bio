# OF3T LEDGER

The campaign's correction channel. Every row reads this before proposing anything, and the
orchestrator writes it. R-entries are **refutations of the charter** — things the campaign was
chartered on that did not survive a check. K-entries are **knowns** worth not re-deriving.

Every entry names how it was verified. An entry without that is not an entry.

---

## R — refutations of the charter, pass 1 (2026-09-19)

### R1. `openfold3/tests/test_training_full.py` does not exist. The protocol's spine item 1 is void.

The charter calls it *"the single most credible artifact available and the cheapest"* and made it
deliverable 1 of `of3t-equivalence`. It is not in the package.

Verified: `pip download openfold3==0.4.3 --no-deps --no-binary :all:`, extracted. `find . -name
'test_training*'` returns nothing. The 40 test files under `openfold3/tests/` include
`test_inference_full.py` and `test_model_runner.py` — the latter is 133 lines of **mocked OOM
handling** (`mock_forward_with_name_based_oom`), not a training run. `grep -rln
"training_step|\.backward\(\)|fit\("` over `openfold3/tests/` hits only `test_of3_model.py`,
`test_kernels.py`, `tests/utils/test_utils.py` and `core/data/tools/test_rscb.py`. **No upstream
test trains end-to-end.**

### R1-AMENDED (2026-09-19, same day). The fact holds for 0.4.3; my conclusion was wrong. The test exists in 0.5.0.

`of3t-equivalence` corrected me, with evidence, on `wk/of3t-equivalence` at `f010f46b8`. Both
halves of the record now stand:

- **0.4.3 has no test that trains.** The row enumerated all 67 `test_*.py` independently and
  agrees; `test_model_runner.py` is mocked OOM over `predict_step`. My check was sound.
- **0.5.0 ships `openfold3/tests/test_training_full.py`, 210 lines.** It invokes the real
  `run_openfold train --runner-yaml` console script and asserts exit 0 plus a written `.ckpt`.
  The wheel ships `tests/` and the console script. sdist sha256
  `a43357fddd4758dfb557e5ce801758f6e3069fc5422323d3a4dd91c33fbe2f6a`.

So **deliverable 1 is a version bump, not a deletion**, and I was too quick to void it. What it is
NOT is a free pass: run as shipped on qb1 the test reports **2 skipped, "Requires cuda; found
cpu"** (`@skip_unless_cuda_available`). The row is right not to call that a pass. A test that
module-skips is the exact trap the campaign already carries a lesson about.

**Version decision, mine, recorded here and reversible.** The reference stack and the training
reproduction target **0.5.0**. Three reasons: it is the only version whose training path is
testable; its training stack is the more complete one; and our port already implements v0.5.0
behaviour, since `openfold3_weights.py:is_openbind` selects "the two v0.5.0 behaviour changes"
for the OpenBind checkpoint. **This does NOT bump tt-bio's production pin** — `NOTICE:45` pins
0.4.3 for the vendored host data pipeline and that stays where it is. The 0.5.0 stack is an
evidence-side dependency living under `perf/of3t_reference/`. A production bump is a different
question, it is release-gated, and it goes to Moritz rather than being taken by a row.

**Which checkpoint is being reproduced must be stated** in every result, because the two are not
the same model: `of3-p2-155k.pt` (preview2, 0.4.3 family) against OpenBind (0.5.0, MODEL_VERSION
2.0.0), which differ by a hoisted diffusion-transformer LayerNorm and two behaviour changes.

Their injection point, found by the row and worth not re-deriving: `ModelRunner.__init__` takes
`model_class` and does `self.model = model_class(self.config)`
(`core/runners/model_runner.py:51`); `training_step` is `self.model(batch)` then `self.loss(...)`.
**One injection point, and their loss, LR schedule, `grad_manager`, optimizer and EMA all stay
theirs.** That is what makes "their test, our model" a real artifact rather than a rewrite.

Consequence: `of3t-equivalence` keeps all four deliverables, with deliverable 1 restated as *their
`test_training_full.py` from 0.5.0, with our model injected via `model_class`, and the CUDA skip
reported honestly rather than worked around silently*. PROTOCOL §2's factorisation stands on its
own merits and is not a replacement for it; the two are complementary, and §2 remains the reason N
can be 20.

### R1 original consequence (superseded by R1-AMENDED above, kept for the record)

`of3t-equivalence` loses deliverable 1 and keeps 2, 3 and 4; PROTOCOL §3-§7 replaces it. Its brief
is edited, not just this ledger (see the binding lesson at the foot of this file).

### R2. Our loss weight table does not cover OpenFold3.

The charter: *"The loss weight table is already per-example-overridable and already covers
OpenFold3 alongside AF2, Protenix, Boltz-1, Boltz-2 and BoltzGen. Extend it; do not branch it."*

Verified: `tt_bio/train/losses.py:63` `LOSS_WEIGHTS` has exactly two keys, `"pretrain"` and
`"finetune"`, and they are **Protenix's** constants — `SIGMA_DATA = 16.0  # generator.py:40`,
`DISTOGRAM_GRID  # loss.py:534-536`, `LDDT_RADIUS  # ProtenixLoss.lddt_radius, loss.py:1465-1468`.
It is keyed by *stage*, not by model. `grep -n "openfold3|of3"` over `train/objectives.py`,
`train/losses.py` and `train/catalogue.py` returns **nothing**. The only OF3 mention anywhere in
`tt_bio/train/` is `cli.py:30`, `ADAPTABLE = ("protenix-v2", "openfold3")`.

Consequence: OF3's weights are real work, not an extension of an existing row in a table. And they
are shaped differently from ours — theirs are **per-dataset overrides inside each stage config**,
delivered per-example in the batch (`batch["loss_weights"]`, `runner.py:448`). "Extend, do not
branch" still holds as a direction, but it is a table *reshape*, and the reshape is the work.

### R3. Our OF3 is an independent reimplementation, and the weight map fuses. "Per-parameter" needed a decision.

`NOTICE:53-56`: *"Like Protenix-v2 this is an INDEPENDENT ttnn reimplementation; no upstream compute
code is vendored (only the data pipeline above)."* The charter never mentions this, and it decides
whether deliverable 2 is even well-defined.

Verified: `tt_bio/openfold3_weights.py` renames OF3 checkpoint keys onto Protenix key names and
delegates to `tt_bio/protenix_weights.py`, which at lines 25-26 does
`torch.cat([ref_sd["linear_a_g.weight"], ref_sd["linear_b_g.weight"]], dim=0)` into our single
`g_in.weight`, and the same for `linear_a_p`/`linear_b_p` into `p_in.weight`. Our parameter set is
therefore **not in bijection** with theirs at every triangle-multiplication module.

Consequence: settled in PROTOCOL §3a — compare in **their** space, splitting our fused gradient
back along dim 0, and publish a bijection manifest with the unmapped set on both sides enumerated
before any gradient is compared. A fused comparison would let one half's agreement mask the other
half's error.

### R4. OF3 training never exceeds a 768-token crop. The 1024 aa OOM is not on the training path.

The charter hands `of3t-memory` the 1024 aa OOM as its centrepiece.

Verified, `examples/training_yamls/`: `token_budget` is **384** in `initial_training.yml`, **640**
in `finetune_1.yml`, **768** in `finetune_2.yml` and `finetune_3.yml`. Those four configs are the
whole published recipe.

Consequence: the 1024 aa OOM (`of3-1024aa-oom-allocation-count-not-size`) is an **inference** size
limit and reproducing their training does not require clearing it. `of3t-memory`'s ladder is
**384 / 640 / 768**, and its first duty is to find where the taped path actually stops on that
ladder. The count-vs-size lesson still binds — it is how the row must measure — but the 1024 aa
target does not. Brief edited.

### R5. Per-rank batch is 1, on their own authority.

`projects/of3_all_atom/runner.py:390-392`:
`assert len(batch["pdb_id"]) == 1, "Currently only local batch size of 1 per GPU is supported."`

This settles the per-rank ambiguity the campaign's DONE_CHECK guards against, from their code
rather than from an estimate. Every s/step figure in this campaign is **per rank at batch 1**, and
the GPU baseline is measured that way.

### R6. Their runner disables confidence parameters on zero-confidence-weight samples.

`runner.py:364-386`, `_get_sample_disabled_param_names`: when a sample's summed confidence loss
weight is zero, the confidence head's parameters are returned as disabled. `initial_training.yml`
zeroes those four terms on 4 of its 5 datasets, so this fires on most of initial training.

Consequence: on such a batch their confidence parameters have **no gradient**, not a zero one, and
PROTOCOL §3b makes `None`-matches-`None` a compared property rather than an implementation detail.
Filling a missing gradient with zeros to make a comparison run is forbidden; it is the exact defect
in memory `zero-filled-missing-gradient-hides-an-untrained-model`.

---

## K — knowns, verified this pass

- **K1. The PTX work is on `origin/main`, not waiting at the gate.** `git merge-base --is-ancestor`
  is true for `origin/wk/ptx-fastpath`, `origin/wk/ptx-objective`, `origin/wk/ptx-unify` and
  `origin/wk/ptx-interface` against `origin/main` (`bd643929a`). `wk/ptx-crop`, `wk/ptx-diffusion`
  and `wk/ptx-integrate` do not exist on origin. So the charter's *"PTX's branches are already
  waiting on Moritz's gate"* is stale for those four: build on `main` and inherit them.
- **K2. The taped surface is `tt_bio/taped_ttnn.py`, and `_TAPED` in `autograd.py` is not it.**
  `autograd.py:1439` `_TAPED = {"linear": ..., "layer_norm": ...}` is the two-entry `tt_bio/ops.py`
  surface only. The ttnn verb surface is `taped_ttnn._VERBS`: **~36 registered names**, 19 via
  `@_verb(...)` (matmul, softmax, transformer.scaled_dot_product_attention, permute, reshape,
  slice, concat, chunk, sum, max, pad, transpose, squeeze, unsqueeze, deallocate,
  experimental.minimal_matmul, experimental.nlp_concat_heads, experimental.nlp_create_qkv_heads,
  softmax_in_place) plus 17 direct assignments (linear, layer_norm, silu, sigmoid, relu, exp, add,
  add_, subtract, multiply, multiply_, divide, clone, reallocate, to_layout, to_memory_config,
  typecast). Anyone who greps `_TAPED` concludes the tape covers two verbs and is wrong. Recorded
  so no row re-derives it.
- **K3. The shipped pairformer routes none of its calls through `ops.linear`** — it calls
  `ttnn.linear`, covered by `taped_ttnn`. A raw weight among raw handles is not seen by the
  short-circuit, so `autograd.parameter(raw)` is what makes a weight a leaf at all 331 taped calls
  rather than at the four routed ones (`autograd.py:1137-1145`). A model can be fully
  differentiable and still have no trainable weight.
- **K4. The upstream training stack is entirely real**, contrary to nothing in the charter but
  worth pinning: `core/loss/`, `core/utils/lr_schedulers.py` (93 lines),
  `core/utils/grad_manager.py` (406), `core/runners/model_runner.py` (188),
  `core/data/framework/data_module.py` (740), `entry_points/experiment_runner.py` (934) and the
  real `pl.LightningModule` at `projects/of3_all_atom/runner.py` (989). **`core/model/` also
  exists** — upstream ships the compute code, so the GPU reference runs their real model.
- **K5. LR schedule.** `AlphaFoldLRScheduler`: linear warmup `warmup_no_steps=1000`, plateau at
  `max_lr=0.001`, exponential decay `decay_factor=0.95` every `decay_every_n_steps=50000` starting
  at `start_decay_after_n_steps=50000`. Pure function of the step index, no state — which is why
  PROTOCOL §4 verifies it exactly over its whole domain instead of sampling it in a trajectory.
- **K6. EMA is updated every step**, `runner.py:768-769` (`self.ema.update(self.model)`), and
  validation swaps the EMA weights in (`runner.py:588-599`). It is part of the update rule and is
  verified under injected drive in PROTOCOL §5.
- **K7. No single stage fires every loss term.** Per-stage overrides measured from the four yamls
  are tabulated in PROTOCOL §6. `finetune_3` is the confidence-only stage (`mse: 0.0`,
  `distogram: 0.0`, `pae: 1e-4`). Coverage is therefore the **union over stages** with a named
  `(stage, dataset)` per term, not "all terms in one run".
- **K8. The 0.4.3 sdist** is 33 MB from PyPI and extracts to a full tree.
  `of3t-reference` pins it into `perf/of3t_reference/` with a hash; nothing may depend on a copy
  under `/tmp`.

---

## The binding lesson this campaign was given

**A decision written into a state doc does not reach a running row.** `train-i-run` kept a schedule
that had been overruled because the decision went into `state/ask-9115-decision.md` and a TASKS
line, while the row read its brief. So every correction above that changes a row's direction —
R1 for `of3t-equivalence`, R2 for `of3t-equivalence` and `of3t-data`, R4 for `of3t-memory`, R5 for
`of3t-reference` and `of3t-perf` — was written **into that row's brief** in
`~/.coworker/workstreams/<slug>.txt` as well as here. This ledger is the reference; the brief is
what the row actually reads.


---

## R/K — pass 1 additions from the rows themselves (2026-09-19)

### R7. `tape()` is silently blind to a function-local `import ttnn`. Found by `of3t-perf`.

`tape()` installs itself by rebinding the module-global name `ttnn`. `openfold3_confidence.py:163`
and `openfold3_host_prep.py:219` do `import ttnn` **inside a function**, which binds the real
module into the function's scope at call time, so **16 call sites run untaped whatever the swap
did**. They are handed raw handles rather than `autograd.Tensor`s, so **nothing raises** — the
tape's guarantee that an op it cannot follow is loud does not reach this shape of gap.

Worse, and this is why it reaches the protocol: **the confidence head is one of the two blind
files**, and `openfold3_fold.py:415-416` already writes `si_trunk` and `zij_trunk` to HOST before
calling it. So as shipped, **a confidence-loss gradient cannot reach the trunk at all**, and
`finetune_3.yml` is the confidence-only stage, so this is not a corner case.

Consequence: PROTOCOL §6 amended (A2) — coverage is censused at RUNTIME during a real taped
forward, never statically, plus a check that no module on the training path imports `ttnn` inside
a function. **Owned by `of3t-tape`**, which owns those files; `of3t-perf` hands it off and returns
to the stage breakdown.

### R8. The protocol's own §7 was vacuous as first written. Found by `of3t-reference`.

`AlphaFoldLRScheduler` is constructed with `last_epoch=-1`, so `_LRScheduler.__init__` calls
`step()` once and **the first optimizer step runs at lr = 0.0**. Measured on a 2-element parameter
with grad 1.0: step 1 lr=0 delta=0; step 2 lr=1.8e-06 delta=1.79e-06; step 3 lr=3.6e-06
delta=3.58e-06.

The row flagged the first two rungs as uninformative. The defect is larger than that: over 20
warmup steps the weights move ~1e-4 relative, so a relative L2 on `w_k` is dominated by a `w_0`
that is identical **by construction** and would read as a pass for any implementation at all.
PROTOCOL §7a now compares the update `d_k = w_k - w_0`, marks k = 1 and k = 2 non-discriminating,
and runs the trajectory a second time with `warmup_no_steps` scaled down. A gate nobody has
watched fail is not a gate, and this one could not have failed.

### R9. The campaign's DONE_CHECK passed a state doc whose every measurement field said NOT YET MEASURED. Found by `of3t-equivalence`.

The `worst_case` check was satisfied by the word "worst" plus any number matching
`\d\.?\d*e-?\d+` — and **quoting the protocol's own 5.0e-02 bar satisfies it**. The gate could
not tell a quoted tolerance from a measured worst case. This is the recurring class in memory
`donecheck-field-regexes-pass-on-an-honest-attempt`.

Fixed this pass in `_of3t_donecheck.py`: `GRADIENTS` must now name a **located** tensor (a dotted
parameter path), and a new `measured` spec per row rejects placeholder text
(`not yet measured`, `unstarted`, `TBD`, `pending`, ...) inside the fields that must carry
measurements. Verified by re-running the gate: `of3t-equivalence`'s doc now fails on exactly the
three fields it had not measured, where before the fix it passed.

### K9. Upstream argument traps in their preprocessing, from `of3t-reference`. For `of3t-data`.

1. `--preprocessed-dir` wants `structure_files/`, **not** the preprocessing output root. Given the
   root it consolidates zero FASTAs and mmseqs exits 1 with "Error: query createdb died", which
   reads like a broken mmseqs install and is not one.
2. `--allow-missing-alignment` does **not** tolerate chains without an MSA. It sets
   `filter_missing_alignment=False`, skipping alignment-representative assignment entirely; every
   id comes out `None` and the dataset dies later at `msa.py:537` on `Path(None)`. Leave the flag
   off and supply representatives.
3. An MSA file's stem must be a key of `msa.max_seq_counts`
   (`dataset_config_components.py:80`). Anything else is not read, and an empty MSA set raises
   `IndexError` rather than falling back to single-sequence.
4. `n_templates: 0` is not how upstream expresses "no templates": `TemplateSettings` has no
   enabled flag and 0 reaches `get_with_unknown_3_to_idx` on an empty array
   ("cannot call vectorize on size 0 inputs"). Express it as a zero-entry template cache.

### K10. Their own published cache does not parse with their own subset generator. From `of3t-equivalence`.

A bare `"resolution": NaN` in the published cache is rejected by all four ijson backends. The row
shimmed only the parser and left the sampling logic untouched, which is the right shape of fix.
Also: one 404 in their manifest (`val_template_cache/7kud_A.npz`), and **two** NVIDIA kernel
switches to turn off rather than one — `use_deepspeed_evo_attention` and
`use_triton_triangle_kernels`.

### K11. Static CALL-SITE coverage of OF3 is 89.6 %, from `of3t-perf`.

310 of 346 ttnn CALL SITES in `tt_bio/openfold3*.py` are verbs the tape already registers (the row corrected my first write-up, which said 89.6 % of verbs; it is 89.6 % of sites)
(`perf/of3t_perf/verb_census.py`, commit `249a9492b`, read off `taped_ttnn.VERBS` so K2 is
honoured). **A static screen only** — it does not follow shared helpers in `tenstorrent.py`, and a
covered verb still has to survive a real taped forward (see R7 for why that distinction is not
pedantry). The one real verb gap: **`ttnn.embedding`, 5 sites, none a table lookup** — each
replays a host gather on device (`openfold3_atom_transformer.py:107`,
`openfold3_diffusion_decoder.py:76`, `openfold3_diffusion_module.py:105,128,327`), so each
backward is a scatter-add, and ttnn scatter/gather is rate-limited per element. A correctness gap
and a perf risk in the same five sites; price it when writing it.

### R10. `triatt_qkv.py` declines taping at one entry point and not at the three later fusions. From `of3t-memory`.

`qkv_heads` declines under `ops.taping()` (`tt_bio/triatt_qkv.py:74`), but `gate_proj` (170),
`qkvg_heads` (271) and `qkvgb_heads` (374) do not. **OF3's Pairformer reaches `qkvgb_heads`**,
which calls `G.generic_minimal_matmul` -> `ttnn.generic_op`, and the tape raises
"ttnn.generic_op has no tape entry".

The shape of this defect is the point: a decline was added to one entry point, and three later
fusions were added to the same file without it. `mm_dualnoc`, `trimul_tail`, `triatt_sdpa`,
`reblock_permute` and `eltwise_fusion` all decline correctly — this one file is the gap. Same
class as `hf-revision-pin-fix-missed-three-direct-callers`: a fix applied at one call site while
its siblings were written later and never picked it up. **Owned by `of3t-tape`.**

### R11. OF3's `fp32_softmax=True` path has no backward, and Protenix never takes it. From `of3t-memory`.

With R10 worked around, OF3 dies in `_fp32_softmax_tail` (`tenstorrent.py:3758`):
`ttnn.add_(sc, bias_f, input_tensor_a_activations=[MUL_UNARY_SFPU(0.17677669)])` has no entry in
`taped_ttnn._FUSED_UNARY`. The activation is a scalar multiply, so the gradient is `g * c`.

**This is the first concrete limit on the free-transfer hypothesis.** PTX's 331/331 was measured
on Protenix, which does not take the `fp32_softmax` path, so the coverage number does not transfer
to a path Protenix never exercises. The charter's premise survives in the main, but "registered on
the verbs, not on Protenix" does not mean "covers every path another model takes". **Owned by
`of3t-tape`.**

### K12. Anything instrumenting the tape must exclude `tt_bio.taped_ttnn`. From `of3t-memory`, cost it a pass.

`taped_ttnn._Ttnn.__getattr__` decides "is this a nested namespace" with
`isinstance(attr, type(ttnn))`, **reading its own module global**. Any second proxy that rebinds
`tt_bio.taped_ttnn.ttnn` — a memory watcher, a profiler, a census harness — turns that test into a
false negative, and **every `ttnn.experimental.*` verb silently comes back RAW and untaped**.

The symptom is not a tape error. It is a pybind `TypeError` several frames later —
`minimal_matmul` handed an `autograd.Tensor`. Note this is the same underlying mechanism as R7,
seen from the other side: R7 is a function-local import escaping the rebind, K12 is a second
rebind defeating the namespace test. **Module-global rebinding is how this tape installs itself,
and both of its failure modes are silent.** Any row instrumenting the tape excludes
`tt_bio.taped_ttnn` from its own interception.

### K13. First measured OF3 training-size memory, from `of3t-memory`. Per-block checkpointing is not optional.

OF3 dims, 1 Pairformer block, 384 tokens, p300c card 2, untaped shipped forward against the
retained-set upper bound (`ttnn.deallocate` replaced by a keeper, DRAM only). Both numbers
reported, count and bytes, as that row's gate requires:

- shipped: **0.357 GB peak, 122 live DRAM buffers**, largest 0.057 GB
- retained: **3.566 GB peak, 329 live DRAM buffers**, largest 0.151 GB, median 3.54 MB
- 48 block-boundary `(s, z)` pairs at 384 tokens: **1.836 GB**

So 48 x 3.2 GB retained is **154 GB against a 34 GB card**, while boundaries plus one live block is
**~5.4 GB**. **Per-block checkpointing is not optional at any crop**, including the smallest one
their recipe uses. This is a size result at 384, so it does not by itself settle the count-vs-size
question at 640 and 768; the ladder is still running.


### R12. Only the FINAL trunk recycle carries a gradient, and the recycle count is drawn per step. From `of3t-perf`.

From their `model.py`, verified against the source text by
`perf/of3t_perf/their_step_shape.py` so it fails loudly if a later version moves it:

- `enable_grad = is_grad_enabled and is_final_iter and not train_confidence_only`. **Every earlier
  cycle is inside `torch.no_grad()`.** A tape over all cycles holds several times the activations
  and computes a gradient that is not theirs.
- **In training the recycle count is drawn per step from U{0..3}** via a synced generator;
  inference pins it at 3. A training step's trunk is therefore a `no_grad` prefix of random length
  plus one taped cycle.
- `_train_diffusion` differentiates **48** noised structures per step in `initial_training` and
  **32** in the finetunes. That term has no counterpart in our inference path and, on shape alone,
  is the likely dominant cost and the likely memory wall — **not the trunk**.

Consequences, and they change three rows. `of3t-memory`: if the ladder is measured with all
cycles taped, its stopping point is wrong by roughly the cycle count, and its wall is probably in
the diffusion term at crop 384 rather than at any crop. `of3t-tape`: tape the final cycle only;
our `tt_bio/ops.py` recycle hook already exists for exactly this. `of3t-perf`: pin the drawn count
in every arm. PROTOCOL §4a added (A4) — the random draws are inputs to the update rule, pinned
and matched for equivalence, and stated alongside the clock for any timing claim.

### K14. Their step shape, per stage, from `of3t-perf` (`perf/of3t_perf/their_step_shape.py`).

| stage | crop | batch | diffusion samples | mini rollout | confidence-only |
|---|---|---|---|---|---|
| initial_training | 384 | 1 | 48 | 20 | no |
| finetune_1 | 640 | 1 | 32 | 20 | no |
| finetune_2 | 768 | 1 | 32 | 20 | no |
| finetune_3 | 768 | 1 | skipped | 20 | yes |

### K15. 0.4.3 and 0.5.0 agree on everything this campaign reads from the configs. From `of3t-perf`.

The four `examples/training_yamls/*.yml` are **byte-identical** across the two versions, as are
`architecture_defaults` and five asserted grad-scoping facts; both extracts are committed at
`a8575be35` so the agreement is visible rather than asserted. This de-risks the R1-AMENDED
decision to target 0.5.0 considerably: the stage recipe the campaign reproduces does not change
between the version tt-bio pins and the version whose training test exists.


## Instruments B and C are DONE (2026-09-19, `of3t-equivalence`, `wk/of3t-equivalence` at `5247913cd`)

**PROTOCOL §4, the LR schedule: 109,005 step comparisons across four configs against upstream's
`AlphaFoldLRScheduler` driven stepwise, 0 mismatches, exact float equality, all eleven knees.**
Negative control localises to exactly one step.

**PROTOCOL §5, the optimizer: worst relative 2.738e-07 at step 191 on `bias`, against the 1e-06
bar**, over 200 injected-gradient steps with no model. Controls: 1 % on one tensor at one step
flags that tensor only (4.948e-04); a zeroed-gradient drive puts all four tensors over the bar.
Both bars therefore have been watched to fail, which §3e requires and which no bar had yet done.

### R13. Our LR schedule did not express theirs at all. Fixed the unified way.

`af3_lr` carried **Protenix's** `AlphaFold3LRScheduler` — warmup, then a decay exponent counting
from step 0, no plateau. OF3 ships the **AF2-supplement** schedule: plateau until
`start_decay_after_n_steps`, exponent counting from there starting at 1. Different closed forms.
The row added **`plateau_until` as the one parameter they disagree on**, rather than branching per
model: `None` is Protenix unchanged, and a 100,002-step regression guard confirms that path is
bit-identical. That is UNIFIED, NEVER PER-MODEL done correctly and it is the pattern other rows
should copy.

### R14. Their optimizer is `torch.optim.Adam`, not AdamW, and all three hyperparameters differ.

`runner.py:854`; no weight decay. Our defaults are Protenix's: weight_decay 0.01, beta2 0.999,
lr 3e-4, against OF3's **0.0 / 0.95 / 1.8e-3**. An OF3 recipe that does not pin all three trains a
different update rule **with every metric looking healthy**. Expressible as arguments; nothing
needs rewriting. PROTOCOL §5 corrected (A5) — it had said AdamW.

### R15. A one-step schedule offset is worth 1.499e-03, about 1500x the bar.

Lightning calls `optimizer.step()` before `scheduler.step()`, so their step k uses `lr(k-1)`;
ours increments first and uses `lr(k)`. Measured by running the arm both ways. This is exactly the
wiring class §7 exists to catch and it is now a concrete instance for the rewritten §7a.

### K16. Adam is invariant to a uniform per-tensor gradient scaling. The sharpest result of the pass.

The row's first §3e control scaled one tensor's gradient by 1.01 at **every** step. The weight
trajectory moved **2.547e-07** — under the bar, invisible — because `m_hat/(sqrt(v_hat)+eps)`
cancels the factor. It recorded the failed control rather than deleting it, which is the right
instinct and the reason we have this.

**A whole class of gradient error is structurally undetectable downstream of Adam at any
trajectory length.** No N fixes it. This is the standing answer to anyone who proposes dropping
instrument A in favour of running more steps, and it is now PROTOCOL §7a-bis and a line in §8's
DOESNOT.

### K17. Two real gaps on our side, named before they were hit.

- **We have no EMA anywhere in `tt_bio/train/`.** Theirs updates one every optimizer step at
  decay 0.999 (`runner.py:769`). So §5's EMA half is untouched.
- **Our clipping is global-norm** (`train/optim.py:113,168`, `clip_norm=10.0`, cited to Protenix's
  `configs_base.py:80`), while their shipped default is **`per_sample_clipping: True` at
  `clip_val 10.0`**, which their `grad_manager` implements and our `clip_norm` cannot express. So
  §4's `grad_manager` half is untouched.

**Ownership arbitration, mine:** `tt_bio/train/optim.py` goes to `of3t-equivalence` for both. The
TRAIN campaign's rows concluded at 14:12 today so there is no live collision, and the `af3_lr`
edit already made there was additive with a bit-identity guard, which is the right shape. **The
hazard is at composition, not now**: `wk/train-i-run` is 87 commits ahead of `origin/main` and
unmerged, `wk/train-orchestrator` 13. Check `git log origin/main..origin/wk/train-i-run --
tt_bio/train/optim.py` before editing, so we do not walk into
`parallel-branches-independently-fix-same-defect-merge-silently-picks-one`.

### K18. Two rows collided on a generic `/tmp` scratch path and one state doc silently became another's. From `of3t-perf`.

`of3t-perf` staged its state doc through `/tmp/of3t/state.md`. Between two of its edits that file
became **`of3t-equivalence`'s** state doc — same host, same generic path. Its next edit read that,
and its next copy installed it as `state/of3t-perf.md` on both hosts. For a few minutes one row's
state doc was a verbatim copy of another's, **carrying that row's branch, its instruments and its
`VERDICT: PARTIAL`**.

**Nothing errored and nothing was overwritten in git.** The only visible symptom was a DONE_CHECK
suddenly failing on fields that had been fine. And the near miss is the point: **had the gate not
been tightened this pass (R9), the clobbered doc would have PASSED** — a row reporting another
row's verdict under its own name, through a gate that was watching for the right things and could
not see this.

Verified rather than taken on report: the two docs now share **0 substantial lines**,
`of3t-perf.md` is 11,125 B at `VERDICT: BLOCKED` on `wk/of3t-perf` `a8575be35`, and
`of3t-equivalence.md` is 10,427 B at `VERDICT: PARTIAL` on `wk/of3t-equivalence` `5247913cd`.

**Standing rule for this campaign, added to all six briefs:** scratch paths are slug-scoped or
per-process, never a generic shared name. Six parallel rows on two shared hosts is the general
shape, and `sibling-perf-campaigns-need-namespaced-output-paths` already covers the `perf/` half
of it — this is the same lesson for `/tmp`, where nobody had applied it.

### R16. OF3's training memory limit is a COUNT problem, and the crossing is between 384 and 640. From `of3t-memory`.

qb2 p300c card 2, of3-p2-155k dims, ONE Pairformer block, DRAM,
`perf/of3t_memory/out/ladder_1blk.json`. Both numbers at every rung, as the row's gate requires:

| crop | shipped | retained upper bound | largest alloc | median retained |
|---|---|---|---|---|
| 384 | 0.357 GB / 122 allocs | 3.566 GB / 329 allocs | 56.6 MB -> 151.0 MB | 3.54 MB |
| 640 | 0.954 GB / 142 allocs | 13.193 GB / 1878 allocs | 419.4 MB -> 419.4 MB | 0.98 MB |

48 block boundaries: 1.836 GB / 227 at 384, 5.070 GB / 408 at 640.

Growth 384 -> 640: **count 5.71x, bytes 3.70x, largest allocation 2.78x**. That last figure is
exactly N^2, so the largest allocation is the pair tensor tracking the crop and retention makes
nothing bigger. The sharpest form of the result: **the mean retained allocation size FALLS with
the crop**, 10.8 MB to 7.03 MB, median 3.54 MB to 0.98 MB — bytes grow because there are *more*
allocations, not bigger ones. At 640 the largest allocation is **identical**, 419.4 MB, in the
shipped forward and in the retained bound; retention adds nothing larger than what inference
already places, just 13x more of it.

**CLASS is count, with the crossing located between the first two rungs of their own recipe.** At
384 the result still reads size-shaped (10x bytes against 2.7x count) and a row that stopped at
the smallest crop would have called it a size problem. This is `of3-1024aa-oom-allocation-count-
not-size` reproduced on the training path at a crop their recipe actually uses, and it is the
campaign's answer to R4's reframing: the 1024 aa target was dropped, the lesson behind it was not.

**Hypothesis, flagged as one, not asserted:** the shipped kernels' reactive narrowing. At 640 a
single-shot allocation is refused, the kernel retries in row blocks, and every chunk the narrowed
path produces is one more live tensor the tape retains. A shape census on the keeper is running to
say which shapes the retained set holds 1544 of. **If it holds it matters to `of3t-perf` too**:
reactive narrowing would be a memory lever that silently buys bytes by paying in allocation
count, and under a tape the count is what binds.

Two caveats the row raised itself and which are the 

---

## ROTATED 2026-09-25T10:35:59Z

This doc reached 109105 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-LEDGER.20260925-123559.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

self-heal,
because `tape()` never passes `exact_softmax=True`. **And the measured failure is genuinely
repaired**: `pkgarm.py:45` turns the lever on with `with ag.exact_softmax():`, owner
`"exact_softmax"`, which no longer matches the `"install"` an unrelated `uninstall()` passes.

**Two residual findings, filed to the row as an amendment.**

**(a) The unprotected cell of a 2x2 that reads as fully covered.** The owner is a *constant
string*, not a caller identity: `exact_softmax()` records `"exact_softmax"`, and
`install(exact_softmax=True)` records `"install"` — which is the *same literal* that every
unrelated `uninstall()` passes. So turning the lever on through `install(exact_softmax=True)` and
then running an unrelated `install()`/`uninstall()` pair tears it down exactly as before the fix.
The two new tests pin the *safe* two cells — on via the scope CM with an unrelated pair, and on
via `install(exact_softmax=True)` with its OWN teardown — and together they read as "both
directions", which is how the fourth cell stays invisible. It is **latent, not live** (no
production caller uses that entry point), but `exact_softmax()`'s own docstring advertises
`install(exact_softmax=True)`/`uninstall()` as "the same thing without the block", and **the fix
is precisely what made that sentence false**. A constant owner token discriminates between *APIs*,
not between *callers*; distinguishing callers needs a per-call handle or a depth count.

**(b) The arm's provenance stamp names a path the arm does not take.** `pkgarm.py:55` records
`"installed_from": "tt_bio.autograd.exact_softmax() -> install(exact_softmax=True)"`. The arm
calls the context manager, which reaches `_install_exact_softmax("exact_softmax")` **directly and
never calls `install()` at all**. The stamp was harmless while the two paths were equivalent —
and **the fix is what ended that equivalence**, giving them different owners and different
survival under a foreign `uninstall()`. An arm whose defect *was* "the install did not survive"
must stamp the install path it actually took; this one still records the one that would not have
survived. `shipped-artifact-identity-is-digest-plus-recorded-inputs`, in the case where the
recorded input is the sentence rather than the number.

### R181. A per-row workaround around a shipped defect leaves the defect shipped: D247, the fail-fast probe that hangs (pass 414, zero card)

`of3t-verbinstall` lost **230 minutes of card time** — two arms, 115 minutes each, nothing
computed — and wrote the cause in its own state doc without a defect ID. Filed now as **D247**,
and **verified in the shipped file rather than transcribed from the row's sentence**:
`tt_bio/tenstorrent.py:5575`, reached at `:6005` by every `get_device()`.

**The body guards the wrong failure mode.** It wraps `from_torch` / `add` /
`synchronize_device` in `try/except Exception`, so a chip that **throws** is handled exactly as
the docstring describes — closed, re-raised, respawned. A chip that **wedges** never reaches the
`except`: `synchronize_device` blocks indefinitely and there is no timeout, alarm or watchdog in
the function. The docstring's whole claim is that a mis-initialised worker *"fails HERE, at
startup"*. **A fail-fast probe that can hang is worse than no probe**, because it converts the
cheapest failure shape into the most expensive one: a card held by a process every liveness
signal calls healthy. `chip-holder-at-100pct-cpu-can-be-a-corpse`, arrived at from the other end.

**The generalisable half is what the row did next.** It added a bounded pre-flight for its own
launches (`b77e89f27`) — the right local move, and it closes nothing. **The probe is unchanged,
so every other caller on every other model still meets it, and the 230 minutes is a FLOOR rather
than a total.** A row that works around a shipped defect has bought itself out of the blast
radius and left the radius exactly as wide; the workaround also removes the row's own motive to
fix it, and its state doc reads as resolved. This is why the cost belongs in DEFECTS with an ID
and an owner rather than in a row's narrative: **the defect ledger is what outlives the row.**

Handed back to `of3t-verbinstall` with a narrow grant for that one function, since it holds the
reproduction. Bound the probe, not its callers, into the `RuntimeError` path the `except` already
builds — so a wedge and a throw produce the same fast, respawnable outcome. Both D246 and D247
are SOURCE items: no card, no decision, no measurement, and they do not compete with the row's
two outstanding frame scores.

### R182. I sequenced carefully by CARD and was blind to a row already at 991 % CPU on another host (pass 414, zero card)

Dispatching `of3t-cropwall` this pass I reasoned properly about card allocation: qb2 card 2 and
not card 0, because `tt-smi -r` resets the board **pair** and card 1 is `of3t-angle`'s. Then I
checked qb1 for an unrelated reason and found **`of3t-angle`'s `ref_grad.py` at 991 % CPU and
10.9 GB RSS**, building its corrected float64 reference on qb1 — a host its `#DISPATCH` line
does not mention — beside `of3t-verbinstall`'s device arm, at load average **13.58**.

**This is `dispatch-host-grant-does-not-sandbox-which-host-opens-the-device` on the HOST side.**
The known form of that lesson is about which host a row opens a *card* on. The form that bit
here is cheaper and less visible: a row whose card work is on qb2 can put its *host-side* work —
float64 references, and float64 on host is the most core-hungry thing this campaign runs —
anywhere it likes, and nothing in the dispatch record shows it.

**The orchestrator-facing half is the part worth keeping.** My sequencing model is card-shaped:
which row holds which chip, which reset takes which pair. It has no representation of host CPU
at all, so a row can saturate ten cores on a box I believe is owned by someone else and my
allocation reasoning stays confidently wrong. **I was one host away from a number I would have
believed** — `of3t-verbinstall`'s remaining arms are on that box.

**No science was lost and I want to be exact about why, not relieved about it.** Both rows'
outstanding deliverables are accuracy readings — rel, r, cos, angle, and two frame scores —
and an accuracy question is load-insensitive (`a-firing-question-is-load-insensitive-so-a-loud-
box-does-not-block-it`, the same shape). Only durations are corrupted. `of3t-verbinstall`'s
existing "30 minutes of arm time" was recorded at 01:13, **before** this began, so it stands.
Both rows are now told to take no wall time on qb1 while they share it, and `of3t-angle` is told
to stamp host, board and quiet-state **into the artifact** rather than its notes — a reading
whose host is unrecorded cannot later be compared against one taken quiet.

The standing correction to my own practice: **before dispatching, check host load on every box a
live row touches, not just the card map.** A card is allocated; a core is merely taken.

### R183. The campaign carries two definitions of one aggregate triple, and only one of them can carry an angle — A43 (pass 414, zero card)

`of3t-angle` found it while re-reading the softmax ladder, and it is the sharpest instrument
finding since D242. **`of3t-trunkg043/score.py`'s `mass_weighted_rel_l2` is a mass-weighted
QUADRATIC mean of the per-tensor `rel`, while `mass_weighted_norm_ratio` and `mass_weighted_cos`
are ARITHMETIC means of the per-tensor ratio and cosine.** `rel^2 = 1 + r^2 - 2 r cos` holds per
tensor and **not** on those three. Measured on R149's own shipped arm: reported `rel`
0.9153623104186986 against 0.6855 reconstructed from its own reported `r` and `cos` — **a 25 %
residual**. An angle read off that `cos` is not an angle; it is the average of some cosines.

**Why this matters more than it looks.** The whole of the campaign's current position is a
DIRECTION claim — the clause is unreachable by magnitude and reachable only by closing 43.61 %
of an angle. A direction claim resting on an aggregate that does not satisfy the identity would
be a number about nothing, and it would have been very hard to catch later, because
`mass_weighted_cos` is a real quantity that moves in plausible ways.

**Audited, and the published record is clean.** Every direction figure this campaign has
published is in the CONCATENATED definition (`perf/of3t_recut/n384_check.py:40-67`), where the
three are norms of one vector pair: the trunk's **45.763°** (residual 7.1e-16), the pre-repair
**35.561°** (1.1e-16), the float64-space **35.13°** (2.1e-15, and it is `arccos(0.8178953)` from
that split, traced this pass). The `31.92x` contrast between the two spaces' magnitude shares is
also sound — it is quoted by `BF16_SPLIT.json` under `and_they_disagree_sharply` as the evidence
**for** A37's no-cross-space rule, not as a carry across it. **I checked my own VERDICT against
this before writing the clause, rather than after.**

**A43 is the rule**: a `cos`, an angle, a magnitude/direction share or a best-rescaling number
may be read only from a triple whose identity residual is published beside it at float64 noise.
An aggregate that fails the identity is a fine summary of error MAGNITUDE and carries no
direction — quote its `rel`, never its `cos`. And the two definitions may not be mixed in one
comparison: `of3t-angle`'s ladder rungs are mass-weighted because that is the definition R149
published in, while its splits are concatenated, so it reports both per arm and divides neither
by the other. **This is A37's disease on a new axis — not two frames, but two AGGREGATIONS of
one frame.** A reference is part of a measurement's identity; so is the aggregation.

### R184. The package leg of D245 is CLOSED bit-exactly, so the best lever has a shippable path — and my own VERDICT said the opposite for a pass (pass 414, zero card)

`of3t-verbinstall` reports PACKAGE closed. **Verified against the artifact, not the prose**:
`ARMDIFF_PKG_HF3B_vs_CEIL_HF3.json` reads `compared 2736`, `bit_identical 2736`, `differing 0`,
`only_mine []`, `only_theirs []`, `all_bit_identical true`. The packaged install scores
**0.41752141981218177** against float64 — `ceiling_hf3` to seventeen digits — and reach is
banked at **verb 5901, raw 1742** where the first inert arm read verb 0. It is qb1's p150a
reproducing a qb2 p300c arm, so it is a third cross-board A/A as well.

**I verified this one specifically because it is flattering.** It moves the campaign's position
in our favour, and `a-silent-failure-whose-direction-is-flattering` is the entry that says a
result which helps deserves the check a result which hurts would get automatically. The
aggregate alone would not have been enough either — the row is right that a mass-weighted score
over 2,736 tensors can agree to seventeen digits while low-mass tensors differ, which is why
`all_bit_identical` and not the score is the reproduction claim.

**My VERDICT and GAP both asserted the opposite.** They said *"the best softmax arm still has no
shippable path"* and *"BOTH shippable installs have now failed"* — true when written, falsified
by the row within the pass. Corrected in the same pass this time. **This is R157's rot for the
third time in my own fields**, and the pattern is now specific enough to name: the sentences
that rot are the ones that summarise a LIVE ROW's position, because a live row's position is
the only thing in the doc that changes without me. Fields that summarise artifacts do not rot;
fields that summarise rows do.

**Two things must not be conflated because they resolved in the same hour.** The lever now has a
shippable path. Whether the lever is WORTH anything is `of3t-angle`'s open question: its 1.0525x
was taken on the double-counted functional where 62.52 % of the error was magnitude, and the
repaired functional is 0.2749 % magnitude and 99.7 % angle. **A shippable path to an inert lever
is a shippable path to nothing.** D245 accordingly stays UNFIXED with its headline clause struck
and its falsifier still on the card, rather than being closed on the good half.

**And D246 did NOT escalate, which I checked rather than assumed.** The row's VERDICT prose says
the arithmetic *"installs from `tt_bio.autograd.install(exact_softmax=True)` and comes out with
`uninstall()`"* — the unprotected entry point. Grepping the branch, every non-test caller still
uses `with ag.exact_softmax():` (`pkgarm.py:45`, `reachprobe.py:98`), so D246 remains latent.
What that prose shows is finding (b) doing damage on schedule: `pkgarm.py:55`'s wrong
`installed_from` stamp has propagated into the row's own VERDICT sentence. **A bad provenance
string does not stay in the JSON — it becomes the sentence everybody repeats.**

### R185. A refusal's byte count is not evidence of contiguity — 90.6 % of the 2.7 GB was padding, and I repeated the wrong noun in the brief that found it (pass 414, zero card)

`of3t-cropwall`, dispatched this pass, has already overturned D205's mechanism. The
2,717,908,992 B refusal that `of3t-crop768` banked as *"contiguity inside `ttnn::concat`"* was
**tile padding**: `taped_ttnn.py:922`'s qkv-heads vjp scattered into a rank-4 axis of **extent
3**, and TILE layout pads the second-to-last dim to **32**, so the allocator was asked for
10.667x what the gradient needed. **90.625 % of that buffer was padding**, and three such
buffers were co-live. Filed as **D248**; every byte re-derived independently before filing.

**The lesson is about the noun.** A refusal reports *how many bytes were asked for* and *how
many were free*. Neither says the request was **necessary**. `of3t-crop768` had the right
number and the wrong question: it asked why the card could not supply 2.7 GB contiguously, and
the answer was that nothing ever needed 2.7 GB. The diagnostic that separates those is one
question — **what does this buffer LOGICALLY hold?** — and it costs no device time. Here the
answer was 254,803,968 B.

**The fix had to be free, and was shown to be.** The packed width decomposes as `[3, H, dh]`,
so slot `s` is equally a contiguous **last**-axis range, and the last axis is a whole number of
tiles. Same gradient digest `68b639dc693788dc`, max_abs 0.0 against a **float64 host** scatter
rather than a second device expression — bit-identical, which is what a pure re-indexing must
be — at 3.02x less DRAM and a 5.33x smaller largest allocation. The 34.7x wall clock was taken
on a loaded box and the row correctly refuses to call it a perf number.

**And the correction lands on me.** My dispatch brief's contribution was real and was the wedge
the row used: *the odd-32-tile story cannot explain 576, which is EVEN at 18 tiles, so the two
stories are separable and 576 is the arm that separates them.* That held exactly — 544 is the
odd-tile L1-plan collapse into an unblocked `[544,4,544,544]` fp32 score tensor, 576 was
padding. **But I carried crop768's noun across while doing it**, writing that "544 and 576 die
on contiguity". I verified the SEPARATION and not the MECHANISM NAME, which is
`verify-the-rows-noun-not-only-its-numbers` in the one form I had not met: I re-derived the
split that made the finding possible and copied the word that made it wrong. A brief's framing
is inherited by the row that reads it, and this one got the right answer despite mine.

### R186. A pre-registered disjunction can smuggle an unjustified POSITIVE claim into one of its branches, and landing on that branch does not license it (pass 414, zero card)

`of3t-verbinstall`'s FALSIFIER is a model of the form: two sides written down in
`PREREGISTERED.md` at `a06118403`, **2026-09-22 20:30:38Z**, and the arm run four hours later at
00:33Z — so the pre-registration genuinely precedes the measurement, which I checked rather than
accepted. It landed on the second side to four significant figures: `ROUTE_NORAW`
**0.5605474136179824** against `ROUTE_HF`'s **0.5605347900452246**, a move of **+0.0023 %** with
all raw serves gone. The suppression is verified and not assumed — `suppressed_raw 1869`,
`served_raw 0`, `served_taped 9184` — and the row is right that an arm with 0 suppressed would
have been ROUTE_HF wearing a label. **The candidate mechanism is refuted as the carrier.**

**But the pre-registered second side said two things, and only one of them was earned.** It read:
*"the raw serves are not the carrier, **the 472 extra taped serves are**, and the candidate
mechanism above is dead."* Refuting the raw serves establishes the negative half. It establishes
the positive half **only if raw-serves and the-472 exhaust the space**, and they do not:
`ROUTE_NORAW` serves **9,184** taped where the verb arm serves **5,901**, a difference of 3,283,
not 472. The "472" came from an earlier census pairing (`CENSUS_VERB_HF` vs `CENSUS_ROUTE_HF2`,
5,757 − 5,285) taken on different arms, and it was stale by the time these arms ran.

**The row declined the attribution and was right to**: *"which subset of the taped population
carries it is not isolated here and is not claimed."* That is the correct reading of its own
pre-registration, taken against the pre-registration's own words, which is harder than following
them. **The lesson for how this campaign writes pre-registrations**: an "A or B" is a claim that
A and B are exhaustive, and that claim needs its own argument. A branch that says *"not A,
therefore B"* should be written as *"not A; what carries it is then open"* unless exhaustiveness
was established when the two sides were fixed. Otherwise pre-registration — the instrument that
exists to stop post-hoc rescue — becomes the vehicle for a conclusion nobody tested.

### R187. The arm that makes MORE softmaxes exact is the WORSE arm, and that breaks the framing the whole softmax ladder was argued from (pass 414, zero card)

Falling out of the same falsifier, and sharper than the question it was built to answer.

    arm          exact softmaxes                      vs float64
    ROUTE_NORAW  9,184 taped (exact fwd AND Jacobian)  0.5605474136179824
    PKG_HF3B     5,901 verb + 1,742 raw                0.4175214198121818

**The arm that makes more softmaxes exact — and their Jacobians too — is 34.26 % farther from
the true gradient.** So the gap is not raw-versus-taped in either direction, and the "consistent
arm" framing the route was argued from is wrong on *both* sides of it: consistency of the
softmax population does not order these arms, and adding exactness moved the reading the wrong
way. Which subset of the taped population carries it is not isolated and is not claimed.

**Why this matters beyond D245.** The campaign's best trunk number, 1.0525x, is a softmax-lever
number, and the mental model behind every rung of that ladder has been *more exact softmax ->
closer gradient*. That model is now falsified by a controlled within-frame A/B. It does not make
the 1.0525x wrong — it was measured, and the package install now reproduces it bit-exactly — but
it removes the reason anyone had for expecting the lever to generalise. **`of3t-angle` is
measuring whether that lever closes the ANGLE on the repaired functional, and this is the second
independent reason to think the answer may be no**: the first is that the repair left a 99.7 %
direction error where the lever's gains were measured against a 62.52 % magnitude one.

A non-monotone response to exactness usually means errors that partially cancel. Nothing here
establishes that, and it is named as the obvious candidate rather than as a finding.

### R188. The exact softmax CLOSES THE ANGLE — it is a direction lever, not a magnitude one — and the campaign's headline moves 1.0525x -> 1.2224x (pass 414, zero card)

`of3t-angle` concluded **GO** and answered the question the repair opened. On the repaired
injection, frame384 frame, concatenated triple, vs upstream's own bf16 / vs float64:

    arm                         rel                   cos           angle
    shipped `--lever all`       1.5603 / 1.3785       0.3140 / 0.5390   71.6968° / 57.3839°
    verb `ceiling_hf`           0.6515 / 0.7886       0.8106 / 0.7398   35.8461° / 42.2895°
    verb+module `ceiling_hf3`   0.6512 / 0.7881       0.8107 / 0.7400   35.8351° / 42.2685°

**Shipped to verb closes 35.8506° of a 71.6968° angle — 50.003 % of it — in the space the
clause is graded in.** And the counterfactual prices the two axes apart rather than asserting
which mattered: the lever's norm ratio at the shipped direction drops rel by 0.3303 (36.3 % of
the move), its direction at the shipped norm ratio drops it by 0.6153 (**67.7 %**). **It did
not only ever close the magnitude, and no rescaling substitutes for the direction it buys.**

**The A/A floor is exactly 0** — SHIP_A and SHIP_B bit-identical on 2,736/2,736 tensors — which
is the strongest floor this campaign has had under a direction claim, and it does a second job
for free: SHIP_A ran with `host_quiet.py` green and SHIP_B red, and they are **bit-identical**.
That is a direct measurement that load does not move an accuracy reading, which until now this
campaign had only argued (R182, and `a-firing-question-is-load-insensitive`). Arms interleaved
shipped/shipped/verb/module-wide in one session, AICLK sampled DURING at median 1350 MHz, A42
satisfied with one correction cited in every arm, and no timing quoted from a shared box.

**The headline moves, and downward.** What the campaign quoted as **1.0525x** its in-frame bar
reads **1.2224x** on the repaired functional. The best lever is worse than believed and is
still the best lever.

**What this does NOT establish, and the row is the one who said so first.** The clause's
requirement — cos 0.6976277 -> 0.9002917, 43.61 % of a 45.763° angle — is **a different arm on
a different boundary**, and `of3t-angle` explicitly declined to project onto it. 50.003 %
closed there and 43.61 % needed here are two numbers about two frames; the temptation to read
"more than enough" off them is exactly the cross-frame quotient A37 and D218 bar, and it is
more tempting than usual because it points somewhere good. **Dispatched `of3t-modelever` to
MEASURE it** rather than infer it: the package install on the model-frame trunk arm, re-scored
against `CLAUSE.json`'s pre-registered levels, no bar moved.

### R189. I dispatched a row whose name was already concluded; the fleet silently did not launch it, and the reason I dispatched was that the PROSE was stale while the STATUS was right (pass 414, zero card)

I wrote a brief and a TASKS entry for `of3t-tapeamp` onto D30/D58/D129. **A row of that name
had already run and concluded GO on 2026-09-22**, with *"the question is answered and the answer
is that there is no defect here to repair."* Because a concluded marker exists, the fleet never
relaunched it — **so no card was spent, and no warning was raised either.** A brief plus a
`<!--ws:-->` tag for a concluded name is a **silent no-op**: it looks dispatched in every place a
dispatch is recorded, and nothing ever runs. That is worse than a visible failure, and it is the
mirror of `brief-with-no-tasks-ws-tag-never-queues-and-logs-nothing`.

**Why I dispatched, and this is the part worth keeping.** I read D30's ledger entry, whose
heading still opens *"UNFIXED, and it is the campaign's central number"* and still carries
**19.6x**; and the USER-FACING closure plan, which named **`of3t-ditcot`** as D58's owner — a row
that did not answer it — while quoting the same stale pair. Both told me the question was open
and unowned. **Then I checked instead of assuming, and `statuses_by_defect` already returned
CLOSED for D30 and D129 before I touched anything.** The machine-readable half had been right all
along. **Published status right, prose beside it stale — R148's exact shape, and I read the
prose**, because the prose is what a reader reads.

**What the already-concluded row had found**, now finally recorded in the ledger it was about:
the ~20x is two revisions stale at **11.026x**, of which **7.666x is present in upstream 0.4.3's
own bf16 recipe** at the same boundary and reference; upstream's own fp32 recipe shows **9.326x**
at four orders of magnitude lower absolute error, so the factor survives a precision change no
dtype boundary could explain; **0 dtype reconciliations in 1,879 node firings**; and **our arm
beats upstream on both halves** — forward 1.959x, gradient 1.362x — so our factor is the larger
one only because the denominator is the half we beat hardest. D30 closed as not a defect, D129
dissolved by `of3t-ditref`, **D58 narrowed and NOT closed**: the diffusion leg is re-explained,
`msa_module`'s was never measured and no upstream bf16 arm for that boundary exists anywhere.

**One repair, and one guard I built and then threw away.** `of3t-msaamp` is dispatched onto
exactly the unmeasured leg, carrying the prior row's method instead of the stale framing.

The guard did not survive contact. I wrote
`assert_no_brief_for_concluded_row.py` to refuse the silent no-op and tried three signals for
it. **"Brief + live ws-tag + concluded marker" fired on 34 rows** — leaving the TASKS tag in
place after a row concludes is the fleet's normal residue, not an anomaly. **Brief mtime newer
than the marker fired on 103** — git checkouts reset mtime, so it carries nothing. **Git
CREATION date of the brief newer than the marker still fired on dozens**, because the concluded
markers are themselves rewritten by sync and their mtimes are not when the row concluded. **The
disk does not carry a trustworthy "when did this row conclude", so the check cannot be built
from it**, and I deleted the script rather than ship one that fires forty times. A guard that
must be suppressed everywhere teaches everyone to ignore it — this campaign already owns
`a-gate-arm-can-be-permanently-red-on-main-and-then-gates-nothing`, and adding a second one to
feel covered would be the flattering move.

**So the repair is a habit with a cost of one command, not a ratchet**: before writing a brief,
`ls /home/moritz/.coworker/state/concluded/ | grep <row-name>`. I am recording the failed
designs because the next person to want this guard should not re-derive all three.

**And I checked myself on the flattering direction.** Closing defects shrinks counts.
D30 and D129 are campaign-internal so closing them moves no user-facing number, and **D58 — the
user-facing one — stays open**, because the row that narrowed it said one leg was unmeasured
rather than rounding to a close. `reclassifying-out-of-user-facing-is-the-flattering-direction`.

### R190. `of3t-modelever`'s baseline is BIT-IDENTICAL to the clause's own artifact, so its reading will be carryable where `of3t-angle`'s was not (pass 414, zero card)

Two controls landed before the lever arm, and together they remove the ambiguity that has cost
this campaign more than any arithmetic error.

    AA_SHIPA_vs_SHIPB    2736/2736 bit-identical, rel 0.0, cos 1.0   -- the in-session floor
    AA_SHIPA_vs_BANKED   2736/2736 bit-identical, rel 0.0, cos 1.0   -- vs of3t-recut's
                                                                        dev_RENORM_model_n384_external.pt

**The model-frame A/A floor is exactly 0**, the second frame in two days to achieve that, so any
separation the lever produces is real rather than noise. **And the row's shipped arm is the
clause's arm, bit for bit** — `dev_RENORM_model_n384_external.pt` is the artifact the repointed
GRADIENTS clause was scored on at 1.4511706984958472x.

**Why that matters more than it looks.** `of3t-angle` measured a genuine 50.003 % angle closure
and could not carry it onto the clause, correctly, because its baseline was a different arm on a
different boundary and A37/D218 bar the quotient. **`of3t-modelever` will not have that problem**,
and not by argument — by a bit-exactness check against the reference's own artifact. A reading
against a baseline that is bit-identical to the clause's baseline is a reading about the clause.

**The generalisable instruction this yields for any row meant to move a published number**:
bank an A/A against the published artifact ITSELF before taking the arm, and publish the digest
comparison. It converts "is this comparable?" from an argument at the end into a fact at the
start, and it is CPU-only. `a-go-clause-must-be-tested-against-the-references-own-artifact` is
the entry; this is what satisfying it looks like in practice, and it cost the row one
compare-two-banked-artifacts run with no device involved.

Not yet a result: the lever arm is still running and nothing about the clause has moved.

### R191. The lever works on the clause's own arm and is NOT enough: 1.4511706984958472x -> 1.3037867474869442x, a third of the excess (pass 414, zero card)

`of3t-modelever` put the exact softmax on the model-frame trunk arm and re-scored. **The clause
still FAILS.**

    clause        1.4511706984958472x  ->  1.3037867474869442x     bar 0.15210099830945006
    excess        0.451171             ->  0.303787                32.67 % of it closed
    angle closed  14.19 % in the graded space, against the 43.61 % the clause needs
    and against float64 the trunk gets WORSE

**A third.** The campaign's best lever, applied exactly where the clause lives, buys about a
third of what the clause needs, and the clause must still fall a further **23.30 %**.

**The provenance is the strongest this campaign has produced and it is why the number is
believable.** The A/A floor is **exactly 0** — SHIP_A and SHIP_B bit-identical on 2,736/2,736 —
and SHIP_A is bit-identical **to `of3t-recut`'s banked `dev_RENORM_model_n384_external.pt`**,
the artifact the repointed clause was scored on, digests cited both ways. The banked arm
composed through this row's own scorer reproduces the clause at **0.22072451195864032 exactly**,
and the recomposition identity has `rel_difference 0.0`. **The EXACT arm differs from SHIP_A by
the `exact_softmax()` scope alone**, so this is a lever on the clause's arm rather than a
projection onto it — which is what `of3t-angle`'s reading could never be. **No bar moved**: all
five pre-registered levels are unchanged, including `upstreams_own_floor_here` at 0.8525301041731214
and `section_A26_level` at 0.9700522560698159, both of which would PASS.

**And the barred shortcut turns out to have been accurate, which is worth knowing precisely
because we did not use it.** The row pre-registered the in-frame projection as a LEVEL with an
aliveness band — *"within 15 % ... outside that the projection is dead and the campaign must
stop carrying it"* — and measured the multiple at **2.180720496587762** against a projection of
**2.2340903768268046**: **2.4 % off, alive.** A37 still bars a cross-frame quotient as
*evidence*; what this licenses is the cheaper thing — **an in-frame multiple on this frame
family is a sound SCREEN**, good to a few percent, so a future lever can be triaged in-frame
before anyone spends a model-frame arm on it. The discipline was to measure rather than project,
and the measurement is what turned a barred number into a calibrated tool.

**Where this leaves the charter.** Not unreachable: `upstreams_own_floor_here` passes at 0.8525x
and the trunk's norm ratio vs float64 is now **1.0644759387772336** against upstream's own
**1.0568409490651478** on the same scope — we are close to upstream on magnitude and the residue
is still direction. But the best lever the campaign has is now spent on this arm, it delivered a
third, and **nothing else of comparable size is currently identified.** That is the honest state.

**One process note: the D155 warning worked, once, when it came with the fix.** D249 records
three rows warned while live that concluded without acting. `of3t-modelever` was warned at pass
414 **and given the writer-level repair and a row to copy**, and it stamped host, board and card
from the provenance in `pairdiff.py` rather than joining the freeze list. The difference was not
the warning.

**Addendum to R191, same pass — the guard would have failed the one row that complied.**
`of3t-modelever` did stamp its artifacts (verified: `host tt-quietbox2`, `board p300c`,
`card 1`, plus `host_quiet` state), and the D155 guard went on warning about all three. The
guard's `host_of()` read the **top level only**, and the row had put host/board/card in a
**per-compared-artifact provenance block** — which is the *right* place for them, because one
file compares two arms that could come from different boxes.

So the row that finally acted on a warning three rows had ignored was the row the guard was
about to fail, and the freeze it would have forced would then have been cited as **a fourth
instance of D249** — a defect record manufactured out of a checker's blind spot. **A guard that
punishes the one row that listened is worse than no guard**, and the failure mode is specific:
the guard tested *where the field is* when the property it cares about is *whether the host is
recoverable*.

Presence is recursive now, and the exclusion is separately widened to every host/card pair at
any depth — **strictly stronger** than the top-level scan it replaces, since a nested `pc card
0` could previously have hidden from it. And the first attempt at that widening was wrong in a
way worth keeping: I regexed `json.dumps(d)` and **the break control caught it immediately** —
JSON inserts `": ` between `card` and its value, which breaks the adjacency `BANNED` matches on.
The probe that exists so a guard cannot quietly stop guarding is what said so.

### R192. A pre-registered ceiling that promised a CLEARING arm is refuted by measurement, and it was still a live constant in the clause scorer (pass 414, zero card)

`perf/of3t_modelframe/clause.py:37` carried `LEVER_CEILING = 1.8563207917912123`, *"the
pre-registered trunk lever ceiling divisor"*, and `CLAUSE.json` published
`projected_trunk_reading_with_the_lever_ceiling: 0.3584521380145671` from it. That projection
implied the lever would bring the clause to a **clearing 0.9668x**.

**Measured, on the same frame, on an arm bit-identical to the clause's own banked artifact: the
lever divides the trunk by 1.1516970980351204** — trunk 0.7768254196709333 -> 0.6745049727018105
— which is **62.04 %** of the projected divisor, and the clause reads **1.3037867474869442x** and
does not clear. `of3t-modelever` said it plainly: *"that projection is refuted by a measured arm
on the same frame, so the campaign should stop carrying it."*

**Kept, not deleted.** The constant stays in the file with the refutation beside it, and
`clause.py` now also emits a machine-readable `lever_ceiling_REFUTED` block carrying the
projected divisor, the measured divisor, the delivered fraction and both clause values, plus a
`projected_trunk_reading_with_the_MEASURED_divisor` (0.5777579519768513) next to the old line.
Deleting it would have been the tidier move and the wrong one: a removed number reads as
staleness rather than as a retraction (`a-deletion-reported-as-staleness`), and the
pre-registration is the thing that made the refutation meaningful in the first place.

**The pattern to notice is where it was hiding.** This was not prose in a state doc — it was a
**live constant in the scorer that computes the charter's own levels**, feeding a published
field. The five pre-registered levels were all correct and all held; the refuted thing sat
beside them in the same artifact, carrying the same authority, and nothing distinguished them.
**A pre-registered projection and a pre-registered bar look identical once published**; only one
of them is supposed to survive contact with a measurement. Levels are commitments to be held;
projections are predictions to be scored and then marked.

### R193. I dispatched three rows against a DONE_CHECK that did not know they existed, and the critical path spent five iterations discovering it (pass 414, zero card)

`of3t-modelever` finished the measurement the whole campaign was waiting on — the lever on the
clause's own arm — and then could not conclude. Its log: *"the check on qb2 still fails for one
reason: `_of3t_donecheck.py` has no entry for `of3t-modelever`"*, and it deferred itself to
04:35 **waiting for me**. `of3t-msaamp` and `of3t-cropwall` were walking into the same wall.

**The brief's `DONE_CHECK:` line names a script; nothing checks that the script knows the row.**
So a dispatch can be complete in every visible respect — brief, `#DISPATCH`, TASKS tag, a row
that launches and works — and still be unconcludable. The row discovers it only at the moment it
tries to finish, which is the most expensive moment available, and the failure reads to the row
as its own work being rejected. `of3t-modelever` spent five iterations on it and was right every
time.

**Fixed for all three**, with entries keyed to what each brief actually asked for rather than to
generic fields — and their state docs added to `_STAGE_HINTS`, because the script warns that a
path absent from that list *"is never copied and the ssh'd check fails with Errno 2 no matter
how correct that row's work is"*. `of3t-modelever` passes immediately on the work it had already
banked; the two live rows have been told their exact field names rather than left to guess at
regexes.

**The rule this yields, and it belongs beside the concluded-name check from R189.** Dispatching
a row has two halves and only one of them is visible: writing the brief, and teaching the gate
the row exists. **Before dispatch: `grep <row-name> workstreams/_of3t_donecheck.py` and
`ls state/concluded/ | grep <row-name>`.** Two commands. This pass I skipped the first three
times and the second once, and both cost a row's time rather than mine — which is exactly why
they are easy to skip.

**A guard here is worth more than the one I abandoned at R189.** That one failed because the
disk carries no trustworthy conclusion timestamp; this one needs no timestamp at all — every
`of3t-*.txt` brief carrying a `DONE_CHECK:` line that names `_of3t_donecheck.py` must have a
matching key in that script's `EXTRA`. It is a set comparison on two files, and the script
ALREADY asserts the mirror of it (`EXTRA` slugs must appear in `_STAGE_HINTS`). The missing
direction is the one that bit.

**Addendum to R193, same pass — the guard is built and it fires.** `_assert_every_dispatched_brief_is_gated()`
lives beside `_assert_stage_hints()` in the check itself and asserts the mirror direction: every
`of3t-*.txt` carrying a `#DISPATCH:` line and a `DONE_CHECK:` naming this script, and not yet
concluded, must have a key in `EXTRA`. Break control run rather than asserted: **silent against
the tree as it stands, and naming `of3t-msaamp` the moment that row's entry is removed.** It is
a warning, not a failure, for the same reason the hint check is — one row's missing entry must
not refuse a different row.

**Two guards were proposed this pass and only one was built**, which is the useful comparison.
R189's wanted to know *when a row concluded* and the disk has no trustworthy answer, so three
designs each fired on dozens of healthy rows and it was deleted. This one asks *does the gate
know this row exists*, which is a set comparison between two files with no time in it at all.
**The buildable check was the one whose question had an exact answer on disk**; the abandoned
one kept trying to infer a fact nothing records.

### R194. The lever's stated mechanism is refuted by the campaign's own measurement: upstream 0.4.3 computes its softmax in bf16, so "we move toward upstream's recipe" cannot be why it helps (pass 414, zero card)

`of3t-modelever` explained its float64-space result this way: *"The lever moves our trunk toward
upstream's bf16 step, which computes its softmax in fp32 under autocast, and away from the
float64 truth."* It labelled that sentence honestly — *"That is inference; this row measured the
direction, not the mechanism"* — and the inference is **wrong on a fact this campaign already
measured**.

`of3t-fp32islands` enumerated the precision islands against both upstream trees and the autocast
policy header, and its durable finding is explicit: **"OpenFold3 0.4.3 runs LayerNorm and
attention softmax in bf16 by explicitly disabling autocast; 0.5.0 restores both to fp32."** And
the graded reference is 0.4.3: `of3t-refprec` built all four arms, `arm4_bf16_autocast` among
them, *"all upstream OpenFold3 0.4.3"*, against the `grads_f64_043.pt` float64 reference.

**So on the very op the lever changes, upstream's graded step is bf16 and our lever is float64 —
maximally far from upstream — and it moves us CLOSER to upstream in the graded space anyway.**
Recipe-matching cannot be the mechanism. Whatever the lever is doing, it is not converging on
upstream's arithmetic on that op.

**This mattered before it was interesting.** I was about to dispatch a row on exactly that
premise — enumerate where our trunk's dtype recipe differs from upstream's bf16 recipe and match
the next one — which would have spent a card chasing a mechanism the campaign had already
refuted, and the enumeration itself is *also* already done and concluded (`of3t-fp32islands`,
PARTIAL: every single-op forward island is rounded back to bf16 by upstream at both versions and
we are already at the floor there with `precise_config()`). Two R189s in one pass, avoided by
the two commands R189 says to run.

**What is actually open.** The lever closes 14.19 % of the graded angle and opens the float64
angle by 10.12 %, and no explanation the campaign currently holds accounts for both signs.
`of3t-modelever`'s alternative — that making one component exact while the rest of the trunk
stays bf16 loses an error cancellation the shipped trunk had — survives this refutation and is
untested. **It is also the more consequential hypothesis**, because if the gain is cancellation
rather than accuracy then it is fragile: it would not compose with a second lever, and stacking
levers is precisely the campaign's remaining plan for the other two thirds.

**And it puts a boundary-version stamp back on the critical path.** `of3t-fp32islands` handed
the orchestrator a standing instruction I had not enforced: *"every of3t gradient figure needs
its boundary version stated beside it"*, because a port compared against the wrong version's
boundary *"shows a 30,000x gradient gap with no defect present"*. Neither `of3t-modelever`'s
state doc nor its `CLAUSE_EXACT.json` names the version anywhere.

### R195. A new defect's default triage class is CAMPAIGN-INTERNAL, which is the flattering direction (pass 415, zero card)

`stamp_row_counts.py` reconciles `UNFIXED_TRIAGE.json` from the defect union and gives any defect it has not seen the class CAMPAIGN-INTERNAL with the reason "no user-facing claim made". D250 is an inference forward gap in shipped OpenFold3 and went in as internal until I read the reason. The default is defensible (the stamper cannot read prose) but it errs toward the smaller user-facing count, so every new defect's class is set by hand in the pass that files it, not left to the stamper. Corrected: USER-FACING 7 -> 8.

### R196. The clause clears on a lever arm, and a clearing reading is not yet a shipped one (pass 417, zero card)

`of3t-stackexact`: exact softmax + exact LayerNorm on the clause's own arm reads 0.9822570327981535x the bar. I recomputed it with `perf/of3t_modelframe/clause.py` from `MODEL_SL_composed3660_n384.json`: same digits, `clears: True`. The ladder composes super-additively (f(S) 0.3267, f(L) 0.3708, f(SL) 1.0393, I +0.3418) and the float64 contrast space improves with it, which a rounding-matching gain would not do. Held off GO for three reasons named in VERDICT: perf-script lever, one 56-real-token boundary and one seed, 1.77 % margin. Dispatched `of3t-stackship` and `of3t-stackbound`. The compose also caught `stackarm.py` inserting paths at a fixed index in a loop (D149's mechanism); no module name collides across those directories so the arms stand, and the loop was rewritten in search order on `wk/of3t-stackexact` (`a005dda39`) before stackship branched.

### R197. A clearing arm made default opens a scope the clause never covered (pass 418, zero card)

`of3t-stackship` reproduced the clause arm through the shipped tape 2736/2736 bit-identically, so the clause is now a property of the training default. The same row's full-step A/B moves |g|^2 9.786 -> 15.317 and the confidence head 2.95 -> 4.22; the clause grades the model-frame trunk only, so it says nothing about which full-step gradient is right. Dispatched `of3t-fullstep64` with an FD-validated float64 reference and a STOP if OFF is nearer. `of3t-stackbound` relocated qb1 -> qb2 card 3: a HOST-GUARD block on a dark box is not a reason to hold an accuracy question whose comparison arm was measured on qb2, and the float64 reference re-built on pc must reproduce f7659dcd, a free cross-host check.

### R198. A lever's clearing gain can belong to its boundary's pad population (pass 420, zero card)

`of3t-stackbound` re-ran the 5nw3 ladder on 4hhb, 384 of 384 tokens real, bar 0.2224 from that boundary alone. SHIP already clears (0.9029x); SL adds 0.0151 (under T 0.02), L alone 0.0052, S alone 0.0362, and the S/L interaction is -0.0264 against 5nw3's +0.34. On 5nw3 (56 real of 384) the LayerNorm gain and the super-additive interaction came from pad rows. SL stays the default because it is the only stack measured to clear on both boundaries; SLZ (fp32 pair residual) is best on 4hhb (+0.070 over SL, trunk float64 error 0.2634 -> 0.0679) and is held until the full-step MSA defect is fixed, since that defect feeds the pair track. Side finding: qb1 and qb2 float64 CPU references differ at 1.9e-14 while bf16 autocast was bit-identical across them; digest both kinds per host.

### R199. A defect switched off and not numbered is invisible to every gate (pass 422, zero card)

`of3t-trainfwd` defaulted the one-step denoise arm off because 3 of 3400 gradients were non-finite and recorded it as "an unowned object". It got no D-number, so no triage, GAP line or DONE_CHECK ever saw it, and every full-step reading since (`of3t-fullstep64`, `of3t-inproj`) differentiated no part of the diffusion module. The GAP line said the diffusion reading "stands from earlier rows", which was true of a separate harness and false of the shipped step. Filed as D257, with D256 (84 lazily uploaded atom-transformer weights registered too late) behind it; `of3t-denoise` dispatched. The overflow was measured before D253's pad fix, and its signature (a dW over all padded pairs, MSA max|g| 1.7e+05) is D253's, so the row measures before it root-causes.

**R200** (pass 423) The `of3t-denoise` A40 floor was attributed to mse's Kabsch stop-gradient and FD3 froze it. FD3 at h=1e-4 reads 15.897746748793073 against FD2's 15.897749718841725 (1.9e-7 relative) and both read 2.847e-05 at h=1e-5: the freeze is inert, as the envelope theorem predicts when the alignment minimises the loss it feeds. FD2's ladder 2.498e-04 / 7.634e-05 / 2.847e-05 falls ~h^0.5 without a plateau, which a missing gradient path (constant bias) cannot produce. h=1e-6 then read 6.972e-10 and h=1e-7 4.303e-09: a V with one kink crossing between 1e-6 and 1e-5, so A40 PASSES. Separately, `of3t-denoise` found D259 (`ttnn.multiply(bf16, fp32 [...,1])` nondeterministic, up to 16,061 wrong elements) and D258 (mixed-dtype `transpose_a` matmul); both are fixed on the training path only, and `of3t-bcastaudit` measures whether inference reaches them.

- **R209** (pass 450) **THE CAMPAIGN'S BASELINE WAS STALE BY ~65x AND EVERY SHARE BUILT ON IT IS
  VOID, INCLUDING TWO OF MY OWN LEDGER ENTRIES.** `of3t-tapedfwd` (GO) found it. The 466.702 s
  step was banked at `451ed56f4` (2026-09-21 18:07:08Z, from the artifact's own `env.commit`);
  `502ed112e` (2026-09-23 03:59:45Z, *"training tape: exact softmax and layer norm on by
  default"*) then put a **HOST float64** softmax and layer norm inside every `tape()`.
  `git merge-base --is-ancestor 502ed112e 451ed56f4` is **false** and `502ed112e` is an ancestor
  of HEAD, so the feature did not exist when the number was taken and has existed ever since.
  Measured: a taped forward is **251.66 s** today against **3.415 s** banked, reproduced
  independently at 228.814 s, and **247.8 s of it is the exactness** — arms on identical routes
  at identical counts read 3.8884 s against 251.6566 s, teardown 0.0000 s, and a `--declare` arm
  ruled out leaf registration at 270.162 s. **It reaches the backward too**, because
  `autograd.backward` opens `with _training_exact("backward")` for itself once the tape block has
  closed. **Void until re-taken: 58-67x, 98.6/1.4, 466.702 and 456.668, 356.00 s / 168,922 calls
  / 2.107 ms / 44.3x, the 97.7 %, the 100.668 s residual, every JOBS share, my own R205 and R206,
  and BACKWARD.md §4c's 3.32x ceiling.** **Survives: R207** (of3t-tapedfwd's A/B is untaped on
  both arms, so neither installs the exact ops — 3.2381x on the forward, ~1.1 % of today's taped
  forward, and the conclusion holds on both trees), **R208** (a smaller share of a larger step is
  a smaller share), and **K22** (a board fact). **The lesson, and it is aimed at me: an artifact's
  `env.commit` is the only thing that dates a number, and a campaign that re-reads a banked
  baseline every pass never re-checks it.** I built two ledger entries, a ceiling, a retirement
  and a dispatch order on a constant nobody had re-dated in four days. **A banked number needs an
  expiry check, not just a provenance field** — cheapest form: assert the baseline artifact's
  `env.commit` is still an ancestor of HEAD before quoting it, which would have caught this the
  first time any row read it after 09-23.

- **R208** (pass 449) **J3 is RETIRED from the sprint, and holding it rather than dispatching it
  is what made that decision possible.** I held `of3t-triattbw` at pass 445 because its size was
  a 14x bracket (2 SDPA score blocks per call = 10.7 s, or 40 = 153.7 s) and briefing the largest
  authoring job on the page against a 14x uncertainty is a mistake this campaign had already paid
  for twice. `of3t-bwattrib` resolved it from source with the arithmetic shown -- at
  `SDPA_SCORE_BUDGET = 256 MiB`, `per = min(384, 268435456 // 1179648 = 227) = 227`, so
  `ceil(384/227) = ` **2 blocks** (4 if operands arrive fp32) -- and corroborated 1,179,648 B as a
  real allocation class independently: 247 live buffers of exactly that size in
  `perf/of3t_stepfloor/out/d164_probeoff_384.json`. **So J3 is the 10.7 s job: 2.29 % of the
  466.702 s step, 1.0235x, falling to 1.22 s and 1.0026x if J0 lands (R206).** Against J1 at
  46.4 s / 1.1104x and J2 at 32.8 s / 1.0756x, J1 is **4.34x** J3 and J2 is **3.07x** it -- while
  J3 is the largest authoring effort on the list (a new `AttentionMaskType::AdditiveBias` in both
  `sdpa_fw` and `sdpa_bw`, an lse output on the shipped forward, a strided KV work split, an L1
  bias accumulator) and its forward change is release-gated across four models because it serves
  **560 of 560** triangle-attention calls on the Boltz-2 512 aa fold. **The most expensive and
  riskiest job on the page for the smallest win on the page.** It may not even apply here:
  `tenstorrent.py:10395` has OF3 passing `fp32_softmax=True` at all four triangle-attention
  sites, routing the forward to `_fp32_softmax_attention` rather than `fused_sdpa`, so the taped
  `_v_sdpa` may never fire in OF3's trunk. **Retired, not deleted** -- brief complete, gate entry
  in place, the hard thinking banked -- with a falsifiable reopen: if the runtime `sdpa_shapes`
  count comes back near 40 rather than 2, it returns to the top of the list. That confirmation
  comes out of the FORWARD leg inside the first ~15 s of device time, so it is cheap and near.
  **The general form: when a job's size is a bracket wider than its rank, hold it and buy the
  measurement -- the cheap number that resolves the bracket is worth more than the expensive work
  the bracket would have authorised.**

- **R207** (pass 446) **R203 is CORRECT and worth 1.00508x, and I should have priced it before
  reordering a sprint around it.** `of3t-tapedfwd` (GO) measured what R203 located: untaped with
  every lever 1.2054 s, forced onto the taped route set 3.9032 s -- **3.2381x on the forward**,
  medians of 3 at 0.21 %/0.24 % spread, pc card 0, 1350 MHz sampled DURING. Those routes sit in
  **3.415 s of a 466.702 s step**, so the tape gives up **2.3604 s**; the whole taped forward
  including its diffusion is 3.702 s, so a forward costing NOTHING would be **1.0080x**. The
  mechanism was even bigger than I described -- **zero** fused forwards run under a tape, thirteen
  call sites asking 1,528 times per cycle, of which only five are `generic_op` -- and the answer is
  still half a percent. **A big ratio on a small slice.** Worse for the original hope: the guards
  cannot touch the backward at all, because every `with ag.tape():` closes before `backward()` is
  called, so the backward always ran with the full lever set. **The lesson is mine, not the row's:
  a located mechanism is not a win until it is priced, and thirteen honest comments are no more
  evidence of size than one was.** R203's finding stands; its billing does not. BACKWARD.md §4
  rewritten to lead with the price.

- **R205** (pass 446) **`of3t-bwsurvey`'s headline "the step near 51 s, and that alone is 9.1x"
  omits a 100.668 s term, and the omission is REFUTED by this campaign's own hardware floor.**
  Its own partition (`of3t-gpugap` PARTITION, arm B rep 2) is step **466.702 s**, backward
  **456.668 s**, so non-backward is **10.034 s**. Of the backward, **356.00 s** is the 168,922
  ttnn verb calls at 2.107 ms, which leaves **100.668 s of NON-VERB backward time**. Repricing the
  verbs at 0.24 ms gives 40.5 s, so the step becomes 10.034 + 100.668 + 40.5 = **151.24 s =
  3.09x**, not 51 s and 9.1x: 51 s is 10.034 + 40.5 with the 100.668 s silently dropped. **The
  cross-check that settles it without re-measuring anything:** 50.58 s against H200's 7-8 s is
  **6.3-7.2x**, which is BELOW the ~8.5x compute / ~11x bandwidth silicon ratio — the 9.1x
  reading requires us to beat the hardware. 3.09x leaves 18.9-21.6x, comfortably above the floor.
  **A projection that breaches your own floor is arithmetic, not optimism — check a headline
  against the floor before it orders a sprint.** J0's rank does not change (3.09x still dwarfs
  every kernel job); what changes is what it may promise.

- **R206** (pass 446) **J0 and the kernel jobs are SUBSTITUTES, not complements, so the JOBS
  list's seconds must never be summed.** Every kernel saving on that list is
  `(verbs_removed) x (cost per verb)`, priced at today's **2.107 ms**. J1's ">=46.4 s" is
  (31,104 - 9,072) verbs x 2.107 ms; if J0 succeeds and a verb costs **0.24 ms**, the identical
  job saves **5.3 s**. Same for J2 and J4. So running J0 beside three kernel rows means three of
  them may be optimising work J0 is about to make nearly free, and adding 3.09x to
  J1+J2+J4's ~94 s double-counts the same seconds — that total lands suspiciously exactly on the
  hardware floor, which is the tell. **Rule for this sprint: every saving is quoted against a
  STATED per-verb cost, and the sprint total is a re-measurement, never a sum.** This is the
  sub-additivity rule (A46 clause 6) in its sharpest form: here the overlap is near-multiplicative
  rather than merely sub-additive. Not a reason to cancel the kernel rows — J0 may fail, and a
  kernel that removes CALLS helps at either per-verb cost — but a reason they must each say which
  per-verb cost their number assumes.

- **K22** (pass 448) **`card=-` sends a Blackhole campaign to a Wormhole galaxy, and the rows
  find out, not the dispatcher.** I dispatched four device rows for the backward sprint as
  `host=any card=-`. `card=-` resolves to whatever is free, what was free was whglx, and **whglx
  is a Wormhole galaxy** (`tt-smi -ls`: every chip `Wormhole / tt-galaxy`). This sprint's whole
  baseline is Blackhole -- the 466.70 s crop-384 step, the 115.685 TFLOP/s roof, the 1350 MHz
  clock rule -- so none of those cards can produce a comparable number. **The only reachable
  Blackhole card is pc card 0** (`Blackhole / p150a`): qb1 does not answer ssh with all four cards
  blocked, qb2 card 0 reads 800 MHz so a timing there is an artifact by the standing clock rule
  and has no reset path while card 1 holds a live arm, and qb2 1/2/3 are held by BCX rows with a
  28 September deadline. So three kernel rows serialised on one card and `of3t-lnbw` went
  **BLOCKED**, correctly, and named the cause better than my brief did: *"three of3t rows are
  queued on one card on a host whose brief assigned them cards 2, 3 and 0 when the host has
  exactly one. That is a dispatcher fact, not a row's."* **Rule: a campaign whose baseline is one
  board class pins its device rows to that class; `card=-` is only safe when any board will do,
  and the big idle Galaxy is exactly what `card=-` will hand you.** The mitigation that keeps the
  queue productive rather than merely ordered: **a formula-level check and a negative control are
  board-insensitive and can run anywhere; the graded VJP number and every timing number are
  Blackhole-only** -- so a blocked row finishes its instrument, its float64 reference and its FD
  validation while it waits, and says which half it has. Order set to J0 -> J1 -> J2 -> J4 by
  value, with the last two parked behind it so they stop burning passes on a card they cannot get.

- **K21** (pass 446) **a host relocation silently resets a row's worktree, and the row cannot tell
  a first pass from a restart.** `of3t-intensity` ran 23 minutes on qb1 card 3 (16:15:03-16:37:59);
  qb1 then went unreachable (`state/qb1-offline` set, ssh times out), the fleet relocated it to
  whglx card 6, and its worker log records *"fresh worktree ... from origin/main"*. Its pass-1
  commits are in `/home/ttuser/.coworker/wt/of3t-intensity` **on the offline host**, its pass-1
  state doc likewise, and **`origin/wk/of3t-intensity` does not exist** — checked against origin,
  not inferred. The new worktree is clean at `f0db89ef5` and carries none of it. **The row's own
  RESUMING clause then reads exactly backwards for it: it will look for prior progress in a
  worktree guaranteed not to have any, find nothing, and be right for the wrong reason.** The
  relocation is logged, but the LOG is not something the agent reads. **Defense: push every pass,
  not every row** — the branch is the only artifact that survives a host going away, and on this
  fleet that is not rare. It is also the only thing that distinguishes a genuine first pass from a
  silently-reset one for a successor. Two of the three near-miss classes here were already known
  (`worker-push-claim-not-verified-against-origin`, `qb-daily-disconnect-is-silent-host-hang`);
  what is new is that the relocation RESETS rather than preserves, so the loss is invisible from
  both ends.

- **K20** (pass 446) **a dispatch that edits `workstreams/queue.tsv` by hand never happens, and the
  correct edit can be lost too.** `queue.tsv` is REGENERATED by `reconcile_tasks.sh` from the open
  `<!--ws:SLUG-->` tags in `TASKS.md`, and the script's own header says so: *"The orchestrator
  NEVER edits queue.tsv by hand (any manual edit is overwritten next run). To queue a task: write
  workstreams/SLUG.txt + tag the TASKS.md line."* Four rows appended to `queue.tsv` at pass 445
  were gone two minutes later, with **no error and no log line** -- briefs, gate entries and
  stage hints all correctly in place, and nothing to launch. **The second half is the part that is
  not documented and is a real race:** `reconcile_tasks.sh` does `mapfile -t all_lines < $T` and
  later `mv $tmp $T` with no lock, and `fleet.sh` runs it from cron every two minutes, so a
  correct `TASKS.md` edit landing inside that window is silently reverted as well -- which is what
  happened on the retry. **So: queue through `TASKS.md`, and VERIFY the tag survived rather than
  assuming the write did.** The patch is kept as an idempotent re-appliable script
  (`perf/of3t_orchestrator/bwd/apply_tasks_rows.py`) precisely so re-applying is one command
  instead of a re-derivation. General form: **when a file is owned by a periodic regenerator,
  writing to it is a request, not a change -- read it back.**

- **R204** (pass 445, backward sprint) **the blocker R203 found is not an infrastructure gap, and
  the facility that routes around it is already in production.** `tt_bio/autograd.py`'s tape is
  not a verb registry: `_tape(out_value, parents, make_fn, reads=None)` wraps any forward value
  with any hand-written VJP closure, so a `generic_op` result can carry a backward with no
  nanobind, no C++ build and no bridge to `ttml::autograd`. `autograd.triangle_attention`'s
  `value=` argument is the seam and its docstring states the intent -- *"what lets the shipped
  fused SDPA share this backward instead of getting a second copy of it"* -- so the fused forward
  and the authored backward already end up in one tape node at `taped_ttnn.py:862`. **It is a
  one-off: one op, one call site.** The lesson is the inverse of R203's and worth as much: R203
  read a policy off thirteen honest comments and made the problem look like a kernel-porting
  project; twenty minutes in the same file showed the mechanism to fix it shipped months ago and
  was never generalised. **A comment that explains WHY something cannot be done is evidence about
  that call site, not about the codebase -- grep for the thing already doing it before sizing the
  work to build it.** The job list is therefore "author or adapt the backward maths, then wire the
  fused forward through the `value=` seam", eight modules, in Python.

- **R203** (pass 445, backward sprint) **an engine-wide fused-kernel bypass hid behind a
  per-module comment.** `ttnn.generic_op` has no backward, so every one of tt-bio's eleven fused
  kernels declines under taping BY DESIGN: thirteen guard sites in eight modules (`triatt_qkv` 4,
  `reblock_permute` 3, `triatt_sdpa` 2, `softmax_generic` 2, `swiglu_fused`, `trimul_tail`,
  `mm_dualnoc`, `eltwise_fusion` 3), each with a correct comment saying so, and none of them wrong
  in isolation. The campaign's shared finding set carried this as *"one taped call switched fused
  triangle attention off for the whole process"* -- a cached state-dependent refusal, one op, a
  bug. It is neither cached nor one op nor a bug; it is the architecture, and a taped training
  step therefore runs a DECOMPOSED, DRAM-RESIDENT model (`tenstorrent.py:9016` takes DRAM where an
  untaped run takes L1). **The general lesson: a per-site comment that correctly explains one
  decline is the best possible camouflage for a policy, because every reader who checks one site
  leaves satisfied.** Count the sites before believing the anecdote -- the grep took ten seconds
  and moved the sprint's whole job list. The shape of the fix is in the same tree:
  `triatt_sdpa`'s comment says *"the stock fused SDPA verb is taped"*, so a fused kernel survives
  taping when it is a taped ttnn VERB with a registered backward and not when it is a tt-bio
  `generic_op`. Seconds owed by `of3t-tapedfwd`; a code read says the guard exists, only the run
  says what it costs.

- **R202** (pass 426) a resolver's routing premise ('the plain block is inference only') was a claim about a DEFAULT, and the default made it false: `z_fp32_residual` is off, so every taped step took main's block, whose `add_` discard is invisible untaped. Only a composed-tree training arm could see it, which is why composed64 existed. Evidence: perf/of3t_composed64/PROBE_TRANSITION_TAPE.json on wk/of3t-composed64. D264.
- **R201** (pass 425) `upstream_unplaced` with no fused twin in `device_unmatched` = a leaf the step does not have, i.e. trains as a constant; D263 (93 input-embedder atom-encoder tensors) found this way from SCORE_DN384R.json. 12 template tri-att q/k/v had a fused twin: scorer gap only. Evidence: perf/of3t_denoise/BIJECTION_DN.json on wk/of3t-denoise.
- **D268** (pass 442, open, `of3t-sigma`) A44 draw 3 (seed 3, sigma 102.9, identity checks pass) reads `mass_at_or_better_than_bf16` 0.8364 < 0.95; draws 0/1/2/4 hold. Trunk sections only (msa 0.570 vs bf16 0.264, template 0.692 vs 0.293), diffusion module's own sections all under bf16; trunk |g| ordinary, our abs error up and bf16's down. pae itself settled: five-draw mean 1.32x (6.07/1.07/1.10/0.93/1.73). Draw 5 still owed.
- **R203** (pass 442) a pre-registered every-draw clause caught what the headline term's mean hid: A44 was written for pae, the first draw it went looking at pae on found a different section failing a different field. Grade every field on every draw, not just the term that motivated the draws.
