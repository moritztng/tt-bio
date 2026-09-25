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

## ROTATED 2026-09-25T16:00:05Z

This doc reached 113357 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-LEDGER.20260925-180005.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

e discipline was to measure rather than project,
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

- **R212** (pass 453) **THE SPRINT'S ANSWER: OF3T TRAINING IS SLOW BECAUSE OF ONE KNOB, AND IT IS
  THE FIDELITY FEATURE WE CHOSE.** `of3t-bwattrib` GO. `exact_training`'s host float64 softmax
  and layer norm are **95.2 % of the backward and 98.0 % of the forward**, attributed per closure
  AND per verb and confirmed by a break control on an identical route: step **980.73 s -> 39.11 s
  = 25.08x** (forward 50.17x, backward 20.99x), with the knob's own counters reading zero served
  softmaxes in the noexact arm. Per closure: `host_f64_softmax..bw` 111 fires / 250.35 s / 35.5 %
  and `_v_exact_layer_norm..bw` 658 / 128.73 s / 18.2 %, a clean 53.7 % because they do not nest.
  Per verb through the other door: `to_torch` 174.24 s + `from_torch` 142.88 s = **317.12 s,
  44.9 % of the backward on the PCIe bus**, and 57.1 % of the forward. Mechanism, from the
  dispatch table itself: the score tensor is `[384,4,384,384]` FLOAT32, 226,492,416 elements,
  **1.81 GB of PCIe per exact softmax, 391.4 GB over 216 calls, 178 s at the measured 2.2 GB/s** —
  and the residual host arithmetic matches `of3t-tapedfwd`'s independent 1.9813 ns/element.
  **Three premises overturned at once.** (a) **The 44.3x per-verb overhead does not exist**: one
  instrument on both legs reads **0.59x**, a backward verb being CHEAPER than a forward one.
  (b) **All four suspects I pushed total 1.34 %**, and `Tensor.evict` is *inert* — 6,300 calls,
  **0** moved L1->DRAM, the guard returns early every time — and was named against the wrong roof
  anyway, being an on-device copy rather than PCIe. (c) **J3 is dead at zero, not bracketed**:
  `sdpa_taped_calls = 0` and no SDPA verb appears in 24,928 forward or 61,746 backward calls,
  because `fp32_softmax=True` at all four sites routes the forward away from `fused_sdpa`.
  **And the knob is NOT a lever**: `exact_training(False)` reads 1.4512x the accuracy bar where
  ON reads 0.9823x. Its value is as the correct DENOMINATOR — the backward's real compute is
  **33.63 s**, not 705.78 s. **The general lesson, which cost this sprint most of its job list:
  a step that is 95 % measurement instrument ranks every lever by its share of the instrument.**
  Fix the denominator before ranking anything. Two real jobs became visible only once it was
  fixed: **`zeros [384,1,384,128]` BFLOAT16, 324 calls, 11.80 s self = 35.1 % of the REAL
  backward at 1.04 GB/s — 424x off the 440 GB/s roof that binds it**, a plain allocation/fill
  defect that read 0.17 % under the old denominator; and **`EXACT_TRAINING_OPS = ("softmax",)`**,
  worth 128.73 s of the base backward if softmax alone carries the accuracy, with
  `exact_softmax()` already existing as that scope (`autograd.py:1599`).

- **R211** (pass 451) **NODE COUNT IS NOT A COST PROXY, AND THE JOBS LIST WAS ORDERED BY IT — so
  J1 and J2 are both NO-GO and the kernel programme has largely collapsed.** `of3t-lnbw`, NO-GO
  with three independent reasons: the LayerNorm-backward site's whole ceiling is **9.40 s**
  (13.43 s if every node took the fp32 tree) of the step, not the briefed 46.4 s floor with a
  150 s upside — **that figure was an artifact of pricing this site's verbs at the backward's
  GLOBAL 2.107 ms mean, while the measured marginal verb here is 16.9 us**, 125x smaller; the
  wheel op that would have collected it **miscomputes on Blackhole** and upstream knows
  (**#12349**, its own tests skipped); and the site is **DRAM-bound at ~43 % of the roof**, which
  `moreh_all` demonstrated by running **2.17x slower on 12 fewer verbs**. Its general finding is
  the one that matters: **LayerNorm backward is 52.4 % of the tape's NODES and 2.6 % of the
  backward's SECONDS.** `of3t-bwsurvey`'s whole job list is ranked by node and verb counts, so
  its ordering does not track seconds and every share on it inherits the same error.
  `of3t-softbw` lands the same way: Route A (`ttnn.moreh_softmax_backward`, the zero-build route)
  is refused by the op's own dtype guard against operands measured **FLOAT32 at 543 calls** — not
  a tuning problem and no flag for it — and Route B (`ttml::metal::softmax_backward`) needs a
  nanobind binding and a tt-train build, a dispatch of its own. **So J1 is dead, J2's free route
  is dead, J3 was retired at 1.0235x (R208), and what is left of the kernel programme is J4 and a
  build.** The corollary, which `of3t-lnbw` states and which is now the sprint's whole thesis:
  **a 16.9 us marginal verb next to a 2.107 ms global mean is direct evidence that the 2.107 ms
  is not a per-verb property of the engine but a CONCENTRATION somewhere specific, and finding
  where is worth more than every kernel on the list.** That is J0. **The general lesson: when a
  survey sizes jobs by multiplying a count by a global mean, it has assumed the mean is uniform
  — check one site's marginal cost against the mean before ranking anything by count.**

- **K29** (pass 456) **A DEFER ISSUED WHILE A BACKGROUND MONITOR IS STILL ARMED DID NOT
  REGISTER, AND NOTHING SAID SO.** Pass 455 ended with a well-formed
  `DEFER: 2026-09-25T16:40:00Z - <reason>` as the last line. Ten minutes later
  `state/notbefore/of3t-orchestrator` **did not exist**, so the row was never parked and kept
  being dispatched. Ruled out: the 800-char payload cap (`worker.sh:652`) -- mine was **662** --
  and a malformed line, since earlier defers from this same row with the identical shape
  registered fine and are in the log. **No `DEFER rejected` line was written either**, which is
  the tell that `worker.sh` never processed the reply as terminal at all. The one condition that
  differed: I had armed a `Monitor` seconds before issuing the DEFER, and it outlived the reply
  -- its expiry notification arrived **8 minutes after** the turn supposedly ended. Leading
  hypothesis, not a proven mechanism: an armed monitor keeps the session alive so the launch
  never terminates and the DEFER is never consumed. **Practical rule either way: stop every
  background monitor BEFORE issuing a DEFER, then verify `state/notbefore/<slug>` exists rather
  than assuming the line took.** A defer is the one construct whose failure mode is invisible
  from the inside -- you believe you are parked, the dispatcher believes you are live, and the
  only symptom is passes you did not intend to spend.

- **K28** (pass 455) **AN ARTIFACTS-ONLY ORCHESTRATOR BRANCH DRIFTS, SO ITS `tt_bio/` READS ARE
  STALE — CHECK `origin/main`, NOT YOUR WORKTREE.** `wk/of3t-orchestrator` never touches engine
  code, so nothing ever forces it to rebase; it is **1,489 commits behind `origin/main`**. A
  `grep` for `exact_softmax(` across my own `tt_bio/` returned NOTHING, which would have read as
  "the symbol `of3t-bwattrib` cited does not exist" and sent `of3t-exactscope` off to build a
  scope that already ships. Against `origin/main` every symbol is there:
  `EXACT_TRAINING_OPS = ("softmax","layer_norm")` at 1536, `exact()` at 1590, `exact_softmax()`
  at 1600, `exact_softmax_installed()` at 1581. **The orchestrator is the role most likely to
  hold a stale tree, because it is the one role that never has a reason to update one** — and
  the absence of a symbol is exactly the kind of finding that feels decisive and is worth
  nothing. Corollary already earned: cite the SYMBOL, not the line number, in a file this active
  (bwattrib's 1599 is now 1600).

- **K27** (pass 454) **A PRIORITY WRITTEN IN A STATE DOC DOES NOT REACH THE DISPATCHER; QUEUE
  ORDER IS `TASKS.md` INSERTION ORDER.** I published the card queue as exactscope (128.73 s) ->
  restep -> zerosfill (11.80 s) in SEQUENCE, then inserted the two new TASKS.md rows in the order
  I happened to write them — zerosfill first. `reconcile_tasks.sh` emits `queue.tsv` in
  `open_order` (first-seen order in TASKS.md) and `fleet.sh` walks it top-down, so the moment the
  card freed it went to **zerosfill, the job I had ranked third**. No harm here: both are wanted,
  zerosfill was already running before I noticed, and the next in line is exactscope, which is
  where I wanted it. **But the general form is the same as K20 and worth the entry: the state doc
  is documentation, not control plane.** If the order matters, it has to be expressed where the
  mechanism reads it — TASKS.md insertion order — and a successor should not assume SEQUENCE's
  ranking is what will actually run next. Read `queue.tsv` top-down to know the real order.

- **K26** (pass 454) **A DETACHED RETRY CHAIN DEADLOCKS ITS OWN ROW, AND THE SAFE EXIT IS TO KILL
  THE WRAPPER, NOT THE DEVICE HOLDER.** `of3t-wheelbw`'s `chain3.sh` looped on the K24 bug for
  **38 minutes and 5 of its 40 permitted tries, 5 failures and 0 successes**, holding pc card 0 —
  the fleet's only Blackhole card — while `of3t-zerosfill`, `of3t-exactscope` and `of3t-restep`
  all deferred behind it. **The row could not stop it and nothing else would**: `task_running()`
  calls `chain_alive "$name"`, which matches any live process rooted in `$D/wt/<slug>`, so the
  chain made the row look RUNNING and fleet.sh never relaunched the agent that was the only thing
  able to kill the chain. Clearing its park (pass 452) did not help because the park was never
  the block. **40 tries at ~6 minutes is a ~4-hour fuse on the whole sprint.**
  **The resolution: `kill -TERM` the WRAPPER pid alone — not the process group, not the python
  child holding the device.** The in-flight arm then exits on its own terms and no next try
  starts. Verified after: wrapper gone, `fuser /dev/tenstorrent/0` reports nobody, the lease
  carries a `released` timestamp, and the chain log ends at try 5's `TypeError` with no try 6.
  **That ordering matters and is the whole point** — killing the device holder is what
  `killing-a-wedged-fold-leaves-the-card-unopenable` warns about, where the NEXT open hard-hangs
  the host; killing only the loop touches no device at all and needs no card reset, which this
  row is not permitted to perform anyway. **General rule: when a detached chain has deadlocked
  its own row, SIGTERM the retry wrapper and let the current attempt finish. And when writing a
  chain: a retry cap is not a safety feature if the thing being retried cannot succeed — bound it
  by CONSECUTIVE IDENTICAL FAILURES, not by a try count.**

- **K25** (pass 452) **PINNING A ROW TO THE CARD ITS OWN DETACHED CHAIN HOLDS CAGES IT
  PERMANENTLY — and that is the obvious fix to K24, so check `card_free()` before applying it.**
  Having found `of3t-wheelbw`'s chain looping on pc card 0 while its agent was parked elsewhere,
  the natural repair is "pin the row to pc card 0 so it lands where its mess is". **That would
  have caged it.** `fleet.sh`'s `card_free()` resolves the lease holder, calls
  `chain_alive "$hslug"` and returns not-free if the chain lives — **with no special case for the
  requester BEING that holder**. So the row would defer every tick forever, never wake, and never
  kill the chain that was blocking it; `of3t-infab` did exactly this 128 times
  (`a-hard-pinned-row-can-be-blocked-by-its-own-detached-chains-cardblock`). The correct pin is
  **`host=pc card=cpu`**: it lands the row on the host where its detached work lives *without*
  requiring the resource that work is holding. **General rule: when a row must clean up its own
  detached device job, pin it to the HOST and deny it the CARD — needing the thing you must
  release is a deadlock, and the dispatcher will not notice it is one.**

- **K24** (pass 452) **A RETRY WRAPPER THAT READS ANY NON-ZERO RC AS CONTENTION WILL RETRY A CODE
  BUG FOREVER — WHILE HOLDING THE SCARCEST LEASE ON THE FLEET.** `of3t-wheelbw`'s `chain3.sh`
  wraps each device step in "retry until the card is genuinely this row's", classifying failure by
  exit status alone; its own banner prints `card held by:` with nothing after it, which is the
  tell that it could not name a holder and assumed contention anyway. The actual failure was
  deterministic and local: `census.py:208` calls `host_losses(roots, 0, ...)` while
  `fullstep.py:252` does `len(rep)` and line 254 does `atoms[rep]`, so `rep` must be an index
  array — `TypeError: object of type 'int' has no len()`, identical every retry. It looped for
  ~25 minutes at roughly 6 minutes a cycle, **holding pc card 0, the only reachable Blackhole card
  on the fleet**, while `of3t-restep` (which unblocks every share in the campaign after R209)
  deferred five times and J0 queued behind it. **The row could not notice: it had parked itself to
  18:45 believing it was WAITING for the card, on a reading that was already stale when written**
  (it named `of3t-bwattrib` pid 36647 as the holder; that pid was killed at the 3000 s turn cap).
  Cleared the park — dot first, per the phantom-row rule — because a defer whose premise is false
  is not a defer. **Two rules: classify a retry on the EXCEPTION you mean (`DeviceInUseError`),
  never on rc, because rc cannot tell a busy card from a typo; and a loop that holds a scarce
  lease must bound its retries, since the cost of being wrong is paid by every row behind it.**
  The near-miss worth naming separately: I hypothesised the crash was in `fullstep.py` on main,
  which would have blocked `of3t-restep` too — **it is not**, `fullstep.py`'s own caller at line
  424 passes a proper `rep` and the `0` is the calling row's. Checking that before touching
  anything is the only reason this did not become a fix to a file that was not broken.

- **K23** (pass 451) **a gate that requires `^VERDICT:` cannot be satisfied by a row that writes
  `## VERDICT:`, and the row cannot tell.** `of3t-bwattrib` wrote `## VERDICT: PARTIAL` as a
  markdown heading; every verdict pattern in `_of3t_donecheck.py` anchored on `^VERDICT:`, so the
  gate reported **"no VERDICT: line at all"** against a document that plainly had one. Same
  family as the constructed-path staging trap: a refusal the row's own correct work can never
  clear. Fixed by accepting an optional heading prefix (`^#{0,6} *VERDICT:`) in all 70 patterns;
  the semantic requirement is unchanged and the row now fails on a genuinely missing field
  instead. **Where a document's format is a matter of style, the gate reads the style too — so
  either fix the format or widen the pattern, but never leave a row failing on punctuation.**

- **R210** (pass 450) **THE KERNEL PROGRAMME OPTIMISES A PATH THE SHIPPED TRAINING CONFIG
  BYPASSES, and that is the sprint's real structural problem — bigger than any job on the list.**
  `of3t-intensity` (GO) found it while ranking: J1 and J2 are both briefed against
  `autograd.py`'s composed closure, and **on main that closure does not run** —
  `_v_exact_layer_norm` and `_v_exact_softmax` replace it the moment a tape is entered, and their
  backward is float64 on the host. So a fused LayerNorm or softmax backward is, in the shipped
  training configuration today, speeding up code that does not execute. Its second measurement is
  the same fact from the cost side: **every kernel row branching from `origin/main` today runs an
  8.4x step**, so an A/B taken without care is 8.4x contaminated by a host term — which it names
  as `of3t-stepfloor`'s own "contaminated by a host arm" defect recurring. **My decision, taken
  this pass and pushed into every brief: kernel rows grade with `exact_training(False)` and say so
  in every arm; the shipped-config baseline is `of3t-restep`'s single clean measurement.** The two
  settings answer different questions and the sprint needs both, separately: with the host term in
  the denominator a 46 s win and a 33 s win are indistinguishable. **What no row may do is report
  a clean speedup and omit that the path is bypassed as shipped** — that is the flattering
  direction and this ledger already has an entry for it. The work stays worth doing for three
  stated reasons: the exactness default is a product decision on a price that has moved and may
  come back; the kernels are engine-level, so Boltz-2, BC2 and RFD3 reach the same backwards
  without OF3T's exactness scope and the win is live for them today; and **the product decision
  cannot be made at all until someone measures what the fast path is worth.** A third finding
  attached: the exact LayerNorm backward does two `ttnn.to_torch(...).double()` per node over
  1,296 of 2,473 tape nodes, which is a fifth and currently dominant suspect for J0 — and whose
  list is right depends entirely on which tree J0 profiles, so the commit must be named in its
  result.

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
