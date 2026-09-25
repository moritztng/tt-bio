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
a warning, not a failure, for t

---

## ROTATED 2026-09-25T18:00:06Z

This doc reached 115825 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-LEDGER.20260925-200006.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

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

- **R213** (pass 454, `of3t-exactscope`) a knob that is 95.2 % of the backward cannot be narrowed, and the narrowing REGRESSES the truth while improving the graded clause. On `of3t-stackexact`'s bar (pre-registered 0425448d4, four minutes before its first rung ran): softmax-only 1.30379x, layer-norm-only 1.28386x, the pair 0.98226x, none-of-it 1.45117x -- only the pair clears. Softmax-only raises the model's squared gradient error against float64 to **1.1760x** of shipping nothing while its clause improves, because the clause is measured against upstream's own bf16 and part of the apparent gain is matching THEIR rounding rather than approaching the truth. **A single-reference accuracy clause can reward a change that moves you away from the ideal; carry a float64 contrast column beside it or you cannot tell the two apart.** The mechanism is legible in the worst tensor softmax-only leaves: `pairformer_stack.blocks.23.single_transition.layer_norm.weight` at rel_l2 23.59 -- leave the layer norm on the device op and the tensor that hurts most is the layer norm's own parameter. Arm S reproduces `of3t-modelever`'s banked arm 2736/2736 bit-identical, so it is a reproduction, not a fresh reading. Evidence: `perf/of3t_exactscope/SCOPE.json` on `wk/of3t-exactscope`.

- **K30** (pass 454, `of3t-zerosfill`) ttnn dispatch is asynchronous, so a blocking op is billed for the queue draining behind it, and a per-verb self-time table therefore ranks levers by WHERE THE QUEUE DRAINS rather than where the seconds are spent. `zeros [384,1,384,128]` was briefed at 11.80 s and 35.1 % of the real backward off such a table, and the fix is worth **1.04 s / 1.0518x**: `zeros` loses 12.018 s in the table while the other verbs grow 10.650 s, 89 % of it. The fix is real and on main at `f0e8f3b71`; the share was not. **Any lever ranked off a self-time profile owes a wall-clock A/B before its share is quoted.** This is the second time this sprint that a ranking instrument, rather than a measurement, sent work to the wrong place -- see R211, where the job list was ordered by tape-node count. Memory `a-blocking-op-is-charged-for-the-queue-drained-behind-it`.

- **K31** (pass 454, orchestrator) the sprint's compose branch `wk/of3t-bwd` read as 29 commits pending and was already fully landed. `git merge-tree --write-tree origin/wk/of3t-bwd origin/main` writes `c8b91db8d` = `origin/main^{tree}` exactly, so the merge changes nothing; the tip is not a commit ancestor because the content reached main through `wk/of3t-zerosfill`, which had merged `wk/of3t-tapedfwd` and `wk/bcx-tapedfwd` before fast-forwarding. **A compose branch's ahead-count measures its own commits, not what main lacks -- compare TREES.** Second occurrence in this campaign after `wk/of3t`, which was 745 behind / 0 ahead and equally finished. Memory `a-branchs-ahead-count-is-not-a-merge-signal`.

- **R214** (pass 455, orchestrator) the campaign's most valuable number does not exist: **nobody has measured a full training step with the exactness ON.** `of3t-gpugap`'s 58-67x is axis-consistent (full taped step 466.70 s, 9,888 nodes, trunk + diffusion + loss heads + optimizer, against a full Lightning step) but was taken at `451ed56f4`, **a tree where the exactness feature did not exist** -- so it describes a configuration whose gradients this campaign has since shown do not clear the accuracy bar. And `of3t-bwattrib`'s 25.08x cannot supply the missing figure: it is a valid A/B on a **trunk cycle** (`config.cycles = 1` of 4, 2,482 nodes, no diffusion module, no loss head, no optimizer), and the knob's share of a step depends on that step's op mix -- the diffusion module is 88.8 % of the compared gradient mass and is absent from both arms. **Whether the exactness costs a full step 25x, 5x or 1.5x decides whether OF3T training on Tenstorrent is a product or a demo, and it is unmeasured.** Owned by `of3t-restep`. Scope table: `state/of3t/BACKWARD.md` §0.

- **K32** (pass 455, orchestrator, caught in my own draft before it shipped) **you cannot convert a measurement from one scope to another by a proxy -- not by tape-node count, not by cycle count.** I scaled `of3t-bwattrib`'s 39.11 s trunk cycle to a full step by the node ratio (2,482 of 9,888 = 25.1 %), got ~156 s, and was one paragraph from publishing "~21x, so the software gap is nearly spent". The measured exactness-off full step is **466.70 s**, 3x my estimate, because the step's other 7,406 nodes are diffusion, losses and optimizer and cost far more per node than trunk nodes -- which is R211's lesson wearing a different hat, and I walked into it while writing the section that warns about it. **The free check that catches this class: a ratio that beats the hardware floor is a scope mismatch every time.** `39.11 / 7.5 = 5.2x` sits below the 8.5-11x silicon floor, which is impossible, and that alone was enough to know the axes were wrong before finding out why. Run it on every ratio.

- **R215** (pass 455, `of3t-exactscope`, NO-GO) the exact softmax **alone** costs **612.09 s of the step -- 25.25x the noexact step by itself** (fwd 195.76, bwd 441.57), against a noexact step of **25.24 s**, three arms on pc card 0 p150a as the box's sole device tenant, crop 384, 1 trunk cycle, taped, AICLK 1350 DURING. **So the layer norm is the SMALLER half of the exactness and the hoped-for narrowing would have bought little even had it cleared the bar** -- it fails at 1.3038x regardless. Two derived facts worth carrying: the softmax path is very nearly the whole of the exactness cost, so work is sized against it first; and `of3t-bwattrib` read **39.11 s** for the identical noexact arm under load against this row's quiet **25.24 s**, a **1.55x contention penalty** that bounds every contended figure taken on that box. Each arm asserts its scope from the mechanism on both legs (`exact_softmax_installed()` while the tape is open, plus the exact verbs' counters differenced per leg), which is what excludes the flattering failure mode of an arm that silently kept both ops and would have read as "softmax alone is nearly free".

- **K33** (pass 456, orchestrator, caught before the row spent a launch) **a brief that asks for a median of N is void if one rep exceeds the pass timeout, and the row cannot discover that without burning its whole budget.** I briefed `of3t-restep` for "median of at least 3" on a full exactness-ON step. `worker.sh` wraps every pass in `timeout 3000` -- **50 minutes** -- and the lower bound from banked numbers on the same board is that the trunk **alone** at the shipped four cycles costs of order **3,500 s** before the diffusion module, the loss heads or the optimizer: `of3t-exactscope`'s base arm read forward **268.27 s** for ONE trunk cycle on a quiet box and its backward never landed, `of3t-bwattrib`'s read **980.73 s** for the same scope under load. Ten passes would each have started a run that could not finish, and the row would have reported ten timeouts rather than one number. **Before briefing a measurement, divide the expected run by the pass timeout.** If it does not fit: one rep not N, started DETACHED from the row's own worktree on its first action so it overlaps the passes that follow, with the row polling the ARTIFACT rather than the clock (a killed run and a running one are identical on elapsed time), and cheap in-pass results sequenced ahead of it so no pass banks nothing. **And say explicitly that not fitting is itself a publishable result** -- "this configuration costs more than N hours on one card" answers the trainability question -- because the alternative a row reaches for under time pressure is quietly shrinking the scope, which is exactly what produced the numbers the row was created to replace.

- **R216** (pass 457, orchestrator, found by checking the run instead of the clock) **the campaign's headline measurement was OOM-killed 7 minutes in, and two rows had already written down the cause without connecting it to anything that would break.** `of3t-restep`'s detached full step died at **19:11:36 CEST on pc, `anon-rss:19917952kB` (19.0 GiB) of a 30 GB box with no swap, `total-vm:57083188kB`, `global_oom`, pid 1795534**, launched 19:04:34. Its log ends on `[losses] root 3 af3_loss done` — the taped forward, all four diffusion samples and all four loss roots completed and it died **entering the backward**, walking a tape that retains every activation in the step (`a-seam-bracketed-dram-read-understates-the-in-step-peak`: a retaining tape dies in the FIRST backward, and a peak read anywhere else understates it). The lease file still named the dead pid, so the row read as live. **The cheapest candidate is bit-identical and free, and its evidence was banked two rows ago**: `host_f64_softmax_values` on main is `torch.softmax(ttnn.to_torch(v).double(), dim=dim)`, monolithic on a `[384,4,384,384]` block — 226,492,416 elements, 0.844 GiB fp32, **1.688 GiB float64**, and `torch.softmax` allocates a second one beside the `.double()` copy — **measured this pass at exactly 2.00x the float64 tensor** (`perf/of3t_orchestrator/bwd/PEAKPROBE.json`, 865.6 MiB of RSS growth over a 432.0 MiB float64 tensor at a quarter of the production rows; the ratio is dimensionless), so one in-flight exact softmax is **3.377 GiB at full shape, 17.8 % of the 19.0 GiB peak** — a real term and not the whole one, which is why the attribution is owed before the fix is sized. `perf/of3t_tapedfwd/exactprice.py` chunks *because* "pc has 30 GB and one such tensor is 1.81 GB in float64"; `of3t-xcost`'s `HOSTPRICE.json` then swept the chunk size over identical total work, found the rate **flat** (rows=8 0.808 s against rows=384 0.871 s forward), proved **every chunked arm `torch.equal` to the monolithic reference**, and filed it as *"a useful fact even though the rate does not move."* At rows=8 the float64 temporary drops 1.69 GiB to ~36 MiB. **The general lesson, and it is the one worth keeping: a hazard noted in a docstring as the reason for a local workaround is a finding about the whole codebase, and nobody owns it.** Two rows independently wrote "this tensor is 1.81 GB in float64" as an aside while the shipped path did the unchunked thing, because each was answering a question about *rate* and the hazard was about *peak*. When an instrument works around something, grep whether production does too. Second instance this sprint of a comment that satisfied every reader who checked one site (R203). **Not yet attributed**: the retained tape is an independent term and 19.0 GiB is a lot of softmax, so `of3t-restep` owes a phase-tagged RSS profile (the run is only 7 minutes, which makes it cheap) before `of3t-xsplit` sizes the fix it now owns. Evidence: `dmesg -T` on pc, `/tmp/of3t/of3t-restep/run1.log`, `perf/of3t_xcost/out/HOSTPRICE.json` on `origin/main`.

- **R217** (pass 458, orchestrator, source + banked artifacts, no card) **the exact softmax RETAINS its float64 output on the host and the exact layer norm deliberately does not — and no banked instrument can tell a retained taped softmax from a recomputed raw one.** This corrects the fix I briefed one pass earlier. Three things verified by reading `origin/main`: (1) `host_f64_softmax`'s backward closure `bw(g)` closes over **`y64`, a host float64 tensor**, and `_tape` stores that closure in `out.node = _Node(make_fn(), ...)` with `reads=None`, which its own comment says "pins everything" — so the float64 output is resident from the forward until the backward consumes it, at **8 bytes/element**. (2) `_v_exact_layer_norm` makes the opposite choice on purpose and says so in its docstring — *"mean and rstd are re-derived in the backward, not held on the host"* — its `bw` re-reads `x` off the card with `ttnn.to_torch(x.value)`. 658 fires, **zero host retention**. The asymmetry between two ops written for one feature is unexplained and nobody noticed it. (3) `triangle_attention` takes the **raw** path: `_scores` calls `ttnn.softmax` on `q.value[b0:b1,:,i0:i1,:]` slices, not on taped `Tensor`s, so the exactness install routes it to `_exact_softmax_raw`, which discards `y64` — and it `ttnn.deallocate(p)` immediately and recomputes `p` in its own backward. That is why `of3t-xcost` found 108 backward crossings with `closure: None`. **Unresolved, and the artifact cannot resolve it:** `hist_384_base.json`'s `by_closure` shows `host_f64_softmax..bw` firing **111** times while `forward_by_shape_class` lists only **48** softmax calls at the small APB shape `[1,16,384,384]` (0.018 GiB in float64) against 108 at `[384,4,384,384]` (**1.688 GiB in float64**). Either ~63 taped fires sit on the large shape, at 1.688 GiB of retained host float64 each, or the shape-class label conflates raw and taped calls at the same verb and shape. **The counters that settle it already exist and nobody has read them out**: `HOST_F64_SOFTMAX_STATS["served_taped"]` / `["served_raw"]` and `EXACT_SOFTMAX_STATS["verb"]` / `["raw"]`. `stats["elements"]` accumulates over both, so a taped-only element accumulator is the one-line addition that turns this from an argument into a number. **Consequence, and it is why this outranks the chunking:** chunking `.double()` cuts the **transient** (2.00x, 3.377 GiB per in-flight call, R216) and does nothing about **retention**. If retention is the larger term, chunking will not make the step fit. The fix pattern is already in the same file one function away — recompute as `triangle_attention` does, and recompute is the ONLY route: `_tape`'s `reads` argument sets `Tensor.pinned`, which is read only by `_free_shared` and the eviction path, both acting on the **ttnn device buffer** — no value of `reads` frees a host object captured in a closure. (I offered `reads` as a second route when first briefing this and withdrew it in the same pass.) **`reads` is a real finding on a different axis**: adopted at **3 of 31 `_tape(` call sites**, with `DROP_DEAD_VALUES` already `True`, while its own comment names **OpenFold3's fp32-softmax tail** as the case it was built for and reports only two of the five tensors a row block leaves behind are ever read again. That is device DRAM, not the host RSS that died. R204 again: the mechanism shipped and was never generalised. Evidence: `tt_bio/autograd.py` on `origin/main`, `perf/of3t_bwattrib/out/hist_384_base.json`.

- **R218** (pass 459, orchestrator, source only, no card — **resolves R217's open question and changes the fix**) **OpenFold3's big triangle softmax is TAPED and therefore RETAINING, not raw. The retained host float64 set is the dominant term in the OOM and chunking is nearly irrelevant to it.** The chain, each link checked on `origin/main`: (1) `taped_ttnn._NEVER_SHIM = ("tt_bio.taped_ttnn", "tt_bio.autograd")` — **those two modules only**, so every other `tt_bio` module, `tenstorrent.py` included, gets the shimmed `ttnn` and its softmax calls route through `taped_ttnn._VERBS`. (2) Under `exact_training` (default ON) that verb is `_v_exact_softmax`, which does `x = _wrap(args[0])` and calls `host_f64_softmax` with a **taped `Tensor`**, so `xt` is not None and `_tape` creates a node holding the closure that closes over `y64`. (3) There is **exactly one `ttnn.softmax_in_place(` call site in all of `tt_bio/`** — `tenstorrent.py:4329` in `_fp32_softmax_attention`, the route the `site_softmax` docstring calls *"the whole OpenFold3 triangle path, AF2's, RF3's atom stacks"* — and `autograd.py` has **zero**. (4) `hist_384_base.json`'s forward records **108 calls of `softmax_in_place [384,4,384,384] FLOAT32`**, and at 226,492,416 elements that is **1.688 GiB of retained host float64 each**. So the ~63 taped fires R217 could not place are on the BIG block, and 111 taped backward fires against 108 + 48 forward calls is consistent rather than contradictory. **`triangle_attention` is the exception that proves it**: it lives in `autograd.py`, which is `_NEVER_SHIM`, so its `_scores` reaches the real `ttnn.softmax`, gets `_exact_softmax_raw`, retains nothing, deallocates `p` and recomputes — which is why `of3t-xcost` saw its 108 backward crossings with `closure: None`. **Bound, stated honestly**: retention is not necessarily simultaneous (57 `checkpoint` fires discard and recreate segments, and a node exists only where `requires_grad`), so the live set is between 1 and 108 — but at 1.688 GiB each **only ~11 need be simultaneously live to account for the entire 19.0 GiB** that was OOM-killed, against **3.377 GiB** for the whole transient (R216). **The fix is recompute, and it is the pattern already in the tree**: rebuild `y64` in the backward from the taped input as `triangle_attention` does, bit-identical because the same input through the same float64 softmax gives the same bits. Sized from `HOSTPRICE.json`'s 0.871 s per call at this shape: **~94 s added to a 705.78 s backward, +13 %**, traded against a step that currently does not complete at all. That is a real trade with a number, and it is `of3t-xsplit`'s to make. Evidence: `tt_bio/taped_ttnn.py:1134`, `tt_bio/tenstorrent.py:4329`, `tt_bio/autograd.py` (`_v_exact_softmax`, `host_f64_softmax`, `_tape`), `perf/of3t_bwattrib/out/hist_384_base.json`.

- **R219** (pass 460, orchestrator — **corrects R218's magnitude, and the meta-lesson is the more useful half**) **The 19.0 GiB is NOT explained, and I should have stopped reasoning two passes ago and taken the 7-minute measurement that was available the whole time.** R218's mechanism survives and R218's magnitude does not. What survives: `host_f64_softmax`'s backward closure retains `y64` on the host; `_v_exact_layer_norm` and `triangle_attention` deliberately retain nothing; `tenstorrent.py:4329` in `_fp32_softmax_attention` is the only `ttnn.softmax_in_place(` call site in all of `tt_bio/`. **What does not survive: "108 calls x 1.688 GiB, ~11 live accounts for the whole 19.0 GiB."** Two things I had not read when I wrote it. **(1) The shim dispatches on the argument, not the module.** `taped_ttnn._taped_verb`'s `call` is `if not always and not _on_tape(args, kwargs): return shipped(*args, **kwargs)` — so a site in a shimmed module reaches the retaining `_v_exact_softmax` only when an argument **is** a taped `Tensor`. Being outside `_NEVER_SHIM` is necessary, not sufficient, and I treated it as sufficient. **(2) The pairformer blocks are CHECKPOINTED.** `ops.checkpoint_segment` at `tenstorrent.py:10721` (the trunk, returning `s, z`) and `openfold3_template.py:144`, with `checkpoint.<locals>...bw` firing **57** times in the banked profile. Under checkpointing a segment's forward tape is built during the backward and dropped after, so the live retained set is bounded by roughly **one segment's** softmaxes, not by the forward's 108 calls. At ~2 per segment that is ~3.4 GiB — the same order as the 3.377 GiB transient, not ten times it. **So neither term as measured explains 19.0 GiB**, and the honest state is that the peak is unattributed. **The meta-lesson, which is why this is filed as its own entry: three passes of source reading produced a mechanism, a wrong magnitude, and a wrong magnitude again, while a decisive instrument sat unused.** `of3t-restep`'s run reaches the OOM in **7 minutes**; a phase- and term-tagged RSS profile of those 7 minutes settles by measurement what four ledger entries could not settle by reading. I demoted that profile to "confirmatory" in pass 459 on the strength of my own inference chain, and re-sequenced the card queue behind it. That was the error, it is reversed this pass, and the rule is the campaign's own: **when a cheap measurement exists, reading the source is how you choose what to measure, not a substitute for measuring it.** Evidence: `tt_bio/taped_ttnn.py` (`_taped_verb`, `_on_tape` short circuit), `tt_bio/tenstorrent.py:4329` and `:10721`, `tt_bio/openfold3_template.py:144`, `perf/of3t_bwattrib/out/hist_384_base.json` (`by_closure`: `checkpoint` 57, `host_f64_softmax..bw` 111).

- **R220** (pass 461, orchestrator, from the kernel's own OOM task list — **measured, and it closes the escape hatch I was hoping for**) **The OOM is tt-bio's own memory on the full shipped scope, not co-scheduling, and the run needs a budget nobody has stated.** Two readings, both from artifacts that already existed. **(1) Co-tenancy is not the cause.** The kernel dumps every task's RSS at an OOM; summed over the 252 tasks listed at 19:11:36, total resident was **23.32 GiB** of pc's 30 GB, of which the run's `python3` was **18.995 GiB — 81.4 % of everything resident on the box.** All 252 other processes together, ~20 `claude` agents and a postgres and a tailscaled included, came to **4.33 GiB**. I had noted `cpuset=docker-...scope` and `task_memcg=/system.slice/cron.service` in the kill line and suspected a multi-tenant squeeze; **the task list refutes it.** A quiet box would have bought roughly 4 GiB, and the run died with ~6.7 GiB of headroom still nominally free, so its peak demand is **above 19.0 GiB by an unknown margin** — that margin is exactly what the RSS profile still owes. **(2) It was the real configuration.** `perf/of3t_restep/out/step_exact_on_384.json` (written incrementally before the kill, `started_utc 2026-09-25T17:04:35Z`, commit `f0e8f3b71`) records `config: crop 384, cycles_pinned 4, diffusion_samples 4, taped true, stage initial_training`, 3,152 optimizer params over 381,302,188 elements. **So this is not an inflated harness configuration that a smaller scope would dodge — it is the shipped step, and the shipped step does not fit on a 30 GB host.** **The budget, stated so a fix can be graded against it rather than against "less":** to run on pc the peak must sit under **~25.7 GiB on an idle box** and under **~19 GiB with the fleet's normal ~4.3 GiB of co-tenants**, against a current peak of >19.0 GiB. A fix that saves 3 GiB does not land this; one that halves the peak does. **And the same arithmetic is the honest framing of the result if no fix is found: OpenFold3 training at crop 384 with the exactness on needs a host with more than 30 GB, which is a hardware statement rather than a defeat.** Evidence: `dmesg -T` OOM task list on pc, `wt/of3t-restep/perf/of3t_restep/out/step_exact_on_384.json`.

- **R221** (pass 461, orchestrator, one `free -g` over ssh) **The campaign's headline measurement may not need a memory fix at all — it needs a bigger host, and the fleet already has one.** pc has **30 GB**; **qb2 has 249 GB total and 186 GB available**. The shipped crop-384 training step with the exactness on wants **>19.0 GiB** and died on pc with ~6.7 GiB nominally free (R220). On qb2 that is not close to a limit. **So the OOM is a host-sizing mismatch before it is an engine defect**, and four ledger entries of chasing it as an engine defect (R216-R219) were chasing the wrong category — one `free -g` on the other host would have said so at any point. **What this does and does not change.** It unblocks `of3t-restep`'s headline, which is the campaign's one missing number and does not depend on any fix landing. It does **not** retire `of3t-xsplit`'s memory work: a 19 GiB peak still matters for anyone training on a 30 GB box, and the retention asymmetry (`host_f64_softmax` keeps `y64` where `_v_exact_layer_norm` and `triangle_attention` deliberately do not) is a real defect on its own terms. It changes that work from **blocking** to **worth doing**. **Three constraints on actually using qb2, none of them trivial and all of them stated rather than discovered later:** its loadavg is **18.60 on 16 cores**, and the exactness is HOST-bound work, so a timing there is an artifact by this campaign's own clock and quiet rules — the memory question is answered but the **timing** needs a quiet window (`a-measurement-precondition-the-arm-itself-cannot-satisfy`: `host_quiet` cannot green when the SUBJECT is host CPU work); qb2 is **p300c** against pc's **p150a**, and board class is part of a number here (the same 512 aa fold reads 17.39 s against 14.59 s at the same 1350 MHz); and qb2's card availability is not established by this entry — cards 0, 2 and 3 carried live cardblocks and card 1 a lease when last checked, so a free card there is a scheduling question I have not answered. Evidence: `free -g` on pc and tt-quietbox2, 2026-09-25 17:4xZ. qb1 (`tt-quietbox`) was unreachable, ssh timed out.

- **R222** (pass 462, orchestrator, one ssh to qb2 — **settles R221's one unestablished link and overturns a belief the campaign formed by breaking its own rule**) **qb2 card 0 is free and is NOT latched at 800 MHz; the campaign's "qb2 card 0 is an artifact" belief was an IDLE sample.** Read off the class node on qb2 itself: card 0 reports `AICLK 0x320` (800) but **`AICLK_LIMIT_MAX 0x546` and `AICLK_ARB_MAX 0x70546`, both 1350 — identical to the busy card's** — while its `ARB_MIN` is 0x10320 (800) against the busy card's 0x10546 (1350). Nothing is latching it; the arbiter floor is simply low because nothing is asking. A loaded sibling on the same box reads `AICLK 0x546` = 1350 right now. **So the 800 is an idle reading, and the standing clock rule's own instruction — sample DURING, not before — is what the campaign failed to apply to its own fleet belief.** Compare the real latch case in memory (`a firmware upgrade latched the max arbiter at 800`), where **LIMIT_MAX itself** would read 800; here it reads 1350. **Card availability, checked on qb2 rather than from pc:** card 0 has **no device holder** (`fuser /dev/tenstorrent/0` empty) and its cardblock says in its own first line *"qb2 card 0 has NO holder as of 2026-09-25T13:05:54Z ... This file is NOT a claim on the card"* — it documents `bcx-mutate`'s pid 8279 exiting on a TT_FATAL DRAM OOM, nothing killed. Cards 1, 2 and 3 are genuinely held: `worker:land-standing`, `bcx-accept`'s `accept_s4` pid 271361, `bcx-mutate`'s `profile_s2_arm` pid 112637. **A FLEET DEFECT found on the way, and it is the `a-card-guard-is-only-as-good-as-where-it-is-readable` shape again with the hosts swapped:** pc has **no** live `state/cardblock-qb2-1` while qb2 **has** one (`worker:land-standing`'s). A dispatcher reading pc sees qb2 card 1 as free and would hand it to a row on top of a live holder. Last time the blocks lived only on pc and qb showed stale; this time it is the reverse. **Net for the campaign: `of3t-restep`'s headline has a host.** qb2 card 0, 249 GB, no holder, arbiter ceiling 1350. The one real constraint left is load — qb2 is at **loadavg 18.60 on 16 cores** and a banked artifact on that box records `host_quiet` refusing at **loadavg1 26.39 over ceiling 2.00** with *"do not measure and explain it afterwards"* — and the exactness is HOST-bound work, so that is not a formality. Evidence: `tt-smi -s` and `fuser` on tt-quietbox2, `~/.coworker/state/cardblock-qb2-{0,1,2,3}` read on qb2, 2026-09-25 17:4xZ.

- **K34** (pass 463, orchestrator — **my own judgement error, caught by the log line my own memory index says is a defect report**) **A stale lease whose holder pid is dead cost `of3t-xsplit` 45+ minutes of zero launches, and I had seen it two passes earlier and decided it could wait.** `state/leases/pc-card0.json` named `worker:of3t-restep` pid **1795534** — the run OOM-killed at 19:11:36 — with `released: null`. `/dev/tenstorrent/0` had **no holder**. `fleet.log` recorded, every two minutes from at least 19:36 to 19:50: *"of3t-xsplit: pinned card 0 not free on [pc] -- DEFERRING to next tick (hard pin, no relocation) -- held by: pc=worker:of3t-restep"*. **The error was not missing it. I saw it in pass 461 and wrote that I was leaving it for the daily `.released-stale-cleanup` because "the row that owns the card is alive and returning to it, and racing its relaunch to tidy a file buys nothing."** Two things were wrong with that. I priced the cost as *tidiness* when the actual cost was a **hard-pinned row getting no launches at all** — a `card=N` row cannot relocate, so a stale lease is not untidy, it is a cage. And the premise expired: pass 462 moved `of3t-restep`'s headline to qb2 card 0 because pc at 30 GB cannot hold the step, so the row I was reserving pc card 0 for **no longer wanted it**, and nothing re-read that decision against the lease. **The general rule, and it is already in this fleet's memory in two forms** (`a repeated "no free card" in fleet.log IS the defect report`; `a hard-pinned row can be blocked by its own detached chain's cardblock`): **a repeated scheduler refusal naming a holder is a defect report, and the holder must be verified live before the refusal is accepted as correct.** Two checks, both cheap and both required — the pid must be alive **and** the device node must have a holder; either alone is a false reading, because a self-hold and a stale lease look identical from the other host. Released this pass as `pc-card0.json.released-stale-cleanup-2026-09-25-of3t-orchestrator-pid-dead-node-free`, after asserting both conditions in the script rather than by eye. **Corollary worth carrying: when a row's target host changes, re-read every reservation held in its name.** Evidence: `fleet.log` 19:36:38-19:50:40, `state/leases/pc-card0.json`, `fuser /dev/tenstorrent/0` empty.

- **K35** (pass 463, orchestrator, following K34's release) **Clearing the stale lease exposed a SECOND gate, and it is arithmetic rather than a stale file: pc's card-row dispatch reserves HALF the host, and pc does not reliably have it.** `fleet.sh`'s `host_mem_ok` computes, for a card task, `need_mb = total_mb / (2 * nc) * (nclaimed + 1)`. pc has `NCARDS=1`, so one card row reserves **`31234 / 2 = 15617 MB`**. pc's available memory is **17005 MB** with ~20 resident `claude` agents. **Margin: 1.4 GB**, so `of3t-xsplit` dispatches only when the agent population happens to be light, and its 19:52:38 refusal changed from *"held by: pc=worker:of3t-restep"* to *"no lease/cardblock names a holder; check host awake/mem/NCARDS"* — the same line for a different cause, which is why the first fix looked like it had not worked. `MAX_CARD_ROWS_PER_HOST=14` against 1 card row on pc, so the slot cap is not involved. **The gate is arguably RIGHT and that is the uncomfortable part**: the reservation of half the box is a good estimate for this workload, because the run that OOM-killed wanted **19 GiB** (R220). pc is not a marginal host for OF3T training by accident; it is a genuinely undersized one, and the scheduler was encoding that before the campaign measured it. **So the two findings compose rather than compete**: K34's stale lease was a real defect and releasing it was right, and behind it sits a real capacity limit that no amount of lease hygiene fixes. **Consequence for sequencing:** `of3t-restep`'s headline goes to qb2 card 0 (249 GB) and `of3t-xsplit` stays on pc, where its probes are much smaller than a full step and will dispatch in a light window — but its brief now says the delay is memory-gated so it does not misread it. **Noted, not acted on:** two live `worker.sh of3t-restep pc 0` processes (pids 1649634, 2496907), the duplicate-launch shape; `card_slots_ok` dedups on the host field so it counts them as one. **And a THIRD gate behind both, which is the one that actually binds: `card_free()` calls `live_worker_on`, which matches any live `worker.sh <slug> <host> <card>` by ARGV** — so `of3t-restep`'s worker occupied pc card 0 with or without a lease, and `of3t-xsplit` is pinned to the same card on a **one-card host**. Two rows on one card was my dispatch error, not a scheduler fault. Fixed by moving `of3t-restep` to `card=cpu` (it cannot use a card today: pc OOMs and qb2 card 0 carries an unrecoverable-open hazard while its board-pair sibling is held). **The change takes effect on restep's NEXT launch** — `live_worker_on` reads argv, so the current `worker.sh of3t-restep pc 0` keeps the slot until that pass ends, and xsplit unblocks then rather than immediately. Evidence: `fleet.sh` `host_mem_ok`/`card_slots_ok`/`card_free`/`live_worker_on`, `config.env` `MAX_CARD_ROWS_PER_HOST=14`, `free -m` on pc, `fleet.log` 19:36-19:54.

- **K36** (pass 464, orchestrator — **a near-miss I am recording because it was one step from a fifth wrong magnitude**) **Do not analyse another row's artifact while that row's run is live.** `of3t-restep` built the RSS instrument this campaign has been owed for four passes — phase-tagged, with `sm_calls` and `sm_inflight` counters and a `MemAvailable` floor guard that **exits before the kernel OOM-killer picks a victim**, so every peak it records is a stated LOWER BOUND. Good instrument. I opened `rss_384_exacton.jsonl` to read the answer out of it and computed **"0.091 GiB per softmax call"** across what I took to be one profile. It was not one profile: the file is append-mode and I read it **while restep was writing it**. Minutes later the same path held **202 rows against 390, one header record against several, and a different pid** (2887082 against 2673408). My slope was computed across concatenated runs, and the follow-up read gave a *negative* slope over the same field. **Nothing of that number was real and none of it reached the ledger, which is the only reason this is a K and not a retraction.** The rule: a live artifact is not evidence, it is a file being mutated; `stat` it twice or check the owning process before you read it, and when the owner is a live row, the analysis is **theirs** — my job is to route it, not to pre-empt it. This is the same failure mode as R219 wearing yet another hat: substituting a fast read for the measurement's own conclusion. **What IS stable across both of today's attempts and is worth carrying**: both breached the floor in the **forward** (`trunk_taped`), at **14.11 GiB** and **13.64 GiB** RSS, with `avail_at_start` of **16.45** and **16.77 GiB** on a **30.5 GiB** box — so roughly **14 GiB was already held by other processes at launch**, against **4.33 GiB** at the 19:11:36 OOM (R220). **pc's available memory at run start swings from ~16.5 to ~26 GiB with the agent population**, which means a run that fits at one hour does not at another, and it makes pc unusable as a *repeatable* host for this step independently of the 30 GB ceiling. That is a sharper statement of R221 than R221 made. `of3t-restep` owns the attribution and reports it.

- **K37** (pass 465, orchestrator) **Two of the four EXIT conditions were already satisfied and I was carrying them as open.** The gate first: `perf/of3t_orchestrator/bwd/batch_gate.sh` — which I wrote at pass 448 *before* the levers it grades, and have been listing since as owed — **was run by `of3t-zerosfill` on the tree carrying its own engine change, and came back GREEN**: all **16 cases** at `rel_l2 <= 1.0e-02` and `cos >= 0.9999`, bf16 quantisation floor **2.76e-03** against the pre-registered 2.8e-3, finite-difference validation of the float64 reference at **3.02e-10 to 3.63e-10**. **All three controls fired**, which is the part that makes it a gate rather than decoration: the float32 arm collapses the error (so the formula is imprecise, not wrong), LoRA's frozen base took no gradient, and `--break-layernorm-axis` was **refused by MEASURING** (`layernorm/x` rel_l2 1.978e+00) rather than by raising before it measured — the exact property I specified when writing it and the one a gate usually fails. And the branch half is vacuous rather than pending: `git diff --name-only origin/main...origin/wk/of3t-bwd` returns **0 engine files**, so there is nothing on the sprint branch for an accuracy gate to grade; the two landed levers (`of3t-zerosfill` at `f0e8f3b71`, `of3t-tapedfwd`) went to main by other paths and each carried its own evidence — zerosfill's VJP is **byte-identical through the changed closure, verified on all three concat slots**, with an AST walk (`assert_backward_only.py`) showing no forward or inference path reaches the changed fill, so no inference A/B was owed. **The lesson is about my own bookkeeping, not the gate: an EXIT list is a claim about the world and decays like any other.** I had re-read and re-published that list for seventeen passes without checking whether a row had discharged an item for me. Check the record before re-publishing a gap — `concluded != merged` has a mirror, which is `owed != actually outstanding`. **EXIT now stands at 2 of 4**: the batch gate is green and the cost answer is drafted (`COST.md`); `of3t-restep`'s full step and `of3t-xsplit`'s 168.51 s split remain. Evidence: `state/of3t-zerosfill.md` VJP section, `git diff --name-only origin/main...origin/wk/of3t-bwd`.

- **K38** (pass 466, orchestrator — **applying K37's lesson to the GAP item that most directly meets users, and finding two things**) **The six "user-facing" defects I have been re-publishing every pass come from a register generated on 2026-09-23 and never re-read; and the one that most directly meets a user is undocumented while the campaign's own record disagrees with itself about its number.** The list is `D32, D55, D184, D205, D210, D250` from `state/of3t/UNFIXED_TRIAGE.json`, whose `generated_from` tops out at an archive stamped `20260923-230004` — **two days stale**, carried verbatim in GAP ever since. Two findings from re-reading it. **(1) D184 is DECIDED, not outstanding, and I can say so without reclassifying anything.** It reads *"the 99.50523 % coverage figure is arithmetically right and belongs to a default-OFF lever, so on the shipped arm coverage is still 97.98499 % ... it is a merge decision, not a measurement."* That merge was **refused on evidence**: `land-standing` verified `TT_BIO_OF3_DEVICE_REFATOM` does not exist on `origin/main` and `of3t-covdefault` NO-GO'd it at +198 ms cold on openfold3. So the lever is not shipping, **97.98499 % IS the shipped coverage**, and the gap between the two figures is a product choice rather than a bug. Checked rather than assumed: **nothing the campaign publishes quotes the default-OFF figure** — it appears only in seven row working docs and `PASSLOG.md`, never in `PROVES`/`COVERAGE`/`COST.md`, whose coverage claim is about loss terms and conditional paths and is a different, correct statement. `reclassifying-out-of-user-facing-is-the-flattering-direction` is a standing trap and this is not that: the classification stands, the *status* is decided. **(2) D205 is live, user-facing, and BLOCKED ON THE CAMPAIGN'S OWN INCONSISTENCY.** It reads *"512 is the largest crop that runs, and the ledger has never said so"* — and `git grep` over `README.md` and `docs/` on `origin/main` finds **no crop or token limit stated anywhere**, so two days later the user-facing docs still do not say it either. My charter is explicit that a capability is not done until the docs match reality. **But the number is not safe to write**: D205 says **512**, D248 (`of3t-cropwall`, fix release-gated and unmerged) says a **576**-token capacity wall, and my own GAP says **crop 640**'s wall. Three figures, one shipped limit. **Writing any of them into user docs today would ship a wrong capability statement, which is worse than an absent one**, so what this closes on is one size ladder on `origin/main` establishing the largest crop that runs today — a card job, deliberately NOT dispatched this pass because the fleet has one contended card and two rows already queued on it, and putting a third there is the error I fixed at K34/K35. Evidence: `state/of3t/UNFIXED_TRIAGE.json`, the D184/D205 headers in `state/archive/of3t-DEFECTS.20260922-113213.md`, `git grep` over `origin/main -- README.md docs/`.

- **R223** (pass 467, measured by `of3t-restep`, arbitrated here — **the memory question is ANSWERED, and it confirms R216's arithmetic while refuting where four entries went looking**) **The largest single term in the step's host memory is the OPTIMIZER, and it is spent before the first forward op runs.** From a 5 Hz RSS sampler tagged with `fullstep.py`'s running phase (crop 384, cycles 4, samples 4, exactness ON), `perf/of3t_restep/out/rss_384_exacton.summary.json`: imports +0.43, `capture_build_fold` (MSA resolve, featurizer) **+3.14**, `capture_prep` +0.94, **`AdamW.__init__` +6.00**, untaped 3-recycle prefix **+0.00**, taped exact trunk cycle **+2.82** (breached). `tt_bio/train/optim.py:164-165` keeps three host fp32 numpy copies per weight — `master`, `exp_avg`, `exp_avg_sq` — over 381.3 M elements: **4.26 GiB of buffers, 6.00 GiB measured** including the download transient. **By design and documented** at `optim.py:121` (*"Masters and moments are fp32 numpy on host"*). It is **31.6 % of the 19.0 GiB the kernel recorded**, it is not the exactness, and it is not a bug. **R216 is CONFIRMED rather than corrected**: the exact softmax transient measured **+2.85 GiB for one in-flight call** against the **3.377 GiB** I computed from `PEAKPROBE.json`'s 2.00x ratio — same order, slightly under, consistent with not every call being at the full `[384,4,384,384]`. Two independent readings agreed (a +2.82 GiB sawtooth over 27 s and 13 calls; a 11.256 -> 14.106 GiB jump in under 2.4 s at `sm_inflight = 1`). **And the control that makes the rest readable: the untaped recycle prefix retains +0.00 GiB over three cycles**, so the growth above it is not "the trunk runs". **Consequence for `of3t-xsplit`, and it is a correction to its brief rather than a cancellation:** chunking removes a real ~2.85-3.38 GiB transient, but on these numbers it **does not on its own make the exactness-ON step fit in 30 GB**, because the floor beneath it is already **10.6 GiB** (6.00 optimizer + 4.08 capture) before the tape opens, with the retained tape and the backward's own temporaries still to come. **The campaign's memory question is closed; the seconds question is not** — no arm has reached the backward, so EXACTPRICE in seconds is still owed. Evidence: `state/of3t-restep.md` PARTITION and EXACTPRICE, `perf/of3t_restep/out/rss_384_exacton.summary.json`, `tt_bio/train/optim.py:121,164-165`.

- **K39** (pass 468, orchestrator — **a trigger I wrote one pass earlier was unsafe, caught before it fired**) **"The card is free" and "the card is not in use" are different tests, and a bounded arm makes the first one lie.** I told `of3t-restep` to flag me when qb2 card 1 is free, which would authorise taking card 0 — its board-pair sibling — under the qb2-0 hazard block's own CLEAR WHEN. Checked this pass: card 1 is held by pid 542435 whose parent is **`timeout 400 .../run_round.py --rounds 3 --seed 100 --out .../land_final`**. A **bounded 400-second arm**, one of a series: the same row wrote `land_verify` earlier, `land-standing` is still open with its worker alive and its doc written 20:03. **So card 1 goes free between arms, repeatedly, while the work continues** — and an instantaneous `fuser` reading during that gap reads identically to the end of the row. The hazard needs the second meaning: cards 0 and 1 are one p300c board, so while card 1 is in use `tt-smi -r` on the pair is unavailable and a failed open on card 0 has **no recovery path** on a box that hard-hung on exactly that at 09:03:31Z today. **A reset path that exists for twenty seconds and then vanishes is not a reset path.** Corrected trigger, written into the brief: a `state/concluded/land-standing` **marker** (the row done, not an arm done) **AND** `fuser /dev/tenstorrent/1` empty at the moment of taking card 0 **AND** the block deleted on **both** hosts, since it exists on pc and qb2 and a guard is only as good as where it is readable. **General form, and it generalises past cards: when a condition authorises an irreversible action, test the thing that must stay true for its duration, not a snapshot that happens to be true now.** Same family as `a-defer-waiting-on-an-orchestrator-action-must-not-ride-a-wall-clock`, where a card reported free two hours earlier had been given away in the gap. **And the campaign's remaining measurement is now formally capacity-blocked rather than engineering-blocked**: pc cannot *dispatch* a card row reliably (`host_mem_ok` reserves `total_mb/(2*NCARDS)` = **15,617 MB** against **15,710 MB** available, a **93 MB** margin) and R223 shows it cannot *hold* the step; qb2's only free card is half the hazarded pair; qb1 is unreachable. Evidence: `pgrep -af run_round.py` and `fuser` on tt-quietbox2, `free -m` on pc, `state/cardblock-qb2-0`.

- **R224** (pass 469, `of3t-restep` concluded PARTIAL, arbitrated here) **The forward half of an exactness-ON crop-384 step is `<= 386 s` against `5.919 s`, so `<= 65.2x` on a matched one-taped-cycle axis — and it is a BOUND from two durable clock stamps, not a timed run.** The row says so itself and I am carrying it that way. It **withdrew its own earlier ~299 s / ~54x derivation as unreproducible**, having overwritten its state doc before reading it on a `state/` tree that is gitignored, so the working is gone; the replacement is re-derived from 421 s wall between `started_utc` and the kernel's OOM stamp minus 35.2 s of self-timed setup, with `of3t-exactscope`'s independently banked 268.27 s base forward sitting inside it. **`COST.md` had been carrying the withdrawn figure and now carries the bound with its provenance.** Three further things worth keeping. **(1) No clean second exists for any OF3T exactness-ON arm**: all four runs died before `fullstep.py` writes `env.aiclk_during`, so not one has a DURING-sampled AICLK, and `host_quiet` was RED at loadavg1 3.33 on the run that got furthest. The row states the load-sensitivity split rather than glossing it — host memory is load-insensitive, so the decomposition stands while no seconds figure is a clean step time. **(2) It confirmed K39 independently**: qb2 card 1's holder moved **466323 -> 542435**, i.e. the card stayed busy *across* an arm rotation, so the pair's reset path never opened and amendment 11's condition was correctly not met. It did not clear that cardblock. **(3) It released its own dead pc card 0 lease** (pid 3013249, checked dead AND the node free) so `of3t-xsplit` stops being refused — the second time that exact hygiene has been needed on this row (K34), and this time the row did it rather than me. **Decision recorded separately** in `state/ask-of3t-optmem-decision.md`: **do not move AdamW's 6.00 GiB to device**, because device DRAM is where this fleet already OOMs (`bcx-mutate`'s TT_FATAL DRAM OOM is why the qb2-0 block exists; D248 is a 10.667x DRAM waste on the taped backward's largest buffer). The better and unasked question is **ordering** — `AdamW.__init__` runs *before* the trunk forward, and the state is only touched in the optimizer step, so constructing it after the capture may be worth 6.00 GiB at the peak for no redesign. Card-free to measure; deliberately not dispatched while the real number is one card-hour away on qb2.

- **R225** (pass 472, orchestrator, git only — **the campaign's headline claim is stamped on a tree main has moved past, and the op-level evidence is NOT**) **Two gradient claims, two different trees, and only one of them still covers `origin/main`.** Checked by blob hash rather than by diffstat, because a diffstat of an unchanged path and a diffstat that was never run look the same. **(1) The op-level gate covers main exactly.** `of3t-zerosfill` ran `batch_gate.sh` on `f0e8f3b71` — **16/16 cases** at `rel_l2 <= 1.0e-02`, `cos >= 0.9999`, all three controls firing — and `tt_bio/{autograd,taped_ttnn,tenstorrent,openfold3_trunk,openfold3_data}.py` are **byte-identical** between `f0e8f3b71` and `origin/main` (`ce158e016`). Five blob hashes, five matches. So the gradient path main ships today is the one that gate passed on, and that is a stronger statement than "the gate was run recently". **(2) The whole-model per-parameter claim does NOT.** `PROVES` asserts, in the present tense, that OpenFold3's training gradient is upstream 0.4.3's to within upstream's own bf16 error on every section over the whole step — 4,152 tensors, 21 of 21 sections within the 3x bar, global rel 1.55834e-01 against bf16's 9.48232e-01. That was stamped on **`f01fa0813`**, and between it and main the gradient-bearing files moved **1,032 insertions / 297 deletions**, of which **673 lines are in `autograd.py` itself**. `autograd.py` and `tenstorrent.py` both read DIFFERS against the stamp tree. **Nothing has re-scored the whole-model claim since.** **This is `a-suite-green-on-the-branch-tip-does-not-verify-the-merge` pointed at our own headline**, and the reason it is worth filing rather than panicking over is that the two halves genuinely differ in coverage: a per-op gate and a per-parameter model score are different instruments, and the op gate's byte-identity result does not extend to the section-level or coverage claims. **What closes it:** re-run `perf/of3t_orchestrator/charter/charter_evidence.py`'s scoring arm on current main. A card job, and `of3t-stepqb2` will be holding qb2 card 0 with 249 GB, so it belongs in that visit rather than in its own. **I nearly reported the alarming version of this.** The first diff — 44 engine files and 1,032 lines since the stamp — reads as "the headline is unverified on main". The second diff, against the gate's tree, is empty. Both are true; only the pair is informative. **Diff against every tree that carries evidence, not just the oldest one.**

- **K40** (pass 473, orchestrator, git only — **a refinement to something I have been asserting for a dozen passes**) **The three held-out OF3T branches all merge CLEAN into main, and BCX's competing rewrites have not landed — so the contention I keep citing is QUEUED, not active, and "clean" is the one result that proves nothing here.** Measured: `git merge-tree --write-tree` writes a tree for **all three** of `wk/of3t-{lnbw,softbw,wheelbw}` against `origin/main` (`ce158e016`) — no conflict in any. All three are **78 behind**, all forked at `f0db89ef5`, and main's own churn in `autograd.py` since that fork is only **28 insertions / 9 deletions**. Meanwhile `wk/bcx-heads` is **33 ahead and NOT in main**, `wk/bcx-bwdplan` **47 ahead, not in main**, `wk/bcx-stack` **24 ahead, not in main**. **What I have been writing is slightly overstated.** I have said repeatedly that `autograd.py` and `taped_ttnn.py` are "contested — BCX is rewriting the head-verb backwards in both", and used that as a live reason to hold. BCX does have that work and it is real, but **it is unmerged**, so today there is no competing content in main to conflict with, which is why the merges are clean. **The hazard is ordering, and it is not decaying — it is waiting.** Whichever set lands second inherits the other's edits to the same functions, and `land-d264` established that a merge error in exactly these two files is **invisible to inference and visible only to a taped arm**. So a clean `merge-tree` is a statement about text, not about gradients, and it is the single most misleading green this campaign can get. **My hold decision is unchanged by this check — and that is the point of having run it.** A conflict would have forced action; clean means the reasons to hold are still the stated ones (two NO-GO rows needing a purpose that outlives their job, `of3t-softbw`'s wiring never run on a card) rather than a merge mechanic. **What it does change: the wait may be long.** BCX's branches are 33-47 commits ahead and unmerged, so "we will re-run their bar when they land" is not a plan with a date on it. Evidence: `git merge-tree --write-tree`, `merge-base --is-ancestor`, `rev-list --count`, `diff --numstat` against `origin/main ce158e016`.

- **K41** (pass 474, orchestrator) **The campaign's answer document was citing evidence that did not exist on main.** `COST.md` publishes the host-memory decomposition — `AdamW.__init__` +6.00 GiB, 31.6 % of the peak, the +0.00 GiB untaped control — and cites `perf/of3t_restep/out/rss_384_exacton.summary.json` for it. That path had **nine files on `wk/of3t-restep` and zero on `origin/main`**: the row concluded, its branch was never composed, and nobody noticed because the number was right and the citation looked fine. **A published number whose artifact is unreachable is a claim, not evidence.** Composed onto `wk/of3t-bwd` (now `bd86d4c9b`, pushed): the 5 Hz phase-tagged RSS profiler, its reducer, the step harness, the profile and its reduction, the OOM evidence and two step artifacts. **Nine files, zero engine files**, so no accuracy, OOM or perf risk; base caught up from `ce158e016` with the catch-up half verified byte-identical to `origin/main^{tree}` before committing. **One file deliberately excluded, and it is a judgement rather than an oversight: `state/of3t-restep.md`.** It sits on that branch too, but tt-bio main carries **0 files under `state/`** and does not gitignore the path, so composing a status doc there would set a new precedent against the standing rule that planning and status docs live in `~/.coworker` rather than the code repos. Built the tree with a temporary index over `origin/main` plus the nine `perf/` blobs rather than taking `merge-tree`'s result wholesale, precisely so the exclusion was explicit instead of hoped for. **The general point: "the row concluded" and "its evidence is where a reader can reach it" are different states, and a concluded marker asserts only the first.** Same family as `concluded != merged`, with the artifact rather than the code as the thing left behind.

- **K42** (pass 475, orchestrator — **a negative result, recorded so the next pass does not re-investigate it**) **The campaign's evidence audit is not broken; it refuses ad-hoc runs by design, and its two halves have different costs.** `perf/of3t_orchestrator/audit_evidence.py` exits 2 unless HEAD is branch `wk/of3t`, and the reason is in its own comment: run from a single row's worktree it found **24 of 25 "drifts" false at pass 182, with the one real drift buried among them**, so a wrong-tree run must be a refusal rather than a drift report — *"a drift report is indistinguishable from artifacts having actually gone missing."* I nearly filed this as a stale guard, on the grounds that `wk/of3t` was absorbed into main. It is not stale: `compose_verify.sh:134` checks the composition out as that branch, so the intended entry point still works, and the guard is doing exactly its job. **What the investigation did establish, and it is worth keeping.** `compose_verify.sh` is **CPU-only** — no `TT_VISIBLE_DEVICES`, no device open — and it says so about itself: *"It does NOT verify behaviour: on a host without ttnn no test executes."* It composes, checks every row is an **ancestor** of the result (not merely that six merges were clean, "because a row can push mid-compose"), diffs test-collection **error sets** against a detached `origin/main` on the same interpreter, and **recomputes the CPU-only instruments rather than believing their committed JSON**. So it is a composition-integrity check, **not** the gradient re-score R225 says is owed — that needs a card, and `of3t-stepqb2`'s brief already points at the right tool (`charter_evidence.py`'s scoring arm). No correction needed there. **And a reason not to reflexively run it**: it does `git worktree add -q "$CO" wk/of3t`, which would move a branch that is now finished history fully contained in main, on a fleet where `two-worktrees-on-one-branch-turns-a-commit-into-a-mass-deletion` is a filed incident. Re-running the compose is a decision, not a health check.

- **R226** (pass 477, measured by `of3t-xsplit`, arbitrated here — **the sprint's first engineering win since `zerosfill`, and my own sizing of it was 10.6x wrong in the useful direction**) **Moving the TILE/ROW_MAJOR layout conversion from host to device is worth 32.55 s, 4.61 % of the 705.78 s backward, 1.048x — bit-identical, with the binding roof named.** Measured on the card at the production shape `[384,4,384,384]` FLOAT32, 0.90597 GB per call, median of 3, in the same process as the bit-identity check (`perf/of3t_xsplit/out/split_384.json:layout`). Out-leg: `to_torch` from TILE **0.76305 s** against device `to_layout` **0.00447 s** plus `to_torch` from ROW_MAJOR **0.69212 s**, net **+0.06646 s/call**; return leg net **+0.08422 s/call**; over the banked 216 crossings each way, **32.55 s**. **The roof argument is the result, not decoration.** Both routes move the same 0.906 GB over PCIe, so this is **not a bandwidth saving**: it relocates the layout conversion from a host untilize running at **12.8 GB/s** to a device `to_layout` running at **202.7 GB/s** — the same bytes on a machine **15.8x** faster for them, because the card's DRAM roof sits far above the PCIe roof that binds the crossing itself. The crossing stays host-DMA-bound at **1.19-2.25 GB/s** either way. That is A46 clause 2 discharged properly rather than a seconds figure with a roof bolted on. **Two corrections to numbers I put in the brief, both found by measuring in the live path rather than in isolation.** I priced the device op from a banked **19.21 GB/s** and predicted ~0.047 s per 906 MB; it reads **202.7 GB/s and 0.0045 s** — **10.6x cheaper**, so the device op is **6.3 %** of the host term it removes rather than the 35 % I implied. And the **44.90 s** upper bound I carried does not survive either, in the opposite direction: the host untilize costs **0.0709 s/call** in the live ttnn path against the **0.13417 s** `layoutprice.py` measured host-only, while the host tilize costs **0.0887** against a banked **0.07369**. In-process host layout is **34.48 s**, of which the device route returns 32.55 s. **The lesson, and it is this campaign's own in a new costume: a banked GB/s from one op is not a rate for another, and a host-only microbenchmark is not the live path.** Both of my errors came from reusing a number measured under different conditions because it was to hand. Same family as `a-ratio-is-not-a-defect-until-the-references-own-ratio-is-measured`. **Still in flight**, and the larger term: `SPLIT`'s 168.51 s arm and the mantissa census. Row VERDICT is PARTIAL and it is mid-pass, so the 32.55 s is banked and the row is not concluded.

- **R227** (pass 478, orchestrator, `lspci` + sysfs, no device opened — **the fleet's PCIe links are negotiated well below capability, and the two hosts differ by 2x on the axis that binds the exactness**) **pc's Blackhole runs at Gen4 x8 against a Gen5 x16 capability; all four qb2 cards run at Gen4 x4 against Gen4 x8.** Read from `/sys/bus/pci/devices/*/current_link_{speed,width}` against `max_link_*`: pc `01:00.0` **cur 16.0 GT/s x8, max 32.0 GT/s x16** — about **15.8 GB/s** per direction now against **63 GB/s** capable, **4x** on the table. qb2 `01-04:00.0` all **cur 16.0 GT/s x4, max 16.0 GT/s x8** — about **7.9 GB/s** now against 15.8 capable, **2x** on the table, on every card. **Why this matters to this campaign specifically, and it is not the obvious reason.** It does **not** explain the 168.51 s. **CORRECTED at pass 484, and my original phrasing here was misleading**: I wrote that the crossings at 1.19-2.25 GB/s sit *~7x below even pc's degraded link*, implying that much headroom. `of3t-xsplit` then measured the board's **achievable** host-DMA rate in the same process — **1.265 GB/s live against 1.187 GB/s isolated** — so the crossings are **at the achievable roof**, and the headroom I implied does not exist at the DMA level. **A link's negotiated capability and its achievable rate for a given transfer pattern are different numbers**, and quoting the first as a bound on the second is the same error as reusing a banked GB/s from another op (R226). What survives: link bandwidth is not the *binding* constraint in the sense of a misconfiguration to fix, and the 2x host asymmetry below stands. **What it does bind is the comparison `of3t-stepqb2` is about to make.** The exactness is a host round trip per softmax and per layer norm — its floor is the PCIe link — and **qb2's link is half of pc's**. The board-class caveat already in that brief (p300c against p150a, 17.39 s vs 14.59 s on the same fold) is about **compute**; this is a second, independent axis, it runs the other way, and nobody had stated it. A host-bound crossing figure from qb2 is therefore **not** comparable to one from pc even at matched clock and matched crop. Amended into the brief this pass. **And the general form is one this campaign has now hit twice in one day**: the environment carries decisive facts that cost thirty seconds to read and that no amount of source reading produces — first `free -g` on the other host (R221), now `current_link_width`. **Read the platform before pricing the software.** Not actionable as a fix by this campaign: whether pc's x8/Gen4 is a slot limitation, a BIOS setting or a training fault, and whether qb2's x4 is by design, is a hardware question for whoever owns those boxes. Filed so it is on the record before someone quotes a transfer rate as a hardware ceiling.

- **K43** (pass 480, orchestrator — **the trigger paid for itself a second time, and a cleared cardblock turns out not to mean what it looks like**) **`of3t-stepqb2` is still correctly gated, and the obvious test would have fired today.** Every surface signal said go: my pc copy of `cardblock-qb2-1` was cleared at 20:37, the original on qb2 is `.cleared-...-land-standing-bc2-verification-finished`, `.cleared-...-land-standing-gate-finished` and `.retired-...-nih-perf-finished-card-free-twice`, and `fuser /dev/tenstorrent/1` read **free on two observations five seconds apart**, as did card 0. The qb2-0 hazard note's own CLEAR WHEN is *"having either confirmed card 1 is free so the board pair can be reset, or accepted that risk knowingly"* — and card 1 **was** free. **The marker test refused anyway, and it was right.** `land-standing` has no `state/concluded/` marker, its worker is alive, and its state doc says in its own words: *"card 1 is this row's grant and is free, `cardblock-qb2-1` cleared"* and *"DEFERRED-TO: next pass. **First thing**: run `scripts/full_parity_gate.py`"*. It is **between passes and plans to return to that card**. **The distinction, which I had not had words for: a cleared cardblock means SAFE TO TOUCH, not AVAILABLE TO TAKE.** A row that clears its own block while keeping its grant is practising good hygiene — it is telling the fleet the chip is not wedged — and a reader who treats that as a release takes a card out from under a row that is merely mid-cycle. Sibling of the new `a-cardblock-copied-to-the-dispatchers-host-is-a-snapshot`: there the copy outlived the original, here the *absence* of a block outlives nothing at all, because it never meant ownership. **Second time K39's refinement has paid** — the instantaneous `fuser` reading would have authorised taking half a board pair whose sibling is claimed. **And a self-inflicted near-miss worth one line**: `pgrep -c -f run_round.py` returned 1, which I nearly read as a live arm; it was **my own ssh command line matching itself**. Same class as the `pkill -f` earlier this session that killed its own shell (exit 144). **A `pgrep`/`pkill` pattern that appears in the command running it always matches.** Evidence: `fuser` twice on tt-quietbox2, `state/land-standing.md`, the cardblock name lists on both hosts.

- **K44** (pass 481, orchestrator) **The campaign's entire written record had two documents with no backup path at all, and the rest were 18 entries stale — on a tree that is gitignored with no restore.** `compose_verify.sh` publishes the live docs into `perf/of3t_orchestrator/record/` on the branch, and its own comment states the reason exactly: *"the source is gitignored, on one machine, with no backup."* Audited the published copies against the live ones rather than assuming: **LEDGER 121,152 B live against 83,716 published — 18 entries behind, newest published `R203` against a live `K43`**, so everything from the backward sprint, the OOM investigation, the PCIe link finding and the trigger refinements was unprotected; **DEFECTS** 90,280 against 85,621; **EVIDENCE** current apart from the banner; **PROTOCOL** current. **And two documents were not in the publish list at all: `BACKWARD.md` (30,460 B) and `PASSLOG.md` (82,295 B)** — the first is the backward sprint's shared finding set, the document every sprint row is told to read first, and the second is the campaign's narrative record. Neither had ever been published and no future compose would have published them. Both added to the loop; they live under `state/of3t/` like the four already there, so they need no special-casing. **Two things worth separating.** The staleness is **not** a defect in that script — it is a consequence of the compose not having been run, which was my own decision at K42 for a stated reason (it does `git worktree add` on `wk/of3t`, now finished history in main). The **omission** is a defect, and it is the kind that hides behind a working mechanism: the list looked complete because everything in it was handled correctly. **`EVIDENCE.md` was deliberately NOT overwritten** when I refreshed by hand — its published copy is *larger* than the live file, and a backup bigger than its source means an overwrite destroys something. It turned out to be the provenance banner, which is the right answer and one I would not have had if I had assumed. **Check the contents of a backup list against the thing it claims to back up; a mechanism that works perfectly on the items it knows about tells you nothing about the ones it does not.** **Postscript, pass 482:** verified the six published copies against `origin` rather than against my local worktree — all present — and the pushed LEDGER's newest entry read **K43** while the live one read K44. I had copied the file and *then* appended the entry describing the copy. **A backup is only as current as its last copy, and appending after copying silently re-opens the gap you just closed.** Refreshed, and the ordering rule is: publish LAST in a pass, after the ledger is written, not first.

- **R228** (pass 483, measured by `of3t-xsplit`, arbitrated here — **the sprint's largest remaining lever is CLOSED, and the row improved on the instrument I specified while catching a defect in my brief**) **The 168.51 s is not queue drain: 2.54 s is drain and 165.99 s is real transfer sitting at the roof.** Measured backward-only at `[384,4,384,384]` FLOAT32, 0.90597 GB, per-crossing medians: `to_torch` SYNC drain **0.01278 s** against SYNC transfer **0.71639 s** and PLAIN **0.69109 s**; `from_torch` drain **0.00007 s** against an **empty-queue sync floor of 0.00009 s** — so `from_torch`'s drain is **zero by measurement rather than by argument**, and `to_torch`'s is 1.85 % of its crossing net of that floor. `a-blocking-op-is-charged-for-the-queue-drained-behind-it` is the campaign's own standing lesson and **it does not apply here**; the row checked rather than assumed, which is the only way that could have been established. **It rejected my briefed instrument, for a stated reason, and was right.** I asked for the base arm run twice, interleaved, median of >= 3: six 26-minute runs whose two arms never share a contention window, on a box that moved loadavg **1.99 -> 5.84 inside one pass**. It alternated the arms **within a single backward**, per crossing, fixed seed — **n=110 SYNC against n=106 PLAIN** for `to_torch`, 109/107 for `from_torch` — so both populations see the same clock, the same load and the same tape, and it recorded **exactly 216 crossings each way**, the banked population to the call. **And it found a site my brief missed.** I named `autograd.py:1313` as *the* insertion point. The census shows **three** sites of 108 crossings each: forward `host_f64_softmax_values` at `:1315`, backward the same at `:1315`, and **backward `bw` at `:1366`, which my brief never named** — so, in the row's words, *"a patch at :1313 alone would have split half the seconds and reported it as all of them."* **Second time this session a row has corrected my sizing** (R226 was the 10.6x device-op mis-price). **Consequence for the campaign: the 168.51 s is attributed and 98.5 % of it is not recoverable by removing overhead**, because it is bytes moving at the achievable rate. **The transfer half is a roof and not a defect**, established by measuring the board's own achievable host-DMA rate in the same process: **1.265 GB/s live against 1.187 GB/s isolated**. **And the mantissa census came back dirty, so candidate 2 is dead**: censused on the real `v` at real sites during the real backward — **324 crossings, 73,383,542,784 elements**, three distinct sites, every block index the run produced, no synthetic tensor anywhere — **48,203,960,098 words (65.69 %) carry at least one of the low 16 mantissa bits set**, the mean number of set explicit mantissa bits is **8.47** against the <= 8 the candidate required, and **20.76 % of the mass sits on bit 0 alone**. A bf16 transfer would be a rounding, not a bit-exact narrowing, and this row makes no fidelity trades. **So the only live lever on that axis is the 32.55 s layout relocation already banked.** `of3t-xsplit` concluded **GO** with nothing merged.

- **R229** (pass 483, measured by `of3t-optorder`, arbitrated here — **the optimizer holds FIVE host copies, not three, and the campaign's 6.00 GiB was 1.10 low**) **`AdamW` costs 7.105 GiB of host memory at crop 384, and 4.263 GiB of it can go at construction with the gradients bit-identical.** `of3t-restep` attributed the cost to *"three fp32 numpy copies per weight — master, exp_avg, exp_avg_sq"*, and I carried that into `COST.md`. `optim.py` builds **five** dicts, and the three resident before any compute are **`master`, `init_master` and `init_device`**. Direct census at crop 384 (3,152 parameters, 381,302,188 elements, qb2 CPU, card-free): **4.000-4.003 bytes per element per dict, 7.105 GiB total**. **The campaign's 6.00 GiB was a 5 Hz sampler's phase boundary — 1.10 GiB low** — which is a good reminder that a sampled trajectory bounds a phase, it does not census one. **Two mechanisms, both found by reading what the calls actually do.** `init_master` and `init_device` were built from the same unwritten `t.value` three lines apart and **held identical bytes**, the second carrying a reshape that reshaped an array to its own shape — one copy now, with `displacement()` reading it for both terms. And **`np.zeros_like` is `empty_like` plus `copyto(0)`, so it writes every page**: the moments were **fully resident from construction**, not a cheap untouched mapping, which is exactly the assumption my brief told the row to verify before implementing. `_Moments` now materialises them at first use, inside `step()`. Result: **construct 7.105 -> 2.842 GiB (-4.263, -60.0 %)**; **after one step 7.105 -> 5.684 GiB (-1.421, the duplicate, permanent)**. `_reduce` also built a `zeros_like` for every parameter to fill gaps in a few; it builds them only for the gaps now. **Bit-identity proven rather than argued**, which is what an optimizer change owes: `optfid.py` runs HEAD's `optim.py` and the new one side by side over the same parameters, gradients and seed across **six arms** (batch / per-sample clipping / a disabled parameter, each with and without the AF3 schedule), **six steps each**, comparing masters, both moments, device weights, every `step()` report and `displacement()` by `np.array_equal` on raw bytes — **0 differences**, card-free, with a host stand-in doing the bfloat16 round-to-nearest-even the PCIe write performs.

- **K45** (pass 484, orchestrator — **my own compose was deleting a file from main, and `--name-only` hid it**) **If you name a commit as a parent, your tree's omissions are deletions.** At K41 I composed `of3t-restep`'s artifacts onto `wk/of3t-bwd` while **deliberately excluding** `state/of3t-restep.md`, on the correct policy that tt-bio main carried **0 files** under `state/` and a status doc there would set a new precedent. I expressed that exclusion by **building the tree with a temporary index while naming `origin/wk/of3t-restep` as a PARENT** — so git records the branch as having *deleted* that path. Main has since gained the file (9,926 B, from restep's own commit `9b6c53520` merged directly), and the three-way merge then resolves base-has / mine-deleted / theirs-has **in favour of my deletion**. **The policy call was right; expressing it by omission against that parent was not.** Same family as `two-worktrees-on-one-branch-turns-a-commit-into-a-mass-deletion`: a tree built by construction carries an implicit deletion for everything it does not include, relative to every parent it names. **Two process failures made it invisible, and both are cheap to fix.** I read the compose back with **`git diff --name-only`**, which prints three paths and tells you nothing about whether they are additions or removals — `--name-status` shows `A A D` at a glance, and there is no reason to ever use the former for a merge check. And I **dropped an assertion I had used one pass earlier**: at pass 474 I verified the catch-up tree equalled `origin/main^{tree}` and printed the confirmation; this pass I omitted it, and it would have failed instantly — the catch-up read **`66a9ccb53`** against main's **`d3155abf9`**. **A verification step removed because it passed last time is a verification step you no longer have.** Rebuilt from main's tree plus only the paths main lacks: **two additions, zero deletions, zero engine files**, `wk/of3t-bwd` now `d51477a91`. Verified with `--name-status` after the push, not before it.

- **R230** (pass 485, `of3t-optorder` concluded GO, arbitrated here — **the memory wall moved, so the campaign's last missing number is reachable on the host every other figure came from**) **`AdamW` gives up 4.263 GiB at construction (60.0 %), bit-identical, and the step's entry into the backward drops from 19.0 GiB to ~14.7 — under the 16.5 GiB pc has at its worst.** Two mechanisms, both found by reading what the calls do rather than what they are named: `init_master` and `init_device` were built from the same unwritten value three lines apart and held **identical bytes**; and `np.zeros_like` is `empty_like` plus `copyto(0)`, so it **writes every page** — the moments were fully resident from construction, not a cheap untouched mapping, which is exactly the assumption the brief told the row to verify before implementing. Proven across **six arms and six steps** by `np.array_equal` on raw bytes over masters, both moments, device weights, every `step()` report and `displacement()`: **0 differences**. **Two carry-forwards it filed that are worth as much as the fix.** (1) **The tape's HOST memory is never released**: `_retire` frees the device buffer and leaves the node, so every exact softmax's float64 `y64` and every exact layer norm's saved input stand from the forward through `opt.step()` — `autograd.py:623-658`, plus a `del roots` before the step, is where a row would start. **That is the retention I chased through R217-R219 and mis-sized twice**, now located and, by the optimizer work, made the *largest* remaining term. (2) `tt_bio/train/recipes.py` **never calls `ag.release_pins()`** where eight perf harnesses do, so checkpoint pins in the production training loop are held to process exit — reported, not changed, because a production training-loop change is a different review. **Dispatch consequence, taken this pass**: `of3t-stepqb2` re-pointed from a gated qb2 slot to **pc card 0**, freed by `of3t-xsplit` concluding, and **pc is the right host rather than a fallback** — every banked OF3T figure is pc p150a, and pc's link is **Gen4 x8 (~15.8 GB/s)** against qb2's **Gen4 x4 (~7.9)**, which binds because the exactness is a host round trip. Taking it on pc **removes** the comparability caveat instead of managing it. It runs on `wk/of3t-optorder`, not main: the fix is release-gated and unmerged, so main still peaks at 19.0 GiB and would OOM exactly as before.
