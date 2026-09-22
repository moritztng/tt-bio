# of3t-orchestrator — reproduce OpenFold3 training, with proof

TASK TYPE: VERIFY/BENCHMARK (campaign design + evidence arbitration) | PLAYBOOKS loaded: ALWAYS-ON
+ verify | memories read: zero-filled-missing-gradient-hides-an-untrained-model,
negative-control-must-break-what-check-reads, parity-gate-builds-both-sides-from-the-same-source,
unified-solution-not-per-model-patches, orchestrator-must-verify-merge-claims-against-git,
sibling-perf-campaigns-need-namespaced-output-paths, moritz-directed-task-silently-retired-
mid-flight, parallel-branches-independently-fix-same-defect-merge-silently-picks-one

Pass 1, 2026-09-19. Baseline `origin/main` at `bd643929a`. No card taken; CPU only. My first duty
was the proof protocol, not a dispatch, and writing it meant checking the charter's ground truth
against upstream. **Six of its stated facts did not survive that check**, one of which deletes a
deliverable from the campaign's central row.

PROTOCOL: `~/.coworker/state/of3t/PROTOCOL.md` (**99 KB** as of pass 287 — the 30 KB this line read until then was three times stale, thirty amendments, all recorded in §9 with the row that asked and whether a number already existed). Originally 16 KB, written pass 1 **before any row was
dispatched and before any number existed**. What counts as complete proof, and the tolerances,
both fixed in advance:

Its load-bearing idea is a factorisation. The update rule `w_{k+1} = U(w_k, b_k, k, s_k)` has five
components, and **four of them carry no model and no device**: the LR schedule and the gradient
clipping are pure functions, verified EXACTLY over their whole domain on CPU in float64; the
optimizer and the EMA are closed-form recurrences, verified under an injected synthetic gradient
drive for 200 steps. Only the gradient itself needs a card. Driving the state-free factors with a
controlled input is strictly better evidence than an N-step trajectory gives, because a trajectory
that agrees cannot say which factor was right, while an injected drive isolates each one and
reaches the corner cases a real batch never happens to hit — the clip threshold, the warmup knee,
a zero gradient, a disabled parameter. That is why N is 20 rather than 2000, and it is my answer
to Moritz's actual question, "the most simple way to be completely sure".

Tolerances, fixed in advance with their justification. Gradients: per-tensor relative L2 bar
**5.0e-02**, median-over-tensors **2.0e-02**, worst tensor always named and located, computed in
float64 against a reference itself validated by float64 central finite differences. bf16 carries 8
mantissa bits, unit roundoff 3.9e-03; the trunk composes 48 pairformer blocks and error through a
depth-d chain grows about sqrt(d), so sqrt(48) x 3.9e-03 = 2.7e-02 is the floor bf16 alone
predicts — the median bar sits just under it, the per-tensor bar one doubling above. PTX's
single-op SDPA backward measured 6.37e-03/6.40e-03/6.02e-03/6.35e-03 against a float64 reference
validated by finite differences to 1.76e-09, which is the calibration. LR schedule and clipping:
**exact float64 equality** and **1e-12** respectively, because both sides compute the same closed
form and nothing accumulates. Optimizer and EMA: **1e-06** per parameter per step, the fp32
master's own floor, not a bf16 argument. A tensor over a bar is the finding; no row may move one,
and PROTOCOL §9 records any amendment with its date and whether a number already existed.

Three design decisions in it that the charter did not anticipate and that a row would otherwise
have had to improvise. **§3a, the comparison space**: our OF3 is an independent ttnn
reimplementation whose weight remap FUSES two of their tensors into one of ours, so the comparison
is made in THEIR parameter space by splitting our fused gradient back along dim 0, and a bijection
manifest with the unmapped set on both sides is published before any gradient is compared. **§3b,
presence before magnitude**: a missing gradient and a zero gradient are different and are compared
as such, because their runner genuinely produces absent gradients. **§7, the growth law**: the
20-step trajectory's bar is not a magnitude but the shape of divergence in k — linear or
sub-linear passes, super-linear fails at any magnitude, including inside the per-step bars.

LEDGER: `~/.coworker/state/of3t/LEDGER.md`, 10 KB, six refutations and eight knowns, each naming
how it was verified. R1: **`openfold3/tests/test_training_full.py` does not exist** — I downloaded
and extracted the 0.4.3 sdist; `find . -name 'test_training*'` returns nothing, and no upstream
test trains end-to-end. It was the charter's cheapest and most credible artifact and it is
fictional. R2: our loss weight table does **not** cover OpenFold3 — `train/losses.py:63` has two
keys, "pretrain"/"finetune", holding Protenix's constants, and of3 appears nowhere in
objectives.py, losses.py or catalogue.py. R3: our OF3 is an independent reimplementation and
`protenix_weights.py:25-26` fuses by `torch.cat`. R4: **OF3 training never exceeds a 768-token
crop** (384/640/768/768 across their four stage configs), so the 1024 aa OOM is an inference limit
and is not on the training path. R5: per-rank batch is 1, asserted in their own runner. R6: their
runner disables confidence-head parameters on zero-confidence-weight samples. K1: the PTX work is
already an ancestor of `origin/main` for fastpath, objective, unify and interface, so it is
inherited rather than rebuilt. K2: the taped surface is `taped_ttnn.py` at ~36 verbs, and the
two-entry `_TAPED` in autograd.py that a grep finds first is a different surface. K7: no single
stage of theirs fires every loss term, which reshapes the coverage requirement into a union over
stages.

ROWS: **one hundred eight dispatched, one hundred four concluded** as of pass 358 — one hundred
eight `of3t-*` briefs on disk including this row's own, and one hundred four concluded markers
excluding it. `of3t-covdefault` concluded 11:46Z (NO-GO on the default flip, and its BAR proposal
refused as D212); `of3t-blk4544` concluded 11:18Z having excluded the trunk's backward;
`of3t-trajfull` concluded 08:35Z and closed TRAJECTORY. Dispatched this pass: **`of3t-refcov`**
(the coverage leg, from the training adapter) and **`of3t-vjpln`** (blk4544's named next arm).
`of3t-trunkact` is live. Four rows live against a cap of five — Moritz, 20:29Z 2026-09-21:
*"we're still burning money too fast. do exactly the same with fewer agents in parallel."*

Every row owns a disjoint artifact namespace `perf/of3t_<row>/` and is based on `wk/of3t` via
`CONTINUES_FROM:` so the DISPATCHER resolves the base rather than the row spending ten minutes
detecting it (D188). A row may not edit another row's namespace; where a shared instrument has to
change (`perf/of3t_trajwide/trajwide.py` is the one that keeps coming up) the change must be
**opt-in by default**, so every existing arm reproduces what it produced before —
`of3t-trajfull`'s `--namemap structural` and `--census-out` are the model for that.

**This field is CURRENT STATE, not a dispatch log.** The per-pass narrative it carried for 350
passes is in `state/archive/of3t-orchestrator.*` and in the branch history, which are the durable
record; re-reading it every pass was costing more than it told anyone (pass 357 replaced ~15 KB of
it with this paragraph).

**Pass 357 dispatches `of3t-trunkact`** onto GRADIENTS' accuracy clause, which is entirely the
pairformer trunk: 5.8282 % of the model's gradient mass at **6.402x** upstream's own bf16, where
the other nine sections sit between 0.945x and 2.176x, and correcting it alone moves model scope
from 0.5201 to 0.1240 — inside the bar. It takes `of3t-blk4544`'s own named next arm and its
measured exclusion: with **every** matmul, softmax and `triangle_attention` backward in blocks 45
and 44 pinned to float64 (518 substitutions, `errors {}`, live at 5.17e-02 of our own norm) R44
read 2.203113 against a 2.201450 baseline and a 1.90 refutation line, and the mass injection
survived at `norm_ratio` 1.722041 → 1.722830. **A quantity exact backward arithmetic cannot touch
did not originate in the backward**, so the row is pointed at the forward activations. Namespace
`perf/of3t_trunkact/`, `CONTINUES_FROM: wk/of3t-blk4544`, gate entry, stage hint and TASKS ws-tag
added.

**Pass 357 also dispatches `of3t-covdefault`**, the other half of GRADIENTS and the last charter miss
that had no owner. Namespace `perf/of3t_covdefault/`, `CONTINUES_FROM: wk/of3t`, gate entry, stage
hint and TASKS ws-tag added. Sized before the brief was written: the shipped arm covers
97.98499306866148 %, the seventeen host-applied tensors are 1.5202384841128946 %, and the sum is
99.50523155277437 % — so the clause clears the 99.2594 bar by 0.2458 points, but only with
`TT_BIO_OF3_DEVICE_REFATOM` ON, and that is **default OFF** with a docstring saying why. The row is
therefore a DEFAULT decision with an inference A/B attached, not a measurement, and the brief
forbids the shortcut this campaign already paid for once: **grading a lever-on arm and calling the
condition met is D177.** It also carries the fallback, because the bar has the D209 shape if the
default stays off — 99.2594 was set as `100 - 0.74055` and the host-applied share is 1.52024 %, so
the shipped ceiling is 98.4797615158871 % and a bar above its own ceiling can never be met. The row
may PROPOSE that repoint; it may not edit the gate.

SEQUENCE: **pass 358** — the two rows dispatched this pass are disjoint and neither has a `DEPENDS_ON`. `of3t-refcov` owns the coverage leg and touches only `tt_bio/train/`; `of3t-vjpln` owns blk4544's backward-VJP arm and `of3t-trunkact` owns the forward-activation arm, which is the same object approached from two sides and the reason `of3t-vjpln`'s brief says explicitly not to take the forward. Neither blocks the other: one is a build on qb1, the other a device re-score. Earlier: **pass 318** — `of3t-barresolve` has no `DEPENDS_ON`: it is CPU-only, contends for nothing, and it gates a claim that is currently being quoted, so holding it behind anything would be holding the audit behind the thing it audits. It is the only row whose result can REMOVE a MET condition, which is why it is dispatched now rather than after the merge story closes. Earlier: with every row already running, sequencing became a set of rules in briefs rather than
`DEPENDS_ON` holds, which would have been inert. `of3t-reference` is the spine and everything
compares to its frozen bundle. `of3t-tape` gates real work in `of3t-equivalence`, `of3t-perf` and
`of3t-memory`, because all three need a structurally complete taped OF3 step. I **corrected the
charter's sequencing for `of3t-perf`**: it said the row may not start until there is a correct
step to measure, which is too strong, since profiling where time goes needs a complete step rather
than a verified-correct gradient. It builds its per-stage breakdown now; the narrower rule is
absolute and stands — no lever proposed or landed until equivalence has passed, and any lever
re-verified at §3d's bars rather than assumed to survive. `of3t-data`'s determinism feeds coverage
breadth, not the step-1 gradient check, so it does not block the proof.

Sequencing of the three rows still live: `of3t-gradients` carries the central claim and holds
qb2; `of3t-l1` decides at which crops that claim can ever be restated (its measurement can be
made today only below the smallest crop upstream trains at, which is a **scope caveat on the
result, not a blocker on the row**); `of3t-updaterule` is CPU-only by construction, so it
contends for nothing and can run alongside both. No `DEPENDS_ON` hold between them would buy
anything — they read different instruments and write disjoint namespaces.

How the corrections reached the rows. The lesson I was given binds here and it cost a day last
night: a decision written into a state doc does not reach a running row, because the row reads its
brief. So every correction that changes a row's direction was written **into that row's brief** as
an append-only amendment block inserted after its DONE_CHECK line — the body each row has already
built against is untouched, since rewriting a brief under a live row inverts its work. Six briefs
amended. Because a brief edit only lands on the next launch and these rows are mid-pass, I also
sent each of the six live sessions the row-specific correction directly; the 14:10 and 14:12
launch groups map exactly onto four and two peer sessions, so the targeting is not a guess.

BRANCH: **`wk/of3t`, one reviewable branch, recomposed from `origin/main` every pass and pushed.**
The pass-318 staleness note this field carried for forty passes is retired: the compose has been
producing a successor every pass since, and pass 357 composed `of3t-trajfull` in, which is the
branch state the charter's TRAJECTORY verdict is read from. **Never run a compose without reading
its exit line** — pass 357's first attempt merged every row and then FAILED at the D149 ratchet on
`perf/of3t_trajfull/trajfull_score.py`, so the merges were done and nothing was published; the
repair and the re-run are in `PASSLOG:`. The figures below describe the composition as last built.
Regenerate with
`perf/of3t_orchestrator/compose_verify.sh`, which recomposes from `origin/main` rather than
merging into a stale branch, asserts per-row ancestry **after** the merges, attributes file
ownership by commits unique to each row, warns when a row's worktree is ahead of origin and says
why (`rc=124` means its turns are being killed before the push), diffs the collection error set
against a detached `origin/main`, checks every `docs/*.md` the diff mentions exists, recomputes
the three CPU-only instruments, and re-reads 127 scoreboard figures from the artifacts, including the PROVES/DOESNOT summary
itself. For a row whose worktree is on another
host it says so rather than skipping the unpushed-work check in silence (K60). Its fetch asks `git ls-remote`
which refs exist rather than naming ROWS blind, so a row that is dispatched but has not pushed
yet is skipped by name instead of failing the whole compose (K58). Rows
commit to their own `wk/of3t-<row>`; none merges anywhere itself. Nothing goes to the merge gate
without Moritz.

BRANCH-VS-GATE (pass 20, established by trial merge rather than by intersecting file lists):
`wk/of3t` touches 205 files, the TRAIN campaign's branches 126, and the intersection is exactly
**one — `docs/training.md`**. No TRAIN branch touches `tt_bio/train/optim.py`, which closes the
cross-campaign hazard I flagged in pass 3 and never verified. A trial merge of `wk/train-i-run`
still conflicts on three files, but **two have nothing to do with this branch**: `wk/of3t`
touches `README.md` and `cli.py` zero times, and `train-i-run` bases at `350773d09`, an older
main that has since taken 4 commits to `README.md` and 1 to `cli.py`. **Merging this campaign
does not make TRAIN harder to merge** — TRAIN is stale against main either way, so the gate can
take them in either order. Method note (K49): intersecting three-dot diffs predicted one
conflict where the trial merge produced three, because a three-dot diff is computed against each
branch's own merge base; check the bases first, and prefer a trial merge when it matters.

PROVES: the **state-free half of OpenFold3's update rule, exactly and against upstream's own
objects** — including **PROTOCOL A12's clip/optimizer seam**, the one place the two state-free
factors interact: with the clip BINDING, our step agrees with upstream's to a worst relative error
of **1.804e-07** over 50 steps (worst tensor `bias`,
`perf/of3t_orchestrator/instrument_c2_clip_in_step.json`), with the clip inactive reproducing
identity and a negative control that clips AFTER the moments separating — — and, for the model-dependent half, **four scopes now MEASURED rather than a
machinery claim**: 38.1862 % passes on the shipped path, `aux_heads`' 2.8431 % passes at
2.271382e-03 once a dropped-mask call site was fixed, the trunk's 5.8282 % is **UNREAD at
the gradient against the revision its checkpoint is bound to** (corrected pass 195; the
5.367727e+00 this field carried was taken at a forward that disagreed on both tracks against a
**0.5.0** reference, and one of those two tracks now agrees), and the diffusion arm's 51.1358 % is
bounded by a softmax ceiling no device lever reaches. So two of the four are readings, one is a
ceiling and one has gone back to unread — which is worse bookkeeping and better evidence than
this field claimed.

**What the trunk IS now, split by track against 0.4.3** (`of3t-trunk043ref`): the **pair** track's
forward **PASSES** at **4.947045e-02**, 0.9894x the 5.0e-02 bar and **9.596x** upstream's own
bf16, where against 0.5.0 it read 2.793661e-01 at 46.67x — **92.08 %** of that figure was the
reference revision, measured directly as 0.5.0's own float64 stack against 0.4.3's. The **single**
track's forward **FAILS** at **1.065338e-01** as shipped — and the cause is settled: it is
`scale_pair_bias=False` (**D1**), not bf16. With the bias pre-scaled, float64 reproduces upstream
0.4.3 to **rel ~1e-16**; on device the flag takes the single track to **1.655263e-02**, inside both
bars, pair track bit-identical. The cliff shape is the same fact read from the other side — the
single track's growth collapses `|q.k|max` from 13916.7 to 873.5 over blocks 8 to 46 while
`|bias|max` barely moves, so a bias 4.899x too small is invisible at 0.4 % of the logits and
dominant at 3 %. Genuine bf16 excess after the flip is **2.48x** upstream's own composed bf16,
35x smaller than the convention was.

**Every load-bearing figure in this field is now traceable to an artifact the orchestrator read
itself, not to the row that produced it (pass 250).** Four numbers carry the campaign's answer, and
all four have been checked:

  * the **closing measurement** — `perf/of3t_wholemodel/MODEL_arms.json`, pass 248: 0.9592x and
    0.93924x confirmed to every digit, with the shipped arm's **52.898x** added as the companion
    the record lacked;
  * **§6 coverage** — `of3t-bondcov`, pass 249: **8 of 8**, `bond` firing at a **14.6902 %** share
    of the squared gradient norm with a 0.0-exactly control, and the GAP line that had understated
    it for forty passes corrected;
  * the **shipped split** — `perf/of3t_orchestrator/DISTANCE_TO_GO_AGAINST_THEIR_STEP.json`, this
    pass: `measured_and_PASSES/pct` **2.8431** (aux_heads at 0.002271382, **0.1136x** the bar),
    `measured_and_fails/pct` **94.8836**, and the artifact's own `arithmetic` field carrying
    *"0.2666 + 2.8431 + 94.8836 + 2.0067 = 100.0000 %"* with `derived_totals` holding the sums
    **because** a pass-175 traceability check had found a figure living only in prose;
  * the **trajectory headline** — checked and **corrected** rather than confirmed (**D136**,
    pass 245): it was the `repin` arm, not the shipped default. **And then RE-MEASURED and it
    passes (pass 307)** — see the next paragraph; the correction is what made the real reading
    possible, which is the argument for doing this at all.

Three confirmed, one corrected. That ratio is the reason the exercise was worth a pass.

**The 20-step trajectory, on the SHIPPED default, at 88.0819 % of the model (pass 307,
`of3t-trajwide`, GO condition 3 — MET).** `rel_d` **2.564253e-01** at k = 20 with a log-log growth
exponent of **-0.27724** (intercept -0.46711, r2 0.9050) over the full **k = 2..20**, zero rungs
dropped, falling monotonically at every rung. **SUB-LINEAR, and §7b's bar is the shape**:
super-linear fails at any magnitude, and this is the opposite sign. **980 of 980** tape resolutions
at every step of every arm, so D126 does not recur at this scope; `d_1` exactly **0** on both
sides, which `lr(1) = 0` requires; **1752.90x** the fp32 differencing floor at k = 20 and never
under 19.20x from k = 2. The controls separate rather than agreeing quietly: the A16 `zero` model
reads exactly **1.000000e+00** at every rung, the break control `permute` reads **+0.23471** against
the treatment's -0.27724, and `shipped_aa2` is bit-identical to `shipped`. **I refit it myself from
the row's own published table** — exponent, intercept and r2 reproduce to all five digits, the floor
ratios to 0.01 % (display rounding on a 4-figure floor). This **supersedes the 36.9462 % /
4.763338e-02 / -0.2482 reading** and retires D136.

**All five of D136's pre-registered fields are now met (pass 315)** — reference resolved
in-process (not from the constant) at 0.4.3 with 0 missing / 0 unexpected, arm `shipped`,
scope-with-mass 88.0819 %, denominator construction published with the uncovered 11.9181 % split
tensor-by-tensor and summing to 100 %, and the fifth — **agreement-and-accuracy against the
reachable bar** — supplied by `of3t-trajbar`.

**The bar, and it is the number that turns a shape result into a reproduction claim.** Upstream

---

DOESNOT: **Nothing a user gets today has changed, and every headline repair in `PROVES:` is a
CONFIGURATION rather than the shipped port.** This is the first line of this field because it is the
sentence most likely to be lost in a summary. Neither `SOFTMAX_BW_RENORM` nor `host_f64_softmax`
exists on `origin/main` at all — verified by `git grep` on the tree, not inferred — so the
defensible claim is *"a configuration we have built and measured reproduces X"*, never *"tt-bio
reproduces X"*, and the two differ by a merge gate that is Moritz's and not this campaign's.

**It does not prove a training RUN.** Everything here is the update rule at k = 1..20. Stability
over 100k steps, precision drift, loss-scale behaviour late in training and rare sample types are
all outside it, and no amount of widening the per-step comparison reaches them. The honest sentence
is *"our training step is their training step, verified per-parameter over the mass named in
`PROVES:`, with full path coverage"* — never *"we reproduced their training run"*.

**TRAJECTORY's gate grades SHAPE, scope and coupling — not magnitude — and the magnitude is now
measured anyway.** The clause passes on shape (log-log exponent over k = 2..20 of
**-0.30639702557322707**, r2 0.9066, falling monotonically; PROTOCOL §7b fails super-linear at any
magnitude) at `rel_d` **0.22750192605686895** at k = 20. **`of3t-trajbar`'s `bar_bf16mixed_all` is
the same measurement for upstream's OWN bf16-mixed loop at the identical 761-tensor, 89.2106 %
scope**, and pass 357 scored the two against each other: **1.2707x at k = 20, never worse than
1.3677x over k = 2..20**, with the reference trajectory `d_theirs_norm` **bit-identical at all
twenty rungs**, so this is one reference and two arms rather than two experiments. That is the
magnitude statement, and it is a good one — our twenty-step trajectory sits 27 % above the
precision floor the reference itself has.

**No magnitude CLAUSE has been added to the gate, deliberately.** The number existed before any
bar could be written for it, and PROTOCOL forbids a tolerance chosen after the result. At a factor
of 1.0 the clause would fail, so adding one is not self-serving — it is simply not evidence. The
proposal belongs in PROTOCOL §9 as an amendment recording that a number already existed, and the
honest reading until then is: TRAJECTORY is MET on the three properties its clause names, and the
fourth property is measured at 1.2707x rather than graded.

**It does not prove the trunk.** 5.8282 % of the model's gradient mass is measured at 6.402x
upstream's own bf16 and is the campaign's one large open accuracy object.

**The 2.0150 % that is unread on the shipped arm is a wiring gap, not a limit** — `of3t-readable-
mass` established that, and D184 names the lever (`TT_BIO_OF3_DEVICE_REFATOM`, default OFF) that
takes coverage from 97.98499 % to 99.50523 %.

GAP: **GRADIENTS, and after this pass it is two things rather than the one the doc was claiming.**

    accuracy  0.520124 against its own A26 bar 0.147353, 3.5298x over, at 97.98499 % coverage.
              ALL of it is `pairformer_stack` at 6.402x upstream's own bf16; the other nine
              sections are 0.945x-2.176x and the counterfactual lands at 0.1240 (0.841x, passes).
              Owner: `of3t-trunkact`, dispatched pass 357 onto the
              FORWARD activations, which is what `of3t-blk4544` excluded the backward in favour of.
    coverage  97.98499 % against 99.2594 % on the SHIPPED arm, and **the bar stands** (D212,
              pass 358). `of3t-covdefault` NO-GO'd flipping `TT_BIO_OF3_DEVICE_REFATOM` on a real
              inference A/B (+198.476 ms cold / +50.231 ms warm on openfold3) and proposed
              repointing the bar to the shipped arm's own 97.9849. Refused: the flag is not on
              any route. `tt_bio/train/openfold3.py:332` calls the HOST `ref_atom_embed`
              unconditionally and its host prep runs ABOVE `with ag.tape():` at line 362, while
              the arm that DID reach the eight built `RefAtomFeatureEmbedder` directly with no
              env var at all. `tt_bio/train/` cannot reach inference by construction, so
              97.98499 +0.75304 (the eight) +0.74051 (`linear_q.0.weight`) = **99.47854**, which
              clears. Owner: `of3t-refcov`, dispatched pass 358, with the accuracy cost of moving
              two host-float32 legs onto the card pre-registered as a possible NO-GO.

**D126 is ANSWERED and CLOSED, and it never reached `origin/main`'s shipped RECIPE** (2026-09-22,
ask 9807, Moritz: *"us your own own judgement, do the right thing"*; record
`state/ask-9807-decision.md`, measurement `state/d126-main-reach.md`). **Three corrections this
row owes, because it is the doc that carried the claim.** (1) The fix is not "two lines in
`Tensor.rebind`" — there is no `Tensor.rebind`, and `of3t-rebind.md` already refuted `lora.rebind`
as the repair. It is a `value` property setter on `autograd.Tensor` that re-keys `_PARAMS` where
the handle changes, ~55 lines with the slot rename and the `__getattr__` guard. (2) The sentence
*"on `origin/main` a training run computes one gradient and then exactly zero forever"* was written
here from a reading taken on **`wk/of3t`** (`of3t-rebind.md`'s VERDICT says so in those words).
`ag.parameter()` has **zero callers in all of `tt_bio/` on `main`**, whose only training route
hands the forward the leaf through `lora._Substitute`. `d126-main-reach` settled it ON A CARD at
9e17ad418: six steps of the shipped recipe, `grad_norm` 1.749e+01 -> 2.802e+01 and never zero,
5 of 5 leaves carrying a gradient every step, cumulative displacement ratio 1.0001, and a run
resumed from a checkpoint stepping normally at `grad_norm` 2.817e+01. **The claim is REFUTED on
`main`, not merely unestablished.** (3) The fix **is** on `origin/main`, as **e1d887a54**, but it
was pushed at 11:44:18 +0200 — DURING `d126-main-reach`'s pass, not before it. That row's opening
`git ls-remote` saw `main` at 9e17ad418 with no `wk/d126-main` ref and briefly recorded the fix as
never landed; that reading was true of its moment and is now wrong. `e1d887a54`'s parent IS
9e17ad418, so every number the row took is the pre-fix control the brief asked for, and main's
head measures **bit-identical** to its parent on the shipped recipe: the same six `grad_norm`s,
the same loss 106.102083, ratio 1.0001, registry resolving 0/5. That is what hardening a seam
nothing on the route reaches looks like.

**D210 is unowned** — 14.2M DiT pad-lane parameters that train and should not; masking them out
of the optimizer is a model change on the shared diffusion path and owes an inference A/B.

**D211 stands as a condition, not only as corrected prose** — GRADIENTS' coverage clause and its
accuracy clause are not satisfiable by any single artifact the campaign holds, and closing it means
closing both legs at one scope, which is what `of3t-trunkact` and `of3t-covdefault` are for.

**The merge story.** `wk/of3t` is the single reviewable branch, recomposed from `origin/main` and
verified every pass by `perf/of3t_orchestrator/compose_verify.sh`. Nothing merges to `main` without
Moritz.

VERDICT: PARTIAL, stamped pass 358, 2026-09-22 — **still working, which is what PARTIAL means.**
The machine-readable exit criterion reads **2 of 3** (`state/of3t/CHARTER_EVIDENCE.json`,
regenerated every compose, spec lifted from the live gate, break control passing): COVERAGE MET at
pass 351, **TRAJECTORY MET at pass 357**, GRADIENTS not. One hundred eight rows dispatched, one
hundred four concluded, four live. **Two hundred twelve defects filed**, **78 UNFIXED** (4
scope-excluded, 9 USER-FACING, 65 campaign-internal) over the UNION of `DEFECTS.md` and its
archives — the live file holds only the tail. `state/concluded` holds **106** of3t files, of which
two (`of3t-orchestrator.falseconclude-20260920`, `.reopened-20260920-225425`) are this row's own
historical markers and not rows, so **one hundred four rows have concluded**. Counts recounted
from disk this pass, not carried forward: 108 `of3t-*.txt` briefs, 106 markers, 2 of them this
row's own.

**The distance still to go, per tensor against upstream's own step**
(`perf/of3t_orchestrator/DISTANCE_TO_GO_AGAINST_THEIR_STEP.json`, denominator 10.279642678524981,
compared union 97.985 %). **48.1831 %** of the mass is AT OR BETTER than upstream's own bf16
training step tensor by tensor, **49.8019 %** is worse, **2.0150 %** has no direct reading. Mass
weighted, the same union reads 0.520124 against the A26 bar 0.147353, **3.5298x** — the two point
opposite ways and both are true, because our failures concentrate in high-mass tensors.

**What is verified.** The update rule's four state-free factors — LR schedule, gradient clipping,
optimizer, EMA — are exact or at 1e-06 under an injected drive, over their whole domain and at the
corner cases a real batch never hits. Every loss term and every conditional path is demonstrated to
FIRE on a real end-to-end step, 8 of 8 and 11 of 11, with inference byte-identical twice. The
twenty-step trajectory is coupled, covers the diffusion module completely, diverges sub-linearly,
and — measured at pass 357 against `of3t-trajbar`'s upstream-own-bf16 loop at the identical
761-tensor scope, with the reference side bit-identical at all twenty rungs — sits **1.2707x**
that loop at k = 20 and never above **1.3677x**. Outside the pairformer trunk the gradient is at upstream's own accuracy: nine of ten
sections between 0.945x and 2.176x of upstream's own bf16 step, most of them at or below 1.1x.

**What fails, and it is one object.** The trunk at n384 reads **2.0151** against upstream's own
**0.3148** on the same reference — **6.402x** — and at 5.8282 % of the model's gradient mass that
one section is the entire difference between GRADIENTS passing and failing. It is localised to a
STEP created in the backward of blocks 45 and 44, it is 100 % ours (the float64 reference is
bit-identical across both widths at all 49 rungs), softmax is refuted as the carrier, and the
LayerNorm affine leaves carry 93.80 % of it with an excess that is width-invariant.

**The honest one-sentence claim today**: *on a configuration we have built and measured, the
OpenFold3 training step reproduces upstream's step per-parameter over 92.16 % of the gradient mass
at upstream's own bf16 accuracy, with full loss-term and path coverage and a coupled twenty-step
weight trajectory over the diffusion module that stays within 1.27x of upstream's own bf16 loop —
and one section, the pairformer trunk at 5.83 % of the mass, is 6.4x upstream's own error and is
not reproduced.* Everything in that sentence is a
measurement in a committed artifact; nothing in it is the shipped default.


PASSLOG: the per-pass narrative. **This field is deliberately uncapped and is not a
summary** — the capped fields above carry current state and this carries how it was
reached, which A30 requires be kept verbatim rather than deleted. Until pass 319 all of
it lived inside `VERDICT:`, which is why that field measured 18,455 characters against a
4,000 cap: there was no ALL-CAPS heading to terminate it, so every pass section appended
below it was read as part of the campaign's verdict.

**Restored at pass 357.** The 06:30:02Z rotation archived the middle of this doc and took
`DOESNOT:`, `GAP:`, `VERDICT:` and this heading with it, which had two consequences the
composition reported separately and which are one fact: the three summary fields read as
ABSENT, and `PROVES:` swallowed every pass section below it and measured **136,078
characters against a 20,000 cap**, because nothing terminated it. Rotation cuts by size and
cannot know which lines are load-bearing headings. The fields below are re-authored with
pass-357 numbers rather than restored verbatim from
`perf/of3t_orchestrator/record/ORCHESTRATOR.md`, because TRAJECTORY changed state in this
pass and the mirror's text predates it; the mirror stays the durable copy of what they said
before.

## ROTATED 2026-09-22T06:30:02Z

This doc reached 334798 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-orchestrator.20260922-083002.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

inst
the REFERENCE's own artifact on the pass it is written**; if the reference fails it, it is void on
arrival and must be recorded as void then.

**Also corrected pass 326:** one line still called D174 "a live candidate for D19's forward error".
Both halves are wrong — D174's gradient mechanism is refuted, and D19 has been closed as a port
defect since pass 196, where the crop-64 2x2 has each arm agreeing with its OWN convention's
reference to 0.50 % (4.947045e-02 against 4.971863e-02), making the 2.792e-01 a cross-convention
figure rather than our error. That also exposed the gap `of3t-padshape` was amended to fill: at
crop 64 both the forward and the gradient are known good, at n384 **only the gradient has ever
been read**, and the forward at each width is the discriminator between a backward-only defect and
a shape-following forward op.

**Pass 326, second finding. D177: the charter was grading a configuration nobody ships.** GRADIENTS
read `MODEL_shipped.json`, the **pre-D56** arm, while `TT_BIO_SOFTMAX_BW_RENORM` has defaulted True
on main since `1aa7070f5` and the compose asserts that default every run. Repointed to
`of3t-ditmodel`'s `MODEL_d56_retake.json`, which is demonstrably the same instrument: same float64
digest, same denominator, same bars, `scope_only: false`, and its d56off arm reproduces the old
headline to six digits (5.551840268594777 against 5.551840268986491).

**On the arm we actually ship:**

    d56on vs upstream's own bf16    0.10066947293993027   against the A26 bar 0.1049544980174316
    d56on over the per-tensor bar   623 of 907            against upstream's own 791
    coverage                        92.15682156952718     against 99.2594

So at THAT scope the gradient is **inside the reachable bar at 0.9592x** and **no worse
per-tensor than upstream's own step**.

**CORRECTED pass 357 (D211), and the correction is load-bearing.** The paragraph above used to end
*"and GRADIENTS now fails on coverage alone"*. Both of its numbers are readings of
`MODEL_d56_retake.json`, which scores 907 tensors over **92.1568 %** of the mass, and the live gate
grades GRADIENTS on `perf/of3t_modelboundary/MODEL_withtrunk_n384.json` — 3643 tensors,
**97.98499 %**, the same three references by sha256 and the same denominator. There the shipped arm
reads **0.520124** against its own A26 bar **0.147353**, **3.5298x** over. **The widening that
closes the coverage clause is what breaks the accuracy clause**, so "coverage alone" was true only
of a scope nobody grades. One section carries all of it: `pairformer_stack`, 5.8282 % of the mass at
**6.402x** upstream's own bf16, where the other nine sit between 0.945x and 2.176x. Hold the nine
and put the trunk at upstream's own level and model scope falls to **0.1240**, which is 0.841x the
bar and passes.

**Two clauses moved in one pass, both from fail to pass, both by me.** That is the pattern that
should draw scrutiny, so the guard is written into PROTOCOL §9 rather than left implicit: the test
each had to survive is *does this let the campaign declare GO*, and neither does — coverage reads
92.1568 in the old artifact and the new one alike, so GRADIENTS is NOT MET either way and the
charter still reads **0 of 3**. A third adjustment in this area should be treated as the gate being
fitted to the

---

## ROTATED 2026-09-22T09:32:09Z

This doc reached 170706 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-orchestrator.20260922-113209.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

d proving this
adapter changed nothing users get — while it has touched `tt_bio/autograd.py`,
`openfold3_host_prep.py` and `recipes.py` beside the new module. Composing a coverage win while the
"did you break inference" check is outstanding is the wrong order, and a condition flipping to MET
on a live row's artifact is precisely where this campaign has been wrong before. It is a one-line
source addition to `merge_coverage.py` when the row concludes.

**`of3t-trajwiden` found the bar itself is inconsistent (D202).**
`COVERAGE_CEILING_IS_NOT_100.json` derives **99.2594 %** as 100 minus **0.74055**, counting the
`input_embedder` host-applied weight. `READABLE_MASS.json`, in a third namespace, resolves the same
HOST_APPLIED class over 17 tensors at **1.52024 %** — `input_embedder` 0.76720 plus
`diffusion_module.atom_attn_enc` 0.75304. Applied consistently the ceiling is **98.47976 %**, a
0.77964-point overstatement, and it is the bar under BOTH GRADIENTS' coverage clause and
TRAJECTORY's scope clause.

**I checked whether correcting it would move a verdict before saying it should be corrected**, and
that ordering is the whole point: GRADIENTS reads 97.98499 % and TRAJECTORY 88.0819 %; both fail
against 99.2594 % and both still fail against 98.47976 %. A bar correction that moves a verdict is
D181's failure mode — this one moves none, so reconciling the two artifacts is bookkeeping. It gets
corrected when a row re-derives it, not by me editing a number in prose.

The class is not structural either: `TT_BIO_OF3_DEVICE_REFATOM`
(`openfold3_host_prep.py:187`, default-off) ports the op, and with it on inside the instruments the
ceiling is 100 % minus 0.49477 %. It must stay default-off for inference — a device linear in
bf16/fp32 is not bit-identical to the host float32 one — and nothing is proposed for the shipped
default. One tensor, `linear_ref_pos`, is 0.6734 % — 59.7 % of the coupled headroom. And the row
caught one more **`firing != code`** without prompting: the flag is wired on the shipped fold path
while the trajectory builds `OF3DiffusionModule` directly, so *"the flag existing is not the flag
firing here"*.

`of3t-tapeattn` launched onto D191's relocated object and has not committed yet.

**`of3t-trajwiden` concluded late in the pass with a trap it found before running into it (D203).**
The one-line way to widen the coupled scope — swapping `HP.ref_atom_embed` for
`ref_atom_embed_device` at `trajwide.py:559` — fails silently, because
`openfold3_host_prep.py:262` builds the embedder inside the function and throws it away: a walk of
attributes cannot reach the eight weights, and the call sits outside the taped `fwd()` so no
cotangent reaches them. **The failure would have read as progress** — 581 tensors scored instead of
573 with eight frozen, so the scope percentage rises while the trajectory covers less real
training. The row's own sentence: *"That is a number I would have reported as progress."* It is two
standing lessons landing together and it named both — the parameter set comes from a walk, and a
wired flag is not a firing flag.

It also halved its own cost estimate by checking key lists instead of reasoning: 1.9 h to
**0.93 h**, because `theirs/k20.npz` already holds all eight `ref_atom_feature_embedder` entries at
every one of the 20 rungs where the shipped arm's file holds none. **`of3t-refatom` is dispatched**
onto that work, with the row's `NEXT_ACTION_REFATOM.json` named as the specification rather than
background and its seconds-long pre-check made deliverable one — including the assertion that
catches a change of function rather than of place.

## Pass 349 — a concluded row's STOP verdict sat unabsorbed for a hundred passes, and GAP was telling Moritz three conditions were MET while the instrument read 0 of 3

No row committed this pass, so I audited my own bookkeeping and it was worse than I expected.

**`of3t-ditcot` concluded on 2026-09-21 with a STOP and none of it was in the ledger.** Its finding
is not small: the diffusion reference runs ONE shared `layer_norm_z` at all-ones while the
checkpoint's **48 trained per-block tensors are dropped as `unexpected_keys`** — and it is not a
`strict=False` slip, because upstream 0.5.0's `DiffusionAttentionPairBias` has no `layer_norm_z`
member at all, so **the reference cannot be built in the checkpoint's architecture**. The row
priced it instead of stopping at the discovery, and our port is the only side that can, since it
runs both layouts: scope median rel_l2 **ours 0.7055 → reference architecture 0.1065, 6.62x**, 24
of 24 blocks improving at D129's own leaf, and a random-weight control splitting it **2.73x layout
/ 2.43x weight**. **6.62x is larger than the 2.28x D129 was chartered to attribute to an op**, so
that figure cannot survive its boundary and the ablation stayed halted — a per-op attribution there
would have named an op for an architecture difference. The same row also **closed D55's backward
half** on four reductions proved inert, two of which *never execute in either model scope*, which
it measured rather than assumed.

**`of3t-crop768` likewise: 512 is the largest crop that RUNS, and the ledger had never said so.**
Filed as **D205**, USER-FACING. 544, 576, 640 and 768 all refuse, on **two different walls** —
capacity at 640/768 (23.7 MB and 6.2 MB free device-wide) but **CONTIGUITY** at 544/576 with
6.30 GB and 6.67 GB still free, 576 refusing a 2,717,908,992 B buffer inside `ttnn::concat` short
by 77,930,560 B per bank at 88.45 % occupancy. **A capacity extrapolation cannot see that wall**:
the row's own fit said 576 would clear with 14 % of margin. The refusal is deterministic and
card-independent, byte for byte across two cards, and every refused rung's high-water is a LOWER
bound, so 768's 1.558x overshoot is a floor. A rule fell out of it too — odd 32-tile counts (480,
544) narrow the fp32-softmax L1 plan to **0 B** where every even count measured keeps it.

**So I measured the hole rather than assuming it was these two (D204).** Of **99** concluded
`of3t-*` rows, **8** were named nowhere in the DEFECTS union. All six real ones are absorbed this
pass and a shrink-only ratchet now holds the count at **0**. But naming is a weak test and the
worst case proves it: `of3t-ditcot` *was* named while its concluding verdict was not. I absorb rows
I dispatched and rows that report while I am watching; a row that concludes during a pass spent
elsewhere has had nothing watching for it.

**And GAP was contradicting the gate.** An ENDGAME block read *"condition 1 MET, condition 2 MET,
condition 3 MET, only 4 and 5 outstanding"* against a five-condition framing retired long ago,
while `CHARTER_EVIDENCE.json` has read **0 of 3** throughout. It also called `of3t-ditref` *"row
still live"* and `of3t-fwdkcfg` pending; both concluded on 2026-09-21. Replaced with the live
reading. **Then I found the block was DUPLICATED** — a second ENDGAME carrying the identical stale
table further down the same field, so correcting one and stopping would have left the contradiction
standing in the copy I had not read. Removed, keeping its one unique paragraph. GAP went from
40,545 to 38,885 characters in the process, which is the smaller point.

**What this changes about the work, not just the record.** `of3t-ditref` and `of3t-fwdkcfg`
concluding means **D30/D58's ~20x tape amplification, D129 and D55's forward arm are UNOWNED** —
four USER-FACING defects whose closure plan named live owners that are not live. The plan's SET is
checked against the triage mechanically every compose; its per-item `closes_when` prose is not, so
it can name a concluded row and a refuted figure and still pass. Located-and-unowned is still
PARTIAL rather than NO-GO, but it is a dispatch I owe.

Also fixed: `MEMORY.md` had grown past its size limit and was silently dropping its last entry.
Ten index lines I had written over recent passes were 400-460 characters where the convention is
under 200; compressed, detail left in the topic files where it already lives.

## Pass 350 — a training forward crossed a precision boundary the wrong way, every loss looked sane, and the gradient came out eleven orders off

**`of3t-trainfwd` found D206 and fixed it.** A training forward that drives the diffusion modules
directly never passes through the typecast `OF3SampleDiffusion.__call__` does inline, so **fp32
weights met bf16 activations**. Nothing raised. The squared gradient norm read **4.87e+11** against
the model denominator **10.2796**.

**Every loss value looked sane**, and that is the defect rather than a detail of it: the loss is
built from activations that are each individually plausible, so the forward has no reason to
complain, and the damage only appears in the number nobody reads until the end of a training step.
A loss that looks right is not evidence the gradient is right. Fixed as a **shared method both
callers use** rather than a second inline copy, with `ttnn.typecast` given a tape entry so the
boundary is differentiable instead of a hole. Where two callers must cross the same boundary, the
boundary belongs to neither of them.

Worth keeping as a reflex: an arm reporting a squared gradient norm **eleven orders** off its own
denominator is a configuration fault to find, not a finding to write up. No model is that
inaccurate.

**And the dispatch I owed from last pass went out.** `of3t-tapeamp` takes the **~20x
forward-to-gradient amplification that belongs to the tape** — `diffusion` at forward 0.85 % /
gradient 16.6 % = **19.6x**, `msa_module` at 0.82 % / 16.2 % = **19.8x**. Different ops, different
track, the same factor to two significant figures, which is what moved it from a module property to
a tape property in the first place. **Four USER-FACING defects wait on it** (D30, D58, and through
them D129 and D55's forward arm) and all three rows that have owned it concluded without naming it.

The brief writes the spent ground in as spent, because the most expensive mistake available here is
re-running it: `of3t-ditref` concluded having re-priced the denominator without locating the
object, and `of3t-ditcot` concluded on STOP after finding the diffusion reference **cannot be built
in the checkpoint's architecture**, so a per-op attribution against the 0.5.0 reference names an op
for an architecture difference — which that row refused to do and this one must not undo.

**The first thing it is told to price is a lead that did not exist yesterday.** If the ~20x is a
D206-class precision-boundary artifact rather than an amplifier, that explains the
two-significant-figure agreement across unrelated tracks better than any per-op mechanism — because
a boundary crossed the same way in both tracks is the same boundary. It is cheap to test, and
ruling it out is worth the hour either way.

Three constraints carried from what the campaign has already paid for: **call census, not route
reading** (D196); the cotangent per block against the in-frame float64, which is what separates a
per-block injection from a per-block accumulation; and the A/A and wall-clock floors established
before any arm is read, as `of3t-shapekey` did. Plus one that is about the record rather than the
measurement — **say which of the four defects the result closes, narrows or leaves untouched, and
do not describe a narrowing as a closure**, since the closure plan has already carried a concluded
row as a live owner for days.

`of3t-tapeattn` and `of3t-refatom` are live and have not committed.

## Pass 351 — I went after the campaign's best result to see whether last pass's defect reached it, and it does not

Five rows live against a cap of five and none committed, so the pass went to the one thing that
could have quietly invalidated the largest positive claim this campaign has.

**The worry was real and specific.** Pass 349 absorbed `of3t-ditcot` with the sentence *"the
diffusion reference cannot be built in the checkpoint's architecture"*. If that applies to the
pinned float64 bundle, then `MODEL_shipped.json`'s **92.1568 % of the mass at 1.1031x upstream's
own bf16** — the "outside the pairformer trunk the gradient is at upstream's own accuracy" headline
— is scored against a denominator that cannot express the model, and the campaign's best number
goes with it. My own sentence was what made that reading available, so it was mine to close.

**Checked directly rather than argued.** On qb2,
`/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt` (sha256 `1d4ea9225f…`) holds **4,170**
tensors, **143** `layer_norm_z` entries, **28** under `diffusion_module`, spanning **24 distinct
`diffusion_transformer.blocks.N` indices**. It carries the per-block layout.

**So the two are different objects.** What ditcot measured is the module *the diffusion reference
BUILDERS construct*: loading `of3-p2-155k.pt` that way gives `missing_total 3` and
`unexpected_total 48`, all 48 unexpected keys `attention_pair_bias.layer_norm_z`, leaving exactly
**one** all-ones site where ours runs 24 trained tensors at mean 0.276–0.568. The pinned bundle is
not built that way. **The 1.1031x stands, now for a measured reason rather than by omission.**

**And the correction is mine.** "The diffusion reference" is not one object; pass 349 should have
written *the reference builder's module*. A defect scoped to a builder, read as scoped to every
reference, would have put the campaign's best result in doubt for no reason — the mirror image of
the flattering direction, and the same imprecision either way. Both the ledger entry and the GAP
cell now say which one.

Worth noting what made this checkable in minutes: the bundle is pinned by sha256 in the artifact
that uses it, so "which reference" was a file to open rather than a question to reason about.

`of3t-tapeattn`, `of3t-refatom`, `of3t-trainfwd` and `of3t-tapeamp` are all live; none committed
this pass.

**COVERAGE is MET — the campaign's first charter condition, and the criterion now reads 1 of 3.**
`of3t-trainfwd` concluded at `bdbe43a95`: coverage 9 of 11 to **11 of 11**, loss terms 8 of 8.

**It composed this pass and not at 348 because the precondition I set then is now satisfied.**
Inference is **byte-identical, verified twice** — after the adapter and again after the sampler
refactor — `ubq.cif` `6a8a43ca…` and `ubq_model_1.cif` `62d94b47…` matching base `16eac05c1`. I
said at pass 348 that composing a coverage win while "did you break inference" was outstanding is
the wrong order, and the order held.

**Composed upgrade-only, with the guard written to refuse rather than to win.**
`merge_coverage.py` takes `COVERAGE_TRAINFWD.json` as a third source pinned by sha256
(`0fd2c866bb8eb1e9`) and **stops the merge** if that row reports a path uncovered where
`of3t-covpaths` has it covered — two rows disagreeing about one path is a finding, not a merge.
Both upgraded entries carry `superseded: {was: false}` so the artifact records what changed.

**And the row refused the most attractive deliverable in the campaign, correctly.** Its P4 held:
there is still **no unstitched model-scope gradient** and D187 stays open, because the pinned
float64 reference was taken with upstream's own cotangent and replayed draws while this forward
seeds from our loss and draws its own noise. Scoring across that is exactly the cross-frame error
this campaign has already paid for once. It also reported **P5 refuted as a miss** — predicted over
600 s per step, measured **496.9 s** — and left one thing deliberately unshipped: the one-step
denoise arm runs with six of eight terms firing but **3 of 3400 parameter gradients come back
non-finite**, always `sampler.dc.w_lin_z/w_lin_s/w_lin_n`, with the seeds ruled out by measurement
(the seed norm is identical for a structure rotated 1.1 rad and translated 12 Å). Unowned, and it
looks like `of3t-tapeamp`'s neighbourhood.

## Pass 352 — the trap-check ran before the spend, so 581 is the honest 581

**`of3t-refatom` concluded and TRAJECTORY's coupled scope is 88.83498302148425 %**, up from
88.08194359237523, the projected 0.75304 points confirmed to **5.7e-07**, at 1620.85 s — half the
estimate, because the 3335.86 s figure was taken on a shared board pair.

**D203 said a silent failure would score 581 tensors instead of 573 with eight of them frozen, and
the scope percentage would rise while the trajectory got less honest. The row scored 581.** What
separates the two readings is not the total — it is assertions 1 to 3, and they were run *before*
the 0.93 h and committed before they ran (`26b7fb834`, results at `717bf1d20`): eight of eight in
the parameter set, eight of eight participating, `tape_resolves_after_step` **988 = 980 + 8** with
`of_walked` and `rebound` both 988. **4.84 s of taped compute, about 0.3 % of what it protects**,
and driving the real arm's program rather than a replica.

**Assertion 4 failed and the bar was not moved.** `plm` norm ratio 0.9988837 against a 1.0e-3
tolerance, 12 % over, with cosine at 1 − 1.5e-7 over 917,504 elements. The row's own proposed
mechanism died to its own control in seconds — truncation overstates the device's bias 4.2x on both
legs and nearest-rounding gets the sign wrong — and what survived was bracketing: the device sits
between two **width** variants of one function, 1.8x finer than nearest bf16 and 4.2x coarser than
fp32. That answers "place moved, function did not" without re-scoring anything.

**And then it flagged the thing that would have been easiest to keep quiet (D207).** The widening
was supposed to be additive. It was not: the **573 shared** tensors moved too, `rel_d` at k=20
reading **2.246887e-01** against the shipped arm's **2.564253e-01**, worst per-tensor 3.454769e-01
against 8.110072e-01 on a *different* tensor. **In the flattering direction, and unexplained** — so
the row reported it as unexplained rather than as part of the win. A scope gain carrying an
unattributed accuracy change is two results presented as one, and nothing about the direction would
have prompted the question. Separating it needs the host-leg arm re-scored over the same 581 names;
not run, not owned.

Its own durable lesson, which I am adopting as a rule rather than a note: **a change that ADDS
tensors to a scored set must be shown purely additive on the tensors it did not add.**

**The two live rows both advanced without concluding.** `of3t-tapeattn`'s verb census at padded 384
reads the same chain and the same counts as at 64 — 48 firings each per backward on the single
track, 624 captured nodes at both widths, no capture errors — so **`AttentionPairBias` does run on
the tape and nowhere else**, and its structure does not change with width, which constrains D191's
per-block factor to arithmetic rather than a different program. Its instrument note is worth
keeping too: `cmp_cot.py` compares by PAYLOAD rather than file sha256, because the file carries the
arm's report beside the tensors, so a file hash answers *"same run"* and not *"same numbers"* — at
384 the two file hashes differ while all 98 tensors are bit-identical. `of3t-tapeamp` is building
the arm that measures upstream's own forward-to-gradient ratio, which is the control its object has
never had.

**`of3t-tapeattn` concluded mid-pass and the pass-349 ratchet caught it on its first real test** —
the compose refused with *"1 concluded of3t rows are named nowhere in the DEFECTS union, against a
ratchet of 0"*. That is D204's guard doing exactly the job it was built for, three passes after a
row's STOP verdict had sat unabsorbed for a hundred.

**And what it concluded refutes the premise I wrote into its brief.** D191's width growth is **not**
a per-block factor on the single track's cotangent. It is a **STEP created in the backward of
blocks 45 and 44**, then carried and diluted: rung 46 reads 1.042x, rung 45 **1.858x**, rung 44
**2.201x** — against D191's model-scope **2.1795x** — and rung 43's 2.275x decays monotonically to
1.724x at rung 0. **A per-block factor would compound.** All of D191 is present four blocks in and
the other 44 add nothing.

**100 % of it is ours**, measured: the float64 reference is bit-identical between crop 64 and padded
384 on the 56 real tokens, `rel_l2` exactly 0.0 at all 49 rungs, both tracks.

**The softmax candidate fell to a decomposition rather than to a smaller number**, which is the part
worth copying. `VERBS.json`'s pooled **1.7797x** is a reference-norm-weighted RMS, so it moves when
the mass moves — holding weights at 64 and taking 384's errors gives **0.9369x**, holding errors at
64 and taking 384's weights gives **2.0923x**. **Softmax is 6.3 % better per firing at 384**, only
10 of 48 firings worse, `matmul` and `multiply_` flat. The campaign's most-chased candidate was
carrying someone else's mass.

The row also bounded itself twice without being asked: the step is **not attributed to a named op**,
and the census reads only the single track (heads = 16) so **the pair track and the triangle
multiplications run in the same blocks 45/44 and stay live**; and the ladder is **saturated** at
O(1) relative error from rung 47 down, which bounds what it can localise at all.

**`of3t-blk4544` is dispatched onto naming the op**, with the cheap arm the row already wrote as
deliverable two — pin the single track's softmax backward to its float64 VJP and re-score,
`tapecensus.py` computes that VJP — then the two candidates the census could not see. Its brief
carries the saturation limit as a constraint rather than a footnote, and one instruction earned the
hard way: **do not present a mechanism as a repair**, three have been named and refuted for this
object already.

## Pass 353 — the ~20x four user-facing defects were waiting on is the function's, not the tape's, and our arm beats upstream on both halves

`of3t-tapeamp` built the control the object had never had in three rows of ownership: **upstream's
own forward-to-gradient ratio**, same boundary, same float64 reference, same 547 tensors.

    upstream 0.4.3 bf16      7.666x        ours   11.026x
    upstream 0.4.3 fp32      9.326x        and four orders lower in ABSOLUTE error

**Our arm beats upstream's own on both halves** — forward by **1.959x**, gradient by **1.362x** —
and the arithmetic closes: **1.959 / 1.362 = 1.438**, which is exactly the ratio excess. Our factor
is the larger one *because* the denominator is the half we beat it on hardest. A forward-to-gradient
ratio is a quotient of two accuracies, and improving the numerator less than the denominator raises
it. That is why two unrelated tracks agreed to two significant figures: same property of the same
function, not a shared tape component injecting it. **D58 as filed is refuted.**

**Three precisions span 7.7x to 11.0x, and the span is the discriminator.** A dtype boundary cannot
survive a four-order change in absolute error; the conditioning of the Jacobian-transpose product
can, and `of3t-bwdaccum` already measured that on the trunk.

**My own lead, the one I told the row to price first, is dead — by call count.** 1,879 node firings
and **zero dtype reconciliations**. The tape has exactly one place a cotangent's dtype is reconciled
to its forward value's, and on the diffusion scope it never fires: the whole backward runs fp32
against fp32. The only crossings are 96 calls of the model's own explicit typecast verb, which is
taped and differentiated. **There is no D206-class boundary in the shipped diffusion backward**, and
that is a census rather than a route read — which is the standard I put in the brief and it was
turned on my own hypothesis.

**The control that makes the ratio legible is one nobody had run:** rolling the cotangent moves the
gradient **10.058x** and leaves the forward **bit-identical**. A ratio needs a control that moves
one half and not the other, and until this pass the campaign had been reading a ratio without one.
Determinism floor exactly 0 across forward values, gradient median, mass-weighted gradient and the
ratio itself; wall-clock floor 54 % at load 9-13, with **no claim resting on a timing**; D141's
fingerprint guard armed and passing at 761 parameters, 24 per-block `layer_norm_z`, 0 unexpected, 0
missing — so this is not `of3t-ditcot`'s architecture case.

**I am not closing D30 or D58 on this.** The row is live and its gate owes the `DEFECTS:` field
saying which of the four it closes, narrows or leaves untouched. Closing a USER-FACING defect on my
own reading of a live row's commit message is the closure-plan failure this campaign already had,
pointing the other way.

Also this pass: `MEMORY.md` had two index entries for one lesson — the pooled-ratio decomposition —
because the row that found it wrote its own memory seven minutes before I wrote mine. Kept the
row's, which is the fuller one, and deleted mine.

**`of3t-tapeamp` concluded and closed two USER-FACING defects — the first the campaign has
recorded.** Its `DEFECTS:` field states all four in the words the brief asked for, with its own
reason: *"the campaign's closure plan has already carried a concluded row as a live owner for days
and a narrowing described as a closure is how that happens."*

  * **D30 CLOSED as not a defect.** Filed at 19.6x, corrected to 14.75x by `of3t-tapediverge` and to
    **11.026x** by `of3t-ditref`'s repaired denominator; this row prices **7.666x of that inside
    upstream's own bf16 recipe** with the remaining 1.438x arithmetic. Our gradient is **0.734x**
    upstream's own on 547 tensors and **0.5865x** mass-weighted on 761 — *below the reference's own
    floor*. The tail is explicitly **not** closed: a worst tensor at 1.850397e+01 is a different
    object and is unowned.
  * **D129 CLOSED** — already dissolved by `of3t-ditref`, and the row notes my brief was wrong to
    list it as waiting on this one.
  * **D58 narrowed to one leg**: the "belongs to the tape" half is refuted on the diffusion track,
    `msa_module` is unmeasured, and there is no upstream bf16 arm for that boundary anywhere in the
    campaign.
  * **D55's forward arm untouched**, said in those words.

**And I checked the closure's arithmetic rather than accepting it, which caught one thing.** The
0.536x quoted for D129's leaf **mixes statistics** — it divides the MEDIAN numerator 0.0854492105 by
the MASS-WEIGHTED floor 1.5931532097e-01. The like-for-like readings in the same artifact are
**0.6837x median** (which `VS_FLOOR.json` records as `ratio_median`, one field away) and **0.7652x
mass-weighted**. The verdict is unchanged — both are under 1 — but the circulating figure
understates by about 1.3x in the flattering direction, so the ledger carries the like-for-like
pair.

**Two self-inflicted mechanics this pass, both caught by guards I built.** A `### D30 and D58
UPDATE` heading did not parse, because the parser takes one D-number per heading — split. And a
`**NARROWED, not closed**` heading stored **CLOSED** off the lower-case prose, which is the
status-parser trap for at least the third time in this campaign and the second time on a heading I
wrote in the same pass I was warning about it. Reworded to carry `UNFIXED` in capitals and no
status word in prose.

## Pass 354 — the campaign spent four rows on a ratio and never measured the reference's own

`of3t-tapeamp` concluded, and its full reading is larger than the commit I absorbed at pass 353.

**The figure was two revisions stale in my own record.** The "~20x" is **11.026x** (diffusion) and
**10.903x** (`msa_module`) at the repaired denominator — `of3t-ditref` and `of3t-tapediverge` had
already moved it, and **both the brief I wrote and `UNFIXED_TRIAGE.json`'s D58 entry were still
quoting the old number**. The triage entry is corrected this pass; it had been asserting the tape
claim that the same row refuted.

**The measurement nobody had made was upstream's own FORWARD accuracy.** Its gradient was on record
and its forward *seconds* were; its forward *accuracy* was in no artifact in the campaign. One
process, one host, both halves:

    ours, device             forward 8.4748009e-03   gradient 9.3442464e-02   11.026x
    upstream 0.4.3 bf16      forward 1.6601749e-02   gradient 1.2727639e-01    7.666x
    upstream 0.4.3 fp32      forward 1.2939393e-06   gradient 1.2067747e-05    9.326x

**69.53 % of our factor is upstream's own**, and to reach upstream's "better" 7.666x we would have
to make our forward **1.96x worse**. Four rows, and the decisive number was one measurement away the
whole time.

**A third refutation of the dtype story, and it is the cheapest of the three.** Beyond the call
census (0 dtype reconciliations in 1,879 firings) and the factor surviving fp32: **the per-block
cotangent curve has the same shape in both precisions** — same five-block ramp, an 8.191x / 5.726x
step at the *same* boundary 19→18, same plateau — while the absolute errors sit **11,792x to
18,350x** apart. A shape that is precision-independent cannot be a precision boundary, and that
costs one curve rather than a per-op ablation.

**And the shape answers `of3t-bwdaccum`'s discriminator with neither of its two options** — not flat
at the bf16 floor, not monotone with depth, but **ramp / step / saturation**, that discriminator's
*third* pre-registered shape. The step moves relative error 8.191x at 1.399x magnitude, so **it is a
cancellation event**, not an amplification. Set beside `of3t-tapeattn`'s trunk step at blocks 45→44
from pass 352, there are now two precision-independent cancellation-shaped steps in different
scopes. Whether that is one phenomenon is not established and I am not claiming it.

The row also caught a fleet-mechanics trap in its own doc before concluding: it had written
*"`git diff origin/wk/of3t -- tt_bio/` is empty"*, which stopped being true when `wk/of3t` merged its
branch mid-pass and a sibling's 19-line change landed on top. **A `git diff` against a moving branch
tip attributes a sibling's edit to you** — state tree claims against the merge-base.

## Pass 355 — TRAJECTORY's bar was the static instrument's ceiling handed to a different instrument, and it is repointed

`of3t-trajwiden` left a proposed clause two passes ago and I had not read it. It is the most careful
piece of reasoning any row has handed me, and it argues for **lowering a charter bar by 10.05
points** — which is the direction that has to be earned.

**The argument.** 99.2594 % is the ceiling on what the **STATIC single-step** instrument could ever
measure: 100 minus the 0.74055 % of gradient mass on a weight the shipped path applies on the host,
where no device gradient exists. A trajectory needs four things where the static instrument needs
one — a taped device forward **and** backward at scope, a captured 0.4.3 boundary whose cotangent is
complete, upstream's module runnable standalone at arbitrary weights for 20 steps, and 20 affordable
optimizer steps on both sides. **Two instruments of different reach were given the same number, so
the weaker one was unsatisfiable.** That is D181 read from the other side: the rule forbids coupled
clauses reading different scopes, and this was one number read by two scopes.

**What I checked before making the change, because a bar that moves toward passing is exactly where
this campaign has gone wrong:**

    1  does it let us declare success?     NO -- 88.83498302148425 against 89.2106 misses by 0.376,
                                           and `scope.coupled` fails outright, unemitted
    2  is the bar OURS or the reference's?  the reference's -- 89.2106 % is the share of upstream's
                                           own float64 gradient mass inside diffusion_module; it
                                           would read the same if our port did not exist
    3  is anything lost by the swap?        no -- pass 346 declined this repoint suspecting the
                                           `clauses` block would be dropped; the `moves` check
                                           reads `per_step`, and the new artifact carries all 20
                                           entries with both norms. Checked, not assumed
    4  what would raise it?                 a capture spanning more than one section AND a taped
                                           whole-model forward+backward; the reference half is
                                           digest-pinned already, the device half does not exist

**And the row killed its own better-looking number to get here.** Its pass-1 proposal was 97.9849 %,
built by adding the sections whose device arms exist. It then tested that addition and refused it:
every boundary the campaign holds is a frozen capture of upstream's r = 0 step, so an `aux_heads`
trajectory reads upstream's step-0 trunk outputs at every k and never our step-k ones. **A union of
per-section runs tests each update rule in isolation and none of the coupling, and the two are
indistinguishable in a `pct_of_model_sq_grad_norm` field.** Hence the `scope.coupled: true`
requirement, which nothing emits yet — so the repoint makes the clause *stricter in kind* while
lower in number.

TRAJECTORY now reads **88.835 against 89.2106**, unmet by 0.376 points, with a second clause unmet
for a named and fixable reason. The charter stays **1 of 3**.

**One mechanical note worth keeping.** I first wrote the new clause with an `==` operator that
`charter_evidence.py` does not have. Its break control raised `KeyError: '=='` and **refused to
publish** rather than emitting an artifact with a clause it could not evaluate. The vocabulary
already had `is`, doing strict equality with a type check. A guard that refuses on an operator it
cannot evaluate, instead of skipping the clause, is the right failure and it cost one compose.

## Pass 356 — the bar I set last pass could not be met by a perfect artifact, and the one row that can close TRAJECTORY is out

**The bar was unreachable by construction.** Pass 355 repointed TRAJECTORY onto the diffusion
module's own share of upstream's float64 gradient mass and wrote it as `>= 89.2106`. The share is

    9.170528876862544 / 10.279642678524981  =  89.2105802084 %

so the clause demanded more than 100 % of the module it is a property of. An artifact that scored
every one of the 761 reference tensors at that boundary would have read 89.2105802084 and **failed
by 0.0000198 points**. The arithmetic was right; the transcription rounded a CEILING up. Corrected
to **89.2105** (rounded down) in `workstreams/_of3t_donecheck.py`, filed as **D209**.

The correction moves toward passing, so it was checked against the rule for that direction before
it was made: **it flips no verdict.** Today's best artifact reads 88.83498302148425 and misses the
corrected bar by the same 0.3756 points it missed the impossible one by, and the sibling clause
`scope.coupled` is still unemitted. Two substantive failures before, two after.

**How it was found, and why nothing found it.** Not by a guard. I was sizing a row against the
remaining gap and computed what FULL coverage of the module would score — which is the one
arithmetic that exposes it. No composition check tests a bar for reachability and
`audit_evidence.py` cannot: a bar's ceiling is not in any artifact it reads. The general rule, now
standing: **a bar that is a measured CEILING must be rounded DOWN; a floor may be rounded up.** The
direction that is safe for a target is unsafe for a limit, and the two are easy to confuse because
both read as *be at least this good*. This is D201 arriving from the other side — there, two
clauses of one guard were jointly impossible at their limit; here, one clause was impossible
against its own limit. Both were found only by asking what the best possible artifact would score.

**TRAJECTORY's whole remaining miss is now one row.** The 0.3756-point shortfall is not distributed
across the trajectory; it is exactly three families of unscored tensors:

    diffusion_transformer.blocks.N.*          96   0.27152646887360954 %
    atom_attn_enc.atom_transformer.blocks.N.* 42   0.07103250562507443 %
    atom_attn_dec.atom_transformer.blocks.N.* 42   0.03303821241802195 %
                                             ---   -------------------
                                             180   0.3755971869 %

88.8349830215 + 0.3755971869 = 89.2105802084. Cover all 180 and the clause passes with 0.00008
points to spare; cover 179 and it does not. **Pass 356 dispatches `of3t-trajfull`** onto that and
onto the unemitted `scope.coupled`. Namespace `perf/of3t_trajfull/`, `CONTINUES_FROM: wk/of3t` so
the DISPATCHER resolves its base instead of the row spending ten minutes detecting it (D188), gate
entry, stage hint and TASKS ws-tag added. Two rows live against a cap of five, deliberately.

**Sized before the brief was written, not after.** I read the bank on qb2 rather than assuming it:
`theirs/k20.npz` holds **761** tensors and `refatom/k20.npz` holds **581**, theirs-not-in-ours 180,
ours-not-in-theirs 0. So the reference side is complete and must not be re-run — and **our side
never dumped those 180**, which means this is not the pure rescoring job it resembles. That one
check is the difference between a row that starts with a census and a row that discovers the same
fact an hour in. (Read with stdlib `zipfile.namelist()`; qb2's system python has no numpy, and the
`.tenstorrent-venv` does not either.)

**The census is the real first deliverable**, because 856 of the device arm's 988 taped parameters
carry a checkpoint name while only 581 were dumped. Each of the 180 is one of three things:
present-but-never-dumped, FUSED in our tree where the reference splits it (a scoring-side split,
explicitly **not** a model change), or genuinely absent from the device parameter set. The brief
says plainly that the third case means 89.2105 is **unreachable at the only boundary we hold**, and
that reporting it as such is an acceptable outcome — a charter condition that cannot be met is a
finding, not a failure to engineer around.

**The failure mode here reads as progress**, which is why the controls are pre-registered rather
than left to the row. A wrong name map or a wrong fused split scores MORE tensors, so the
percentage goes UP and nothing complains. Committed before any scored number: k=1 bit-identity per
NEW tensor (lr is 0.0 at the AF3 warmup rung, so refatom reads `tensors_bit_identical: 581` there
and every added tensor must match the reference exactly — one that does not means the map is wrong,
not that our port differs), a bit-exact split round-trip, and added mass equal to 0.3755971869 %.
`of3t-refatom` was handed the same trap and it is the reason that row ran a discovery check first.

**On `scope.coupled`:** the property already HOLDS in `traj_refatom.json` — `our_step_log` and
`their_step_log` carry different per-rung gradient norms (ours 0.7157321105277967 at k=1, theirs
1.056923747062683), so each side stepped its own weights from its own gradients. What is missing is
the assertion. The brief requires it be emitted as a CHECKED fact and says a hardcoded `true` will
not be accepted. I considered dropping the clause as a token requirement and did not: dropping it
moves TRAJECTORY toward passing, and it is the only guard against a UNION of per-section
trajectories satisfying a scope number while testing none of the coupling.

Ledger: 209 defects, **76 UNFIXED** — 4 scope-excluded, 8 USER-FACING, 64 campaign-internal. D209
is campaign-internal: it gated our own exit condition, not anything a user runs. Filed against
myself, one pass of exposure. Charter unchanged at **1 of 3** — COVERAGE MET, GRADIENTS and
TRAJECTORY NOT MET.

## Pass 357 — TRAJECTORY is MET, and the reason GRADIENTS is not is one section

`of3t-trajfull` concluded at 08:35Z. Its artifact scores **761 of 761** reference tensors at the
diffusion_module boundary for **89.21058020840096 %** of the model's squared gradient norm, against
the bar of 89.2105 that pass 356 corrected downward (D209). That is the module's whole share, so at
this boundary the clause is now exactly *score every reference tensor*, and
`scope.tensors_in_reference_not_scored` is empty.

**The repoint flips a verdict, so I checked the artifact rather than the row's report.**

    recomputed from its own two sums   9.170528876862544 / 10.279642678524981 = 89.21058020840096
    `moves` clause, read off per_step  19 of 20 rungs non-zero on BOTH sides; zero rungs where
                                       theirs moves and ours does not; k=1 zero on both, which is
                                       D169's AF3 warmup no-op and what the clause was rewritten
                                       at pass 319 to accept
    model change?                      NONE. The row's three commits (92804cd30, eeea50684,
                                       649ab5d98) touch perf/of3t_trajfull/ and
                                       perf/of3t_trajwide/trajwide.py and nothing under tt_bio/ --
                                       read per-commit, not as a diff against a moving branch tip
    shared instrument                  `--namemap structural` and `--census-out` default off, and
                                       the resolver RAISES rather than narrowing when a slot does
                                       not resolve
    can a wrong map inflate it?        No, and this is the clause's best property. A wrong name map
                                       compares our tensor A against their tensor B, which makes
                                       rel_d WORSE; the added 0.3755971869 % is a property of the
                                       REFERENCE's gradient mass for those 180 names and does not
                                       move with our map at all

The row's own three pre-registered controls agree and were committed before the arm: 180 of 180 new
tensors bit-identical to the reference at `w_0`, 48 of 48 fused splits round-tripping bit-exactly at
every rung, added mass within **1.67e-11** of the figure written down in advance. `scope.coupled` is
emitted from six computed checks, the decisive one being that at k=2 Adam's first update is
`-lr*sign(g)` elementwise, so the two sides' displacement signs agree on **93.96 %** of 201,279,424
elements where a side advanced from the other's gradient would agree on 100 %.

**Charter: 2 of 3.** COVERAGE (pass 351) and TRAJECTORY (this pass) MET, GRADIENTS not.

**Why GRADIENTS is not, stated correctly for the first time (D211).** `PROVES:` has been saying it
*"fails on coverage alone"*, and that sentence describes the 92.1568 % artifact while the gate
grades the 97.98499 % one. Same float64 reference by sha256, same denominator, 3643 tensors instead
of 907 — the difference is the trunk — and there the shipped arm reads **0.520124** against its own
A26 bar **0.147353**. **Raising coverage is what breaks accuracy.** Computed this pass from the
graded artifact's own `per_section` block: `pairformer_stack` carries 5.8282 % of the mass at
**6.402x** upstream's own bf16, the other nine sections sit between 0.945x and 2.176x, and holding
the nine while putting the trunk at upstream's own level moves model scope from **0.5010 to
0.1240** — 0.841x the bar, passing. The trunk is not part of GRADIENTS' accuracy miss; it is all of
it.

That has a live owner, which is why this pass dispatched nothing: `of3t-blk4544` is on the step
created in the backward of blocks 45 and 44.

**Also this pass.** The D149 ratchet caught `perf/of3t_trajfull/trajfull_score.py` — a loop putting
two package trees on `sys.path` at a fixed index with nothing reading the resolution back. The
recomputed tree digests the row published answer *are these trees the same content*, which is a
different question from *which tree did this process import*, and the script then calls
`torch.load(..., weights_only=False)` on upstream's checkpoint, an unpickle that imports whatever
`openfold3.*` classes the pickle names. Repaired to read `openfold3.__file__` back and record all
three outcomes, including "not importable", because nothing can have come from the wrong tree if
nothing came from any tree. The ratchet is empty again. **The published artifact predates the
check**, so its tree evidence remains the recomputed digest pair and I am not claiming otherwise.

**D210, and it is the finding of the pass that nothing graded would have caught.** `of3t-trajfull`
reported, outside its own scored set, that our DiT's fused `qkv_w` pads head_dim 48 → 64 and **the
pad lanes are in the optimizer's parameter set**. Exactly 0.0 at `w_0`, up to **3.494e-04** by
k = 20. Adam's scale invariance is the mechanism: a numerically tiny device-backward gradient in a
lane that should have none still takes a full lr-sized step. **14.2M elements that upstream does not
have.** It moves no number the campaign quotes, because the pad columns are outside the reference's
parameter space and are sliced off before v is used — which is precisely why it could sit unnoticed.
Filed USER-FACING and unowned: masking them out of the optimizer is a model change on the shared
diffusion path and owes an inference A/B.

### Pass 357, second finding — the trajectory's MAGNITUDE was one join away and it reads 1.2707x

`DOESNOT:` was about to say the trajectory has no magnitude reading, which would have been the
fourth time this campaign recorded "no bar exists" for a number that was already sitting in an
artifact. `of3t-trajbar`'s `bar_bf16mixed_all` is upstream's own bf16-mixed twenty-step loop scored
against upstream's own float64 loop at **761 tensors, 89.21058020840096 %** — the identical scope
`of3t-trajfull` just reached, which is why the join became possible only this pass.

    k    ours rel_d    their own bf16    ratio
    2      0.430993          0.390393   1.1040
    5      0.391539          0.305162   1.2831
    10     0.325687          0.238187   1.3674   <- worst
    15     0.262649          0.199794   1.3146
    20     0.227502          0.179031   1.2707

`d_theirs_norm` is **bit-identical at all twenty rungs** between the two artifacts, so this is one
reference trajectory and two arms, not two experiments compared by eye. Same boundary sha256
(`ea80f18e651dae517e...`), same checkpoint sha256 (`af09eac4f29cef856...`), same `w0_baseline: own`.

**I did not turn it into a clause.** The number existed before any bar for it could be written, and
PROTOCOL's first rule is that a tolerance chosen after the result is not evidence. That the clause
would FAIL at a factor of 1.0 is what makes declining it credible rather than convenient: the
flattering move here would have been to add the clause at 1.4 and collect a fourth MET property.
PROTOCOL §9 is where the amendment goes, with "a number already existed" recorded against it.

**The lesson, and it is the same one four rows have now paid for.** A bar and a reading are
comparable only at a matched scope, and the scope is the thing that moves. This bar was built at
pass 307 and has been unusable since, not because it was wrong but because nothing reached its
scope until pass 357 — and no guard reports "an artifact you already hold is now joinable". The
check that would have caught it is cheap: when a row widens a scope, grep the other artifacts for
that same scope number.

### Pass 357, third finding — I overwrote a concluded row's brief and the duplicate check I had already run could not see it

Dispatching the trunk row I wrote `cat > workstreams/of3t-trunkfwd.txt`. **That slug was taken**:
`of3t-trunkfwd` concluded 2026-09-20 with a state doc, a marker and a gate entry, on a different
object (*does the trunk's 28 % forward disagreement reach shipped inference?*). The redirect
destroyed its brief, and the only reason it was caught is that the gate answered
`DONE_CHECK: concluded -- of3t-trunkfwd` for a row I had just invented. Restored with
`git checkout HEAD --`; `.coworker` being a git repo is what made that free. The duplicate gate
entry and the stage hint were unwound, and the row is dispatched as **`of3t-trunkact`**.

**The reason my own check missed it is the part worth keeping.** I had run a census minutes
earlier — briefs against markers, both directions — and it reported clean. It scanned *markers with
no brief* and *briefs with no marker*, and `of3t-trunkfwd` had **both**, so it was invisible to a
consistency check by construction. A consistency check cannot answer *is this name free*; only
looking at the path can. **Before `cat >` on a path the fleet keys on, test that the path does not
exist.** `duplicate-check-grep-state-verdicts-not-just-task-names` is the same lesson from the
other side: there the duplicate was a task doing work already done, here it was a filename.

The near-miss also says something about the redirect itself: a concluded row's brief is the only
record of what it was asked to do, and the campaign quotes those briefs. Overwriting one silently
rewrites history in a file nothing audits.

---

## 2026-09-22 11:38 UTC — the park was wrong, and so was the number behind it

This row was parked at 11:00 for "PATHOLOGICAL COST -- $172.81/pass, 32 passes = $5,530, 90% of
the day". **None of that was real.** `total_cost_usd` from `claude -p` is the cost of the whole
SESSION so far, not of the turn, and `worker_reply.py` charged the field verbatim on every turn —
so a 34-turn launch was billed for turns 1..n all over again at every n.

The launch's own record settles it: the session transcript carries 32 `cost-state` entries whose
`totalCostUSD` series is *exactly* the ledger series (17.94, 28.16, 38.66 … 347.95), strictly
increasing, ending at **$347.95 for the entire launch**. The ledger summed those 34 cumulative
figures to $5,537.65 — a **15.9x over-count**. Fleet-wide today: $6,140 reported, ~$771 real.

Real per-turn cost was ~$10.9, which is high-ish for a doc-heavy orchestrator and entirely normal
for the fleet. The park is cleared. `worker_reply.py` now charges the delta, and today's ledger
carries a −$5,369 correction line.

Two things that were nonetheless worth doing and stay done: `state/of3t/` went 640 KB → 424 KB
(DEFECTS and LEDGER rotated, PROTOCOL deliberately exempt — it is authored, not appended), and
`maxit` stays at 40.


---

## Pass 358 — a bar-repoint I nearly took, refused because the route it was priced on is not the route the code takes

`of3t-covdefault` concluded 11:46Z on **NO-GO: do not flip the default**, and left me one decision:
repoint GRADIENTS' coverage clause from **99.2594** to **97.9849**, or read it UNMET and dispatch
the wiring. **Refused, and D212 records why.**

The row's NO-GO half is right and stands. `TT_BIO_OF3_DEVICE_REFATOM` ON costs **+198.476 ms cold /
+50.231 ms warm** on openfold3 and **+192.515 / +48.185** on openbind at 1350 MHz with an A/A floor
of 0.0 A, and it moves the fold digest. Against the 2026-09-21 hard constraint that is a refusal
twice over.

Its BAR half reasoned that *"every route that reaches it requires the flag ON... The clause can
only be satisfied by breaking the hard constraint it sits beside."* **I read the code instead of
the prose**, and the flag is not on any route:

    tt_bio/train/openfold3.py:332       the training adapter calls the HOST `ref_atom_embed`
                                        unconditionally. It never reads the flag.
    tt_bio/train/openfold3.py:324-332   and its host prep runs ABOVE `with ag.tape():` at line
                                        362, so even flag-ON yields no cotangent.
    perf/of3t_diffusion/device_gradient.py:586-605
                                        the arm that DID reach the eight built
                                        `RefAtomFeatureEmbedder` directly and called
                                        `ag.parameter()`. No env var, no shipped code.

`tt_bio/train/` is training-only (`main.py:1694` lazy-maps the `finetune` CLI entry;
`autograd.py:1116` imports `train.losses` lazily inside a tape closure), so a change confined to it
**cannot reach inference by construction** — the gating principle D137 established for the float64
softmax. From the row's own `COVERAGE_CEILING.json`: 97.98499306866148 **+0.7530394291090192**
(the eight diffusion ref-atom linears) **+0.7405050029283714** (`linear_q.0.weight`) =
**99.47853750069887**, clearing 99.2594. Dispatched as **`of3t-refcov`**.

Also dispatched: **`of3t-vjpln`**, `of3t-blk4544`'s named next arm — `_ref_layer_norm` and
`_ref_linear` as float64 VJPs, then `allref2` over blocks 45 and 44, on the falsifier that row
pre-registered (R44 ≤ 1.33 names the carrier, ≥ 1.90 refutes it). Its option 2, the forward
activations, is already `of3t-trunkact`'s and the brief says so.

**Checked before claiming:** `CHARTER_EVIDENCE.json`, `_of3t_donecheck.py`, `PROTOCOL.md` and
`UNFIXED_TRIAGE.json` all still carry 99.2594, so nothing had adopted the repoint and there was
nothing to unwind. Both new slugs were tested for an existing brief, marker and state doc before
`cat >` (pass 357's third finding), all three free. Gate entries and `_STAGE_HINTS` added for both
— the gate's own warning caught the missing hints, which is the check working.

**The lesson.** The repoint was the flattering direction and it was argued from a real,
well-measured number, which is exactly what made it credible; `of3t-trajwiden` killed its own
97.9849 % bar on this same ground and the campaign still nearly took it a second time. **When a row
prices "the only route" to a bar, check that the route it priced is the route the code takes.** A
cost measured on the wrong route is a true number about the wrong thing.
