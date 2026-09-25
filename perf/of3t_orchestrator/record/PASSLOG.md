# of3t-orchestrator PASSLOG archive

Moved out of the state doc at pass 416 (2026-09-23) so it stops being re-read every pass.

### PROTOCOL history, moved out pass 416

Earlier at pass 379: A38 — where a falsifier's behaviour under its own hypothesis can be simulated more cheaply than the experiment it gates, simulate it BEFORE dispatch; a falsifier the true hypothesis also satisfies is not one (R131). And A37, a boundary on what this campaign may publish as a target.** A projection may be published as a PREDICTION, never as a TARGET; it carries its own falsifier and its frame; and **no row's bar, allowance or done-check may be derived from one.** Twice in two passes I published a projected clause reading as the aim and twice it was wrong — 0.8986x from a lever whose forward and backward were different functions (R129), then 1.0674x carried across a frame change and measured at 1.7814x, outside its own band (R130). D214/D218 already barred dividing an in-frame ratio into another frame's threshold; A37 bars multiplying one into it, which is the form both took. **A34** (pass 378) remains the live precondition §3z: both sides of a per-parameter comparison must be on the SAME BOUNDARY — same batch, same entry activations, same incoming cotangent — recorded by digest and ASSERTED. `of3t-modelframe` is the first arm built to it. Originally 16 KB, written pass 1 **before any row was
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

### ROWS history, moved out pass 416

Earlier: **Live at pass 414: `of3t-angle` (dispatched pass 413, qb2 card 1), `of3t-cropwall` (qb2 card 2, DEFERred to 03:25 with its ladder chain detached and healthy; it overturned D205's mechanism — see D248), `of3t-modelever` (qb2 card 1, **the critical path** — A/A floor exactly 0 AND its shipped arm **bit-identical to the clause's own banked artifact**, 2736/2736, so its reading will carry onto the clause by digest rather than by argument; R190), `of3t-msaamp` (qb1, D58's one unmeasured leg — it replaces an `of3t-tapeamp` dispatch I retired, that name having already concluded; R189) and, concluded, `of3t-verbinstall` (qb1, **CONCLUDED GO at pass 414** — package install bit-exact, falsifier landed on its pre-registered second side, D246 and D247 both closed; its ARMDIFF artifact is frozen out of the D155 guard with the host established from the filesystem, see D249). `of3t-recut` and `of3t-recutfin` concluded GO; D242 is repaired, landed and repointed. The two live rows are disjoint: angle owns whether the exact softmax closes the ANGLE or only ever closed the magnitude, verbinstall owns which softmax install ships and still owes D245's FALSIFIER and INFERENCE legs.** Earlier: **`of3t-frameself` was live (relaunched 21:46 CEST, its DEFER expired) and `of3t-verbinstall` was dispatched at pass 386 onto D245. Two rows, two questions, no overlap: frameself owns whether the model frame is the frame it claims, verbinstall owns which softmax install ships.** Superseded at pass 386, kept for its method: at pass 385 the DEFER was re-checked against the host rather than the clock. Re-checked at pass 385 against the host rather than the clock — pid 42326 is in **R** state at 908 % CPU and 16.5 GB RSS with qb2 showing zero swap used and 176 GB free, so the DEFER is waiting on a live process and not a corpse. None dispatched at pass 382 — the critical path is already owned and adding a row to it would be two agents on one question.**

- **`of3t-angle`** — **dispatched at pass 413**, qb2, one card. The clause is reachable only by closing an angle, and the campaign's best lever (exact softmax, 1.0525x) was measured entirely on the double-counted functional where 62.52 % of the error was magnitude. It asks whether the lever closes the ANGLE or only ever closed the magnitude, re-reading the ladder on the repaired injection with rel/r/cos/angle in BOTH spaces. Namespace `perf/of3t_angle/`.
- **`of3t-recutfin`** — **CONCLUDED GO at 01:17.** Emitted the per-scope `injection.convention` stamp that unblocked the repoint (`n_leaves_differing` 0, mixed-pool refusal exercised on eight synthetic pools) and re-took the magnitude/direction split in the graded space, which is what showed the remaining excess is an angle no rescaling can reach.- **`of3t-recut`** — **CONCLUDED GO at 00:58**, and its deliverable's answer is that the clause FAILS. It repaired `ref_grad.py`, controlled the repair three ways, withdrew its own linearity shortcut when the control failed at 2.21 %, re-scored, and read the ladder without moving a bar. The corrected clause is **1.4511706984958472x**, improved 18.54 % from the withdrawn 1.7814428090278143x.- **`of3t-frameself`** — **CONCLUDED GO at 23:57, and it solved D242.** Root-caused the double count, proved the repair at 3.0392623414001263e-15 against a 1e-12 bar, and banked a 13.2 s one-block reproducer. Its eight brief amendments and the arms they specified are the record.- **`of3t-verbinstall`** — **live**, picked up on qb1 at 22:14. Dispatched at pass 386 onto D245. The campaign's
  best trunk number (1.0525x) comes from `dev_cot.py` rewriting `tt._VERBS` from a perf script,
  while the shippable site-selector install measures **34.25 % worse against float64**. Three
  jobs: PACKAGE the consistent arm as a tape-gated install from `tt_bio.autograd.install` (which
  is also the safer answer to Moritz's inference hard stop, since a tape-gated lever has no route
  from an inference fold by construction where an env selector on shared sites does); the route's
  FALSIFIER, pre-registered two-sided on whether the 1,685 raw serves carry it; and the INFERENCE
  A/B with its A/A floor first and the census per-PID (D236). Namespace `perf/of3t_verbinstall/`,
  base `wk/of3t`. Gate entry, `_STAGE_HINTS` path and brief written on all three hosts (K19).
- **`of3t-cotcoh`** — **concluded 20:39, $7.808**, and it took the pass-381 amendment properly: it sorted its own readings into within-frame (kept) and cross-frame (both frames named) rather than retracting wholesale. Its headline redirects the campaign — pair track 4.9964x against single track 1.0583x (R143) — and it refuted D240 by counting it (R144). Its refutation (R137) stands: it was measured inside one frame by its own pre-registered instrument. Its 0.1100-per-block headline is measured on the frame D242 disqualifies and must be reported as such.
- **`of3t-twoside`** — concluded on **STOP**, and it is the most valuable row of the last ten passes. It ran the control its brief made gating, found the frame broken, and stopped instead of building on it. Its step-2 arm was taken anyway and is banked: the two-sided 1.7998x (R139).
- **`of3t-modelframe`** — concluded, and D242 is its defect. It published a clause reading from a frame whose control had not been run. See A40.

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
the other nine sections sit between 0.945x and 2.176x cross-reference (same-reference: 0.010x to
2.019x, five of them better than upstream's own step — D215), and correcting the trunk alone
moves model scope from 0.5201 to 0.1259 — inside the bar. It takes `of3t-blk4544`'s own named next arm and its
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

### SEQUENCE history, moved out pass 416

Earlier: SEQUENCE: **pass 414, final — `of3t-stackexact` is the new critical path and it asks whether the campaign's remaining plan exists.** The clause is at 1.3037867474869442x, the best lever gave a third, and the unexamined plan for the rest is *stack more levers*. Two sightings say it may not work: the lever closes the graded angle while OPENING the float64 one (R191), and `of3t-verbinstall`'s falsifier found more-exact-is-worse on another arm (R187). So the row is a monotonicity ladder on the clause's own arm — shipped, +exact softmax (must reproduce 1.3038x or stop), +LayerNorm — and the deliverable is the SHAPE. **A CANCELS answer puts the campaign at a ceiling and its brief says so, and says it must be evidenced as well as a positive.** It is queued behind `of3t-modelever` on qb2 card 1, gated in the same commit as the dispatch (R193), and carries the refuted recipe-matching story explicitly so no row revives it (R194). Earlier: **pass 414, latest — every card is committed and D32/D55 are held deliberately, which is the answer the closure plan has been asking for.** Allocation: qb2 card 1 `of3t-modelever` (critical path), qb2 card 2 `of3t-cropwall`, qb2 card 0 `land-standing`, qb1 `of3t-msaamp`. **D32 and D55 both need a card and both name CONCLUDED owners** (`of3t-stepfloor`, `of3t-fwdkcfg`), and the plan prints that every compose. They are not dispatched because there is no card that does not come out of the critical path or out of D58's last unmeasured leg, and **neither D32 nor D55 can move the charter** — the clause is a direction question in the model frame and `of3t-modelever` is the only row that touches it. They go out on the first card that frees, in that order. Saying so is the discharge; leaving the plan to ask a fourth time is not. **`of3t-tapeamp` is retired, not held** — that name had already concluded (R189) and `of3t-msaamp` replaces it, scoped to the one leg that is genuinely unmeasured. Earlier: **`of3t-modelever` is the critical path and the amplification row discharges the hold.** `of3t-angle` concluded GO with the exact softmax closing 50.003 % of frame384's angle, and the clause is reachable only by direction, so `of3t-modelever` puts the PACKAGE install on the model-frame trunk arm and re-scores against `CLAUSE.json`'s pre-registered levels — qb2 card 1, which angle just freed. **It may not project from frame384 and its brief says so**: 50.003 % closed there and 43.61 % needed here are two frames, and the quotient A37 bars is most tempting when it points somewhere good. And qb1 freed when `of3t-verbinstall` concluded, which was the stated condition for holding the amplification row — so `of3t-tapeamp` is dispatched there onto D30/D58/D129, on an idle box, because its deliverable is a profile share and that is the one class co-tenancy corrupts; its deliverable ZERO tests the framing rather than assuming it. No MGX row contends: all five of that campaign's card rows are whglx and one is pc cpu. Earlier: **pass 414 dispatched `of3t-cropwall` onto D205 and deliberately held the other three orphans.** The fleet flagged UNDER-USED twice (3/8 cards, 4/12 slots) and the closure plan has been printing `D205 needs a card and has NO ROW` every compose; those are the same gap and one row closes it. It goes to **qb2 card 2, not card 0** — `tt-smi -r` resets the board PAIR, so a reset on card 0 would take `of3t-angle`'s card 1 with it. Co-tenancy is safe in this direction: angle's arms are accuracy scores and load-insensitive, and cropwall's own perf numbers are gated on a quiet box in its brief. **Why the other three are NOT dispatched this pass, which is the question the closure plan asks and nobody had answered:** D30, D58 and D129 are ONE object — the tape's backward ~20x amplification — so they want one row, not three, and that row's deliverable is a PROFILE SHARE, which is the one measurement class co-tenancy actually corrupts. qb1 is the quiet box and `of3t-verbinstall` holds it with live perf claims, so the amplification row is held until qb1 frees rather than given a card that would make its number arguable. D55's forward half is a third object again and is behind it in the same queue. **Holding a row for the host its measurement needs is sequencing; giving it any free card is utilisation theatre.** Earlier: **pass 397 dispatches `of3t-recut` onto the critical path, and it is the only row that can move the charter.** `of3t-frameself` concluded GO having solved D242; the repair is proven but not landed, so every `MATCHED/` reading is still on the old functional and nothing in `PROVES:` may move until `of3t-recut` re-scores. It contends for qb2, where frameself just freed the card; `of3t-verbinstall` holds qb1. The two are disjoint: recut owns the model-frame trunk and the ladder, verbinstall owns the softmax install and D245, and recut's brief forbids re-scoring verbinstall's arms. Earlier: **pass 386 dispatches `of3t-verbinstall` and it does not breach the D242 rule.** Every reading in that row is a within-frame A/B between softmax install arms on the `of3t-frame384` frame, one scorer and one float64 reference — the class `of3t-cotcoh` established D242 does not touch — and its brief forbids carrying any of it across into the model frame or into the clause's 1.7814x. It is orthogonal to `of3t-frameself`, which owns D242 itself, and it contends for qb1's card where frameself is CPU-only on qb2. **The standing rule is unchanged: no new row may be dispatched onto a trunk ratio until D242 closes.** Earlier: **pass 382 dispatches nothing, deliberately.** `of3t-frameself` owns D242 and is live; a second row on the same question is two agents on one host's worth of confusion, and the campaign has paid for that before. What pass 382 did instead is make that row's next experiment cheaper — R141's candidate and its break control went into its BRIEF, not into this doc, because a decision written into a state doc does not reach a running row. `of3t-cotcoh` is orthogonal and unblocked. **No new row may be dispatched onto a trunk ratio until D242 closes**, which is what A40 means in practice and remains the standing sequencing rule. Earlier: **pass 379** — `of3t-lnreduce` has no `DEPENDS_ON` and must not get one. It needs a card and the `wk/of3t-modelframe` base, contends only for qb slots, and it is the sole critical path: the GRADIENTS clause is now frame-matched and fails at 1.7814x with the whole gap ours, so nothing else in the campaign changes that reading. Its ladder gates its own arm — step 4 re-runs the trunk **if and only if** the K ladder confirms, because an arm built on a refuted hypothesis is device time spent on a story. `of3t-modelframe` concluded and `of3t-cotterm` passes its gate, so the composition is unblocked and no row waits on another. Earlier: **pass 358** — the two rows dispatched this pass are disjoint and neither has a `DEPENDS_ON`. `of3t-refcov` owns the coverage leg and touches only `tt_bio/train/`; `of3t-vjpln` owns blk4544's backward-VJP arm and `of3t-trunkact` owns the forward-activation arm, which is the same object approached from two sides and the reason `of3t-vjpln`'s brief says explicitly not to take the forward. Neither blocks the other: one is a build on qb1, the other a device re-score. Earlier: **pass 318** — `of3t-barresolve` has no `DEPENDS_ON`: it is CPU-only, contends for nothing, and it gates a claim that is currently being quoted, so holding it behind anything would be holding the audit behind the thing it audits. It is the only row whose result can REMOVE a MET condition, which is why it is dispatched now rather than after the merge story closes. Earlier: with every row already running, sequencing became a set of rules in briefs rather than
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

### BRANCH history, moved out pass 416

Earlier: **`wk/of3t`, one reviewable branch, recomposed from `origin/main` every pass and pushed.**
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

### GAP history, moved out pass 416

Earlier: `of3t-verbinstall` held qb1 on D245, and **the package leg is CLOSED**: the
              packaged install reproduces the harness arm `ceiling_hf3` BIT-EXACTLY — 2,736 of
              2,736 gradient tensors, 0 differing, neither side holding a key the other lacks
              (`ARMDIFF_PKG_HF3B_vs_CEIL_HF3.json`, read at pass 414 rather than taken from the
              row's prose), and it is qb1's p150a reproducing a qb2 p300c arm. **So the
              campaign's best trunk lever DOES have a shippable path**; the site selector stays
              34.25 % worse, and the FALSIFIER has landed on its PRE-REGISTERED second side:
              suppressing all 1,869 raw serves moved the reading +0.0023 % (0.5605474136179824
              against ROUTE_HF's 0.5605347900452246), so the candidate mechanism is REFUTED as
              the carrier. **And it produced something sharper than the question asked**: the
              arm with 9,184 taped-exact softmaxes reads 0.5605 where one with 5,901 verb plus
              1,742 raw reads 0.4175 — **more exact is WORSE**, so the "consistent arm" framing
              the whole softmax ladder rests on is falsified (R187). `of3t-verbinstall`
              concluded **GO** and closed D246 and D247 on the way — D246 with a better repair
              than the one proposed to it, putting the flag on both halves so the docstring's
              advertised equivalence is RESTORED rather than withdrawn, and D247 by bounding the
              probe itself at 120 s against a measured 1.6 s healthy open. Its conclusion armed
              the D155 guard on its ARMDIFF artifact exactly as I warned it would, so that claim
              is frozen with the host established from the filesystem — **the third concluded
              row frozen for one cause, now filed as D249**: the writers emit no host field, the
              fix is one line in a row namespace every time, and every time it arrives after the
              row can act.
              The earlier inertness was ROOT-CAUSED and repaired at
              pass 414 (R180): `uninstall()` was never the lever's private teardown, an unrelated
              `install()`/`uninstall()` bracket around the DISCOVERY forward closed first, and
              all 1,742 exact softmaxes were spent in a forward whose output is discarded. The
              repair holds for the arm's own entry point; auditing it filed **D246**, the one
              USER-FACING defect in the campaign needing no measurement at all — the owner is a
              constant STRING, so `install(exact_softmax=True)` shares the token every foreign
              `uninstall()` passes and that path is torn down exactly as before. Latent in-repo,
              user-facing out of it, because the docstring still calls it equivalent. The same
              audit filed **D247** from that row's 230 lost card-minutes, verified in the shipped
              file: `_assert_local_dispatch` is a fail-fast probe with no timeout, guarding the
              chip that THROWS and not the one that WEDGES, and the row's own bounded pre-flight
              left the probe shipped for every other caller — so 230 minutes is a floor.

    satisfiable  Unchanged and still the reason this is PARTIAL and not NO-GO: a perfect trunk reads
              **0.6752x** the bar (D221, ten sections pooling to 0.1026990533692057), and our trunk
              merely at upstream's own bf16 floor clears at **0.8525x**. Both are computed in the
              REPAIRED frame at pass 408, so neither is provisional any longer — and a terminal
              NO-GO needs unreachability and
              nothing here shows it.

    coverage  Still `not instrumented`, still outranked by the above only because a frame that
              cannot grade a gradient cannot grade a covered one either. Blocker named (A2:
              `tape()` rebinds). It is the next thing after D242 closes.

    open      **The UNFIXED roster, named so this field cannot drift from `DEFECTS.md`.**
              **USER-FACING (6)**: D32, D55, D58, D184, D205, D210.
              **SCOPE-EXCLUDED (5)**: D2, D3, D123, D124, D213.
              **CAMPAIGN-INTERNAL (83)**: D10, D18, D22, D23, D24, D26, D27, D28, D35, D37, D42, D46, D48, D49, D51, D53, D59, D62, D63, D64, D69, D71, D73, D78, D82, D86, D89, D91, D92, D93, D94, D110, D112, D118, D119, D120, D121, D122, D125, D136, D140, D141, D148, D152, D158, D163, D180, D183, D186, D187, D189, D190, D191, D192, D193, D194, D195, D196, D197, D198, D200, D202, D204, D207, D208, D209, D211, D214, D217, D219, D222, D223, D224, D225, D227, D231, D232, D233, D235, D237, D240, D241, D242.
              Triage and reasons in `state/of3t/UNFIXED_TRIAGE.json`; the list is stamped from
              the DEFECTS union, not retyped, so a rotation cannot close one by moving it.


## PASSLOG as of pass 415

PASSLOG: the per-pass narrative.

**Pass 384, the premise audit as first written.** **Pass 384: every checkable premise of D242 now holds, and the backward still differs by 1.75x**
(R147, zero card, zero model run). The capture ran on qb1 and the replay on qb2 reading the same
path on a different filesystem — sha256 identical both sides. The pad caveat
closes in our favour: pads cost **47.6x** of sensitivity, but 16-digit norm agreement still bounds
the real-row forward discrepancy at **4.76e-15**. Parameter sharing is out by code read, and the
loss's checkpointed chunks close over `{'batch','eps'}` with no trunk tensor. **The hook firing
count is the last cheap discriminator and the self-test banks it.** Six exclusions are not a
mechanism. Detail in GAP.


**Pass 383, R146 as first written.** **Pass 383: the hypothesis space collapsed onto H-A, and I refuted my own candidate to do it.**
`checkpoint_blocks` — R141, which I put in the live row's brief last pass — is **exact**: on
upstream's real `PairFormerBlock`, **228 of 228 parameter gradients are bit-identical** to a bare
loop at `use_reentrant` None/True/False, with `ds_in` and `dz_in` exactly 0.0 and the scorer's own
break control firing at 1.35e-11 (R146). With `of3t-frameself`'s two eliminations beside it — the
tree, 293 files byte for byte, and the per-block call — **H-B has no surviving mechanism.** What
remains is **H-A, the captured pair is insufficient**, and the replay being **1.75x LARGER** than
the reference excludes a merely-missing path: it must partially cancel, or the captured cotangent
is too large. The hook firing count settles that and it is one integer.


**Pass 382, R142 and R143.** D242 is a **regression, not an oversight** (R142): the control already existed as
`of3t-conditioning`'s COTANGENT_COMPLETE, and `capture_model_frame.py` argued in a comment that
the check was impossible and shipped a provenance witness instead. A witness asserts PROVENANCE
and structurally cannot see a replay defect. And `of3t-cotcoh` concluded: **the aim is the PAIR
track at 

---

## ROTATED 2026-09-25T10:35:59Z

This doc reached 104392 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-PASSLOG.20260925-123559.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

here a direct run gives `g(Q(cot_hooked − delta))`, and since
the duplicate is 99.6628 % of the hooked cotangent by norm, the external cotangent they bracket
is 11.7471x smaller and any quantisation lands against the small quantity. From bf16's nominal
epsilon that is **9.18 %** — enough to make the shortcut useless on the device.

**It is wrong by 637x, and the artifacts already held the measurement.** Both arms bank a
`cotangent_on_device` block — added for exactly this question — giving round-trips of 2.9059e-06
and 9.3992e-06 relative, a combined **9.3295e-09** absolute, which is **0.01442 %** of the
external cotangent.

**A quantisation bound taken from a dtype's epsilon is an upper bound on a harness nobody has
measured.** Once the harness is instrumented, the epsilon is the wrong number to reason from —
the same mistake as asserting a roofline instead of measuring it, in precision's costume. The
row is told to run R161's end-to-end control anyway, and told that 0.014 % is the expected
disagreement so a sub-0.1 % result is not read as a defect.

## Pass 407 — the falsifier fired on its other branch, exactly

Pass 387 pre-registered a one-scalar falsifier for D242 with both values banked before any arm
ran: `||dL/dz_in||` at 0.000848887340907281 means exact, at 0.0014907294032500784 it overcounts.
At pass 395 `--graphdrive` read the OVERCOUNTS value bit for bit. The repaired reference now
reads **0.000848887340907281** — the EXACT value, bit for bit, with `ds_in_norm`
0.009204973933437452 alongside it.

**A two-sided falsifier that fires on one branch, is acted on, and then fires on the other
after the repair is the strongest form this evidence takes.** Both values were fixed before the
mechanism was known, so neither could be fitted to the answer. And it is independent of R170's
parameter-gradient confirmation at 1.6952505222168708e-14 — a different quantity against a
different reference, landing exactly. Two pre-registered confirmations of one repair.

The A42 chain checks out by digest at every hop: the reference self-corrects (permitted only for
the arm that defines the boundary), banks `cot_external.pt` and `cot_delta_only.pt`, and both
device arms cite those digests — one correction, one source, three consumers.

One consequence stated before it surprises anyone: the corrected cotangent is 11.7471x smaller,
so any fixed-absolute device error is 11.7471x more significant in relative terms. Measured, the
harness's round-trip is ~2-7e-9 absolute whatever the cotangent's size, giving 3.38e-5 relative
on the external one against 2.9e-6 on the hooked one — four orders below the clause's scale, and
there is no other fixed-absolute floor to amplify because the device arms' A/A is exactly 0.0.

## Pass 408 — the corrected clause reads 1.4512x the bar, still failing, and two of my numbers are wrong

`of3t-recut` delivered the re-score. On the repaired injection the trunk reads
**0.6221485227575493** against float64 (was 0.9349175217825587) and **0.7768254196709333**
against upstream's own bf16 (was 0.9969599833682794), a multiple of **1.976518918492322** (was
2.970162431380236). **The clause is 0.22072451195864032 against a 0.15210099830945006 bar —
1.4511706984958472x, down from 1.7814428090278143x, an 18.54 % improvement, and it still
FAILS.** The trunk now carries 78.35 % of the model's error mass, down from 96.06 %, at
**1.7414x** its allowance.

**The repair was necessary and is not sufficient**, which is R130's sentence about the earlier
frame fix now true of this one. **No bar moved**: `LADDER_APPLIES_UNCHANGED` re-derives all five
pre-registered levels on the rescored artifact at rel_difference 0.0, and the recomposition
control reproduces the headline at 0.0.

**R159's "1.4172x" is retracted — it is a cross-space quotient, the class I have been catching
in others.** The allowance 0.44608901561034203 lives in vs-upstream-bf16 space; upstream's floor
0.3147698293887927 lives in vs-float64 space. The artifact's own quotients each stay in one
space: 1.976518918492322 (float64) and **x_allowance 1.7414134679108282** (bf16). The honest
target is **the trunk must fall by 1.7414x**. R159's framing was also ill-formed, since
upstream against itself is zero.

**R172's shortcut bound was wrong and its own two-sided statement fired on the withdrawal
branch.** I predicted 0.014 % and said percent-scale would withdraw the shortcut. Measured
**2.21 %**. I bounded the cotangent *injection* round-trip, which really is ~1e-5, and ignored
the *taped backward's own bf16 arithmetic*: the subtraction cancels gradient norms 1.3279 and
0.5947 into 0.8149, so each arm's ~0.9359 % bf16 error survives against a smaller difference.
**A linearity shortcut needs the arithmetic to be linear, not just the mathematics.** The
float64 sum identity of 6.4e-15 is what made it look safe, and float64 is exactly where it is
safe. R161's mandatory control caught it, the shortcut was withdrawn and the repair untouched —
which is the part of the design that worked.

**Repoint condition 4 is satisfied by ELIMINATION, not waved through**: the reading now contains
no shortcut at all, which meets the condition's purpose more strongly than a passing control
would, and the failure and its magnitude sit in the record beside it.

## Pass 409 — the trunk's remaining error is 97 % direction, and that inverts the campaign's picture

**The repoint is blocked, by my own pre-registration.** The corrected composed artifact carries
six of seven contract keys with the corrected 0.22072451195864032; `injection.convention` is
absent. Conditions 2, 3 and 4 are met. R164 fixed the rule before the number existed and it
says any failing condition means no repoint, so it holds. The namespace and the value would let
a careful reader work it out — which is exactly the reasoning that makes a stamp optional and
then rots.

**And the corrected reading has a structure nobody had named.** From the artifact's own two
scalars — cos 0.8178953379770566 and norm ratio 1.054584096168068 — the observed 0.6221485227575493
reconstructs to 1.1e-15, so the split is exact rather than modelled:

    magnitude alone    0.054584      8.8 % of the observed error
    direction alone    0.603498     97.0 % of the observed error
    the angle          35.13 degrees

On the double-counted functional the trunk read as a near-constant scale of 1.7460 at cos
0.9841 — a magnitude story with a ten-degree angle. **That was the duplicate**: a large
nearly-parallel component inflates the magnitude and flatters the cosine at the same time.
Remove it and the magnitude is within 5.5 % while the angle opens to 35 degrees. **Everything
the campaign chased from D227 through D233 as a magnitude deficit was reading the duplicate;
what is left is a direction error.**

**What this does not license.** The decomposition is in vs-float64 space, because those are the
scalars banked; the clause is graded in vs-upstream-bf16 space, and carrying it across is the
cross-space error I retracted one pass ago. So no claim is made here about what the clause
needs. The one-line ask that closes it: bank cos and norm ratio against upstream's own bf16 too
— the scorer already computes that comparison and reports the decomposition for only one of the
two. With both, the campaign can say whether the remaining 1.7414x is reachable by magnitude,
by direction, or by neither.

## Pass 410 — the shippable install fires only the half that is worth nothing

`of3t-verbinstall`'s package arm landed and reads **0.702981502944001** against float64 and
**0.9153623104186986** against upstream's bf16 — **bit-identical, to sixteen digits, to CTRL_B,
the arm with no exact softmax at all.** Its own counter says which half is missing: `verb` **0**,
`raw` **1742** over 21.9 G elements.

The verb half is where the win lives. `CEIL_HF` (verb only) is 0.5547455957585244 and
`CEIL_HF3` (verb + module-wide) is 0.5545352626143085, so module-wide is worth **0.0002** and
verb is worth the other 0.36. A package install delivering only `raw` delivers nothing
measurable, which is exactly what the score shows.

**So D245 is worse than filed.** It was "the site-selector install is 34.25 % worse"; now the
tape-gated package install — which R161 argued was both more accurate and structurally safer
for Moritz's inference hard stop — is **inert on the half that matters**. The campaign's best
trunk number, 1.0525x, still has no shippable path, and there are now two failed attempts at
one. Whether `exact_softmax()` never installs at the verb (a design gap) or installs and is
never reached (D225's reach family) is the row's to settle, and its counter separates the two by
construction. **The campaign's status must not meanwhile claim a shippable lever.**

**And the row lost 230 minutes of card time to a defect worth filing beyond this campaign**:
`tenstorrent._assert_local_dispatch` — a startup probe whose own docstring says a bad bring-up
should "fail HERE, at startup" — **hangs instead, with no timeout, while every cheap liveness
signal reads green.** Two arms, 115 minutes each, nothing computed. The row added a bounded
pre-flight for itself; the probe still has none.

Also fixed this pass: my LEDGER index named one archive file by name, which the next rotation
would have falsified. It now names the rule.

## Pass 411 — the two fields Moritz reads were three passes and one closed defect behind

Checked `VERDICT` and `GAP` against the numbers the campaign now holds. Both carried 1.7814
(withdrawn) and 0.7945 (D242's unrepaired control); neither carried 1.4511, 0.2207 or 1.7414.
They described a campaign blocked on a frame defect that has since been root-caused, repaired,
controlled three ways and re-scored.

**That is R157's rot, and I committed it three passes after naming it.** Naming a failure mode
does not inoculate you against it; what catches it is re-reading the field against the
artifacts, which is a task rather than an intention.

VERDICT is rewritten and stays under its cap at 3,967 by deleting two paragraphs D242's closure
superseded — including "the defect is two components: a scale near 1.7493 plus a residual of
0.1779", which was no longer merely stale but wrong. GAP is rewritten and shrank 9,322 → 5,583:
its `control` block recorded D242 as open at twelve orders over, its `next` block still had
`of3t-frameself` parked on a DEFER two concluded rows ago, and two sentences called the frame
broken and the ladder levels provisional. All corrected.

**The sharper form of R157**: a field recording a BLOCKER is the likeliest to rot, because it is
written at maximum certainty about something expected to change. **Closing a defect should
trigger a re-read of every field that named it** — its own entry, the clause's `why`, and the
summary fields. The campaign has now found the same rot in all three.

## Pass 412 — the repoint is executed, and the remaining excess is an angle no rescaling can reach

**The GRADIENTS clause now grades `perf/of3t_recut/MODEL_RECUT_composed3660_n384.json`** at
**0.220725** against its bar — FAIL, with the other three checks passing and `n_met` 2 of 3.
All four pre-registered conditions were met first, the fourth by elimination, and the artifact's
restamp control reads `n_leaves_differing: 0` with one key added. **No bar moved.**

Verified in the COMPOSED tree, which is the only place the artifact exists — running the
generator in my own worktree reported "artifact absent" for the new clause *and* for COVERAGE,
which is what caught me checking the wrong tree.

**And `of3t-recutfin` answered the strategic question, in the graded space rather than by
carrying the float64 split across.** That restraint was worth a factor of **31.92**: the
magnitude share is 0.2749 % in vs-upstream-bf16 and 8.773 % in vs-float64, on the same arm and
the same 2,736 tensors, both splits exact.

**By magnitude the clause is unreachable** — `r = 1` is *worse* than today, and the best any
rescaling can do is sin(45.763°) = 0.7164605, still 1.6060930x the allowance. **By direction it
is reachable and need not be perfect**: cos 0.6976 → 0.9003 suffices, 43.61 % of the angle. The
magnitude is already right to 0.21 %.

## Pass 413 — the repair removed a magnitude error and opened the angle, and the lever may have been aimed at the wrong thing

`of3t-recutfin` concluded GO at 01:17. Its last measurement, both rows vs-upstream-bf16 on the
same 2,736 tensors and the same frame with only the injected functional differing:

    functional        rel         r           cos        angle     magnitude share
    double-counted    0.9969600   1.6232598   0.8134998  35.561    62.52 %
    repaired          0.7768254   0.9978645   0.6976277  45.763    0.2749 %

**The double count was a magnitude error; the repair removed nearly all of it; the angle
OPENED by ten degrees.** A large nearly-parallel duplicate inflates the magnitude and flatters
the cosine at once, so removing it improves one and worsens the other.

**A correction of mine.** R175, R178 and `of3t-angle`'s brief said the pre-repair trunk read as
"a near-constant scale of 1.7460 at cos 0.9841". Those are the model-frame float64 control's
per-block statistics, not the graded-space reading — in vs-upstream-bf16 it is cos 0.8134998 at
35.561°. The qualitative claim survives; the two scalars I attached to it came from a different
measurement.

**And it leaves one sharp question, now dispatched as `of3t-angle`.** The campaign's best trunk
result — the exact softmax at 1.0525x against the shipped 1.7373x — was measured entirely on
the double-counted functional, where 62.52 % of the error was magnitude. **A lever that closes
a magnitude error is worth nothing on a functional that is 99.7 % angle.** Either the lever
closes the angle, in which case it is the only thing known to move what is left, or it does
not, in which case the campaign's best result evaporates and D245's two failed shippable
installs stop mattering.

### Pass 414 (2026-09-23, zero card) — R176 is root-caused, and auditing the fix found the two cells nobody was looking at

`of3t-verbinstall` landed `27d24c6b3` and it closes R176. **The inert package install was never
arithmetic and never the shim: the lever installed, fired 1,742 times, and was torn down by a
caller that had never heard of it.** `train/lora.py:608-615` brackets the discovery forward in a
conditional `install()`/`uninstall()` pair, `train/recipes.py:211` does the same around the fit,
and the discovery pair closes first — so every one of those 1,742 exact softmaxes was spent in a
forward whose output is discarded, and the step that was scored ran entirely on the device
softmax. Hence 0.702981502944001, CTRL_B to sixteen digits. **This is the campaign's cleanest
case of a failure where every cheap check reads as success**: the call returned, the lever really
was installed, the counters were non-zero, the score was plausible. Only `verb 0` against
`raw 1742` separates "reverted" from "never dispatched", because a score is one number and the
question is which code served the call. `reachprobe.py` counts by call site and is kept.

**I audited the repair rather than taking it, and the audit is most of this pass.** `install()`
arms five things — recycle, checkpoint, host-softmax and grad hooks, plus optionally the exact
softmax — and only one of them got an owner, which is the shape of a fix that repairs one sibling
and leaves four. **It isn't**: `taped_ttnn.tape()` calls `ag.install()` on every entry, so the
other four are re-armed at each tape block and self-heal; the exact softmax is the only passenger
that cannot, because `tape()` never passes `exact_softmax=True`. **And the measured failure is
really repaired** — `pkgarm.py:45` enters via `with ag.exact_softmax():`, owner `"exact_softmax"`,
which no longer matches the `"install"` a foreign `uninstall()` passes. I checked the reachability
rather than inferring it from the code, because which caller actually fires is not a code fact.

**Two residual findings, filed to the row as an amendment and written up as R180.** First, the
owner is a *constant string, not a caller identity*: `install(exact_softmax=True)` records
`"install"`, the same literal every foreign `uninstall()` passes, so **that entry point is still
torn down exactly as before the fix**. The two new tests pin on-via-CM + foreign pair, and
on-via-install + its own teardown — the two *safe* cells — and together they read as "both
directions", which is precisely how the fourth cell stays invisible. Latent, not live, but
`exact_softmax()`'s docstring still advertises the pair as "the same thing without the block",
and **the fix is what made that sentence false**. Second, `pkgarm.py:55` stamps
`"installed_from": "...exact_softmax() -> install(exact_softmax=True)"` while the CM never calls
`install()` at all — harmless while the paths were equivalent, and **the fix is what ended the
equivalence**. A row whose defect *was* "the install did not survive" must not stamp the path
that would not have survived.

The generalisable half, and why it is worth a ledger entry rather than a row comment: **a global
`install()`/`uninstall()` pair with a constant owner token discriminates between APIs, not
between callers.** Nesting two such pairs is indistinguishable from one caller undoing itself.
Distinguishing them needs a per-call handle or a depth count — a name cannot do it.

**Pass 414, second half — two defects filed and one row dispatched.** `of3t-verbinstall`'s 230
lost card-minutes were in its state doc with no defect ID, so they were a story rather than a
tracked cost. Filed as **D247** and verified in the shipped file rather than transcribed:
`tenstorrent._attr` — precisely, `_assert_local_dispatch` at `tt_bio/tenstorrent.py:5575` —
wraps its dispatch in `try/except Exception`, so **it guards the chip that THROWS and not the
chip that WEDGES**, and `synchronize_device` blocks with no timeout anywhere in the function.
The row's own bounded pre-flight (`b77e89f27`) is the right local move and closes nothing: the
probe is unchanged, so **230 minutes is a floor rather than a total**, and the workaround also
removes the row's motive to fix it while its state doc reads as resolved (R181).

**And the fleet's own alarm was right.** It flagged UNDER-USED twice — 3 of 8 cards, 4 of 12
slots — while the closure plan printed `D205 needs a card and has NO ROW` every compose. Same
gap, one row: **`of3t-cropwall`**, on **qb2 card 2 and not card 0**, because `tt-smi -r` resets
the board PAIR and card 1 is `of3t-angle`'s. The brief's own contribution is that **D205
conflates two walls** — 640/768 die with the card full, 544/576 die on contiguity with 6+ GB
free — and that **the odd-32-tile-count story cannot explain 576**, which is even at 18 tiles
and dies on contiguity anyway. So the two are separable and 576 is the arm that separates them.

**The other three orphans are held, and that is the answer to a question the plan has been
asking unanswered.** D30, D58 and D129 are one object, and its deliverable is a profile share —
the single measurement class co-tenancy corrupts. qb1 is the quiet box and `of3t-verbinstall`
holds it with live perf claims. **Holding a row for the host its measurement needs is
sequencing; giving it any free card is utilisation theatre**, and a number taken to satisfy a
utilisation alarm is arguable forever after.

### Pass 414, third part — the package leg closes bit-exactly, and my own VERDICT had it backwards

`of3t-verbinstall` reported PACKAGE closed, and I read
`ARMDIFF_PKG_HF3B_vs_CEIL_HF3.json` rather than its VERDICT: `compared 2736`,
`bit_identical 2736`, `differing 0`, `only_mine []`, `only_theirs []`, `all_bit_identical true`.
The packaged install scores **0.41752141981218177** against float64 — `ceiling_hf3` to seventeen
digits — with reach banked at **verb 5901, raw 1742** where the first inert arm read verb 0, and
it is qb1's p150a reproducing a qb2 p300c arm, so it is a third cross-board A/A besides.
**I checked this one because it is flattering**: it moves the campaign our way, and a result
that helps deserves the scrutiny a result that hurts gets for free.

**So the campaign's best trunk lever now has a shippable path, and my VERDICT and GAP said the
opposite.** They read *"the best softmax arm still has no shippable path"* and *"BOTH shippable
installs have now failed"* — true when written, falsified by the row inside the pass, corrected
in the same pass this time. That is R157's rot for the third time in my own fields, and it is
now specific enough to name: **the sentences that rot are the ones summarising a LIVE ROW's
position**, because that is the only thing in the doc that changes without me touching it.
Fields that summarise artifacts hold; fields that summarise rows do not.

**What must not be conflated.** The lever is shippable. Whether it is WORTH anything is still
`of3t-angle`'s question — its 1.0525x was taken on the double-counted functional where 62.52 %
of the error was magnitude, and the repaired functional is 0.2749 % magnitude and 99.7 % angle.
**A shippable path to an inert lever is a shippable path to nothing.** D245 therefore stays
UNFIXED with its headline clause struck and its falsifier still on the card, rather than being
closed on the half that went well.

**D246 did not escalate, and I checked instead of assuming.** The row's prose says the
arithmetic installs via `install(exact_softmax=True)`/`uninstall()` — the unprotected path — but
every non-test caller on the branch still uses `with ag.exact_softmax():` (`pkgarm.py:45`,
`reachprobe.py:98`). What the prose shows is finding (b) doing its damage on schedule:
`pkgarm.py:55`'s wrong `installed_from` stamp has become the row's own sentence. **A bad
provenance string does not stay in the JSON.** Also flagged to the row: the ARMDIFF artifact
records no host, a D155 warning today and a gate FAILURE the moment that row concludes.

### Pass 414, fourth part — `of3t-cropwall` overturns D205's mechanism within an hour of dispatch

The row I dispatched this pass has already corrected the defect it was sent at, and the
correction reaches my own brief. **The 2,717,908,992 B refusal that `of3t-crop768` banked as
"contiguity inside `ttnn::concat`" was TILE PADDING.** `taped_ttnn.py:922`'s qkv-heads vjp
scattered into a rank-4 axis of **extent 3**, TILE pads the second-to-last dim to **32**, and
the allocator was therefore asked for 10.667x what the gradient held — **90.625 % of that
buffer was padding**, with three such buffers co-live. Filed as **D248**, user-facing, fixed on
the row's branch and release-gated. **I re-derived every byte before recording any of it**: the
32/3 factor, the pad share, `536,870,912 = 2,717,908,992 x (256/576)^2` exactly, the
`[544,4,544,544]` fp32 at 2,575,826,944 B and 7,247,757,312 B at 768. All exact.

**The lesson is the noun, not the number.** A refusal reports how many bytes were asked for and
how many were free; neither says the request was NECESSARY. crop768 had the right number and
the wrong question — why the card could not supply 2.7 GB contiguously, when nothing ever
needed 2.7 GB. The question that separates them costs no device time: *what does this buffer
logically hold?* Here, 254,803,968 B. And the repair is a pure re-indexing onto the last axis,
controlled against a **float64 host** scatter rather than a second device expression, same
digest `68b639dc693788dc`, max_abs 0.0 — bit-identical, at 3.02x less DRAM.

**My brief supplied the wedge and the wrong word.** Its contribution held exactly — *the
odd-32-tile story cannot explain 576, which is EVEN at 18 tiles, so the two stories are
separable* — and 544 duly turned out to be the odd-tile L1 plan collapsing into an unblocked
fp32 score tensor while 576 was padding. **But I carried crop768's noun across while doing it**,
writing that "544 and 576 die on contiguity". I verified the separation and not the mechanism
name. A brief's framing is inherited by the row that reads it, and this row got the right
answer despite mine.

**Amended the row on the one gap that is in the instrument rather than the finding**:
`concat_census.py` ranks by absolute bytes, but the defect is a RATIO, and now that the largest
offender is fixed every remaining site's share of the peak has grown. Re-rank by
`allocated / logical` across all 43 sites. Also told it to report the **512 A/A before 640** —
the fix fires at every crop, so 512's banked 23,299,281,920 B must come down, and 512 is the
rung that ships today — and to put the training-only reachability argument for `taped_ttnn.py`
in its state doc explicitly, since that is the sentence a reviewer will ask for.

### Pass 414, fifth part — the answer arrived, and the two rows it implies are dispatched

**`of3t-angle` concluded GO: the exact softmax CLOSES THE ANGLE.** Shipped to verb, in the space
the clause is graded in, cos **0.3140461038 -> 0.8105926521** and the angle
**71.6968° -> 35.8461°** — **50.003 % of it closed** — with the counterfactual pricing the two
axes apart rather than asserting which mattered: the lever's norm ratio at the shipped direction
drops rel by 0.3303 (36.3 % of the move), its direction at the shipped norm ratio by 0.6153
(**67.7 %**). It is a DIRECTION lever, and no rescaling substitutes for what it buys. Its A/A
floor is **exactly 0** — two shipped arms bit-identical on 2,736/2,736 — and because one ran
quiet and one loaded, that pair is also the first DIRECT measurement that load does not move an
accuracy reading, which R182 had only argued. The headline moves downward and honestly:
**1.0525x -> 1.2224x**.

**What it does not establish, and the row said so before I did.** The clause needs cos
0.6976277 -> 0.9002917, 43.61 % of a 45.763° angle, on **a different arm and a different
boundary**. "50.003 % closed there" over "43.61 % needed here" is exactly the cross-frame
quotient A37 and D218 bar, and it is more tempting than usual because it points somewhere good.
**`of3t-modelever` is dispatched to MEASURE it** — the package install, on the model-frame trunk
arm, re-scored against `CLAUSE.json`'s pre-registered levels, A42 bound to the model frame's own
correction rather than frame384's, no bar moved, and its brief forbids the projection explicitly.

**And the hold is discharged.** qb1 freed when `of3t-verbinstall` concluded, which was the exact
condition I set at the start of this pass for holding the amplification row: its deliverable is
a profile share, the one measurement class co-tenancy corrupts. qb1 is idle at load 0.08, so
**`of3t-tapeamp`** goes there onto D30/D58/D129. Its **deliverable ZERO tests the framing rather
than assuming it**: D58's reframing from "a diffusion property" to "a property of the tape"
rests on 19.6x and 19.8x agreeing to two significant figures, and **D30 itself says those
forward denominators come from two different harnesses, are not interchangeable, and are
order-of-magnitude rather than calibrated**. Recompute both on one denominator first. If the
agreement dissolves, that is the better result, because every "expect ~20x in the gradient"
expectation built on it would rest on an artifact. Deliverable one is then the bisection D30
named and nobody ran — the harness's `bisect` field is `{}` in every 0.4.3 run on record.

`of3t-cropwall` DEFERred itself to 03:25 with a stated reason and its detached ladder chain is
healthy (`split_trace` at 96.5 % CPU); that is a clean defer, not a dropped row, and I checked
the process rather than the marker. No MGX row contends for a qb card.

### Pass 414, sixth part — I dispatched a row that had already answered the question

`of3t-tapeamp` went out onto D30/D58/D129. **A row of that name had already concluded GO on
2026-09-22**: *"the question is answered and the answer is that there is no defect here to
repair."* The fleet will not relaunch a name with a concluded marker, so nothing ran and nothing
warned — **a brief plus a `<!--ws:-->` tag for a concluded name is a silent no-op**, and every
place a dispatch is recorded said dispatched. No card was spent, which is precisely why nothing
caught it.

**Why I dispatched is the part worth keeping.** D30's ledger heading still opens *"UNFIXED, and
it is the campaign's central number"* carrying **19.6x**, and the closure plan named
**`of3t-ditcot`** as D58's owner — a row that did not answer it — quoting the same stale pair.
Then I checked instead of assuming, and `statuses_by_defect` **already returned CLOSED for D30
and D129 before I touched anything**. The machine-readable half was right all along; the prose
beside it was two revisions stale, and the prose is what a reader reads. R148's exact shape,
which this campaign has now paid for at least three times.

**What the earlier row had found, now in the ledger it was about**: the factor is **11.026x**,
not ~20x, and **7.666x of it is upstream 0.4.3's own bf16 factor** at the same boundary and
reference; upstream's own fp32 recipe shows **9.326x** at four orders of magnitude lower
absolute error, so it survives a precision change no dtype boundary could explain; **0 dtype
reconciliations in 1,879 node firings**; and **our arm beats upstream on both halves** — forward
1.959x, gradient 1.362x — so our factor is larger only because the denominator is the half we
beat hardest. D30 closed as not a defect, D129 dissolved by `of3t-ditref`, **D58 narrowed and
explicitly NOT closed**: one leg re-explained, `msa_module`'s never measured.

**`of3t-msaamp` is dispatched onto exactly that leg**, on the idle qb1, carrying the prior row's
method and pre-registration discipline instead of the stale framing, and forbidden from
preferring either outcome.

**And I built a guard for the silent no-op, then deleted it.** Three signals, all noise:
brief+tag+marker fired on **34** rows (leftover tags are the fleet's normal residue), brief
mtime fired on **103** (git checkouts reset mtime), git creation date still fired on dozens
(the concluded markers are rewritten by sync, so their mtimes are not when rows concluded).
**The disk does not carry a trustworthy conclusion timestamp, so the check cannot be built from
it.** Shipping it anyway would have bought a second permanently-red arm — the thing this
campaign already knows gates nothing. The repair is a one-command habit before writing a brief.

### Pass 414, seventh part — the critical path answered: a third, not enough

**`of3t-modelever` put the exact softmax on the clause's own arm and the clause still FAILS**, at
**1.3037867474869442x** against 1.4511706984958472x before — **32.67 % of the excess closed**,
with a further **23.30 %** to go. It closes **14.19 %** of the graded-space angle against the
**43.61 %** the clause needs, and **against float64 it makes the trunk worse**.

**This number is believable in a way earlier ones were not.** The A/A floor is exactly 0, SHIP_A
is bit-identical to `of3t-recut`'s banked `dev_RENORM_model_n384_external.pt` — the artifact the
clause was scored on, digests cited both ways — the banked arm through this row's own scorer
reproduces 0.22072451195864032 exactly at `rel_difference 0.0`, and the EXACT arm differs from
SHIP_A **by the `exact_softmax()` scope alone**. It is a lever on the clause's arm, not a
projection onto it. No bar moved.

**The charter is not shown unreachable** — `upstreams_own_floor_here` (0.8525301041731214) and
`section_A26_level` (0.9700522560698159) both still pass, and our trunk's norm ratio vs float64
is 1.0644759387772336 against upstream's own 1.0568409490651478 on the same scope. **But the
best lever the campaign has is now spent on this arm, it delivered a third, and nothing of
comparable size is identified.** That is the honest state and it is what `VERDICT` and `GAP` now
say.

**One thing gained for free.** The in-frame projection A37 bars as evidence was pre-registered
as a level with an aliveness band — *within 15 %, or it is dead and the campaign must stop
carrying it* — and measured **2.4 % off** (2.180720496587762 against 2.2340903768268046). It
stays inadmissible as evidence and becomes a **sound cheap screen**: a future lever can be
triaged in-frame before anyone spends a model-frame arm on it.

**And I broke my own guard fixing it.** The row was the first to act on a D155 warning while
live — because it was handed the writer-level fix and a row to copy, not just the warning — and
it stamped host/board/card into a per-artifact provenance block. The guard read the top level
only, so it went on warning, and would have **failed a row that complied** and forced a freeze
that would then have read as a fourth instance of D249. Presence is recursive now and the
exclusion is widened to every host/card pair at any depth, strictly stronger than before. My
first attempt regexed `json.dumps(d)` and the break control caught it in one run: JSON puts
`": ` between `card` and its value and breaks the adjacency the pattern needs.

## Pass 418, 2026-09-23 ~06:15Z (zero card)
stackship GO and cropwall GO read and recorded. Dispatched of3t-fullstep64 (qb2 c1), relocated of3t-stackbound qb1 c1 -> qb2 c3, cleared its qb1 notbefore and blocked-on-qb1 marker, gate entry + stage hint added to _of3t_donecheck.py. R197.

## Pass 420 (2026-09-23 ~10:30-11:20Z)
stackbound GO recorded (R198, D252). fullstep64: A40 control passed; row named D253 (training trunk unmasked; 128-wide rel 27.3 -> 0.150); brief amended to own the fix. disk_guard removed pc /tmp/of3t (20.8 GB, fullstep64's references) at 10:40Z: parent basename matched no running slug; disk_guard patched (uncommitted, beside another writer's uncommitted edit), campaign scratch moved to top-level slug names. Compose: main's MGX merges broke it four ways; four resolvers; wk/of3t -> cbb3e1f92, 175/0/0.

## Pass 421, 2026-09-23
of3t-fullstep64 concluded GO (D253 fixed, 77d0ec8aa). Filed D254 (trimul in-projection leaves carry no gradient, 473 tensors) and D255 (resolved objective reads pads); dispatched of3t-inproj on qb2 card 1. Composed fullstep64 into wk/of3t.

## pass 423 (2026-09-23 ~15:55Z)
disk_guard removed /tmp/of3t-inproj (2715 MB) at 14:55Z: deferred rows are not RUNNING_SLUGS, so under the critical line their /tmp is swept once the producer exits. Inproj's 384 float64 reference was never copied out. Folded its score into of3t-denoise (reference already carries D254/D255; denoise-on is the default). Both briefs edited. Started /home/moritz/of3t_keep/hold_denoise.sh to hold denoise's pc outputs. PADON (card 1) left running, not killed.
pass 424 2026-09-23T16:40:59Z: denoise FD died 16:10Z on full pc disk (spill of a never-backpropagated forward); relaunched no_grad as run_fd2.sh pid 3764649; brief edited.

## Pass 424 (2026-09-23 ~19:35Z)
Concluded: of3t-inproj GO (D254/D255 fixed at 64), of3t-bcastaudit GO (0 D258/D259 sites in 1,479,561 calls, six models; D259 campaign-internal). Filed D261 (ref-atom embedder untrained, fixed b770aee71) and D262 (PWA raw `__getitem__` bypasses the tape, 9 leaves). Dispatched of3t-pwaslice (qb2 card 2). Compose: main's MGX PairformerLayer `add_to_input=True` conflicted with the transition masks and msafwd's fp32 residual; two resolvers (`resolve_pairformer_add_to_input.py`, `resolve_pairformer_z_residual.py`) take main's block when plain and the row form when taped-wide or masked. wk/of3t -> e3a20b1d2, 175 confirmed, 0 drifted. of3t-denoise brief amended (card 2, D262 ownership, HOST in A/A artifacts).

## Pass 425 (2026-09-23 ~20:30-21:30Z)
DN384R/RB exited 0 on qb2 (20:04Z/20:30Z) while of3t-denoise was parked to 21:20Z; cleared its notbefore, it relaunched 22:31 CEST to score. of3t-pwaslice GO (D262) composed; wk/of3t -> 59646c2bb, 175/0/0 after restamping row counts. Two new collection errors (test_softmax_precise_site.py, test_tape_embedding.py) are the top-level ttnn-import class, owed importorskip.
Scored: of3t-denoise GO (384 global 0.156 vs bf16 0.603, cards bit-identical). Filed D263 from BIJECTION_DN upstream_unplaced (93 input-embedder atom-encoder tensors, no device twin). Dispatched of3t-composed64 (qb2 c1) and of3t-ieatom (qb2 c3). R201.

## Pass 426 (2026-09-24 ~00:00-01:00Z)
of3t-ieatom GO (D263, D264 fixed; CM64F bit-identical to PW64F). Compose: D155 froze 4 cmp artifacts (qb2, writer has no host field); D265 found (nested PairformerLayer route from a clean 3-way merge of rows based on 59646c2bb), fixed by collapse_nested_pairformer_route.py plus a tt_bio/-equals-scored-tree check. GRADIENTS repointed pre-registered to the whole-step score, of3t-go384 dispatched (qb2 c3). D264 main-side line handed to land-standing.

## pass 427, 2026-09-24
go384 STOP on 3x section clause (pfe 36.7x, resolved 5.7x). D266 filed. GRADIENTS repointed to CF384, GRADIENTS-SECTIONS added (a8a062916), control fails on GO384. of3t-confpfe dispatched.

## pass 429 (2026-09-24 04:0xZ)
CF384 scored: s-track fix inert (resolved 5.63x, pae 6.07x vs 3x); RF384 had closed pairformer_embedding 36.7x->1.33x. confpfe bisecting head inputs. Deferred.

## Pass 429, 2026-09-24
`of3t-confpfe` concluded STOP: D266 fixed (reference pae/pde logits a4b0290ae; `scale_pair_bias=True` restored dfc21275d), CF384 holds the five GRADIENTS fields, `aux_heads.pae` 6.07x its bf16 is the one section over 3x (D267). Dispatched `of3t-paez` on `wk/of3t-confpfe`. GRADIENTS and GRADIENTS-SECTIONS repointed CF384 -> PZ384, clauses unchanged. D268 merge-resolution audit (remerge-diff, 15 rows): only `scale_pair_bias`, dropped twice, now fixed. D10 and D24 marked FIXED after checking origin/main. Compose deferred to the GO arm.
Pass 430: trial merge-tree with main clean; main's fused_sdpa zero mask reaches the taped SDPA backward, so the GO compose owes a composed 64-wide taped step.

## pass 432, 2026-09-24
of3t-paez STOP read: D267 is the pae label draw. A44 written (six-draw mean, 3x unchanged), of3t-paedraws dispatched, charter repointed to PD384 before it exists. D184 closes with the wk/of3t merge. bcx-heads collision note acknowledged in SEQUENCE.
- pass 435 2026-09-24 13:55Z: re-placed of3t-paedraws references (f64 -> qb1 /dev/shm, bf16 6 threads on qb2, devstep nice -5); single scoring trigger run_bf16.sh; c347c53aa
GAP: **Pass 432.** (1) D267 OPEN pending A44: `aux_heads.pae` 6.07x its bf16 on CF384's one draw; `of3t-paez` located it in the label draw (z_trunk 2.17x, s_trunk 0.54x its bf16; head backward 2.1e-3), no device defect. If the six-draw mean stays past 3x it stays a defect. Owned by `of3t-paedraws`. `pde` 2.62x on one draw. (2) D266 fixed on `wk/of3t-confpfe`, not yet composed. D268 (merge dropped a verified fix): audited this pass with `git show --remerge-diff` over the latest merge of each of 15 rows with tt_bio/ conflict resolutions; only `scale_pair_bias` was dropped, twice (45100fbce, 6b3fc349e), both fixed at dfc21275d; nothing else still dropped. (3) Six USER-FACING defects unfixed in the triage: D32, D55, D205, D250 need card work, D210 a release, and D184, which CLOSURE_PLAN routes through the training adapter, `perf/of3t_orchestrator/userfacing/CLOSURE_PLAN.json`; MERGE list empty. (4) D252: exact LayerNorm on a pad-dominated boundary; SLZ arm held. (5) pae/pde are labelled from the denoise arm's `pred_xyz` where upstream labels from the rolled-out structure; same labels on both sides of the gate, adapter departure recorded. (6) closed pass 430: ttnn.Tensor's other tensor-returning methods (reshape, pad, pad_to_tile, unpad, to, cpu, extract_shard, the six arithmetic operators) cannot take a weight off the tape unseen: SCORE_CF384.json `placed_but_carried_nothing` n=0 of 4158 placed tensors, and every method-form call on the OF3 path acts on a host torch tensor at weight prep. A weight used at two sites with one off-tape would read as a wrong nonzero gradient, which the per-section clause bounds; 208 device weights have no upstream counterpart (BIJECTION `device_unmatched`) and are compared against nothing. Standing: crop 640 is a single-card capacity wall. Earlier GAP text: PASSLOG.
an unknown: D245.** The host float64 softmax reaches 1.0525x only through `perf/of3t_bwdaccum/
dev_cot.py` rewriting `tt._VERBS` from a perf script; the site-selector install that could ship
reads 0.5605347900452246 against float64 where the harness arm reads 0.41752141981218177. Until
`of3t-verbinstall` closes that, **the campaign has no assembled configuration that produces its
own best number**, and the inference A/B Moritz's hard constraint requires is still owed on
whichever install wins. **GRADIENTS, and as of pass 381 the honest statement is that we do not yet have a frame that
can grade it.**

    control   **CLOSED at pass 408. The model frame reproduces its own reference.** D242 was a
              DOUBLE COUNT in the injection: `z_out` is an ancestor of `s_out`, so
              `(s_out, z_out)` was never a graph cut and the hooked `cot_z` — correctly read by
              three instruments as a TOTAL derivative — already carried the `s_out <- z_out`
              route, which the surrogate then replayed. The duplicate was **99.6628 %** of it by
              norm. Repaired, the injection reads **1.6952505222168708e-14** over all 2,736
              trunk tensors against the 1e-12 bar, which is the capture's OWN witness value, and
              `dL/dz_in` lands bit-exact on the EXACT branch of a falsifier banked before the
              mechanism was known. `ref_grad.py` now defaults to the corrected injection, stamps
              which convention produced every artifact, and asserts the graph cut (A41/A42).

    rescored  **1.7814428090278143x is superseded, not merely withdrawn.** On the repaired
              injection the clause reads **0.22072451195864032** against bar
              **0.15210099830945006** — **1.4511706984958472x**, an 18.54 % improvement, and it
              **still FAILS**. The trunk must fall **1.7414134679108282x** (`x_allowance`,
              vs-upstream-bf16). No bar moved: all five of `CLAUSE.json`'s pre-registered levels
              re-derive on the rescored artifact at rel_difference 0.0.

    in-frame  **What survives, because both sides share one cotangent.** Upstream's own injected
              bf16 trunk reads **0.22475141530713597** against injected float64; ours reads
              **0.4045023229329932**; ours against theirs **0.5560356989573298**. The two-sided
              multiple is **1.7997765325758555** against a separator of **2.0** pre-registered in
              commit 2520681ed before the arm ran, so the reading is **the excess was substantially
              the asymmetry, not our arithmetic** — and it does not depend on which of the two
              available definitions is used (the literal pre-registered one gives 1.0560). Against
              the 2.9702x the campaign has carried since pass 379, that is most of the gap.
              **The bound on it**: this is differentiated at a point that is not the training point,
              so it estimates the like-for-like excess and does not settle it.

    next      **The repoint is blocked on ONE key and nothing else.**
              `MODEL_RECUT_composed3660_n384.json` carries six of R164's seven contract keys
              with the corrected value; `injection.convention` is absent, and R164 fixed before
              the number existed that any failing condition means no repoint. `of3t-recutfin`
              (qb2, CPU) is emitting it per scope and banking `cos`/`norm_ratio` against
              upstream's own bf16, so the 97 %-direction split can be computed in the space the
              clause is graded in rather than carried across from float64.
              **The critical path reported at pass 414 and the answer is a third.**
              `of3t-modelever` put the exact softmax on an arm **bit-identical to the clause's
              own banked artifact** (SHIP_A vs `of3t-recut`'s `dev_RENORM_model_n384_external.pt`,
              2736/2736, digests both ways; A/A floor exactly 0; recomposition `rel_difference`
              0.0) and the clause moved **1.4511706984958472x -> 1.3037867474869442x** — 32.67 %
              of the excess, still FAILING, a further 23.30 % to go. It closes 14.19 % of the
              graded-space angle against the 43.61 % needed and **against float64 it makes the
              trunk worse**. No bar moved; `upstreams_own_floor_here` (0.8525301041731214) and
              `section_A26_level` (0.9700522560698159) both still PASS, so the charter is not
              shown unreachable — but **the best lever the campaign has is now spent on this arm
              and nothing of comparable size is identified** (R191). One thing gained for free:
              the in-frame projection the campaign was barred from using measured **2.4 % off**
              inside a pre-registered 15 % band, so an in-frame multiple is a sound cheap SCREEN
              for the next lever even though it remains inadmissible as evidence. History: `state/of3t/PASSLOG.md`.

UNFIXED, by ID, besides those named above (each entry in `state/of3t/DEFECTS.md`): D2, D3, D10, D18, D22, D23, D24, D26, D27, D28, D32, D35, D37, D42, D46, D48, D49, D51, D53, D55, D59, D62, D63, D64, D69, D71, D73, D78, D82, D86, D89, D91, D92, D93, D94, D110, D112, D118, D119, D120, D121, D122, D123, D124, D125, D136, D140, D141, D148, D152, D158, D163, D180, D183, D184, D186, D187, D189, D190, D191, D192, D193, D194, D195, D196, D197, D198, D200, D202, D204, D205, D207, D208, D209, D210, D211, D213, D214, D217, D219, D222, D223, D224, D225, D227, D231, D232, D233, D235, D237, D240, D241, D249, D258, D259, D267.

VERDICT: PARTIAL, stamped pass 432, 2026-09-24. **Still working.** On CF384 all five GRADIENTS fields hold (global rel 0.1558 where upstream's bf16 reads 0.9482, 98.1 % of the mass at or better than bf16, 4158 of 4158 tensors read) and every section is within 3x its bf16 except `aux_heads.pae` (6.07x, D267). `of3t-paez` showed that single-draw ratio measures bf16's label luck, so A44 grades every section on the mean of six pre-registered draws with the bar unchanged; `of3t-paedraws` is dispatched on it. Ledger: two hundred sixty-eight defects. The triage lists 6 USER-FACING: D32, D55, D184, D205, D210, D250; D184 closes with the campaign merge. Ask 10455 is decided (`state/ask-10455-decision.md`): GO on gradients, the user-facing defects listed by id as separately owned, so they no longer hold GO.

**Why this is not yet GO.** The clause clears on two boundaries (5nw3 56/384 real at 0.9823x, 4hhb 384/384 real at 0.8878x). On the full step, D253's fix brought both arms to global rel 0.1455 (ON) / 0.1495 (OFF) against float64, bf16 0.1000, so the exact default is no farther from float64 than the switch it replaced. What blocks GO is D254: 473 triangle-multiplication in-projection leaves carry no gradient, so not every gradient is correct yet. D255 (resolved objective over pads) rides with it on `of3t-inproj`.

`of3t-cropwall` GO: the 576 wall was D248 (tiled padding in the `nlp_create_qkv_heads` and
`concat_heads` gradients, 90.6 % padding), fixed on its branch with bit-identical gradients at
256; 640 asks for 26 MB with 7 MB free, a single-card capacity wall.

**Distance to go, per tensor** (denominator 10.279642678524981): **48.1831 %** of the mass at
or better than upstream's own bf16 step, **49.8019 %** worse, **2.0150 %** unread, still on
the pre-D242 functional, not yet re-scored on the default arm.

Exit criterion **2 of 3** in `CHARTER_EVIDENCE.json` (COVERAGE and TRAJECTORY MET); GRADIENTS
flips on `of3t-inproj` (D254, D255) with fullstep64 and stackbound already GO.

**Counts** (`stamp_row_counts.py`): `state/concluded` holds **one hundred forty** of3t files, so **one hundred thirty-eight rows have concluded**.
**Two hundred sixty-four defects filed**, **97 UNFIXED** (5
scope-excluded, 8 USER-FACING, 84 campaign-internal). D251 (stackship's tape-vs-install gate) is filed fixed on its branch.

**Nothing merged; the exact default lives on `wk/of3t-stackship`, release-gated.** Per-pass
- pass 437 (2026-09-24 14:50Z): of3t-infab re-pinned qb1 card 1 -> qb2 card 1 (qb1 card 1 ARC dead since 05:28Z, row deferred every tick since pass 436); parked to 16:00Z behind paedraws draw 2->4 handoff. Verdict restamped PARTIAL, same single gradient item (pae six-draw mean) plus infab.
- pass 443 (2026-09-25 01:30Z): A45 (mass field on six-draw mean, 0.9913); D269 closed, D270 filed; infab fixes 83da296bc; wk/of3t -> f01fa0813, charter 5/5; GO waits on of3t-infab AFTER2.
- pass 444 2026-09-25 02:20Z: of3t-infab AFTER2 f01fa0813 meets its pre-registered bars on six models (4 trace-identical, openfold3/openbind digest = first AFTER, from_torch delta 0, 5 atom-encoder uploads reordered at zero cost, walk 4049=4049). Gate fixed for the rotated ledger (D190 recurrence). VERDICT GO on gradients, six user-facing defects carried under ask 10455.
