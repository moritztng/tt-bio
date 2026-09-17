# Landing the 0.6118 s: what it takes, and the one prerequisite a default flip alone would miss

C12's only measured win is **+0.6118 s / 826.0 Mcycles at the fold** (95 % CI [+0.4127, +0.8109]),
the silu + cond-hoist stack, at a forced and during-sampled 1350 MHz. It is **two independent
recoveries of the same session** (the row's own and the orchestrator's, agreeing to the digit), all
witnesses fired, every arm bit-identical across its reps, A/A correctly unresolved from zero,
sub-additive at 0.915 of its own in-session singles.

**Both flags default to `False` on `origin/main` and nothing has flipped.** That is the standing
`unshipped-flag-repeats-without-escalating-to-a-landing-task` / `merged-lever-defaults-off-is-not-a-
landed-win` failure class, so this file is the landing task rather than another repetition of the
number.

## The two flags are NOT symmetric, and that is the whole point of this document

    lever        flag                        site on main                 tt_bio/ diff vs main
    silu         TT_BIO_UNFUSED_SILU         tenstorrent.py:305           EMPTY  -> pure env flip
    cond-hoist   TT_BIO_DIT_COND_HOIST       tenstorrent.py:1415          26 insertions, 2 deletions

**silu lands as a one-line default change.** `git diff origin/main origin/wk/c12-unfused-silu-bh --
tt_bio/` is empty: the code already ships. `False` -> `True` at `:305` is the entire change. The
measured rule is *"unfuse silu when the output is L1-resident"*, not "unfuse silu" — the DRAM site
loses 0.1661 ms/call — and that routing is already in the shipped code, which is why the diff is
empty.

**cond-hoist does NOT, and a default flip alone would make the first fold of every process slower.**
On `main`, `__init__` sets `self._cond_w = None` and `_cond_weights()` builds lazily on first call
(`:9668`, memoised, called from `:9707`). So with the flag on, the **first hoisted fold pays the
0.34-0.57 s build inside the fold** to save 0.2415 s/fold. `origin/wk/c12-cond-hoist-block-timing`
is the branch that fixes it — its own comment at `:1425` says *"`_cond_weights()` is now built in
`__init__` when this is on, so its 0.34-0.57 s lands at model [load]"* — and it calls
`self._cond_weights()` from `__init__` at `:9690`. **That 28-line change is not on main.**

Consequence, stated concretely because it decides whether the flip is a win or a regression:

    long-lived worker (JapanFold serves many targets per process)   amortises after ~2 folds, clear win
    one-shot CLI fold (a user folds one target and exits)           NET LOSS of ~0.10-0.33 s without the fix

So the landing order is: **merge the 28-line eager-build change first, then flip both defaults.**
Flipping defaults on today's `main` would ship a first-fold regression to exactly the single-fold
case a new user hits.

## Accuracy, as it actually stands

Decision fixture is `cdk2x2_298`, which the 0.35 pass / 0.60 hold bar is written for:

    worst stack deviation        0.38302 A all-atom   (CA 0.28832)
    that fixture's seed floor    0.80218 A            -> the stack is 0.477x it
    A/A control                  0.00000 A, digests identical
    verdict field in the JSON    HOLD  (above the 0.35 pass bar, below the 0.60 reject bar)

At 512 aa the stack is **inside the seed floor on both metrics at all five seeds** — plDDT worst
0.80x the sampler's own 0.0221 scatter, all-atom worst 12.41 A inside a base-vs-base band of
5.35-21.91 A. An earlier "5.82x the seed floor" reading was retracted: it scored one seed against
the narrowest of ten seed pairs (see `../acc512_seeds/README.md`).

So it clears the standing bar — *"a perf lever that moves the digest is fine if the STRUCTURE clears
the kill bar"* — but it sits in the **HOLD band, not the PASS band**, and HOLD is by construction the
band where a human decides. Hence the ask rather than the flip.

## What is NOT claimed here

The fold-level delta is measured; the **absolute** fold time after flipping is not. s3's own base
median was 14.971 s under co-tenancy, which is not a number of record, so the projection is
14.881 - 0.6118 = **14.269 s** by applying a paired delta to the quiet-box base. A quiet-box absolute
has never been measured below 14.881 s. And the composed delta is **2.87-2.98x its own session A/A
floor against a pre-registered 3x gate** — `c12-compose-fold`'s s5 (48 reps, running 22:24Z) is
buying that margin. None of this changes the sign or the accuracy picture; it changes how tight the
error bar is.

Nothing in this file has been merged. Both flags still ship `False`.
