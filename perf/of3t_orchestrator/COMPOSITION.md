# OF3T composition: the six row branches as one reviewable branch

`wk/of3t` is what the OF3T campaign hands to the merge gate. It is composed by `of3t-orchestrator`
from the six row branches; no row merges anywhere itself, and nothing here goes to `main` without
Moritz.

This file records what was checked, so the composition is reviewable as a claim and not only as a
diff.

## Composed 2026-09-19 from `origin/main` at `bd643929a`. Snapshot head `9aaef9462`, 28 commits ahead. Regenerate with `compose_verify.sh`.

| row | branch head | files outside its own `perf/` namespace |
|---|---|---|
| `of3t-reference` | `ef4c3dc2c` | none |
| `of3t-tape` | `d30eb9903` | `tt_bio/taped_ttnn.py`, `tt_bio/tenstorrent.py`, `triatt_qkv.py`, `triatt_sdpa.py`, `softmax_generic.py`, `swiglu_fused.py`, `openfold3_confidence.py`, `openfold3_host_prep.py`, `tests/` |
| `of3t-equivalence` | `e513adcd1` | `tt_bio/train/optim.py` |
| `of3t-data` | `3af76e477` | `NOTICE`, `scripts/of3_port/`, `tt_bio/_vendor/openfold3/` |
| `of3t-perf` | `a8575be35` | none |
| `of3t-confidence` | *dispatched, no branch yet* | `tt_bio/openfold3_confidence.py`, the confidence path in `openfold3_fold.py` |
| `of3t-memory` | `0f0de49e9` | none |

**Row branches move under a composition.** Between merging the first six heads and pushing, three
rows had each pushed a further commit, so two of them read as *absent* from the branch about to
be published. Six clean merges does not mean six rows present. Assert
`git merge-base --is-ancestor origin/wk/of3t-<row> HEAD` for every row **after** the merges and
before every push; this file's table is the heads that assertion passed against.

## One ownership line that had to be drawn mid-campaign

`of3t-tape` already edits `tt_bio/openfold3_confidence.py` — its R7 fix removed a function-local
`import ttnn` there — and `of3t-confidence` was then dispatched to port that same file's s-path
onto the device. That is two rows on one file, which is how
`parallel-branches-independently-fix-same-defect-merge-silently-picks-one` happens.

**Resolved by handover rather than by sharing.** `of3t-tape`'s change there is landed and it does
not touch the file again; `tt_bio/openfold3_confidence.py` and the confidence path in
`openfold3_fold.py` are `of3t-confidence`'s from 2026-09-19, and it builds on `wk/of3t` so it
starts from the fixed file rather than from `main`. The ownership table above is the current
truth, not the original charter.

## What was verified

**Ownership was disjoint in fact, not only on paper.** The six branches touch **zero files in
common**. Each row's artifacts are confined to its own `perf/of3t_<row>/` namespace, and the three
rows that reach shared source reach disjoint parts of it. That is what
`sibling-perf-campaigns-need-namespaced-output-paths` asks for and it is why the merge is
reviewable rather than a negotiation.

**All six merge clean**, 15 commits ahead of `origin/main`.

**A clean merge is not a working tree, so the composition was tested against a control.** Test
collection on the composed tree against a detached checkout of `origin/main`, same interpreter:

- `origin/main`: 2303 tests collected, 106 errors
- `wk/of3t`: **2333 tests collected, 106 errors**
- the two error sets are **identical**, line for line

So the composition introduces **no new import-level breakage** and adds thirty tests. The 106 errors
are this host's missing `ttnn` and `torch` extras and are present on `main` too.

**No test executes on the orchestrator's host**: without `ttnn` everything errors on the missing
extras or skips, so this control proves the absence of new import breakage, not behaviour.
Behaviour was covered separately and on hardware — `of3t-tape` ran the composition in a detached
worktree on **qb2 card 0**, where `tests/test_tape_reach.py` and `tests/test_fused_unary_param.py`
gave **13 passed** and the composed tree reproduced **11003 / 11003** routed calls. Re-run that on
a card after any recompose touching the tape; the CPU control does not substitute for it.

**The finished evidence was recomputed, not read.** Both completed instruments were re-run from
scratch on the composed tree rather than having their committed JSON believed:

- **PROTOCOL §4, LR schedule** — 109,005 step comparisons across four configs (100,002 + 4,001 +
  3,001 + 2,001), **0 mismatches**, exact equality. Negative control perturbing step 37,000 by 1 %
  flags exactly one step. **Protenix regression guard: 100,002 steps, 0 mismatches**, so adding
  `plateau_until` left the Protenix path bit-identical — UNIFIED, NEVER PER-MODEL, demonstrated.
- **PROTOCOL §5, optimizer** — worst relative **2.738e-07 at step 191 on `bias`** against the
  1e-06 bar. Negative control (1 % on `proj` at step 50) flags exactly `proj` at 4.948e-04; the
  zeroed-gradient control puts all four tensors over the bar.
- Both reproduce the committed numbers exactly.

Two readings that came out of that re-run and belong with it. A one-step schedule offset is worth
**1.499e-03**, about 1500x the bar, which is the size of the wiring error §7 exists to catch. And
a uniform x1.01 on one tensor's gradient at every step moves the trajectory **2.547e-07**, under
the bar and invisible, because Adam cancels a uniform per-tensor scale — the standing reason the
per-parameter gradient check cannot be replaced by a longer run.

## What a reviewer of this branch must not conclude from it

**This branch does not train OpenFold3 yet, and it is easy to read the numbers as though it
does.** As of 2026-09-19 the tape routes **11003 / 11003** calls on the OF3 trunk and the taped
backward completes without error — but call routing and parameter reachability are different
things, and **2314 of 2531** reachable weights carry a gradient, 91.4 %. The 217 that do not are
named: 208 are pre-fused projections that are unused by construction while a tape is open, and
**9 `PairWeightedAveraging` weights are unexplained and open**. Before `of3t-tape`'s
`3aef07969`, **0 of TriangleMultiplication's 8 weights was an autograd leaf** — the backward
returned, the input got a gradient, and no parameter did.

**Report leaves against total, never leaves alone.** 2119 looked healthy; 2119 of 2531 was
actionable. A count with no denominator is how that survived a 100 %-call-routing result.

What is verified, and at what scope — the full table is `~/.coworker/state/of3t/EVIDENCE.md`:

- **Whole domain, exact**: the LR schedule, 109,005 comparisons over four configs, 0 mismatches,
  plus a 100,002-step guard confirming the Protenix path stayed bit-identical.
- **200 steps, no model**: the optimizer under injected gradients, worst relative 2.738e-07
  against a 1e-06 bar.
- **Module scope only**: gradients against finite-difference-validated float64 references
  (8.71e-03 to 2.00e-02), and a 20-step trajectory whose divergence *decays*, exponent -0.465.
- **Stack scope, and it FAILS**: every pair-track sub-module passes alone (0.0092 to 0.0172
  against a 0.05 bar) while the assembled block reads **4.3e-01 to 1.4e+00**. Passing parts do
  not compose into a passing block, and every earlier result on this branch is module-scope.
- **Not run at all**: the whole-model per-parameter comparison against the frozen bundle, the
  whole-model trajectory, coverage, their own training test, and any s/step figure at any clock.

One number a reviewer should carry away about method: turning `fp32_softmax` off moves the
triangle-attention weight gradient **3.2x** (1.449e-01 to 5.239e-02) while moving the forward
**12 %**. A forward comparison cannot see that class of error, which is why this campaign
separates the two instruments and does not let a forward reading stand in for a gradient one.

Our clipping is not theirs: their global norm **excludes disabled parameters** and ours does
not (clip 0.108 against 0.662 on a measured case, and their runner disables the confidence head
on 4 of 5 datasets in their own `initial_training` config), and their shipped default is
**per-sample** clipping, which changes the direction of the accumulated update rather than only
its length. Clipping feeds the update, so both are live defects. The EMA, by contrast, **is not
on the update path** — it is updated after `optimizer.step()` from the model and read back only
for validation — so the fact that we have none does not affect a weight trajectory, though it
does block any claim about validation metrics or about matching a published checkpoint.

**The honest status: the state-free half of OpenFold3's update rule is verified, and the
model-dependent half is verified only on single modules.**

## What this branch fixes, and what it has found but not fixed

`~/.coworker/state/of3t/DEFECTS.md` is the full list with attribution and measurements. In
short: this branch **fixes** weight discovery that was blind to anything a module built for
itself (on four models, not just OF3), clipping that differed from upstream in two independent
ways, and four silent tape blind spots. It **does not fix**, and deliberately leaves
release-gated, OpenFold3's trunk pair bias arriving at 20 % of its intended value in all 48
blocks — because one flag drives three attentions and two of them need opposite values, so the
fix is to split the flag and the end-to-end Angstrom cost has not been measured yet. It also
records that **AF2 receives no gradient at all** and cannot train, which is outside this
campaign and needs its own row.

## Defect found by composing, and it is not a merge artifact

`NOTICE` cited `docs/openfold3-vendor.md` for the eleven flag-gated vendor modifications, and
that file existed nowhere — not on the composition, not on `wk/of3t-data` itself, so it was the
row's own gap. No single row could have found it, because each branch looked fine alone.
**CLOSED** by `of3t-data` at `dd649bf60`, and `compose_verify.sh` now checks every `docs/*.md`
the diff mentions, so what is guarded is the class rather than the instance.

Also worth a reviewer's eye rather than a fix: the vendored tree is now **mixed-version** —
`core/utils/relpos.py` from 0.4.5, everything else 0.4.3. `of3t-data` states this in `NOTICE`
rather than hiding it, which is the right call, but a mixed vendor is a provenance hazard and
`scripts/of3_port/audit_vendor_provenance.py` is what has to keep it honest.

## Recomposing

The row branches move. Recompose from `origin/main` rather than merging into a stale `wk/of3t`,
re-run the two instruments, and re-diff the collection error set against `origin/main` on the same
interpreter. The control is the point: an absolute error count says nothing.
