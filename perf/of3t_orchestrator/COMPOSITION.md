# OF3T composition: the six row branches as one reviewable branch

`wk/of3t` is what the OF3T campaign hands to the merge gate. It is composed by `of3t-orchestrator`
from the six row branches; no row merges anywhere itself, and nothing here goes to `main` without
Moritz.

This file records what was checked, so the composition is reviewable as a claim and not only as a
diff.

## Composed 2026-09-19 from `origin/main` at `bd643929a`. Head `c84287335`, 23 commits ahead.

| row | branch head | files outside its own `perf/` namespace |
|---|---|---|
| `of3t-reference` | `ef4c3dc2c` | none |
| `of3t-tape` | `5a2efa001` | `tt_bio/taped_ttnn.py`, `tt_bio/tenstorrent.py`, `triatt_qkv.py`, `triatt_sdpa.py`, `softmax_generic.py`, `swiglu_fused.py`, `openfold3_confidence.py`, `openfold3_host_prep.py`, `tests/` |
| `of3t-equivalence` | `aa23b558d` | `tt_bio/train/optim.py` |
| `of3t-data` | `fbdbd52da` | `NOTICE`, `scripts/of3_port/`, `tt_bio/_vendor/openfold3/` |
| `of3t-perf` | `a8575be35` | none |
| `of3t-memory` | `0966364a6` | none |

**Row branches move under a composition.** Between merging the first six heads and pushing, three
rows had each pushed a further commit, so two of them read as *absent* from the branch about to
be published. Six clean merges does not mean six rows present. Assert
`git merge-base --is-ancestor origin/wk/of3t-<row> HEAD` for every row **after** the merges and
before every push; this file's table is the heads that assertion passed against.

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
- `wk/of3t`: **2307 tests collected, 106 errors**
- the two error sets are **identical**, line for line

So the composition introduces **no new import-level breakage** and adds four tests. The 106 errors
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

**This branch does not yet train OpenFold3, and the gap is easy to misread.** As of 2026-09-19
the tape routes **11003 / 11003** calls on the OF3 trunk and the taped backward completes without
error, but `of3t-equivalence` measured on qb1 card 3 that **0 of TriangleMultiplication's 8
weights is an autograd leaf** — the backward returns, the input gets a gradient, and no parameter
does. The same construction sits at every triangle multiplication in the trunk, the MSA stack and
the template stack.

The two headline numbers mean different things and neither substitutes for the other:
**11003/11003 is call routing; 0 of 8 is parameter reachability.** Quote both or neither. The fix
is assigned to `of3t-tape`; the root cause is in `Module.__init__`'s `torch_to_tt` (weights
arrive as raw handles, so the tape sees constants) and in `lora.weights_for` (it discovers only
sites routing through `tt_bio.ops.linear`, which `tenstorrent.py` calls at 4 sites in 12k lines).

**Report leaves against total, never leaves alone.** A count with no denominator is how this
stayed invisible underneath a 100 %-coverage result.

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
