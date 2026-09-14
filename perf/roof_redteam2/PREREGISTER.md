# roof-redteam-2 — pre-registration

Written and committed before a single number was computed. The only inputs so far are the published
claim (`perf/roof_budget/ROOF_BUDGET.md` + `roof_budget_table.py` on `origin/wk/roof-budget`) and
round 1's state doc. No capture has been opened, no script has been run.

Round 1's rule applies: every number states its instrument and its roof. A prediction that turns out
wrong is a result, not a failure — the point of writing them down is that I cannot quietly move the
goalposts afterwards.

## Attack 1 — does contention scaling reorder the ranking? Predict ROBUST.

The waste column is `s_per_fold * 0.7011 - s_at_roof`, with one global scalar and an unscaled roof.
Per-unit scalars `k_i` would move it. Two things make me expect the top-3 order survives.

First the algebra has a constraint: the per-unit times must still sum to the cell, so `k_i` cannot
all move freely. Second the physics points the wrong way for a reorder. `k = benchlocked/unlocked`,
so a unit that suffers MORE from host load has a SMALLER `k`. The denoiser is the dispatch-heavy
side (4800 `DiffusionTransformerLayer` at 0.838 ms, 9600 `AdaLN` at 0.118 ms); the pairformer block
is 51.9 ms of device work per call. So I expect `k_denoiser < 0.7011 < k_pairformer`, which WIDENS
the 5.410 vs 3.867 gap rather than closing it.

Predicted: the denoiser needs `k_den/k_pair` of roughly 1.2x or more to overtake the pairformer, and
the expected sign of the effect is below 1.0. MSA overtaking the denoiser needs something near 2x.
CONFIRMED, with the reorder threshold quoted.

Residual risk I am naming up front: if a large slice of the pairformer row's 51.9 ms is itself
dispatch gap rather than device time, the sign argument collapses. I will check the per-call op count
before leaning on it.

## Attack 2 — do the three top-level units tile the fold? Predict SPLIT: disjoint yes, exhaustive no.

`roof_budget_table.py` names them in a comment: `PairformerLayer|1x512x384,1x512x512x128`,
`MSALayer|...`, `DiffusionModule|`, "every other captured unit is a child of one of these". The
nesting the brief points at is real but lands on rows that are NOT in that list —
`Diffusion|1x4480x3,1` (200 calls, 4035.5 MB vs the module's 4035.8) and `DiffusionTransformer|`
(290.0 GFLOP/call) look like the same 200 denoiser steps seen one and two levels down, and
`PairformerLayer|1x512x512x128` (16 calls) is very likely the pair-only stack INSIDE the 16
`MSALayer` calls. If so the sum is not double counting and the table is right to exclude them.

The claim I expect to break is exhaustiveness, not disjointness. `13.708 + 8.226 + 2.710 = 24.644`
session seconds scale to 17.278 against a 17.340 s cell: 99.64 %. That leaves 0.36 % for the input
embedder, the recycling embedder, the pair/MSA initialisation, the confidence head, the structure
output and every host gap between them. I do not believe a real fold has 0.36 % outside those three
units. Predicted: the three are disjoint, the 99.6 % is flattered by the single contention scalar
being fitted on the same three units, and the honest statement is "the three units the capture set
covers", not "the units that tile the fold".

## Attack 3 — does the floor take max(traffic, compute) per unit? Predict BROKEN, by 5-15 %.

Read of the source, not a computation: `binding_floor_s = fold_B / stream_roof` where `fold_B` is
the byte sum over exactly those three units. There is no `max` anywhere in it. At the granularity
of the three units this happens not to matter, because all three have arithmetic intensity far under
the 247.1 balance (75.2, 61.9, 78.6). The defect is one level down. A roofline floor evaluated on an
aggregate is always at or below the sum of the floors of its parts, because `max(sum) <= sum(max)`.
The three compute-bound `Transition` rows at 303-380 FLOP/byte are children of these units and their
arithmetic does not overlap the rest of the block's traffic in any way the aggregate can see.

Predicted: computing the floor as `sum over ops of max(F_i/104.93 TF/s, B_i/424.7 GB/s)` lands above
6.934 s, in the 7.3-8.0 s band, and 10.406 s of headroom becomes something closer to 9.5-10.0 s. The
direction is certain; the magnitude is the prediction. Note this still prices compute-bound ops at
the dense-cube rate, which `roof-budget` itself calls a rate no op in this fold reaches, so the
corrected number remains a floor. If `roof-pair-transition` lands a shape-honest rate for the pair
Transition I will use it and say so; if not I will say I could not.

## Attack 4 — is the 35-row MSA a benchmark artifact? Predict BROKEN, the fixture is not typical.

35 aligned sequences is what you get from a tiny or failed search. A real JapanFold request that goes
through an MSA server gets hundreds to thousands. If that is right, two things follow and the second
is the bigger one: `roof-msa-ladder`'s 1.305 s prize is mostly a benchmark artifact at depth 35 and
shrinks toward zero as real depth approaches the 1024 pad, AND the published 17.340 s cell is itself
optimistic for a real fold, because a deep MSA fills padding the cell currently pays for but does not
use. Predicted BROKEN. I will take the pass condition seriously though: if the fixture's depth is
what the deployed path actually produces for this target, that is a clean CONFIRMED and I will say so.

## Attack 5 — do the two byte instruments still disagree? Predict CONFIRMED, with a third rule found.

Round 1 established `census.py:split_io` drops the read of a read-modify-write (8.49 % low on the
block) while `baseline_attrib.py:Census.charge` does not. The tip's 2.9449 TB comes from
`real_traffic.py`, a third file, which dedupes on buffer address. Deduping on address is the right
fix for double-charging a buffer read twice, and it is also exactly the rule that would charge an
in-place op's destination ONCE when the truth is a read and a write. Predicted: `real_traffic.py`
does not have `split_io`'s defect but has its own, of the same sign; the three instruments do not
agree and none of them says so. I expect to restate at least one published fusion prize.

## What a STOP looks like

If attacks 1, 2, 4 and 5 all confirm and attack 3's correction is under 2 %, the floor stands and the
right output is "the table survives, stop poking it". I would rather write that than manufacture a
finding, and I am writing this sentence now so that it is not a post-hoc excuse either way.
