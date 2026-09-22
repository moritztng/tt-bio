# pvx-didittransfer — did Boltz-2's optimizations already reach Protenix? Four folds, one card

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: §ACCELERATE "THE PERF METHOD" + ALWAYS-ON |
memories read: `qb2-aiclk-governor-sets-fold-time-not-cotenancy`, `perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`,
`perf-page-cell-is-historical-not-live-baseline`, `op-ab-must-interleave-arms-compile-warmup-bias`,
`benchlock-protects-co-tenants-not-just-the-caller`, `eligibility-firing-condition-is-not-a-code-fact`,
`sibling-perf-campaigns-need-namespaced-output-paths`, `ssh-remote-background-launch-stdin-hang`,
`perf-page-matched-batch-protocol-recurrence`, `one-size-tuning-is-a-standing-defect-class`.

Branch `wk/pvx-didittransfer`, worktree `/home/ttuser/.coworker/wt/pvx-didittransfer` on qb2.
Arms folded on **qb1 card 1**, artifacts `perf/pvx_didit/` (mine alone).

VERDICT: NO-GO on the premise. **The optimizations did not transfer.** Over the same 33-day window
in which Boltz-2's 512 aa fold fell from 26.770 s to 17.839 s (**1.5006x**), Protenix-v2's fell from
53.894 s to 51.207 s (**1.0525x**). Same card, same pinned clock, same instrument, same timed
region, seven clean sessions, two per Protenix arm. Protenix kept 95 % of the time it had on 2026-08-15. The `78ed5a1e`
accurate-softmax fix that landed inside the window is not hiding a transfer either: it costs
0.268 % of the fold, so the window's gross Protenix gain is 1.0557x instead of 1.0525x and the
answer does not move.

This is a NO-GO on the campaign's *opening premise* ("maybe we already transferred the
optimizations"), and it is simultaneously the result that keeps `pvx-inventory`'s screen honest:
a top-down measurement that never enumerates a lever lands on the same order of magnitude as the
bottom-up one. What it does NOT say is that Protenix has 1.5x of headroom sitting behind a flag.
Those are two different claims and only the first is measured here.

SCOPE: **`predict_one` — featurize, fold, write the CIF — one process, timed per fold after a cold
fold, host and device wall clock, `hoist=False`.** That is `tt_baseline.build_fold`'s default
region (`scripts/gpu_vs_tt/tt_baseline.py:437-447`, `timed_region="predict_one (featurize + fold +
CIF write)"`), and it is the region `pvx-baseline` used for every cell of its own. One definition,
all four arms, no exceptions.

**It is NOT the perf page's device-only forward, so no number in this document is compared with the
published 50.543 s or 23.504 s cells.** My old-tree Protenix arm is the tree the 50.543 s cell
shipped from and it reads 53.935 s here: different region, different board, different clock, and
chasing that 6.7 % is a separate job from the one this row was given. Every ratio below is read
between two arms of mine that share the region, the card, the clock and the instrument, which is
what makes it a ratio rather than a quotient of two numbers from two documents.

ARMS: four sessions, alternating old tree and new tree, on **qb1 card 1, Blackhole p150a** (qb1 is
four single-ASIC p150a boards, so no board partner shares a power budget with this card; the other
three boards were idle and every fold recorded **zero foreign `/dev/tenstorrent` holders**).
**AICLK pinned at 1350 MHz and sampled at 4 Hz DURING every fold: mean 1350.0, min 1350, zero
re-asserts on all four sessions.** ttnn 0.68.0, grid [11,10], benchlock held for every session,
box verified idle before the first fold (loadavg 0.09, no other logged-in user).

| arm | tree | commit | date | n | median | A/A floor | digest | plDDT |
|---|---|---|---|---:|---:|---:|---|---:|
| `ptx_old_s1` | Protenix at the published cell | `61d05cc5` | 2026-08-15 | 3 | **53.935 s** | 0.130 s (0.24 %) | `7cea4db45445d1f1` | 0.82857 |
| `ptx_old_s2` | the same tree, second session | `61d05cc5` | 2026-08-15 | 3 | **53.853 s** | 0.054 s (0.10 %) | `7cea4db45445d1f1` | 0.82857 |
| `ptx_new_s1` | Protenix on `origin/main` | `ec4b66412` | 2026-09-19 | 3 | **51.188 s** | 0.065 s (0.13 %) | `22bc3eaafd886c17` | 0.810638 |
| `ptx_new_s2` | the same tree, second session | `ec4b66412` | 2026-09-19 | 3 | **51.227 s** | 0.054 s (0.11 %) | `22bc3eaafd886c17` | 0.810638 |
| `b2_old_s1` | Boltz-2 at the published cell | `0d69dc1de` | 2026-08-17 | 3 | **26.770 s** | 0.185 s (0.69 %) | `a8f25b20002032a9` | n/a |
| `b2_new_s1` | Boltz-2 on `origin/main` | `ec4b66412` | 2026-09-19 | 3 | **17.839 s** | 0.227 s (1.27 %) | `57995ebbbe68ff1a` | 0.844645 |

Each arm does the model's own full work, read off the tree under test and recorded per run:
Protenix 10 recycles / 200 sampling steps / seed 0 in **both** arms, Boltz-2 3 recycles / 200 steps
/ seed 0 in both. Nothing was shortened to make a number.

**Provenance is checked against git, not assumed.** Both extracted trees were verified file by
file: `md5sum` of `tt_bio/protenix.py` and `tt_bio/tenstorrent.py` on disk against
`git show <commit>:<path> | md5sum`, four of four, with the two trees differing on every file.
The Boltz-2 old arm reuses `pvx-baseline`'s already-verified `0d69dc1de` extraction rather than
making a second copy of it.

**Instrument:** `pvx-baseline`'s `cell.py`, byte-identical (md5 `21e0770f080a4d965203b59a193107be`)
across both of my trees and that row's own tree, committed here as `perf/pvx_didit/cell.py`. It is
that row's pass-5 copy, which carries the per-fold foreign-holder census and per-fold `loadavg1`
but not the three-way contention split that row added afterwards; per-fold loadavg ran 1.27-2.55,
which is this fold alone on an otherwise empty box.

**Three independent cross-checks that the arms are what they claim to be, and all three pass.**

1. My `b2_new_s1` digest `57995ebbbe68ff1a` and `b2_old_s1` digest `a8f25b20002032a9` are the
   digests `pvx-baseline` got from the same two trees on the same card class. Same computation,
   not merely a similar one.
2. My `ptx_new_s1` digest `22bc3eaafd886c17` is that row's Protenix digest as well, even though
   its tree was `bd643929a` and mine is `ec4b66412`. The only engine change between them is
   `pvx-eligibility`'s matmul-config keys, which claimed bit-exactness by keeping `K_block` equal
   to the whole contraction, and the digest agrees.
3. Same pair, timed: 52.141 s at `bd643929a` against my **51.188 s** at `ec4b66412`, **1.86 %**,
   bit-exact. That is the eligibility lever showing up as time on a fold that produced identical
   output, measured by two rows on two days.

BOLTZ2-RATIO: **26.770 / 17.839 = 1.5006x**, one clean session per arm, A/A floors 0.69 % and
1.27 %. `pvx-baseline` has the same quantity from three clean sessions per arm and reads
**1.5326x**; the two are **2.1 %** apart, which is the size of that row's own cross-session spread
plus mine. Both rows, independently, say the published **1.63x is high** and the defensible figure
is **1.50-1.53x**. I take that row's 1.5326x as the better estimate, because three sessions per arm
beat one, and I record my 1.5006x beside it rather than averaging them into a new number nobody
measured.

PROTENIX-RATIO: **53.894 / 51.207 = 1.0525x. This is the answer, and it is a no.** Both Protenix
arms carry two sessions, so the ratio is read across sessions rather than out of one:

| arm | session medians | mean of medians | cross-session spread |
|---|---|---:|---:|
| `61d05cc5` | 53.935 / 53.853 s | **53.894 s** | 0.082 s (0.152 %) |
| `ec4b66412` | 51.188 / 51.227 s | **51.207 s** | 0.039 s (0.076 %) |

    53.894 / 51.207 = 1.0525x     pooled over all 12 folds: 53.879 / 51.227 = 1.0518x

Protenix-v2 took **2.687 s** out of its fold in the window that took **8.931 s** out of Boltz-2's.
In proportion: Boltz-2 kept 66.6 % of its time, Protenix kept 95.0 %.

    if the window's Boltz-2 rate had reached Protenix:  53.894 / 1.5006 = 35.91 s
    Protenix actually reads                                               51.21 s
    unreached                                                             15.29 s

The floors make this unambiguous rather than marginal. Within-session A/A is 0.10-0.24 % on the
four Protenix sessions, and the cross-session floors are 0.152 % and 0.076 %; the effect is
**5.25 %**, more than thirty times the larger of the two. For the verdict to flip, one arm would
have to move by thirty times anything four sessions have shown. Both trees also returned **one
digest across their six folds** (`7cea4db45445d1f1` and `22bc3eaafd886c17`), so neither drifted
between its two sessions.

**Why this was worth measuring even though `pvx-inventory` already screened it.** That row priced
3 of 7 blocked levers bottom-up and bounded the lot at 1.056x. This row enumerates nothing: it
folds the two trees. The two numbers are close, and the coincidence is worth stating carefully
because it is easy to misread. **They are different quantities.** 1.056x is what Protenix *would*
gain if every blocked shared lever fired; 1.0525x is what Protenix *did* gain in the window. They
are not the same measurement and the closeness is not confirmation of either. What the top-down
number does establish, and the screen could not, is that **no unenumerated lever is hiding in the
window**: whatever the shared stack gave Boltz-2, 94.9 % of Protenix's fold did not receive it,
and no lever list had to be complete for that to be true.

**What the shared `PairformerLayer` argument was worth.** The premise behind the campaign is that
Boltz-2 and Protenix instantiate the same `PairformerLayer` at five sites in `protenix.py`, so one
model's wins should reach the other. The measurement says sharing a class is not sharing a win.
`pvx-baseline`'s block census on these same two models has the mechanism: Protenix's Pairformer
costs 3.883x Boltz-2's, and that is 2.157x more calls times 1.800x more time per call. A lever that
is a fraction of Boltz-2's Pairformer is a smaller fraction of a Pairformer that is called twice as
often at twice the width, and the window's largest Boltz-2 wins (the device-residency family and
the L7/L6 diffusion levers) live in `boltz2.py` and are reachable from no other model.

SOFTMAX-ARM: **attribution, not a proposal.** `78ed5a1e` (2026-08-21) turned the accurate softmax
on per site for Protenix-v2 and OpenDDE, inside the window this row measures, so part of the
window's Protenix time could be a correctness fix cancelling a transfer. It is not. Fifth session,
same card, same pinned 1350 MHz, same instrument, today's tree with only the two Protenix softmax
sites put back in their pre-`78ed5a1e` state:

| arm | tree | softmax | n | median | A/A floor | digest | plDDT |
|---|---|---|---:|---:|---:|---|---:|
| `ptx_new_s1` | `ec4b66412` | shipped default, on | 3 | 51.188 s | 0.065 s (0.13 %) | `22bc3eaafd886c17` | 0.810638 |
| `ptx_nosm_s1` | `ec4b66412` | forced off at both sites | 3 | **51.051 s** | 0.045 s (0.09 %) | `50f474e0aaf62e0f` | 0.810921 |

    accurate softmax costs   51.188 - 51.051 = 0.137 s = 0.268 % of the fold
                             against the two-session mean 51.207 s it is 0.156 s = 0.306 %
    window gain without it   53.894 / 51.051 = 1.0557x   against 1.0525x with it

**The lever is counted, not read off a gate.** `TT_BIO_ACCURATE_SOFTMAX_AB=-protenix.trunk,-protenix.confidence`
was verified in two independent ways. In the tree under test,
`accurate_softmax_site("protenix.trunk", default=True)` and the same for `protenix.confidence`
return **False** under that environment and **True** without it, and those two tokens are the only
`accurate_softmax_site(...)` calls `protenix.py` makes (`tt_bio/protenix.py:1460` and `:2465`, with
`softmax_scope="protenix"` for protenix-v2). At the fold, the arm returns a **different CIF digest
and a different plDDT** from the default arm, so the flag changed the computation rather than
merely parsing. That digest change is the negative control: an override that fired on nothing would
have reproduced `22bc3eaafd886c17`.

0.268 % also agrees with what `78ed5a1e` measured for itself at 512 aa (+0.508 %), same order, and
the two readings differ by less than the difference between the harnesses. **So the fix accounts
for 0.137 s of the 15.29 s Protenix did not gain, 0.9 % of it.** The hypothesis that a
correctness-driven slowdown masked a transfer is refuted, and refuted on the arm designed to
support it.

Nothing here is a proposal to turn the accurate softmax off. It is the cheapest correctness fix in
the tree (`ttnn.softmax` rows sum to 0.977 instead of 1, a multiplicative error on every weight in
the row) and it costs a quarter of one percent.

SHIPPED: **0.0000 s.** This row measures, it does not ship a lever, and that is by design rather
than by shortfall. No merge, nothing flipped on `origin/main`, nothing to gate. The only commits on
`wk/pvx-didittransfer` are `perf/pvx_didit/` — instrument, run script, and the five arms' JSON.

## What the next row should take from this

1. **The premise is settled and the campaign does not need another measurement of it.** Protenix
   did not receive Boltz-2's window. Any further Protenix second has to be found in Protenix's own
   fold, which is `pvx-protenix-specific`'s census, not in a shared lever that was assumed to have
   been missed.
2. **1.53x, not 1.63x**, for the Boltz-2 figure Moritz intends to publish. Two rows now agree
   within 2.1 % and both land under 1.54x.
3. **`ec4b66412` is 1.86 % faster than `bd643929a` on Protenix, bit-exact.** That is
   `pvx-eligibility`'s landed lever showing up on a fold, which is the only thing this campaign
   has actually put on `origin/main` so far. `pvx-land` owns whether more of that exists.
4. The window gave Protenix **2.687 s**. It is not nothing, and it is not a transfer.
