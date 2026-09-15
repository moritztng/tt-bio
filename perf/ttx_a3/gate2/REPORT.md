# Re-gating `TT_BIO_SDPA_FUSED_LARGE_S` default-ON on current main

Verdict: **GO on the lever. HOLD the merge**, on one blocker that belongs to main and has an owner.

The previous pass (`ttx-a3-fused-sdpa-default-ship`, report in `perf/ttx_a3/gate/REPORT.md`) held for
two reasons. One is gone, one is replaced by a sharper one.

| previous blocker | now |
|---|---|
| esmfold2 / esmfold2-fast / esmc-6b could not load weights — 8 of 15 gate reds | **CLEARED.** All 8 green. |
| qb2 reset 5x in 3h, so four arms read "did not complete" | **WORKED AROUND**, not fixed. qb2 still resets; the arms are chunked per model and a crontab entry restarts the driver. |
| — | **NEW:** main's own `docs/size_ladder_baseline.d/` p300c entries are stale, so the size-ladder arm is red for every branch. Owned by `tt-bio-sizeladder-p300c-refresh`. |

## Where the lever actually fires, measured rather than argued

The route is gated strictly above the 1024-token cap. The lever census makes that concrete instead
of leaving it as a claim about the source:

    census_rf3-1024-rep0    SDPA_FUSED_LARGE_S  resolved True  served 0     declined 0
    census_rf3-1088-warmup  SDPA_FUSED_LARGE_S  resolved True  served 1088  declined 0

`SIZE_LADDER_EXTRA_RUNGS` is `{"rf3": (1088,)}`, so across the whole release gate exactly two kinds
of cell sit above the cap: rf3's 1088 ladder rung, and the capacity gate's 1536-token roster. Every
other arm — pytest, ux, perf, the 44-leg parity gate, and every other ladder rung — is at or below
1024 and cannot see this lever at all. Reading a green arm as coverage it does not have is the
easiest mistake available here.

## rf3 at 1088: 1.1556x, fully separated, 0 hangs in 8 reps

`perf/ttx_a3/attr_rf3_1088.sh`, four off and four on alternating, separate processes, 600 s cap per
rep with anything past it recorded as HUNG rather than crashing the run.

    arm   walls (s)              served  resolved   rc
    off   213  215  207  234     0       False      0 0 0 0
    on    191  189  186  186     1088    True       0 0 0 0

    mean off 217.2   mean on 188.0   ratio 1.1556x
    max(on) 191 < min(off) 207               fully separated
    off/off spread 13.0%   on/on spread 2.7%

The off arm is a real negative control, not an assumption: `served 0` and `resolved False` in all
four reps, so it genuinely takes the stock ladder. The on arm serves all 1088 calls in all four.

This is the lever's second independent perf reading and the first inside the gate's own fixtures:
1.1556x on a 6-step rf3 fold at 1088, beside the banked 1.1856x on a 200-step boltz2 fold at 1536.
Both are trunk savings, which is why a 6-step and a 200-step fold agree.

## Capacity at 1536 tokens: 13 PASS, 2 FAIL of 15

    boltz2         PASS   7.35G/23%   219.8 s      rf3            PASS   8.98G/28%   333.9 s
    esmfold2       PASS  19.36G/61%   506.5 s      esmc-300m      PASS   0.64G/2%      4.4 s
    esmfold2-fast  PASS  15.43G/48%   273.5 s      esmc-600m      PASS   1.10G/3%      5.3 s
    protenix-v1    PASS   6.79G/21%   227.4 s      esmc-6b        PASS  11.91G/37%    14.6 s
    protenix-v2    PASS  10.15G/32%   701.1 s      saprot-35m     PASS   0.08G/0%      3.1 s
    openfold3      PASS  13.67G/43%   679.3 s      saprot-650m    PASS   1.26G/4%      5.5 s
    openbind       PASS  14.79G/46%   597.8 s
    opendde        FAIL   size_guard    1.8 s      opendde-abag   FAIL  size_guard    1.8 s

Every model whose own ceiling allows 1536 allocates and completes on the shipped default. The two
failures read `screen/size_guard ceiling~1024`, refused in 1.8 s with no peak DRAM. They come from
`tt_bio/size_limits.py` `CEILINGS`, where opendde on blackhole is `residues=1024, pass_at=1024,
fail_at=1536, mechanism=TRUNK_FREEZE`, recorded 2026-09-10 on qb1 card 0 and qb2 card 1. A static
table lookup cannot depend on a call-time kernel-selection flag, so neither cell owes an off-arm
control. They are also the answer to main's standing BASELINE GAP red, which names exactly
`p300c/opendde` and `p300c/opendde-abag`: the fleet has no p300c cell for either because both refuse
this gate's token count.

## The two stalls, and why neither is the lever

Two folds hung during this gate, both above the cap, both on the shipped default. Both were treated
as candidate NO-GOs and both were run down.

**openbind at 1536.** Stalled at `trunk 0/4`, no progress event for 15 minutes, 241 % CPU
throughout, killed by `capacity_gate.py`'s own 900 s stall detector. off/on/off as separate
processes, plus the gate arm's own rerun:

    off1 PASS 14.79G/46% 634.7 s     on1       PASS 14.79G/46% 597.5 s
    off2 PASS 14.79G/46% 597.7 s     gate arm  PASS 14.79G/46% 597.8 s

Peak DRAM identical to the byte across all four. The off arms span 6.2 % and both on readings lie
between them.

**rf3 at 1088.** `census fold timed out after 1800s`, on a model whose other six rungs read clean
against the baseline with no drift row at all (20.6, 37.8, 52.5, 69.3, 107.6, 122.6 s). Four on-arm
reps at that exact shape then completed at 186-191 s.

So at 1088 the tally on the shipped default is six completions against one timeout, and at 1536 it is
four against one. In both intervals qb2 watchdog-reset. Both stalls also match the `TRUNK_FREEZE`
evidence string already in `size_limits.py` — trunk advancing steadily then stopping forever,
110-160 % CPU, not an OOM and not the 0 %-CPU wedge — which is a documented Blackhole 1536-token
class that predates this branch. And in the one place a rate could be measured, the on arm is the
STEADIER of the two arms, 2.7 % against 13.0 %, which is the opposite of an intermittently wedging
kernel.

## The blocker: main's size-ladder baseline is stale, so the arm is red for every branch

`docs/size_ladder_baseline.d/{boltz2,esmfold2}.json`'s p300c entries were recorded at commit
`1a2ba4f6`, 895 commits behind current main. boltz2, esmfold2 and protenix-v1 are all red at every
rung against nine already-merged, already-default-shipping levers: `REBLOCK_PERMUTE`,
`REBLOCK_PERMUTE_GATED`, `B2_TOKEN_DIT_SDPA`, `TRIMUL_MASK_AFTER_MOVE`, `APB_CONCAT_HEADS`,
`ATOM_AXIS_BUCKET`, `TRANSITION_H_CHUNK`, `PAIR_PROJ_MINIMAL_MATMUL`, `TRIMUL_TAIL_F1`.

This branch's entire code diff against `origin/main` is the flag default, the
`SDPA_FUSED_LARGE_S_STATS` counter in `scripts/lever_census.py`, one docs row and one test. It
touches none of those nine, so the arm is red on main with the flag off.

This branch's own row is the counter-only case `--size-ladder-record-lever` exists for: a guard that
already shipped, given a `*_STATS` pair so the census can finally see it. That splice is one fold per
(model, rung) instead of a 60-fold re-record. It refuses while any other lever mismatches —
`release_gate.py:3233` runs `_size_ladder_compare_levers`, the same comparator check mode runs at
`:3040`, and refuses the model if anything but the named flag's message remains. Check mode has
measured what remains.

A full `--size-ladder-record` would clear the arm but re-measures the card type's whole timing
baseline and blesses nine sibling levers this task never measured, which is precisely the laundering
the refusal exists to prevent. So it is not this task's call. `tt-bio-sizeladder-p300c-refresh` now
owns it, scoped to boltz2 and esmfold2 on qb2/p300c.

Five ladder cells were skipped rather than run: protenix-v2, openfold3, opendde, nesso1 and openbind
all top out at 1024, so the lever is unreachable at every one of their rungs and can only report
"new lever not in the baseline", which three models had already measured. The rows are in the
progress file as `rc=SKIPPED-CANNOT-SEE-LEVER`, never as `rc=0`, with the rationale above them and
the instruction to re-run any of them by deleting its row.

## Arms

| arm | result | above the cap? |
|-----|--------|----------------|
| pytest | 7 failed / 3640 passed — **0 attributable**, all 7 red on main | no |
| ux | **GATE PASS**, 21 of 21 surfaces | no |
| capacity @1536 | **13 PASS, 2 FAIL of 15** — both fails the models' own ceiling | **yes** |
| rf3/1088 A/B | **1.1556x, fully separated, 0 hangs in 8** | **yes** |
| size-ladder | RED, and red on main: 9 of 10 drift levers are main's | rf3/1088 only |
| perf_regression | **16 of 16 measurable models PASS**; 3 reds all attributed, boltzgen parked | no |
| full_parity_gate | running, resumes per leg; 10 of 44 legs cached | no — every leg under the cap |

## perf arm: complete, 16 of 16 measurable models PASS

    boltz2           +3.1%   esmfold2        -2.5%   esmfold2-fast   -6.9%   protenix-v1  -2.4%
    protenix-v2      -3.4%   openfold3       +2.9%   openbind        +2.9%   opendde      -1.6%
    opendde-abag     -1.0%   rf3            +16.1%   rfd3            -5.4%   pxdesign     +8.4%
    nesso1          -12.5%   esmc-300m-single +0.6%  esmc-600m       -1.5%   saprot-650m  +0.6%

Threshold +-15%. rf3's +16.1% is past it in the favourable direction and is not this lever: its
perf target is trpcage at 20 aa, three orders under the cap, and the baseline was taken at v0.7.2
against a 0.8.0 tree.

Three reds, none of which survives attribution, and a fourth cell parked:

| cell | attribution |
|---|---|
| boltz2-affinity | off -54.4% vs on -52.3%. `perf_regression.py:122-137` records that qb2 card 0 cannot satisfy this p300c cell by -28 to -33% whatever the code does, and prescribes exactly the same-card A/B run here |
| esmc-300m | off -21.1% and -4.6% straddle on -22.5% and -5.6% |
| esmc-6b | bimodal; both arms hit both modes, refuted by reversing the arm order |
| boltzgen | parked: 3 single-shot reps at ~255 s is ~13 min against a ~9 min boot |

No cell on this roster can reach the route: the largest input is 107 residues.

### esmc-6b, and why strict alternation was not enough

    off-first:  off1 -1.1%   on1  -38.7%   off2 -1.1%   on2  -36.7%
    on-first:   on1  -1.2%   off1 -33.4%   on2  -1.3%   off2 -1.3%

The off-first run looks like a clean 1.58x arm effect, reproducible, on a quiet box. It is not.
Strict alternation puts every off arm in an odd position and every on arm in an even one, so the arm
is confounded with the position; reversing the order separates them and the slow mode then lands on
an *off* arm. Pooled, both arms span both modes: off reads -1.1/-1.1/-1.3/-33.4% and on reads
-1.2/-1.3/-36.6/-36.7/-38.7%. The cell is bimodal at ~1.5x and the flag is irrelevant to which mode
it picks.

Two independent facts agree. The leg's spec is 8x ubiquitin at 76 aa. And a census on a run that
genuinely computed (rc=0, "Done -- 1 sequence(s), d_model=2560", with `SPLIT_SWIGLU` recording 80
decisions as the live-instrument control) reads `SDPA_FUSED_LARGE_S served 0 declined 0` with the
flag ON. An earlier census attempt returned `served 0` from a run that had died on an input-format
error before computing anything; that zero was worthless and was discarded rather than used.

## The perf arm's first red, closed

    arm                   baseline   current   delta    reps (s)        load0 -> load1
    off1 (flag forced 0)  0.02413    0.01099   -54.4%   87 / 91 / 192   1.37 -> 6.22
    on1                   0.02413    0.01152   -52.3%   83 / 87 / 88    1.16 -> 11.40
    on  (gate arm)        0.02413    0.01063   -55.9%   82 / 94 / 102   1.82 -> 10.58

boltz2 in structure mode reads 1.76 -> 1.813 structures/s, +3.1%, PASS. boltz2-affinity reads -55.9%
and fails the +-15% threshold. The off arm regresses by the same amount, so the flip is not the
cause — which is the expected answer for a reason the census can state exactly: affinity_fkg.yaml
is FKBP12+SB3 at L107 and the route is gated strictly above 1024 tokens.

The residual is not a regression either. `scripts/perf_regression.py:122-137` already records this
cell: "qb2 card 0 cannot satisfy the p300c cell (-28 to -33%) whatever the code does, and card 2
clears it comfortably ... Treat a FAIL here as unproven until a same-card same-session A/B against
the merge base reproduces it." The baseline keys on card TYPE while this single-shot leg carries a
per-card-INDEX offset, and this gate's grant is card 0. The prescribed A/B is what the three rows
above are, and it reads neutral.

The warm-median legs are stable to under 0.5%, so boltz2's +3.1% stands. The three single-shot legs
on the roster — boltzgen, rfd3, boltz2-affinity — each need the same-card A/B before a card-0 FAIL
counts as anything.

## Host ceiling

qb2 reset at 08:55:45, 09:26:08, 09:38:28, 09:52:33, 10:07:30, 10:29:55, 10:47:53 and 10:54:59 UTC:
a mean interval of about 17 minutes, worst 7, while `tt-bio-sizeladder-p300c-refresh` runs a 60-fold
record on cards 1 and 2. Arms that are one fold per model (perf) or resume per leg (parity) still
make progress inside a window; the 14-fold rf3 ladder arm needs about 50 minutes and cannot, so it
is parked with its reasons recorded rather than restarting from rung 256 forever.

## Attribution method

Every red was run against a control. For a flag-dependent red that control is the same tree with
`TT_BIO_SDPA_FUSED_LARGE_S=0`, as separate processes, never flipped inside one device context. For a
tree-state red an env-var control proves nothing, so those were attributed against `origin/main`
directly: main's `tt_bio/tenstorrent.py` contains both `test_perf_citations` citations and neither
artifact is in main's tree; `artifacts` and `patches` are tracked directories in main's root; the
size-ladder drift levers are absent from this branch's diff.

## Recommendation

Merge the flip once `tt-bio-sizeladder-p300c-refresh` lands, then re-merge main, re-run the ladder
arm for boltz2 and esmfold2, and splice this branch's own census row with
`--size-ladder-record-lever SDPA_FUSED_LARGE_S`. The perf and parity arms are neutrality controls for
the fallthrough — every one of their targets is below the cap — and should be green before the tag,
but neither can produce evidence about the route.

Nothing measured here argues against the lever. Below the cap it is byte-identical at 298, 512 and
1024 aa with one CIF digest per size across off/on/off (banked, not re-measured). In its own regime
it is 1.1556x at 1088 with every on reading faster than every off reading, 1.1856x at 1536, and
1.007 Å of all-atom displacement against a 0.60 Å bar at a size where changing the seed moves the
same structure 36.6 Å.
