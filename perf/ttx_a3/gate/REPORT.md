# Release-gating `TT_BIO_SDPA_FUSED_LARGE_S` and flipping it default-ON

Verdict: **GO on the lever. HOLD the merge.** Every gate arm that can run on this host, and every
arm whose result can be attributed, clears. Two things independent of this lever stop the gate from
being called green, and both are named below.

The route is gated strictly above 1024 tokens (`q_len > _triatt_sdpa._Q_SPLIT_MAX_S`, cap 1024).
That single fact is what makes the flip cheap to reason about: no length that folds at or below
1024 can reach it.

## What the flip buys

| | |
|---|---|
| 200-step 1536 aa fold | **1.1856x**, 174.178 s -> 146.915 s, against a 1.21 % same-session A/A floor |
| attention op at 1536 tokens | 4.23x (136.143 -> 32.184 ms), 2.678x at 1920, 1.865x at 2208 |
| lengths served fused, 1024-2592 | 36 of 50 on an 11x10 grid at 4 heads, against 0 of 50 before |

Not re-measured here. Both numbers come from the merged `ttx-a3-1536-200step-fold` and
`ttx-a3-triatt-mask-extend` reports.

## Accuracy above the cap

1.007 Å all-atom at 1536 aa against a 0.60 Å kill bar, and pLDDT moves the favourable way
(0.786016 -> 0.789493). The bar was calibrated at 512 aa, where re-running with a different seed
moves the same structure 1.84 Å; at 1536 aa a seed change moves it 36.6 Å. Against an fp32
evaluation of the same bf16 operands the fused pair is marginally *closer* than the ladder it
replaces (0.402555 vs 0.402814). No shipped digest exists above 1024 tokens to break, because
nothing reached this kernel there before.

## Neutrality below the cap: measured, not argued

One process per fold, 200 sampling steps, 3 recycles, off/on/off per rung.

| size | off1 | on | off2 | CIF sha256, all arms | pick |
|------|------|----|------|----------------------|------|
| 298 aa | 15.646 s | 15.296 s | 15.306 s | `a8c6fd65f70f4418` | q320 k64 fused |
| 512 aa | 23.166 s | 23.044 s | 23.026 s | `45781db716ebf020` | q512 k256 fused |
| 1024 aa | 57.977 s | 59.067 s | 57.775 s | `9a049ee58a5a0f7e` | q256 k256 fused |
| 768 aa | 47.196 s | wedged | wedged | `9e1a1fdd392e0b4c` (off1) | q384 k256 fused |

One digest per size across all three arms, including at 1024 aa, which is the cap boundary itself
and therefore the last length where the flip must change nothing. The pick is unchanged in every
arm, so the stock ladder still serves the call and the above-cap route stays unreachable.

The 1024 aa on-arm read +1.9 % on time. That is noise and is recorded rather than smoothed: load
does not explain it (off2 ran at loadavg 11.49 and was the fastest of the three, so the 0.35 %
off/off spread spans a 3.6x load range), the code cannot (three extra comparisons on 560 calls), and
the sign flips by size — at 298 aa the on arm read 2.2 % *faster* than off1. A 1.21 % A/A floor is
on record and one on-arm sample against it is not a measurement.

### The 768 aa rung wedges on both arms, so it is not the lever

768 aa hangs in fast dispatch rather than completing: three on-arm attempts and an off-arm attempt
all hit their timeout, against one off-arm completion at 47.196 s. Because it happens **with the
flag off**, the wedge belongs to the rung and not to the flip. Two further facts agree: the earlier
`ttx-a3-1536-200step-fold` gdb put the equivalent hang in `fetch_queue_reserve_back` under
`MatmulDeviceOperation` / `ttnn::operations::matmul::linear`, an ordinary linear rather than the SDPA
route; and 768 aa carries a caught L1 `TT_THROW` at `(768, 768, 768, 256, 2)` that fires on both
arms, which is a plausible dispatch-wedge mechanism owned by the rung.

An earlier pass attributed this hang to switching the flag inside one live device context. That
attribution does not survive: single-arm processes with no flag flip in them hang too.

## Gate arms

| arm | result | reaches above the cap? |
|-----|--------|------------------------|
| pytest | 12 failed / 3626 passed — **0 attributable to the flip** | no |
| ux | GATE FAIL on 3 legs — **0 attributable to the flip** | no |
| capacity, boltz2 @ 1536 | **PASS on both arms** | **yes** |
| capacity, full roster | did not complete, host reset | yes |
| size-ladder | did not complete, host reset | rf3's 1088 rung only |
| perf_regression | did not run | no |
| full_parity_gate | did not run | **no — every one of its 44 legs is under the cap** |

The accuracy gate cannot see this lever at all. Its largest legs are HSA at 585 aa and 9ncy at 505
tokens, so a green parity run would be a neutrality control for the fallthrough, not evidence about
the route. Do not let a green parity run imply coverage it does not have.

### Attribution method

Every red was run against a control that is **the same tree with `TT_BIO_SDPA_FUSED_LARGE_S=0`** —
exactly and only the default flip, so a failure present in both arms is not this branch's. The
control self-validates: `test_the_above_cap_route_is_strictly_above_the_cap` asserts the flag is
`True`, so it fails in the control and only there. Without that check a no-op control would have
produced the same reassuring subset result and meant nothing.

- **pytest**: flag-on 12 failed, flag-off 14 failed. `comm` on the sorted `FAILED` lists gives an
  empty only-in-ON set. The control's two extras are that self-validating assertion and a
  lock-release timing test flaking at loadavg 24.
- **ux**: same three legs fail on both arms, all sixteen others pass on both, wall-clock apart.
- **capacity boltz2 @ 1536**: on -> screen FAIL 22.7 s, residency PASS 412.0 s, overall PASS.
  off -> screen FAIL 22.3 s, residency PASS 514.9 s, overall PASS.

The control isolates the flag but **not** tree edits, so the three tests whose verdict depends on
tree state were attributed by hand instead (below).

## The two things that are not this lever and do block a green gate

**1. esmfold2 / esmfold2-fast / esmc-6b cannot load weights** (`KeyError: 'weight'` at
`tenstorrent.py:5398`, a weights-Mapping `__getitem__`). This single cause produces 5 of the 12
suite failures *and* all 3 red ux legs — 8 of the 15 reds in the gate. Main already carries
`095ae476c esmfold2: pin the hub revision, because upstream moved main under production`, and the
fleet lease directory shows an esmfold2 outage relocated off qb1, so it is live and owned elsewhere.
The flip cannot be involved: those tests fold 98-aa inputs, three orders below the 1024-token cap.

**2. Baseline and repo debt that predates this branch.**

| failure | attributed to |
|---|---|
| `test_capacity_gate` x3 | p150a cells stale against `size_limits.CEILINGS`; `p300c/opendde` and `p300c/opendde-abag` have no recorded cell |
| `test_perf_citations` x2 | `tt_bio/tenstorrent.py` cites `perf/b2z2_adaln_sdpa/chunks_wh_c12.json` and `perf/b2z2_layout/PER-SITE-TABLE.md`; both absent from `origin/main` too, and this branch's diff of that file contains no `b2z2` reference |
| `test_repo_root_clean` x1 | `artifacts` and `patches` are *tracked* directories in main's repo root |
| `test_size_ladder_gate` x1 | the known p150a 896/1024 coverage short, ruled non-blocking by `tt-bio-release-bh-0-8-p5` |

The capacity **screen** tier's `4 vs 12 mismatch at non-singleton dimension 1` in `boltz2.py:381`
was treated as a candidate NO-GO because 1536 tokens is this lever's regime and a dim-1 mismatch is
head-shaped. It is not the lever: that line is a pure-torch `einsum` path the ttnn route never
reaches, `boltz2.py` is untouched by this branch, and the mismatch **reproduces with the flag off**
at 22.3 s against 22.7 s. The residency tier passes either way.

## Host ceiling

qb2 hard-reset five times while this gate ran — 23:58, 00:23, 01:10, 02:06 and 02:49 Z, a mean
interval of about 43 minutes, mostly at loadavg 15-27 with four campaigns sharing 16 cores. Every
remaining arm needs more than that: size-ladder walks 9 models x 7 rungs, the capacity roster is 15
models at 1536 tokens, and the 44-leg accuracy gate is hours. This is
`qb2-watchdog-reset-frequency-scales-with-load`, and it is why the arms above are split into "clears"
and "did not complete" rather than into pass and fail.

Two harness defects were found and fixed rather than worked around. `capacity_gate.py`'s `rc=4` was
not a gate failure: the driver let it default to card 0, which a sibling held, and it refused
correctly instead of resetting a card. And `wait_quiet` was gated on the below-cap ladder's end
marker, which only prints once all four rungs finish — the ladder had never got that far, so both
timed arms would have parked their full 3 h and logged `SKIPPED-LOADED`, which reads as a load
refusal rather than as the harness defect it was.

## Recommendation

Keep the flip on the branch. Merge it once the esmfold2/esmc-6b weights outage clears and
size-ladder, perf and the capacity roster have completed on a host that stays up long enough to run
them — expecting exactly one new size-ladder finding, this row's own census entry
(`SDPA_FUSED_LARGE_S: new lever not in the baseline`), which wants a `--size-ladder-record` on a
quiet box and not a NO-GO. Any *other* lever moving `resolved` or going dark is a NO-GO.

Nothing here argues against the lever. The evidence that the flip is safe below the cap is a digest
per size, the evidence that it is safe in its own regime is a capacity PASS on both arms at 1536
tokens, and the evidence that it is worth having is 1.1856x at 15.3x its own noise.
