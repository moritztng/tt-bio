# The fixed cost is the campaign's largest single unknown

`c10-bare-baseline` measured the current tree at a pinned, during-sampled 1350 MHz:
**14.881 s at 512 aa and 9.680 s at 298 aa**, 16 accepted folds each, repeatable to 0.010 s and
0.039 s between sessions, with an exactly zero same-seed structural floor
(`origin/wk/c10-bare-baseline` `bc66f7d6d`).

Those two numbers constrain the clock-immune fixed term but do not pin it. Two equations, three
unknowns: a shared fixed term F plus one work term per size. Choose the work ratio r = W512/W298 and
F follows.

| r = W512/W298 | fixed F | F as % of the 512 fold | cut needed for 10 s | with F cut to 1 s |
|---|---|---|---|---|
| 1.600 (tokens padded to 32: 512/320) | 1.011 s | 6.8 % | 35.2 % | 35.1 % |
| 1.718 (raw token ratio) | 2.436 s | 16.4 % | 39.2 % | 27.7 % |
| **1.839 (what the clock-sweep intercept implies)** | **3.481 s** | **23.4 %** | **42.8 %** | **21.1 %** |
| 2.000 | 4.479 s | 30.1 % | 46.9 % | 13.5 % |
| 2.500 | 6.213 s | 41.7 % | 56.3 % | **already there** |
| 2.950 (a pure N² pair term) | 7.013 s | 47.1 % | 62.0 % | **already there** |

Read the last column. **If the work ratio is 2.5 or above, cutting the fixed cost to 1 s reaches
10.0 s on its own, with no device cycle deleted at all.** If it is 1.6, the fixed cost is nearly
nothing and the whole 35 % has to come out of kernels. The campaign's answer is somewhere between
"the target is a host problem" and "the target is a kernel problem", and nothing measured so far
distinguishes them.

One independent cross-check exists. The clock sweep re-fitted in `../fit_reexam/` gives an intercept
of 3.478 s from a completely different dataset on a different commit, and that lands on r = 1.839
here — slightly above the raw token ratio, which is what you would expect from a fold dominated by
diffusion steps that scale with atoms plus a modestly superlinear trunk. Two unrelated datasets
agreeing at F ≈ 3.5 s is encouraging, and it is still not a measurement.

## The experiment that settles it, and it is cheap

Vary the CLOCK, not the size. The fixed term is by definition the only part that does not scale
with the clock, so two pinned clocks at one size give two equations in two unknowns and F falls out
directly — on today's tree, with today's timer boundary, from all-clean data. The floor (800 MHz)
and the burst (1350 MHz) are both reachable with the accepted force implementation, so the minimum
version of this is one interleaved session, two clocks, one size. Adding 298 aa tests whether F is
really size-independent, which it may not be: featurization and CIF writing scale with the target.

Expect roughly 22 s per fold in the 800 MHz arm, so the session is longer than the baseline's but
still short. That row is `c10-fixed-cost`.

## Caveats

- These are medians of wall time, not a device-cycle census. `elapsed × clock` is a
  clock-equivalent, not measured device work; the 512 aa fold's 20,089.8 Mcycle-equivalents include
  every host gap.
- F is "clock-immune", not "host". It includes clock-immune device and dispatch cost.
- The current tree reads 14.881 s at 512 aa where the historical pinned arm on commit `0df13ad9`
  read 14.554 s. That is +0.327 s, +2.2 %, across a commit change AND a possibly different timer
  boundary, so it is an open question and not a regression claim. Whoever next touches a 512 aa
  timing row should settle which it is.

## Reproduce

    python3 two_size_family.py                       # prints two_size_family.json
    python3 -m pytest test_two_size_family.py -q     # 5 known-answer controls

## Update: the fixed cost was already measurable, and it is 3.95 s

`b2z2-aiclk-default-decision` ran an A/B on commit `0df13ad9` that alternated a forced 800 MHz arm
with an unforced ~1339 MHz arm in one process on one fixture, six non-warmup folds, and nobody ever
solved it for the fixed term. Two clocks, two unknowns:

    fixed term 3.952 s     work 14,418 Mcycles     worst residual 33 ms
    leave-one-out: fixed 3.930 - 3.993 s, work 14,384 - 14,445 Mcycles
    800 MHz arm 21.981 s median, ~1339 MHz arm 14.738 s median

The residuals are inside the A/A timing floor the baseline later measured (55 ms at 512 aa). An
independent clean session on the same commit, which is not in the fit, checks it: the solve
predicts that session's 1339-to-1350 MHz gap to **11.5 ms** and its forced arm to 77 ms.

The historical harness times `state.predict_one` with `perf_counter`, which is the same boundary
`c10-bare-baseline` used, so the two are comparable.

### What it means for the target

Carrying that fixed term onto the current tree's measured 14.8813 s at 1350 MHz:

| | |
|---|---|
| clock-immune fixed cost | **3.95 s, 26.6 % of the fold** |
| work term | 14,755 Mcycles, +2.3 % against `0df13ad9` |
| 10.0 s with the fixed cost untouched | a **44.7 %** cycle cut |
| 10.0 s with the fixed cost cut to 2 s | a 26.8 % cycle cut |
| 10.0 s with the fixed cost cut to 1 s | a **17.7 %** cycle cut |
| cutting the fixed cost to 1 s and nothing else | **11.93 s** |

A 44.7 % cycle cut is larger than the sum of every lever this project has ever landed. A 17.7 % cut
on top of a host-side fix is an ordinary optimization campaign. That is the difference the fixed
cost makes, and it is why `c10-fixed-cost` confirms it on the current tree with a quiet host before
any kernel work starts.

That +2.3 % work term is also worth a look on its own: `0df13ad9` read 14.554 s pinned where the
current tree reads 14.8813 s, same fixture, same config, same timer boundary, both sessions clean.
It is a cross-run comparison so it is a signal and not a verdict, but the direction is a regression
and `5e1886b4f` enabling the above-cap fused SDPA route is the obvious suspect.

### Limits

- Every fold in the two-clock session had one foreign TT holder. It is in both arms, so the
  comparison is controlled, but an additive per-fold holder cost inflates the fixed term.
- Two clocks and two parameters leave no degrees of freedom. The within-arm scatter and the
  independent cross-check are the evidence that the model fits, not a residual test.
- Clock-immune is not host CPU. It includes clock-immune device and dispatch cost, and only a
  device-side split separates them.

    python3 two_clock_session.py                      # prints two_clock_session.json
    python3 -m pytest test_two_clock_session.py -q    # 6 known-answer controls

## A mean-clock label cannot re-price an old lever

The campaign's premise invites an arithmetic shortcut: take a lever recorded at 1.03x, look up the
mean AICLK of the run it was measured on, and rescale. This tests whether that works. Take the model
solved above from one clean interleaved session, and ask it to predict folds from three other
sessions on three other commits, each fold labelled with its own recorded mean AICLK.

| session | commit | folds | mean error | worst error |
|---|---|---|---|---|
| force_ab | `149c8a97f` | 16 | +0.400 s | +1.253 s |
| maxclk_ab | `ab9e3275` | 16 | +0.305 s | +1.044 s |
| pin_ab | `33feece9` | 4 | +0.461 s | +1.101 s |
| **pooled** | | **36** | **+0.364 s** | **+1.253 s** |

Inside the clean interleaved session the same model fits to **33 ms**. Across sessions it is off by
**364 ms on average and up to 1.25 s**, and the error is almost always positive: the co-tenanted,
clock-varying folds run longer than their mean-clock label says they should.

That is the answer to the shortcut. Every byte-deletion lever in this corpus landed between 1.02x
and 1.05x, which at this fixture is 0.3 to 0.7 s — **smaller than the error of the clock label you
would have to use to rescale it**. So the underpriced-lever hypothesis cannot be settled by
arithmetic on the record, only by re-measuring at a pinned clock with interleaved arms. `0.364 s`
is also a useful floor on how much any cross-run comparison in this project's history is worth.

    python3 clock_label_error.py     # prints clock_label_error.json
