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
