# of3t-p10lnbw — a correct AND fast layer-norm backward

VERDICT: GO. The layer-norm backward defect is `ttnn.mean`'s `1/K`, the fix is one op wide,
it heals both damaged targets 9 times out of 9, and it costs +0.164 s of a 29.698 s backward.

RESULT: `ttnn.mean` carries its own reciprocal and hands it back short at a K that is not a power
of two, so all four means of the closure inherit one signed constant. Replacing each with
`ttnn.divide(ttnn.sum(precise), K)` takes the K=384 gradient from `dx` 2.089e-03 / `dgamma`
2.090e-03 to 1.399e-04 / 1.389e-04 in fp32 (**14.9x, 15.1x**), takes both from BIAS to NOISE in
bf16 as well, and leaves every K=128 number bit-identical to shipped. Nothing on the op reads BIAS
any more.
COMMIT: `d2d75496b` (branch `wk/of3t-p10lnbw`); the graded gradient sweep ran on `4770dae52`
CARD: qb2 (tt-quietbox2) card 0 for the arms, card 1 for the step A/B, lease
`worker:of3t-p10lnbw` on both
AICLK: 1350 MHz median, sampled DURING every run quoted here: n=262 on the step A/B, n=47 on the
D1 arm, n=14 on the graded gradient sweep. The `min 800` each sampler also reports is the sample
taken at launch, before the card is under load.

ROUTE: fix the arithmetic of the composed closure, on the card, in the same op count plus one
scalar divide per mean. Not the host float64 instrument, which is correct and 34x.

REJECTED:
- **`ttnn.moreh_layer_norm_backward`** — not on OF3T's path at all (`grep -rn moreh_layer_norm_backward
  tt_bio/` hits two docstrings and nothing else) and wrong on Blackhole anyway, `dx` 2.741e+06
  relative L2 bf16, upstream #12349. Re-confirmed by of3t-p10grad; not re-opened here.
- **The composed host-float64 closure as the shipped path** — correct and 34x (3938 s against
  115 s on the two-step arm). That is the brief's explicit non-answer.
- **fp32 on the failing reduction** (`precise_config()` on the three unconfigured `ttnn.mean`
  calls) — inert, bit-identical. The error is not accumulation width, it is the constant
  `ttnn.mean` multiplies by, so widening the accumulator cannot reach it. of3t-p10grad's MEANCFG
  arm and of3t-p10trainout's `bwcfg_reductions.patch` reached this independently.
- **`rstd` as `reciprocal(sqrt(v))` instead of `rsqrt(v)`** — inert in fp32, 1.25x WORSE in bf16
  (RSQRTSPLIT arm, 5.392e-03 against 4.317e-03).
- **`ttnn.sum * (1.0/K)`** — fixes fp32 and makes bf16 slightly worse. The python float `1/384`
  is rounded to bf16 at the op boundary, which is the exact constant that was wrong. Measured, kept
  as `PEROP_LNFIX.json`. `384` IS exact in bf16, so dividing carries no constant.

GRADERR: `perf/of3t_p10grad/PEROP_LNDIV.json`, the shipped taped verb, 32 draws, against a
finite-difference-validated float64 reference. `sys` is the systematic scale over the draws; the
bar calls it BIAS at `|mean| >= 1e-3` and `sigma >= 5`.

| case | dtype | grad | before | after | sys before | sys after | verdict |
|---|---|---|---|---|---|---|---|
| single K=384 | fp32 | `dx` | 2.089e-03 | **1.399e-04** | +2.078e-03 | +1.359e-04 | BIAS -> NOISE |
| single K=384 | fp32 | `dgamma` | 2.090e-03 | **1.389e-04** | +2.079e-03 | +1.361e-04 | BIAS -> NOISE |
| single K=384 | bf16 | `dx` | 4.317e-03 | **3.904e-03** | +1.914e-03 | -4.100e-04 | BIAS -> NOISE |
| single K=384 | bf16 | `dgamma` | 4.048e-03 | **3.873e-03** | +1.585e-03 | -7.187e-04 | BIAS -> NOISE |
| single K=384 | both | `dbeta` | unchanged | unchanged | — | — | NOISE |
| pair K=128 | both | all six | bit-identical | bit-identical | — | — | NOISE |

`dbeta` is the one layer-norm gradient with no mean in it, and it does not move in any arm, the
control the explanation predicts. K=128 not moving at all is the second: `1/128` is a power of two
and already exact. Every graded gradient repeated bit-exact in-process.

Why divide and not multiply: `perf/of3t_p10grad/meanprobe.py` measures five spellings of a row
mean against a float64 row mean. At K=384 in bf16 the fitted scale reads **-3.725e-03** for
`ttnn.mean`, **-4.140e-03** for `sum * (1/K)` and **-4.363e-04** for `sum / K`. At K=128 all five
agree to the digit.

STEPTIME: **the fix costs +0.164 s on a 29.698 s backward, +0.55%, and nothing readable at the
step.** 13 reps of ONE warm process on qb2 card 1, rep 0 cold and discarded, the fix alternated per
rep so six warm reps grade each arm against the same card, the same host memory and the same clock.
Two separate runs cannot resolve this: the 384-token step's own rep-to-rep spread is 1.5 s here.
Artifact `perf/of3t_stepfloor/out/step_rowmean_48_384.json`, AICLK median 1350 MHz over 262 samples
polled DURING.

| paired ON-OFF, 6 pairs | delta | 95% CI | against |
|---|---|---|---|
| step wall | **-0.707 s** | -2.586 .. +1.172 | 49.962 s |
| step phases | -0.566 s | -2.098 .. +0.967 | 46.093 s |
| **backward** | **+0.164 s** | -0.336 .. +0.663 | 29.698 s, **+0.55%** |

The lever fires 26540 times a step and the counter flips clean: 26540 divide / 0 mean with it on,
the reverse with it off, every rep. 0.164 s over 26540 calls is **6.17 us per extra `ttnn.divide`**,
which is one dispatch, so the point estimate is the mechanism rather than drift. It is still inside
the rep-to-rep noise, which is why the bound is quoted beside it and not dropped.

The cost is dispatch, so it transfers as an absolute, not as a ratio. Against the campaign's
headline step, 62.237 s = 8.17x an H200 (`of3t-p10noexact`, qb1 card 1), +0.164 s is **+0.26%** and
the 95% bound +0.663 s is +1.07%: 8.17x becomes 8.19x, at worst 8.26x. qb1 and qb2 are not
interchangeable here, the same invocation reads 49.962 s on this box with the fix off, which is why
the fix is priced in-process and only its absolute cost is carried across.

ABPROOF: nine device-native two-step arms with the fix, seed 0, same corpus and order as
of3t-p10trainout's ten. **Replicates, not one run**: that row's finding is that the held-out metric
is BIMODAL (7vus lands at ~2.48 or ~5.64, 7ohe at ~10.8 or ~18.9, nothing between) and its ten
device-native runs split 6 blown-both / 3 fine-both / 1 mixed, so a single arm measures a coin.

| arm | 7ohe | 7vus | 7kud | 7fb8 | mean | step 1 |
|---|---|---|---|---|---|---|
| start checkpoint | 10.762624 | 2.451977 | 9.277909 | 4.717251 | 6.802440 | — |
| exact trunk, 34x | 10.829831 | 2.479054 | 9.220225 | 4.713989 | 6.810775 | — |
| device-native x10 | 10.79 or 18.9 | 2.46 or 5.64 | ~9.24 | 4.706–4.719 | 6.813708–9.635149 | all ten differ |
| D1 | 10.788486 | 2.469938 | 9.221039 | 4.715689 | 6.798788 | 1.2376639465563595 |
| D2 | 10.824212 | 2.464245 | 9.245464 | 2.153862 | 6.171946 | 1.2293438986258850 |
| D3 | 10.870706 | 2.468330 | 9.228852 | 2.736508 | 6.326099 | 1.2341772071383170 |
| D4 | 10.841912 | 2.463128 | 9.264436 | 2.160254 | 6.182432 | 1.2385116348459222 |
| D5 | 10.817243 | 2.481998 | 9.246984 | 4.711646 | 6.814468 | 1.2428253503050450 |
| D6 | 10.828145 | 2.471913 | 9.247758 | 2.163701 | 6.177880 | 1.2303387148799380 |
| D7 | 10.867416 | 2.467923 | 9.256044 | 4.720372 | 6.827939 | 1.2263480861324407 |
| D8 | 10.821712 | 2.474703 | 9.253331 | 2.733814 | 6.320890 | 1.2290084814637876 |
| D9 | 10.834621 | 2.471282 | 9.262436 | 2.734731 | 6.325768 | 1.2221978404923326 |
| **fixed, n=9** | **10.788–10.871** | **2.463–2.482** | 9.221–9.264 | 2.154–4.720 | 6.172–6.828 | nine values |

**Nine of nine fine on both damaged targets**, against 3 of 10 for the device-native arm.
Fisher one-sided **p = 0.0024**. 7vus reads 2.463–2.482 where the brief's damaged value is
5.645047 and the start is 2.451977, so the whole spread lands nearer the start than the exact
trunk's own 2.479054 does. 7ohe reads 10.788–10.871 against a blown 18.9 and an exact 10.829.

`eval_before` is 6.8024402513580124 on all nine, digit for digit with every arm of every previous
row, and step 0 is 1.7992572181478035, bit-identical to all ten device-native runs. The forward is
untouched, which is what a backward-only change has to show before any of the rest is readable.

**7fb8 moves, and at n=9 it moves DOWN.** All ten device-native runs and both exact arms put it in
4.706–4.719. The nine fixed arms land in three tight clusters of exactly three: 2.154/2.160/2.164,
2.734/2.735/2.737 and 4.712/4.716/4.720. Every cluster is at or below the unfixed band, so this is
a target finding a better mode a third of the time, not damage. It is also the thing that drags
the nine-arm mean (6.438) below the start checkpoint (6.802). 7fb8 is not one of the two targets
this row was sent to heal and the row does not claim to have explained its trimodality.

## What this row does not claim

The rate of the blown mode beyond n=9, and determinism.

`of3t-p10trainout`'s later finding is that the device layer-norm backward is NONDETERMINISTIC run
to run — step 0 bit-identical on all ten runs, step 1 different on every one — and that the
bimodality is downstream of that, not of precision. **This fix is an accuracy fix and it is NOT a
determinism fix — measured, not assumed.** D1–D9 give nine different step-1 losses from one seed,
one corpus, one order, on one card. The weights after a single step are still not reproducible.
Both spellings are bit-exact on repeat at the op (`bitexact_repeat` true on all 36 graded
gradients in all three sweep arms), so whatever varies at step scope is not the arithmetic of this
closure at these shapes, and this row does not close that defect. What the nine arms show is that
with the constant fixed, the nondeterminism no longer reaches a mode that blows the metric.

## Shared `tt_bio/` code changed here

| commit | file | what |
|---|---|---|
| `4ef77300d` | `autograd.py` | `_layer_norm_bw`, one closure where `layer_norm` and `_taped_layer_norm` carried verbatim copies |
| `4770dae52` | `autograd.py` | `_row_mean` divides by K |
| `2cfdf8193` | `perf/of3t_p10trainout/{armrun.sh,trainarm.py}` | the arm runs from the worktree it is in, and refuses a `tt_bio` resolved from anywhere else |
| `b4b01f613` | `perf/of3t_stepfloor/fullstep.py` | a scalar write clobbered `row["mem_available_gib"]` one line before `_mark()` indexed it, so every run died with a `TypeError` after the first backward |
| `d2d75496b` | `perf/of3t_stepfloor/out/` | the 13-rep step A/B |

## Traps paid for here

A `while pgrep -f "of3t_p10trainout/trainarm.py"` wait loop at the top of a chain script matched the
ORPHANED ssh launcher whose argv contained the whole heredoc, including that pattern, so the chain
sat in the loop forever while the card was free. Same shape as the known
`a-pgrep-f-wait-loop-matches-its-own-command-line` case, reached from the launcher side rather than
the watcher side.

`perf/of3t_stepfloor/fullstep.py` had two writers on one key, a per-phase dict and a scalar, and the
scalar landed second. Nobody saw it until a run reached `_mark("after_backward")`, because the
scalar was added by a different row (`cfabae832`, of3t-p10samples) than the dict.
