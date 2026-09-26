# of3t-p10lnbw — a correct AND fast layer-norm backward

RESULT: **The layer-norm backward's defect is `ttnn.mean`'s `1/K`, and the fix is one op wide.**
`ttnn.mean` carries its own reciprocal and hands it back short at a K that is not a power of two,
so all four means of the closure inherit one signed constant. Replacing each with
`ttnn.divide(ttnn.sum(precise), K)` takes the K=384 gradient from `dx` 2.089e-03 / `dgamma`
2.090e-03 to 1.399e-04 / 1.389e-04 in fp32 (**14.9x, 15.1x**), takes both from BIAS to NOISE in
bf16 as well, and leaves every K=128 number bit-identical to shipped. The single track now grades
what the pair track grades. Nothing on the op reads BIAS any more.
COMMIT: `2cfdf8193` (branch `wk/of3t-p10lnbw`); the graded sweep ran on `4770dae52`
CARD: qb2 (tt-quietbox2) card 0, sole holder, lease `worker:of3t-p10lnbw`
AICLK: 1350 MHz min/median/max sampled DURING the graded sweep (n=14 in-process); 1350 median
DURING the two-step arms (n=47 on D1), min 800 there is the sample taken at launch

ROUTE: fix the arithmetic of the composed closure, on the card, in the same op count plus one
scalar divide per mean. Not the host float64 instrument, which is correct and 34x.

REJECTED:
- **`ttnn.moreh_layer_norm_backward`** — not on OF3T's path at all (`grep -rn moreh_layer_norm_backward
  tt_bio/` hits two docstrings and nothing else) and wrong on Blackhole anyway, `dx` 2.741e+06
  relative L2 bf16, upstream #12349. Re-confirmed by of3t-p10grad; not re-opened here.
- **The composed host-float64 closure as the shipped path** — correct and 34x (3938 s against
  115 s on the two-step arm). That is the brief's explicit non-answer.
- **`precise_config()` on the three unconfigured `ttnn.mean` calls** — inert, bit-identical
  (of3t-p10grad MEANCFG arm, and of3t-p10trainout reached the same result independently with
  `bwcfg_reductions.patch`).
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

`dbeta` is the one layer-norm gradient with no mean in it, and it does not move in any arm — the
control the explanation predicts. K=128 not moving at all is the second: `1/128` is a power of two
and already exact. Every graded gradient repeated bit-exact in-process.

Why divide and not multiply: `perf/of3t_p10grad/meanprobe.py` measures five spellings of a row
mean against a float64 row mean. At K=384 in bf16 the fitted scale reads **-3.725e-03** for
`ttnn.mean`, **-4.140e-03** for `sum * (1/K)` and **-4.363e-04** for `sum / K`. At K=128 all five
agree to the digit.

STEPTIME: owed. The instrument is `perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4
--samples 48 --chunk 4 --no-exact`, the one that read 62.237 s / 63.182 s warm median
(`of3t-p10noexact`, qb1 card 1). The fix adds one `ttnn.divide` on a `[..., 1]` tensor per mean,
four per layer-norm backward, ~3,264 a step. That step's own rep-to-rep spread is 2.551 s, so two
separate runs cannot resolve it: the A/B has to be two reps of ONE warm process, the way
`--fp32bw-per-rep` and `--renorm-per-rep` already do it. Wiring `--rowmean-per-rep` is the next
thing in.

ABPROOF: three arms in, six more and the step A/B queued behind them on card 0. Device-native
two-step arms with the fix, seed 0, same corpus and order as of3t-p10trainout's ten.
**Replicates, not one run**: that row's finding is that the held-out metric is BIMODAL (7vus lands
at ~2.48 or ~5.64, 7ohe at ~10.8 or ~18.9, nothing between) and its ten device-native runs split
6 blown-both / 3 fine-both / 1 mixed, so a single arm measures a coin.

| arm | 7ohe | 7vus | 7kud | 7fb8 | mean | step 1 |
|---|---|---|---|---|---|---|
| start checkpoint | 10.762624 | 2.451977 | 9.277909 | 4.717251 | 6.802440 | — |
| exact trunk, 34x | 10.829831 | 2.479054 | 9.220225 | 4.713989 | 6.810775 | — |
| device-native x10 | 10.79 or 18.9 | 2.46 or 5.64 | ~9.24 | 4.706–4.719 | 6.813708–9.635149 | all ten differ |
| **D1** | 10.788486 | **2.469938** | 9.221039 | 4.715689 | 6.798788 | 1.2376639465563595 |
| **D2** | 10.824212 | **2.464245** | 9.245464 | 2.153862 | 6.171946 | 1.2293438986258850 |
| **D3** | 10.870706 | **2.468330** | 9.228852 | 2.736508 | 6.326099 | 1.2341772071383170 |

**3 of 3 fine on both damaged targets**, against 3 of 10 for the device-native arm. 7vus reads
2.464–2.470 where the brief's damaged value is 5.645047 and the start is 2.451977, so it lands
nearer the start than the exact trunk's own 2.479054 does. At n=3 against n=10 that is p=0.08 by
Fisher, suggestive and not yet significant; D4–D9 are queued for exactly that reason.

`eval_before` is 6.8024402513580124 on all three, digit for digit with every arm of every previous
row, and step 0 is 1.7992572181478035, bit-identical to all ten device-native runs. The forward is
untouched, which is what a backward-only change has to show before any of the rest is readable.

**7fb8 is a new finding and it is not obviously good.** All ten device-native runs and both exact
arms put it in 4.706–4.719; D1/D2/D3 read 4.715689, 2.153862 and 2.736508. So the fix appears to
move 7fb8 off a value it was pinned to, and to move it by a different amount each run. That is
what drags the means below the start checkpoint. It needs replicates before anyone reads a
direction into it.

## What this row does not claim yet

The step time, and the rate of the blown mode. Both are owed and both are measurable on the
instruments named above.

`of3t-p10trainout`'s later finding is that the device layer-norm backward is NONDETERMINISTIC run
to run — step 0 bit-identical on all ten runs, step 1 different on every one — and that the
bimodality is downstream of that, not of precision. **This fix is an accuracy fix and it is NOT a
determinism fix — measured, not assumed.** D1, D2 and D3 read step 1 at 1.2376639465563595,
1.2293438986258850 and 1.2341772071383170: three different values from one seed, one corpus, one
order, on one card. The weights after a single step are still not reproducible. Both spellings are
bit-exact on repeat at the op (`bitexact_repeat` true on all 36 graded gradients in all three
arms), so whatever varies at step scope is not the arithmetic of this closure at these shapes, and
this row does not close that defect.

## Shared `tt_bio/` code changed here

| commit | file | what |
|---|---|---|
| `4ef77300d` | `autograd.py` | `_layer_norm_bw`, one closure where `layer_norm` and `_taped_layer_norm` carried verbatim copies |
| `4770dae52` | `autograd.py` | `_row_mean` divides by K |
| `2cfdf8193` | `perf/of3t_p10trainout/{armrun.sh,trainarm.py}` | the arm runs from the worktree it is in, and refuses a `tt_bio` resolved from anywhere else |
