# c12_mm_attrib — what issues the fold's `matmul` keys, and what they actually cost

The c10 fold census found the 512 aa Boltz-2 fold's biggest single launch key,
`matmul|out=1x128x512x512|K=512` at 560 calls and 1.0724 s, and could not say what code issues
it. This directory answers that for every key in the census's `matmul` arm, and then prices each
one against the configuration the fold runs rather than the one the census replayed.

CPU only. No device is opened and nothing new is measured; three committed instruments are joined
on the launch key.

## Reproduce

```bash
# the census's own artifacts live on the row that produced them
git show origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/budget.json > /tmp/budget2.json
git show origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/replay.json > /tmp/replay2.json

python3 perf/c12_mm_attrib/attrib.py                       # -> mm_sites.json.gz, mm_sites.txt
python3 perf/c12_mm_attrib/price.py --budget /tmp/budget2.json --replay /tmp/replay2.json
```

`attrib.py` walks the 26 graph captures under `perf/roof_budget/captures` and pulls every
`ttnn.matmul` / `ttnn.linear` with its executed operand specs, buffer types, transposes,
`core_grid`, output memory config, compute kernel config and enclosing `unit::` module. Op
ownership comes from the committed `perf/b2x_difflayer/itemize.top_level_spans`, so the op set is
the one `roof_budget` and `roof_residual` already count.

`price.py` joins that to the census's per-key prices and to the in-situ Blackhole profile at
`perf/k10_diffusion/src/ops_perf_step_qb2c0.csv.gz`, which carries per-op device kernel duration,
the engaged core count and the program config ttnn resolved.

## Two things to know before quoting a number

**Cycles are the currency.** The profiler's ns are derived at a 1350 MHz constant (its FW cycle
delta over its FW ns is 1.350 on every row), so what it reports is cycles. Seconds below are
those cycles at 1350 MHz, which is the clock the census pinned and during-sampled.

**The cross-instrument control decides whether the two can be compared.** The armed capture is a
different session on a different card with no during-sampled clock. Over the 27 keys both
instruments carry they agree on the class total to 7 %, per-key ratios run 0.31x to 1.78x in both
directions, and on the 21 `linear` keys — none of which are under test here — the in-situ leg is
8.1 % *slower* in aggregate with 13 of 21 inside 10 %. A clock deficit is one-sided and can only
make the in-situ leg slower, so it cannot explain a key that reads 2.3x faster in situ. It also
means the corrections below are conservative.

## What the join says

`PRICED.txt` is the table. The short version: the census replays these keys with ttnn's automatic
program config or a `core_grid`, and for four of the six the fold pins something else, so 0.7986 s
of the census's 1.4794 s booking for this arm is a pricing artifact rather than fold time. One key,
`matmul|out=1024x32x512|K=512`, is also priced on an operand the fold never builds: the census
reconstructs a 1024-batched second operand where the fold broadcasts one 0.52 MB L1 tensor, a 9.00x
byte overstatement. A seventh key the fold issues 200 times, `matmul|out=1x768x512|K=4480`
(`tenstorrent.py:10712`), is missing from the census budget entirely because its recorded K is
`None`.

Findings and verdict: `~/.coworker/state/c12-matmul-key-attribution.md`.
