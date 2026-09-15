# The census could not score the fp32-softmax L1 rectangle

`scripts/lever_census.py` read `calls` and `blocked` off `FP32_SOFTMAX_STATS` for every
`stats-dict` row. Those two keys belong to `FP32_SOFTMAX_BIAS_HOIST`. The same dict also carries
the L1-residency counters (`l1`, `l1_blocks`, `l1_refused`, `l1_cores`), so a row for
`_FP32_SOFTMAX_L1_GRID` built on that reader would have reported the bias hoist's numbers as its
own and read healthy whether or not a single L1 block ever ran.

The fix names the keys per row: a `stats-dict` counter spec is now
`MODULE.NAME:served_key,declined_key`, with no fallback. The L1 rectangle gets a row of its own
reading `l1_blocks,l1_refused`, plus `l1_cores` as a gauge — a last-value counter, unioned across
processes rather than summed, because two workers on 72 cores each would otherwise sum to a grid
that does not exist.

## The A/B

Card 3 on qb2 (p300c), `TT_BIO_FORCE_GRID=8,9` so `_apply_grid_thresholds` takes the small-grid
branch the lever lives in, `BOLTZ2_FP32_SOFTMAX=1` so Boltz-2 reaches `_fp32_softmax_attention`
(the path OpenFold3 and AF2-IG take by default; Boltz-2 is the model whose weights are on this
box). `cdk2x2_512.yaml`, single sequence, 1 recycle, 20 sampling steps. Arms are
`TT_BIO_FP32_SOFTMAX_L1_LIVE_GRID=0/1`.

| arm | resolved | served (`l1_blocks`) | declined (`l1_refused`) | `l1_cores` | BIAS_HOIST row |
|---|---|---|---|---|---|
| off | (8, 8) | 12504 | 0 | 64 | served=408 declined=288 |
| on | (9, 8) | 16248 | 0 | 72 | served=408 declined=288 |

Every field of the L1 row moves with the lever. The bias-hoist row does not move at all, which is
the point: 408/288 is exactly what the old reader would have printed for the L1 row in both arms.

Cross-checked against an instrument the census does not share any code with. The engine's own
`TT_BIO_CAPACITY_CENSUS=<dir>` dumps `FP32_SOFTMAX_STATS` at process exit, and it reproduces the
census row exactly: `l1_blocks` 12504 / `l1_cores` 64 off, 16248 / 72 on, `l1_refused` 0 on both.

This is a forced grid on a Blackhole part, so none of it is a perf reading and none of it is a
Wormhole result. It measures the instrument, not the lever. The lever's own measurement is
`perf/roof_bh_env/`.

`resolved` also stopped unioning in processes that never opened a chip. The launcher imports
`tt_bio.tenstorrent` and never folds, so it still holds the pre-device `(8, 8)` and the on-arm row
read `(8, 8)/(9, 8)` for a lever that resolved to one rectangle everywhere it ran. That is the
`11x10/13x10` grid-stamp false alarm `_compute_grid` already documents, and the grid stamp is the
measured flag, so no new field was needed.

## What 1024 aa showed, and why it is a partial reading

A second probe at `cdk2x2_1024.yaml`, same forced grid, lever on. The fold did not finish: the
worker stopped dumping at 05:20 UTC and qb2 rebooted at 05:34, which took the run's log and its
output directory with it. The per-process dump it had already written is
`results/dump_1024_worker.json`, and the counters in it are real for the work that ran.

    resolved (9, 8)   served 68341   declined 1   l1_cores 64

Two things in one row that the 512 pair could not show.

`declined` moves. `l1_refused` is the sharded softmax refusing its circular buffers around a
block, and at 512 aa it is 0 on both arms, so that half of the row was only covered by the unit
test. Here it fired on device.

`l1_cores` is 64 while the rectangle resolved to (9, 8) = 72. The floating-core search replaced
the tuned answer at this size, so the lever's default is not what ran. That is the whole reason
the gauge exists and is not folded into `resolved`: reading the default would have said the live
grid served this fold, and it did not.

## Files

- `results/census_off.json`, `results/census_on.json` — the two 512 aa arms.
- `results/dump_1024_worker.json` — the 1024 aa worker dump, from a fold that did not finish.
- `tests/test_lever_census_reasons.py` — card-free checks: each row reads its own keys, the
  negative control where the L1 path is dark and the bias hoist's counters are untouched, every
  `stats-dict` row names its keys, a gauge is unioned and not summed, and `resolved` prefers a
  process that opened a chip.
