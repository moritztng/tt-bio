# `TT_BIO_SDPA_WIDE_K`: what is measured, and what is not

`TT_BIO_SDPA_WIDE_K` has been on by default on `origin/main` since 2026-09-20 14:05Z
(`7662fc58b`). It has **no fold-level number**. The 1.1285x that gets quoted for it is a
Protenix-v2 trunk STAGE reading, which is a screen and not a result, and it recorded no board
class, no during-sampled AICLK and no A/A floor.

## `REPLAYED_NOT_A_MEASUREMENT_wide_k_20260920T2321Z.json`

Do not cite this file. It reads `off 26.95 s / on 24.65 s / A/B -8.534 % / outside the floor` and
every one of those numbers belongs to a different run: a `pvx-gate-land` cell from 14:06-14:10 on
2026-09-20, on a different card at an unrecorded clock.

`fold_ab_flip.py` used to give each (model, rung, arm, rep) a fixed `out_dir`, and
`tt_bio.main predict` short-circuits on an `out_dir` that already holds a finished prediction: it
prints `All predictions complete`, exits 0 in about 3 s, and folds nothing. The harness then read
the `results.json` already sitting there. The A/A floor replays along with the delta, so the cell
is self-consistent and looks healthy. The only tell was that the five folds it reported sum to
172.6 s inside 83 s of wall clock, and that the per-fold logs were 25 bytes.

Fixed in `b8cbb85a7`: `one_fold` clears `out_dir` first and rejects a `results.json` older than the
call that should have written it.

## What a real cell needs here

Four clean folds on 2026-09-20 at 23:27-23:30Z (qb2 card 2, p300c, benchlock held, acquired at
loadavg 1.81) gave warm-up 30.0 s, off 26.1 s, on 24.4 s, off 26.1 s: A/A 0.000 %, A/B -6.51 %.
Consistent with the lever and correctly below the 1.1285x stage screen, but not reportable, on two
counts:

* n=1 on the ON arm. The run was killed at rep 2 of 3 when a co-tenant started folding outside the
  benchlock and took loadavg to 8.87.
* `sample_contention.py` samples every 30 s, which was calibrated for size-ladder folds that run
  for minutes. Against a 25 s fold it cannot say the clock was 1350 MHz *during* the fold rather
  than between folds, and card 2 reads both 800 and 1350 in that window.

So run the cell at **1088 aa**, not 352. It fires, it is a size-ladder rung, and its folds run 4-5
minutes, which is long enough for the 30 s sampler to land inside one. Budget ~35 min of device
time at REPS=3.
