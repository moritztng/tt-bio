# opendde / protenix-v2 p300c baseline reseed, 2026-09-09

Both cells sat at 2026-08-24 / tt-bio 0.7.0 because the 2026-09-01 qb2 reseed skipped them, so the
gate read them +26.7 % and +18.4 % against live behaviour. Stale low is the direction that hides a
regression: opendde could have lost a fifth of its throughput and still passed the 15 % gate.

Protocol, same as the 2026-09-01 reseed: qb2 p300c, one fresh process per draw, each draw is
`perf_regression.py --measure <model>` (2 warmup folds excluded, 5 timed folds, median), each draw
under `benchlock.sh` with a quiet-host precheck that refuses to record when another process holds a
`/dev/tenstorrent` node. Five draws on card 0 pick the cell, three on card 3 control it.

- `draws.tsv` — all 16 draws.
- `quietchecks.txt` — the quiet-check verdict and loadavg recorded for each one.
- `postedit_gate.txt`, `postrebase_gate.txt` — the gate re-run against the new cells, at tree 513c921c
  and again on current main, with opendde-abag as the
  already-reseeded control.
- `draw.sh`, `quiet_check.sh` — the runners.

Full write-up: `state/perf-baseline-reseed-opendde-protenix-v2/FINDINGS.md`.
