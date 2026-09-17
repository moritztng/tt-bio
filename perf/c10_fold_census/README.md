# c10_fold_census — what the 512 aa fold's own shapes cost, at a recorded 1350 MHz

One measurement session on qb2 node 1 prices every launch key the Boltz-2 512 aa fold issues,
weights each by the fold's own call census, and measures the compute and bandwidth roofs in the same
process at the same during-sampled clock. Findings and verdict live in
`~/.coworker/state/c10-fold-census.md`; this file is how you reproduce them.

## Reproduce

```bash
cd <this worktree>
bash perf/c10_fold_census/run.sh mysweep 1 --grid-control 40
python3 perf/c10_fold_census/reduce.py perf/c10_fold_census/runs/mysweep
```

`run.sh` pins the node, the lease and the clock; `reduce.py` is CPU-only and re-runnable on an
archived run. Both refuse to produce a table if the known-answer control fails.

## What the session guarantees

- **The clock is sampled during every timed interval**, at about 1 kHz, by a separate process
  (`node_control.py`). An interval that is not min = max = 1350 MHz, or that has a read error or a
  gap over 10 ms, is dropped as an artifact. A header reading is not a measurement.
- **Both counters pass a known-answer control before any timing starts.** A bf16 8192^3 matmul is
  exactly 1,099,511,627,776 matrix FLOPs and exactly 402,653,184 bytes. The byte counter in this
  project has been wrong three times, in two directions, so the census's own identity
  `B == calls * (in_tiles + out_tiles) * 2048` is re-checked too.
- **The roofs are measured here, not asserted.** The dense cube runs twice, once at each end of the
  session, and the two readings bound session drift.
- **Refutation criteria are pre-registered** in `prediction.json` and evaluated in `budget.json`
  under `refutation.fired`. The load-bearing one: the weighted fold seconds must fit inside the
  14.881 s bare fold. If they do not, at least one arm is running a configuration the fold does not
  use and no rate from the session may be quoted.

## Files

| file | what it is |
|---|---|
| `prediction.json` | predictions, decision rule and kill criteria, written before the first arm ran |
| `census.py` | launch keys -> runnable arms; FLOP and byte counters; the known-answer and byte-identity controls |
| `replay.py` | the session: force the clock, start the sampler, measure the roofs and every arm, release in a finally |
| `reduce.py` | CPU reducer: best qualified block per key, roof pricing, class table, cycles-above-roof ladder |
| `node_control.py` | during-interval clock sampler, node identity, holder observation, interval qualification |
| `run.sh` | one bounded session with the node, lease and environment pinned |
| `imports.json` | sha256 of every helper copied in from another row; `replay.py` refuses to run on drift |
| `runs/<name>/` | preflight, raw rows, reduced budget, clock samples, holder samples, sampler log |

## Reading the table

Cycles are the currency: seconds on this fixture depend on AICLK, and 800 MHz reads 21.90 s where
1350 MHz reads 14.69 s on the same work. `above_roof_Mcycles` is measured cycles minus what the
binding roof allows, and `binding` says which roof that was. Two cautions the numbers do not carry
themselves. Launch is a third roof and is not priced at all, so a key flagged `dispatch_shaped`
(within 3x of the session's own per-call dispatch floor) has no meaningful headroom above the
arithmetic and traffic roofs. And this is a standalone replay weighted by a call census, not an
in-situ measurement of the fold, so a key's cost is bracketed by the configurations tried rather
than pinned to the fold's own memory and program config.
