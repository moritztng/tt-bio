# Roofline measurement controls

The CPU byte-accounting controls pass. Hardware validation has not run because a sibling process holds the board; these artifacts establish no Boltz-2 floor or performance ceiling.

[Preflight](preflight.json) records k10 source/build hashes. [Loaded paths](runtime_import.json) identify imported libraries. [Live holder evidence](quiet_board.json) records the unavailable quiet-board prerequisite.

The 8192-cube control expects 1,099,511,627,776 FLOPs and 402,653,184 compulsory bytes. A graph counter does not measure internal rereads or spills. Both counters pass CPU cases for aliases, scratch, L1 outputs, preallocated writes and gated reads:

```bash
python3 -m pytest -q tests/test_perf_traffic.py
```

The device harnesses are prepared and syntax-checked, but unexecuted. On qb2, require an empty live device-holder set, then run from this worktree:

```bash
bash perf/c10_roofline_recheck/run.sh counter
bash perf/c10_roofline_recheck/run.sh dm
```

The wrapper uses benchlock, card 0, existing k10 and the opt-in clock holder from `5c60137c2:tt_bio/aiclk.py`. No production code changes. An independent process samples sysfs during synchronized intervals and checks foreign holders. Any failed count or clock prerequisite stops validation.

The DM harness reuses add/matmul workloads from `4d80e855`; output zones still require calibration against each invocation's own cycles. Its output is not a roof measurement or fold census. The calibrated roofline, profiler perturbation and whole-fold reconciliation remain unmeasured.

