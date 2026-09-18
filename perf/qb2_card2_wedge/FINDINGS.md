# qb2 card 2: does the host-spin wedge reproduce under a real load?

No. 100 consecutive 298 aa folds at a verified 1350 MHz, 1028 s, zero frozen-syscall intervals.
Card 2 was cleared for dispatch on the strength of this run. The full clearance record, with the
reset that was needed first and the tt-smi finding that came out of it, is
`~/.coworker/state/cardblock-qb2-2-cleared-20260918.md`; this file is the harness note.

## Harness

Nothing here folds. `run_soak.sh <node> [max_s] [n_folds]` drives three pieces that already existed:

* `perf/roof_shared/fold_shared.py` for the fold loop, one process, one device open, production
  `_WorkerState.predict_one`, `cdk2x2_298` at 200 sampling steps and 3 recycles. Run as
  `--arms plain` with one seed per fold, so N seeds is N folds and each one writes its record as it
  lands.
* `perf/c12_genop_rate/clk.py` for the clock, ARC message FORCE_AICLK 0x33, sampled off sysfs at
  2 Hz in its own process so the sampler never takes the fold's GIL.
* `iowatch.py`, new here, for the wedge signature itself.

The clock force starts only AFTER the device is open, gated on `"grid"` appearing in the fold's own
output JSON. That ordering is deliberate: attempt 1 died inside `ttnn.open_device` while the clock
was forced, and with the force in place first there is no way to rule it out as the cause. It is
ruled out by construction now.

## Why iowatch samples syscalls and not CPU

The wedge is a process pinned near 100 % CPU with BOTH `syscr` and `syscw` frozen while its output
goes stale. CPU alone cannot see it, because a holder at 100 % CPU can be a corpse and a live fold
at 190 % looks the same from the outside. `/proc/<pid>/io` can: a live 298 aa fold moves at least
44725 `syscw` per 15 s, and the wedge moves zero. So the flag is `d_syscr == 0 and d_syscw == 0`
sustained past `--stall-s`, with CPU only as a corroborating term.

## Reading the output

    out/folds-node<N>.json      per-fold wall, CIF digest, plddt, atom count
    out/iowatch-node<N>.jsonl   15 s samples: syscr, syscw, pcpu, aiclk, output staleness
    out/clock-node<N>.jsonl     2 Hz AICLK, first line records the force status
    out/soak-node<N>.log        open timing, holders before the run, teardown

A pass is: every fold completed, no sample with `WEDGE` true, no interval with both syscall
counters frozen, and every clock sample at the target. All four held on card 2 and on card 3.
