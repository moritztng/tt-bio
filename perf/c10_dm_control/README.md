# Runtime counter calibration

This control checks dense graph accounting and runtime data-movement profiler zones on card 0. It does not measure model performance, a hardware roof or a whole-fold census.

Run on qb2 from this private worktree:

```bash
bash perf/c10_dm_control/run.sh > perf/c10_dm_control/run.log 2>&1
```

The existing k10 build at 1452925b must contain the DM patch from 4d80e855. The wrapper selects k10's ttnn, Tracy and libraries without rebuilding them. It takes the normal benchmark lock with 60-second lock and load waits. A lock timeout stops the run. A load warning remains in the log and restricts any result to instrument calibration.

[criterion.json](criterion.json) fixes the count, clock and separation checks before capture. One device context captures an 8192-cube dense matmul, then alternates three batches each of 8192-square add and 2048-cube matmul. Each batch contains 40 operations. Three synchronized 32-square exp operations mark each boundary. Host monotonic capture boundaries select independent sysfs samples; raw device ticks retain their separate timebase.

FORCE_AICLK is loaded verbatim from 5c60137c2, after checking the actual open node is 0, and released in finally. Each capture needs at least three wholly contained samples at minimum=maximum 1350 MHz, with no sampling gap above 10 milliseconds. Other-chip holders are recorded; another holder on node 0 stops the run.

The dense graph must return exactly 1,099,511,627,776 matrix FLOPs under two per FMA and 402,653,184 modeled compulsory bytes under the version-2 terminal-output rule. This establishes neither physical DRAM traffic nor general FLOP-counter exactness. Rectangular, transposed, broadcast and opaque generic operations remain outside this control.

The DM check divides each operation's summed cycles by its own core count and raw kernel span. Each round must show add minus matmul NCRISC NoC-read fraction at least 0.40, and matmul minus add NCRISC reserve-back fraction at least 0.20. The historical signatures and source rationale are retained with the criterion. Compute wait-front belongs to TRISC0 and reserve-back to TRISC2. Unclassified time is not issuing time; overlapping RISC/core activity is never elapsed time.

No production code changes. Predicted and measured model cycles saved: zero. Passing this calibration only permits fresh quiet-host measurements.
