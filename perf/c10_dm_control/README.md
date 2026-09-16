# Runtime counter calibration

This control checks dense graph accounting and runtime data-movement profiler zones on card 0. It does not measure model performance, a hardware roof or a whole-fold census.

Run on qb2 from this private worktree:

```bash
bash perf/c10_dm_control/run.sh reproduce > perf/c10_dm_control/reproduce.log 2>&1
python3 perf/c10_dm_control/archive.py --dir perf/c10_dm_control/runs/reproduce
python3 perf/c10_dm_control/analyze.py --dir perf/c10_dm_control/runs/reproduce
```

Use a new capture name for each run; an existing name is refused. The committed evidence remains unchanged.

The existing k10 build at 1452925b must contain the DM patch from 4d80e855. The wrapper selects k10's ttnn, Tracy and libraries without rebuilding them. It takes the normal benchmark lock with 60-second lock and load waits. A lock timeout stops the run. A load warning remains in the log and restricts any result to instrument calibration.

[criterion.json](criterion.json) fixes the count, clock and separation checks before capture. One device context captures an 8192-cube dense matmul, then alternates three batches each of 8192-square add and 2048-cube matmul. Each batch contains 40 operations. Three synchronized 32-square exp operations mark each boundary. Host monotonic capture boundaries select independent sysfs samples; raw device ticks retain their separate timebase.

FORCE_AICLK is loaded verbatim from 5c60137c2, after checking the actual open node is 0, and released in finally. Each capture needs at least three wholly contained samples at minimum=maximum 1350 MHz, with no sampling gap above 10 milliseconds. Other-chip holders are recorded; another holder on node 0 stops the run.

The dense graph must return exactly 1,099,511,627,776 matrix FLOPs under two per FMA and 402,653,184 modeled compulsory bytes under the version-2 terminal-output rule. This establishes neither physical DRAM traffic nor general FLOP-counter exactness. Rectangular, transposed, broadcast and opaque generic operations remain outside this control.

The DM check divides each operation's summed cycles by its own core count and raw kernel span. Each round must show add minus matmul NCRISC NoC-read fraction at least 0.40, and matmul minus add NCRISC reserve-back fraction at least 0.20. The historical signatures and source rationale are retained with the criterion. Compute wait-front belongs to TRISC0 and reserve-back to TRISC2. Unclassified time is not issuing time; overlapping RISC/core activity is never elapsed time.

No production code changes. Predicted and measured model cycles saved: zero. Passing this calibration only permits fresh quiet-host measurements.

The recorded pass satisfies both separation criteria in all three rounds at sampled minimum=maximum 1350 MHz:

| Round | Add minus matmul NoC-read fraction | Matmul minus add reserve-back fraction |
| --- | ---: | ---: |
| 1 | 0.781714 | 0.366485 |
| 2 | 0.781916 | 0.367981 |
| 3 | 0.782456 | 0.370753 |

All 241 fenced control invocations have matching raw kernel markers and use 110 cores. The seven synchronized intervals contain 17, 479, 9, 34, 9, 34 and 9 clock samples; the largest gap is 1.101781 milliseconds. The add graph contains 40 additions, exactly 2,684,354,560 elementary FLOPs and 16,106,127,360 modeled compulsory bytes. These are counts and instrument signatures, not throughput measurements.

Replay the committed evidence without opening a device:

~~~bash
python3 -m unittest discover -s perf/c10_dm_control -p 'test_*.py' -v
python3 perf/c10_dm_control/analyze.py
~~~

[analysis.json](analysis.json) retains every control's raw start/end cycles, thread sums, per-operation core count and separate host timestamps. [out/capture.json](out/capture.json) records loaded ttnn/Tracy/library paths and hashes, clock requests, observed nodes, boot IDs and cleanup. [raw_manifest.json](raw_manifest.json) hashes compressed and original profiler files; [kernel_sources.json](kernel_sources.json) hashes the named kernel sources. The criterion and capture harness were committed at ec761a98f before device execution. The wrapper now writes reproduction runs to a new named directory.

Sum profiling disables this build's C++ report. The analysis reads integer kernel markers and accumulator data directly from the raw device CSV, preserving its reported frequency of 1350 MHz. It checks the legacy report's rounded nanoseconds by forward conversion from raw cycles. The historical dm_report helper is retained with corrected TRISC labels and an unclassified residual label.

The profiler's program-hash metadata cache reports 32×32 inputs for all 120 add calls because they share a program with a 32×32 initialization. Their fresh graph records 8192×8192 inputs. The analysis retains this discrepancy and attributes the named controls using that graph and the fenced schedule. Cached profiler shapes do not establish per-call shape identity or full-table exactness.

The independent holder log records another process on node 3. Benchlock acquired with load averages 1.08, 1.60 and 1.38 and emitted no load warning. Neither fact establishes quiet-host timing. No reset, shared-build edit, model transform or production change was performed.
