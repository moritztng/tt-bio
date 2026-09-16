# Burst-clock operator census

Native matmul and binary operations lead the four sampled model windows at observed 1350 MHz, with 53.00 million and 25.85 million raw program-span cycles. The full default fold completed, but automatic host report generation exceeded the resource budget: **STOP with partial coverage**, not a whole-model ceiling.

[The measured table](TABLE.md) contains 1,478 model invocations and 36 control invocations. The four model windows total 161,450,686 program-span cycles. They cover one MSA layer, one Pairformer layer, one diffusion step and the confidence head. Counts are actual captured executions, never medians multiplied by occurrence counts. This selection is not representative weighting for the fold. Native matmul accounts for 32.83% of these captured spans; that percentage is not its whole-fold share.

The fold used the committed cdk2x2_512 fixture and fixed 35-row MSA, 200 sampling steps, three recycles, one sample, seed 0 and no templates. Its [output CIF](runs/census1/out/cifs/cdk2x2_512.cif), actual configuration, module occurrence counts and hashes are retained. There were no production code changes, model optimizations, dependency changes, fewer steps, MSA truncation or tensor parallelism.

## What is measured

A separate process recorded 591,386 during-fold sysfs samples: minimum and maximum 1350 MHz, largest gap 6.987061 ms, no read errors. All 13 retained model/control windows independently pass the predeclared clock criterion. FORCE_AICLK was released and the device closed before postprocessing. Physical node 0 was exclusive to this process; node-3 co-tenancy was recorded. Containment stayed active, the boot ID stayed fixed, and the driver srcversion was `A10759A24565BC5BBE903C5`.

The accepted DM calibration was independently replayed before device work, including original/archive hashes, exact byte counters and all three fixed separation rounds. [The second live smoke](runs/smoke2/analysis.json) passes 16 independent float64 math checks, eight exact off/on comparisons, dispatch counts and descriptor address rebinding. The first smoke correctly stopped on a binding mismatch; `CoreRange` exports `start/end`, not its C++ member names. Missing compiled-code provenance, `opt_level` and `custom_program_hash` remain explicit.

The raw reducer uses integer per-core/RISC start/end markers from the calibrated sum profiler. It joins native operation counters through the recorded `EncodePerDeviceProgramID(base, device)` implementation. On node 0 the profiler ID is `base << 10`. All raw IDs equal each window’s consecutive native counter range plus six fence programs. Generic observer outcomes match their encoded IDs exactly. Fresh graph operands supply shapes; cached profiler shape labels never do.

Every selected native start has one matching end edge. Outer graph nesting is unbalanced, so inclusive enclosing spans are not used. Legacy `SDPAOperation` and `CloneOperation` names are included by recorded identity. The graph has no output tensor edges for 31 head-splitting calls, whose bytes remain unknown. Generic descriptors describe source/runtime inputs; they do not prove cached machine-code provenance.

Matrix shape arithmetic is known for 516 of 1,478 model programs: 1,014,953,345,536 FLOPs under two per multiply-accumulate, with 1,017,716,932,608 conditional tile32 FLOPs. These are different from issued instructions. Head-major generic contractions, attention, reductions and SFPU work remain symbolic or uncounted. Byte estimates model one logical-payload DRAM read per unique input address and one write per unique output address; physical padding, rereads, spills and cache traffic are unknown.

## Matched controls

One process alternated three rounds of four streaming copies, four adds and four dense 8192-cube matmuls. All 36 outputs exactly equal their independent constant references. Dense matmul uses bf16, HiFi4 and FP32 accumulation; control tensors use interleaved DRAM. Each control has the same graph/sum instrument as the model windows. Rates below divide known work by summed raw program spans, excluding inter-program gaps, at sampled 1350 MHz.

| Control | Four-call spans, cycles across three rounds | Modeled bytes/cycle or shape FLOPs/cycle |
| --- | --- | --- |
| Copy | 3,643,374 / 3,637,803 / 3,633,530 | 294.71 / 295.16 / 295.51 bytes/cycle |
| Add | 4,843,153 / 4,859,147 / 4,853,811 | 332.55 / 331.46 / 331.82 bytes/cycle |
| Dense HiFi4 | 48,689,330 / 48,694,899 / 48,684,670 | 90,328.75 / 90,318.42 / 90,337.40 FLOPs/cycle |

These controls are specific to their kernels, layout, fidelity and instrument. Their rates are not indiscriminately applied to the model’s mixed L1/DRAM layouts or custom kernels. No observed binding roof, model utilization, floor or cycles above a roof is established.

## Why the verdict is STOP

The full device capture finished and cleaned up. Tracy then expanded its 2,919,007,827-byte host trace into a 61,753,834,686-byte timing CSV. Its automatic Python postprocessor reached at least 182,320,308 KiB RSS. The owned, device-closed postprocessor was terminated at the resource limit. No foreign process was signalled and no card was reset.

All selected raw device windows and the complete 26,969,586-byte host operation metadata export survived. They are sufficient for the bounded CPU reducer. The oversized original host files remain at the absolute paths and hashes in [oversized_host_artifacts.json](runs/census1/oversized_host_artifacts.json); they are not in Git and are not needed for this replay. Committed compressed clock/metadata copies preserve the originals exactly.

Raw data outside named windows was synchronously drained and excluded with a hash/byte ledger. Thus 259,634 host device-operation records are not 259,634 cycle-measured model operations: they also include fences, controls and setup. The default two-argument Pairformer signature has no separately retained window. No full-fold device-cycle closure or instrumentation perturbation measurement exists. Host wall times are diagnostic only. Unclassified fence gaps are not CPU work. Raw wall-clock register extents and per-core durations do not establish a synchronized host conversion or independently measured cross-core offsets.

Before another capture, bound or bypass automatic host timing export and its in-memory postprocessor, then add disjoint coverage for the remaining module signatures. Native matmul and binary operations are the first classes to investigate within the measured scope. Their ranking does not authorize an optimization or predict its saving. Any later accuracy-changing lever still needs independent float64 controls and paired 512/298 structure checks; the 512 bar is 0.60 Å with a 1.84 Å seed floor, and the 298 target retains its own documented bar.

## Reproduce on CPU

From the repository root:

```bash
python3 perf/c10_burst_census/verify_calibration.py
python3 perf/c10_burst_census/reduce.py --dir perf/c10_burst_census/runs/smoke2 --table perf/c10_burst_census/runs/smoke2/TABLE.md --require-go
python3 perf/c10_burst_census/reduce_census.py
python3 -m unittest discover -s perf/c10_burst_census -p test_*.py -v
python3 perf/c10_burst_census/verify_artifacts.py
```

The census reducer writes STOP while retaining every validated row. Its successful exit means replay completed, not that the measurement gate passed. `run_census.sh` preserves the original hardware command and reproduces the known oversized host-export path; do not launch another census until that resource limit is addressed. Live smoke reproduction is described by `run_smoke.sh` and requires an explicit node-0 grant, normal benchlock and during-interval clock sampling.

[The manifest](manifest.json) hashes committed evidence. Capture files record source revision and dirty diffs separately from loaded-library hashes, actual flags, fixture/MSA hashes, holders and clock brackets. The historical shipped-kernel catalogue was inspected as context; its old whole-fold ranking, utilization and floor remain unvalidated. The dense calibration survives. The inference that bounded device dumps also bound automatic host reports is refuted. No fitted planning coefficient or inverse-clock intercept is presented as measured work or CPU time.
