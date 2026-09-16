# Burst-clock census evidence

The live identity gate stops this census: the observer cannot read per-core runtime arguments from the installed k10 binding. No model fold or matched roof was captured.

[The bounded control table](TABLE.md) preserves 16 generic invocations at sampled 1350 MHz. All 16 output comparisons equal independent float64 CPU matmul/permutation references exactly, and all eight observed outputs equal their observer-off counterparts. Each arm dispatches exactly four generic programs. The observer identifies the audited wrappers and their operand roles correctly.

`CoreRange` exposes `start` and `end` in the recorded binding source, while the observer asks for `start_coord` and `end_coord`. Those getters fail live, which also makes its coordinate-based `RuntimeArgsView` enumeration fail. Direct native indexing returns the constructed `[13, 17]` values. The observer footer says `observation_ok=true` while retaining these gaps; that flag means capture mechanics succeeded, not that execution identity is sufficient. The reducer refuses continuation.

Program-descriptor hashes stay constant across the changing inputs, but missing runtime snapshots prevent validation of matmul address rebinding. All profiler program-cache-hit labels are false. Descriptor hash equality does not prove a compiled-program cache hit. Missing `opt_level`, `custom_program_hash`, storage aliases and binary provenance remain explicit.

## Reproduce without a device

From the repository root:

```bash
python3 perf/c10_burst_census/verify_calibration.py
python3 -m unittest discover -s perf/c10_dm_control -p 'test_*.py' -v
python3 -m unittest discover -s perf/c10_burst_census -p 'test_*.py' -v
python3 perf/c10_burst_census/reduce.py
python3 perf/c10_burst_census/reduce.py --require-go
```

The final command must exit 2. A successful reduction writes a report; it does not turn STOP into GO. The dependency replay verifies archived raw hashes, independently recounts the dense graph, and requires all seven clock-qualified intervals and all three fixed DM separation rounds. Its accepted calibration establishes no model roof.

The evidence is in [runs/smoke1/analysis.json](runs/smoke1/analysis.json), with every raw program span, graph/profiler join, missing identity field and clock bracket retained. [manifest.json](manifest.json) hashes this deliverable. [provenance.json](provenance.json) records the measurement commit, intended fixture/MSA hashes and resource limits. `runs/smoke1/out/capture.json` records loaded libraries, source hashes, dirty patches, flags, device holders, boot ID and cleanup. `raw_manifest.json` inside the run hashes original and compressed profiler files.

## Hardware reproduction

Only run on an explicitly granted qb2 physical node 0, with containment active and driver srcversion `A10759A24565BC5BBE903C5`. From this worktree, choose an unused name:

```bash
bash perf/c10_burst_census/run_smoke.sh reproduce
python3 perf/c10_dm_control/archive.py --dir perf/c10_burst_census/runs/reproduce
python3 perf/c10_burst_census/reduce.py --dir perf/c10_burst_census/runs/reproduce --require-go
```

The wrapper takes normal benchlock with 60-second lock/load waits and selects the existing k10 build without rebuilding. It sets the card grant and holder identity, verifies the opened node, loads FORCE_AICLK from the recorded source, and releases it in `finally`. A separate process samples sysfs during every interval. Captures need at least three samples, minimum=maximum 1350 MHz, and no gap above 10 milliseconds. A lock timeout stops the run.

The measured smoke used one process and device context, four fenced arms, two alternating input sets per arm and four calls per arm. Matmul uses 64-square bf16 inputs with HiFi4 and FP32 accumulation; reblock permutes `[1,64,64,32]` to `[1,32,64,64]`. Inputs are exactly representable small integers. Matmul reuses two output buffers, so repeated checks also inspect those final buffers; raw dispatch multiplicity is checked separately. Graphs include the generic destination twice under the same tensor ID, which the reducer verifies before matching ordered operands. The join is justified only for this bounded serial schedule, with no trace/replay. It does not generalize to arbitrary model traces.

## Limits and next step

Node-3 co-tenancy was observed. Host walls are diagnostic only. Off arms precede on arms and include initial compilation; differences between their envelopes cannot measure observer overhead. Graph capture records `Tensor::cpu` calls in both arms. No bare/profiler perturbation experiment was performed.

Raw integer marker extents remain in their device-counter timebase. No conversion to host elapsed time is established. The table separates program spans from unclassified fenced gaps; gaps are not CPU work. Physical reads, spills, issued arithmetic, utilization and cycles above a roof are unknown. Matrix shape FLOPs use two per FMA; permutation is explicitly not a contraction. Nothing is extrapolated to a fold.

Repair the observer's binding field mapping and revalidate native runtime enumeration, address rebinding and the execution join before capturing any current-default model windows. Then budget graph/profile volume and collect matching controls with the shipped fidelity/layout. The intended cdk2x2_512 fold still requires its fixed 35-row MSA, 200 steps, three recycles, one sample, seed 0 and templates off. None of that work was run or reduced here.

The dense calibration and the need for during-interval clock samples survive. The inference that CPU-only observer tests certify installed runtime identity is refuted. Historical model ranking, binding roofs, utilization, floor and campaign ceiling remain unmeasured by this capture. The fitted planning coefficient is not a measured cycle total and its intercept is not measured CPU time.

No production code changes, model optimization, dependency change, reset, flash or firmware/kernel change. The branch is unmerged.
