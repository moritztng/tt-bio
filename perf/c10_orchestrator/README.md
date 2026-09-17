# C10: what the campaign measured

Boltz-2 at 512 aa, targeting 10.0 s. Everything here is CPU-side: analysis, provenance checks and
consolidation over artifacts other rows measured. Nothing in this directory opened a device.

**Baseline.** 14.8813 s at 512 aa and 9.6801 s at 298 aa, at a pinned AICLK sampled during every
fold, min = max = 1350 MHz across all 36 folds, 16 accepted per size, zero same-seed structural
deviation (`c10-bare-baseline`, `bc66f7d6d`).

**Where the fold's time actually is.** `c10-fixed-cost` measured both terms of `T = F + W/f` at
four pinned clocks: **F = 3.9830 ± 0.1181 s is clock-immune** and **W = 14665.0 ± 121.1 Mcycles**
scales with AICLK. Two things then happened to the campaign's reading of `F`. The one lever sized
against it, a trace of the diffusion loop's host dispatch, was predicted at +3.04 s and **measured
-0.0214 s — nothing**. And `F` turns out to be size-*dependent*, scaling at N^1.3, which host
dispatch cannot do. So `F` is not collapsible overhead, and the byte axis is the only known lever
against it.

**Where this is heading, on evidence in flight.** `c10-size-scaling`'s 512 → 768 aa leg puts both
terms near N²: `p_work` = 1.83 and `p_fixed` = 1.91, against 0.63 and 1.32 on the 298 → 512 leg. If
that holds, the sublinearity was 298 aa under-filling a 110-core grid, and the fold at 512 aa is
dominated in *both* terms by the N² pair representation — which is also where
[`arithmetic_free_traffic/`](arithmetic_free_traffic/) found 31 % of the fold's bytes moving through
ops that compute nothing. Two independent lines converging on the same tensors is the most useful
thing this campaign has to hand Phase 2.

## Findings

| | |
|---|---|
| [`arithmetic_free_traffic/`](arithmetic_free_traffic/) | **Half the fold's bytes move through ops that do 0.049 % of its arithmetic** — 201,744 calls, 1.401 TB. Roof-free: both capture artifacts agree on the call total exactly and the byte total to 3 ppm. Concentrated in three classes, not a tail: `multiply_`, `layer_norm` and `add_` move **30.8 % of all fold bytes** and compute nothing. Traffic 2.07–2.52 s; realistic prize **0.69–0.84 s** at the one-third return this project has measured for that fusion. Also finds the **third byte-counting defect** of the campaign: the census's own `B = calls × tiles × 2048` identity holds at median exactly 1.000 over 50 shapes and two shapes break it, under-counting 155.8 GB (5.45 %) — correcting it raises the headline to 51.6 %, so the published figures are conservative. |
| [`floor_vs_measured/`](floor_vs_measured/) | **The 2.309 s prize was the clock, and it is retired.** `prize_s = 17.34 − 15.031` and 17.34 s is the ~1063 MHz reading; the same fixture at a pinned 1350 MHz is 14.846 s, **0.185 s below that floor**. The floor's traffic/arithmetic split also lands within 5.7 % and 0.5 % of the measured `F` and `W/f` — interesting, not a validation: its shape rates were measured on **pc's 130-core firmware**, not on qb2. |
| [`size_scaling/`](size_scaling/) | **Likely about to be withdrawn — see its status banner.** Read from the two measured sizes, the clock-scaled work grows at N^0.63, slower than the target does. Work ratio 1.4096 ± 0.0131 against a token ratio of 1.718. Under a non-negative mixture that puts a rigorous floor of **3,301 Mcycles (22.5 % of W512)** under work that does not grow with the target at all, and the pair-tensor reading puts it at 8,220 (56.1 %) — against a deletion target of 6,542. `F` scales at N^1.32 ± 0.07, 19 σ from zero, which is how it is known not to be host dispatch. Could still be 298 aa under-filling the grid, and [`floor_vs_measured/`](floor_vs_measured/) now gives that a number: under an N^2 FLOP model 298 aa need only achieve 48 % of 512 aa's rate to explain the whole thing, which is what 100 tiles on 110 cores would do. `c10-size-scaling`'s in-flight 512 → 768 leg gives **p_work = 1.83**, the artifact range, against 0.63 on 298 → 512. Not yet its verdict; 768 aa is incomplete and 640 aa has not run. |
| [`fixed_cost/`](fixed_cost/) | **3.952 s of the fold is clock-immune**, 27 %, solved from an interleaved 800/1339 MHz A/B that was already in the corpus. Worst residual 33 ms; predicts an out-of-sample arm to 1.9 ms. A mean-clock label carries 0.364 s of error across sessions, which is larger than any lever on record, so old ratios cannot be clock-corrected by arithmetic. |
| [`dispatch_hypothesis/`](dispatch_hypothesis/) | **Refuted by measurement.** The fold issues 465,664 top-level ttnn calls and the diffusion loop is 219,200 device programs, 76 % of them, so a trace of it was sized at **3.02 s**. `c10-trace-lever` ran it: **-0.0214 s at 512 aa, inside a 0.055 s A/A floor**, byte-identical CIF. The call arithmetic was right and the inference from it was wrong. |
| [`floor_mix/`](floor_mix/) | **The byte axis caps at 1.487x** of the floor: delete every byte and 12.706 s becomes 8.543 s. Independently reproduces the lever corpus's 1.470x. The headroom is in the elementwise tail, not the matmuls: `ttnn.linear` is the largest call class in the fold and its entire byte headroom is 0.140 s. |
| [`shape_rank/`](shape_rank/) | **No giant is hiding.** 60 shapes, top 20 hold 31.5 %, none over 3.6 %. A lever has to hit a class across many shapes or remove per-call cost. |
| [`ladder/`](ladder/) | Every candidate with an evidence class. Priced total was 4.16 s reading 10.72 s; **3.02 s of that was the diffusion trace, now measured at zero**, so the ladder's remaining priced value is about 0.2 s. Two Wormhole rows are jointly 0.713 Å against a 0.60 Å bar and cannot both ship. |
| [`remainder/`](remainder/) | Of the 0.93 s the trace was thought not to reach, about 0.612 s is named host work: feature preparation, the input embedder, relative-position encoding, CIF writing. A hypothesis, measured on the wrong CPU, and now the trace reaches none of `F` at all, so the whole 3.98 s is unattributed rather than 0.93 s of it. |
| [`fit_reexam/`](fit_reexam/) | The campaign's 37.6 % target came from the least-supported of four fits. Tested against an arm in none of them, it misses by 284 ms where the clean fit misses by 1.9 ms. |
| [`grid_evidence/`](grid_evidence/) | The second-ranked lever's evidence is **Wormhole**, and on that same sweep the two classes move in opposite directions. **No Blackhole core sweep exists.** The 810 Mcycles was withdrawn as a cross-architecture transfer. |
| [`regression_check/`](regression_check/) | The "+2.3 % regression" on main is not one. The above-cap fused SDPA route needs `q_len > 1024` and 512 aa has 512, so it cannot fire; the gap is smaller than the comparison's own 0.364 s error. |
| [`cpu_prereq_review/`](cpu_prereq_review/) | Independent acceptance of the bounded exporter and the raw cycle stream. |

Every subdirectory has its own README, a script that regenerates its JSON, and known-answer
controls. Run them all:

```sh
for d in perf/c10_orchestrator/*/; do (cd "$d" && python3 -m pytest -q 2>/dev/null); done
```

## The reading

Three independent lines of evidence point at **per-call cost rather than per-kernel cost**: the
clock-immune term is 27 % of the fold, the floor is a 60-shape tail with no dominant member, and
the byte axis is capped at 1.487x with its headroom in the elementwise ops. That is also why this
project's byte-deletion levers have all landed at 1.02-1.05x.

On present evidence 10.0 s needs either the clock-immune term to give up more than a diffusion
trace reaches, or a device-work lever nobody has found. **Nothing here has been measured as a
fold-level win.** Cycles saved to date: zero. Accuracy spent: zero.

## Clock discipline

Every measured second above carries the clock it was taken at. Where a number does not, it is
labelled: the committed floor divides by a compute roof recorded at **loadavg 6.2 with no AICLK**,
and all 29 levers in the corpus ledger were measured at an unrecorded clock. Arithmetic-bound
seconds scale with AICLK and DRAM-bound ones do not, so an unclocked floor cannot be read at burst
in either direction.

## Tools

Run this CPU-only audit from the repository root:

```sh
python3 perf/c10_orchestrator/audit_clock.py
python3 -m unittest discover -s perf/c10_orchestrator -p 'test_*.py'
```

It validates the recorded clock aggregates and holder lists for eight folds per
arm, reproduces the historical two-endpoint fit and its sensitivity to using all
six folds, and calculates conditional work targets. It imports no device code.
Use `--out PATH` to save JSON; `--target-mhz` and `--target-s` change the projection.

`pinned_baseline.json` is an unchanged copy of
`perf/b2z2_aiclk_pin/out/mainab_qb2c1.json` from commit
`463da8e9bd326be091a008da7f886c3ecf5ece20` on
`origin/wk/b2z2-aiclk-default-decision`. That experiment measured commit
`0df13ad983e645722da958e176d22d96fa5067ee`. It records a 14.554 s median at
sampled minimum, mean and maximum 1350 MHz for the forced arm. This is historical
branch evidence, not a measurement of today's tree.

The historical fit uses the existing
`perf/b2z2_cell_recheck/out/cell_main_qb2c1_clock.json`. Its clocks vary during
folds, and three timed folds have foreign device holders. Neither artifact
retains raw sample timestamps, so continuous clock coverage and time-weighted
integration cannot be reconstructed. The script checks the aggregates actually
present; it cannot certify the original instrument.

The published `2.901 + 15355 / MHz` model does not identify removable host time
or a per-op cycle budget. Its work coefficient has units of MHz·seconds, or
Mcycles dimensionally. Reported target work and reductions are conditional
model calculations, not hardware cycle counts. `--baseline` and `--historical`
allow checking copies of these records, including intentionally invalid inputs.

The graph-traffic controls run without a device:

```sh
python3 -m pytest -q tests/test_perf_traffic.py
```

Both traffic counters share the terminal-output rule in
`perf/b2x_difflayer/itemize.py`. The saved 8192-cube bf16 matmul graph must count
402,653,184 bytes: two input reads and one output write. Metadata aliases add no
read; a consuming operation does. Internal scratch estimates remain separate in
`assumed_read_MB`, including opaque buffers in `opaque_buffer_assumed_read_MB`.
`traffic_model_version=2` distinguishes these counts from historical artifacts.
For fold analysis, use `perf/roof_arb/corrected_traffic.py` to retain its L1-output,
preallocated-output and partial-read corrections. A passing matmul control does
not certify unobserved kernel rereads, spills or whole-fold traffic.

`reconcile_cycles.py CAPTURE.json` partitions a single synchronized device
timeline into exclusive program time by class, concurrent program time, and
idle or unobserved time. It reports summed program durations separately so
overlap cannot silently become fold latency. Gaps are not labelled CPU work.

The JSON input contains `device`, `timebase`, `start_tick`, `end_tick` and
`programs`. Every program has a unique string `id`, `op_class`, `device`,
`timebase`, `start_tick` and `end_tick`. All spans must lie inside the capture
and share the same device and synchronized timebase. The script refuses mixed
clocks, mixed devices, duplicate IDs and invalid spans. It does not convert
ticks to seconds or certify telemetry, marker completeness or profiler overhead;
those require the measured controls accompanying the capture. The unit tests
use synthetic timelines only, and make no hardware performance claim.

To import the existing k10 profiler output without its frequency-derived
nanoseconds, join the operation report to `cpp_device_perf_report.csv`:

```sh
python3 perf/c10_orchestrator/import_profiler_cycles.py \
  --ops /path/to/ops_perf_results.csv \
  --device-report /path/to/cpp_device_perf_report.csv \
  --device 0 --timebase capture-boot-and-run-id \
  --start-cycle START --end-cycle END > cycle_partition.json
```

Replace `START` and `END` with independently established boundaries in the same
device timebase as the kernel markers. The importer uses `DEVICE KERNEL START
CYCLE` and `DEVICE KERNEL END CYCLE`, not firmware duration or converted
nanoseconds. Its schema follows `tools/tracy/process_ops_logs.py` from the
existing tt-metal-k10 source at `1452925b033c6608726b731a81500bd3e19f7894`.

Each execution is identified by device, global call count, trace ID and replay
session. Replayed programs therefore remain separate executions. Missing joins,
duplicate identities, mixed devices and spans crossing a boundary stop the
import. Warmup and other spans wholly outside the boundaries are counted and
excluded. Host-only label rows are counted separately. Input paths and hashes
are retained in the JSON output.

A successful import proves the arithmetic partition and identity join only.
It does not establish that the capture contains every dispatched program, that
core clocks are synchronized, or that the capture boundaries cover a complete
fold. Clock samples and a profiler-perturbation control are still required. The
importer has only synthetic CPU validation; no new device result is supplied.

The archived accuracy reference can also be checked without a device:

```sh
python3 perf/c10_orchestrator/audit_accuracy.py --out reference_spread.json
```

This replays the existing fp32 upstream seed-0–3 CIFs, verifies atom identity and
fixture size, and checks the float64 Kabsch results against the archived scorer.
It preserves each CIF hash. Across six seed pairs, worst-domain all-atom RMSD is
0.80128 Å mean (0.74099–0.84656 Å) at 298 residues and 1.66454 Å mean
(0.94540–2.29123 Å) at 512 residues. These reproduce the
[existing reference evidence](../k10_anchor/FINDINGS.md); they are not new folds
or a measurement of current TT accuracy.

Different-seed spread, same-seed A/A repeatability and baseline-to-stack movement
are separate quantities. Keep the campaign's 512-residue 0.60 Å bar and quoted
1.84 Å seed floor explicit; this replay neither changes that bar nor supplies a
new stack verdict. The 298-residue control has its own documented
[0.35 Å all-atom convention](../../docs/implementation-parity.md), so report the
metric as well as the threshold. CA and all-atom RMSD are not interchangeable.
A float64 coordinate scorer is also not a float64 model reference: mathematical
transforms still need independent float64 controls. Compare a new stack as one
stack, with its own paired baseline/A/A folds and both fixture sizes.

Opaque generic calls can be inspected with the [opt-in identity observer](../c10_generic_identity/README.md). Its CPU controls preserve dispatch behavior; live binding and device validation are still required before using its records in a census.

The [runtime DM calibration](../c10_dm_control/README.md) now has archived device
evidence and a CPU replay. The [independent review](dm_control_review.json)
verified all seven raw archives, the dense graph counts, six reducer tests and
identical replay output. Its seven control intervals contain 591 samples at
minimum=maximum 1350 MHz; all three preset DM separation checks pass. This
validates those controls, not a model roofline, fold census or speedup.

For sum-profiling captures, use that calibration's raw per-core/RISC reducer.
This build disables C++ postprocessing in that mode, so the C++ report required
by `import_profiler_cycles.py` is absent. The profiler also reused stale 32×32
shape metadata for 120 graph-confirmed 8192×8192 add calls. A census needs an
explicit join to actual operands before assigning per-call work.

The [burst-clock census](../c10_burst_census/README.md) retains 1,478 model
programs across four windows at sampled 1350 MHz. Its
[independent replay](burst_census_review.json) verifies all 137 artifacts,
12 census/smoke tests and identical reducer outputs. The repaired live observer
passes 16 float64 checks, eight exact off/on checks and runtime-address controls.

Native matmul and binary operations account for 53,004,546 and 25,851,252 raw
program-span cycles in those windows. These are unweighted sampled counts,
not whole-fold shares or optimization prizes. Automatic host reporting exceeded
its resource budget after the complete default fold. Whole-fold closure, matched
model roofs and instrumentation
perturbation remain unmeasured. The retained Pairformer window includes the
two-argument path, as its [actual operands show](census_scope_review.json).
The retained census therefore reports STOP;
no model speedup or campaign ceiling follows from it.

[Generic work contracts](../c10_generic_work/README.md) provide source-conditional
matrix components, permutation/gate counts and logical bytes for explicitly
identified inputs. Parent review reproduced its report and 16 reducer tests;
exact-total requests still refuse. The 15 normalized controls are synthetic,
and the two archived calls are refused. Actual source/flag/ownership joins are
required before these components can enter a measured model table.
