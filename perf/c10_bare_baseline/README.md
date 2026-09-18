# Bare Boltz2 baseline

Warm prediction time and same-seed repeatability on the committed 298- and 512-residue fixtures,
measured on an unprofiled production wheel with the clock pinned at 1350 MHz.

## Result

Two accepted sessions on qb2 node 0 (p300c), 8 accepted warm folds per target per session, every
fold sampled at 1350 MHz throughout.

| target | pooled median | min | max | spread | stdev | between-session median delta |
|--------|---------------|-----|-----|--------|-------|------------------------------|
| 512 aa | 14.881 s | 14.813 s | 14.972 s | 0.159 s | 0.049 s | 0.010 s (0.07 %) |
| 298 aa | 9.680 s | 9.629 s | 9.890 s | 0.261 s | 0.072 s | 0.039 s (0.41 %) |

Per session: 512 reads 14.892 s (baseline2) and 14.881 s (baseline3); 298 reads 9.663 s and
9.703 s. The A/A structural floor is zero. All 18 warm folds per target, across both sessions,
wrote a byte-identical CIF, so the float64 Kabsch RMSD between any two of them is 4e-15 A, which is
arithmetic noise far under the 0.60 A bar at 512 (user seed floor 1.84 A) and the 0.35 A bar at 298
(archived upstream seed pairs span 0.741 to 0.847 A). Same-seed A/A measures repeatability, not
accuracy against upstream and not seed spread.

`baseline.json` holds these numbers, `runs/<name>/analysis.json` the per-fold rows.

## What is measured

Both A/A labels run the same default configuration: 200 diffusion steps, three recycles, one
sample, seed 0, the fixed 35-row MSA, no templates. One cold fold precedes four adjacent A/A pairs
in each target's own process and device context. The timer covers the complete `predict_one` call
including CIF writing, with device synchronization immediately before the start and before the end.

Production source is `5e1886b4fd1bea4211f30c295723c8175c72d63c`, unmodified (each run records an
empty production diff). Its change from census source `71a306a8a` enables fused SDPA above 1024
padded tokens and adds route counters. That route cannot fire at either size, and every fold
records both above-cap counters at zero. `test_controls.py` proves those counters can move, so zero
means the route was not reached rather than the counter being dead. Historical timings and the
instrumented census are different records and supply no ratio against these numbers.

The run uses the installed `ttnn 0.68.0` wheel. Each target records resolved module paths, loaded
shared-library hashes, the actual model configuration, source commit and dirty diff, fixture and
MSA hashes, node sysfs identity and output hashes. No production code changed, nothing was rebuilt,
and no Tracy, graph capture, generic observer, module timers, sum profiling, trace replay or
hoisted model work is enabled.

## Quiet and clock

The normal host benchlock runs with its matcher, threshold and location unchanged and 60-second
waits; its load-timeout warning is rejected by the launcher. During the run one process samples
node 0's clock with monotonic read brackets, a second thread records all-chip device holders, and
`ambient.py` records host CPU every 250 ms. A fold fails if a sampled clock differs from 1350 MHz,
clock coverage gaps exceed 10 ms, a clock read fails, a foreign device holder appears, holder
coverage gaps exceed one second, a process outside the run session burns 20 % of a core, or boot or
containment changes. These are sampled observations, not proof against activity entirely between
samples. Telemetry overhead is included in the times and has not been subtracted.

The host CPU witness was added after run `baseline1` had to be thrown away: a git pack from the
relay fetch overlapped its folds and the device-holder census could not see it. That run is kept
under `runs/baseline1` with its `excluded.json`, and its timings are not part of any accepted
number. In the accepted sessions the loudest foreign process was `stallwatch.py` at 11.7 % of a
core; the rest were system daemons under 8 %.

## Reproduce

```sh
bash perf/c10_bare_baseline/run.sh unique_run_name
/home/ttuser/tt-bio-dev/env/bin/python3 perf/c10_bare_baseline/reduce.py \
  perf/c10_bare_baseline/runs/unique_run_name --out perf/c10_bare_baseline/runs/unique_run_name/analysis.json
```

`criterion.json` records the prediction and kill criteria before measurement, `imports.json`
identifies the unmodified reused helpers, and `test_controls.py` checks timer units, during-interval
clock coverage, foreign-holder rejection, host CPU attribution against a known one-core burner,
route counters, rigid transforms, reflection rejection, invalid coordinates and reordered atom
mapping. Raw telemetry is retained as lossless gzip JSONL alongside every CIF and pLDDT.

CIFs must carry finite coordinates, unique atom identities and the complete fixture residue
sequence. The scorer maps atoms by chain, residue, atom name and residue name before float64 Kabsch
superposition, and reports whole-chain all-atom RMSD plus the maximum independently aligned domain,
splitting the 512-residue fixture after residue 298, as the existing accuracy audit does.

Elapsed seconds multiplied by 1350 MHz are elapsed device-clock-equivalent Mcycles, 20090 at 512
and 13068 at 298. They include host gaps and are not measured device work or a per-op budget. This
baseline establishes no CPU removable cost, roof, ceiling, optimization or upstream-accuracy claim.
