# Bare Boltz2 baseline

This benchmark measures warm prediction time and same-seed repeatability on the committed 298- and 512-residue fixtures. Results are being collected in `runs/baseline1`; no baseline is accepted until the CPU reducer returns GO.

Both A/A labels execute the same default Boltz2 configuration: 200 diffusion steps, three recycles, one sample, seed 0, the fixed 35-row MSA and no templates. One cold fold precedes four adjacent A/A pairs in each target's own process and device context. The timer includes the complete `predict_one` call and CIF writing, with device synchronization immediately before the start and before the end.

Production source is `5e1886b4fd1bea4211f30c295723c8175c72d63c`. Its change from census source `71a306a8a` enables fused SDPA above 1,024 padded tokens and adds counters. That route cannot fire for either target; every fold retains the actual above-cap counters. Historical timing and the instrumented census are different records and supply no speedup ratio here.

The run uses the installed `ttnn 0.68.0` wheel. Each target records the resolved module paths, loaded shared-library hashes, actual model configuration, source commit and dirty diff, fixture/MSA hashes and output hashes. No production code changes or rebuilds are involved. No Tracy, graph capture, generic observer, module timers, sum profiling, trace replay or hoisted model work is enabled.

The normal host benchlock retains its matcher, threshold and location, with 60-second waits. Its load-timeout warning is rejected by the launcher. A separate process samples node 0's clock with monotonic read brackets and records all-chip device holders throughout the run. A fold fails if a sampled clock differs from 1,350 MHz, clock coverage has a gap above 10 ms, a clock read fails, a foreign holder appears, holder coverage has a gap above one second, or boot/containment changes. These are sampled observations, not proof against activity entirely between samples. Telemetry overhead is included and has not been separately subtracted.

CIFs must contain finite coordinates, unique atom identities and the complete fixture residue sequence. The CPU scorer maps atoms by chain, residue, atom name and residue name before float64 Kabsch superposition. It reports whole-chain all-atom RMSD and the maximum independently aligned domain RMSD, splitting the 512-residue fixture after residue 298, as in the existing accuracy audit. The 512-residue bar is 0.60 Å beside the user seed floor of 1.84 Å. The 298-residue bar is 0.35 Å all-atom; archived upstream seed pairs span 0.74099–0.84656 Å, mean 0.80128 Å. Same-seed A/A measures repeatability, not accuracy against upstream or different-seed variation.

Reproduce on the assigned quiet qb2 node 0:

```sh
bash perf/c10_bare_baseline/run.sh unique_run_name
/home/ttuser/tt-bio-dev/env/bin/python3 perf/c10_bare_baseline/reduce.py \
  perf/c10_bare_baseline/runs/unique_run_name --out perf/c10_bare_baseline/analysis.json
```

`criterion.json` records the prediction and kill criteria before measurement. `imports.json` identifies the unmodified reused helpers. `test_controls.py` checks timer units, during-interval clock coverage, foreign-holder rejection, rigid transforms, reflection rejection, invalid coordinates and reordered atom mapping. Raw telemetry is retained as lossless gzip JSONL, alongside every CIF and pLDDT.

Elapsed seconds multiplied by 1,350 MHz are elapsed device-clock-equivalent Mcycles, including host gaps. They are not measured device work or a per-operation budget. This baseline establishes no CPU removable-cost estimate, roof, ceiling, optimization or upstream-accuracy claim.
