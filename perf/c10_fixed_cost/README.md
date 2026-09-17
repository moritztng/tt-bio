# The clock-immune fixed cost of a Boltz-2 fold, measured

`c10-bare-baseline` pinned one clock and got one number per size: 14.8813 s at 512 aa and
9.6801 s at 298 aa, both at a during-sampled 1350 MHz. Two sizes constrain the clock-immune term F
but cannot pin it, and the campaign's own arithmetic makes that gap decisive: at a 512/298 work
ratio of 2.5 or more, cutting F to 1 s alone reaches 10.0 s without deleting a single device cycle.
Before this row nobody knew whether the 10 s target was a host problem or a kernel problem.

F is by definition the part of the fold that does not scale with the clock, so two pinned clocks at
one size give two equations in two unknowns and F falls out. This row does that on today's tree,
with today's timer boundary, at four clocks.

## What it does

Four arms, 1350, 1200, 1000 and 800 MHz, crossed with the committed `cdk2x2_512` and `cdk2x2_298`
fixtures. Identical configuration everywhere: 200 sampling steps, 3 recycles, 1 diffusion sample,
seed 0, the committed 35-row A3M, templates off, production ttnn wheel, no profiler, no trace, no
hoist, no `--fast`. Nothing is optimised and no device cycle is deleted, so the predicted
model-cycle saving is exactly zero.

The arms are **interleaved inside one process and one device context**, alternating
1350/1200/1000/800 on even repetitions and the reverse on odd ones, after one discarded cold fold.
A cross-run comparison would be worthless here: this whole campaign exists because a clock episode
was mistaken for a code path. The harness re-reads `state.model.predict_args`, the job seed and the
MSA row count between every label and refuses the run if any of them moved.

FORCE_AICLK (ARC message 0x33, through tt-kmd's SMC queue) is re-sent and re-verified before
**every** label, not once per process, and released in the `finally` block with the release return
checked. The requested clock has to appear in sysfs and hold for 200 ms before the timer starts,
and then every during-fold sample has to equal that fold's own target: a requested clock the chip
does not actually hold is rejected, never reported. Board power is sampled alongside the clock, so
the arm is witnessed by physics and not only by a telemetry register.

### The 800 MHz arm is an arm, not a clamp

`scripts/aiclk_watch.sh` logs `AICLK-WATCH: qb2 CLAMPED-UNDER-LOAD` for a busy chip under
1200 MHz, so this row trips the fleet's own guard on purpose. A pinned 800 MHz fold is a
measurement; an unrequested one is an artifact. Which one this was is in the raw samples: per fold,
min = max = the requested target, thousands of during-samples, no read error, and board power that
tracks the clock.

## Timer boundary

`ttnn.synchronize_device`, `monotonic_ns`, the complete `state.predict_one` including featurization
and CIF writing, `synchronize_device`, `monotonic_ns`. That is `c10-bare-baseline`'s boundary
unchanged, because the comparison to its 14.8813 s depends on it. `control.py` and `force_aiclk.py`
are byte-identical to its pinned copies; `ambient.py`, `audit_accuracy.py` and
`audit_prerequisites.py` are reused as they are.

The baseline's number was taken on `5e1886b4f` and this tree is two commits later. Both commits only
rewrote comment and docstring prose, and `astsame.py` proves it rather than asserting it: the
changed files are parsed, docstrings are stripped and the ASTs are compared, so the two trees are
shown to be executably identical while a one-token change breaks the check.

## Reproduce

```
bash perf/c10_fixed_cost/run.sh <name> 512 298
python3 perf/c10_fixed_cost/reduce.py perf/c10_fixed_cost/runs/<name> \
        --out perf/c10_fixed_cost/runs/<name>/analysis.json
python3 perf/c10_fixed_cost/report.py perf/c10_fixed_cost/runs/<name>/analysis.json
```

`run.sh` takes the shared benchlock with the existing 60 s waits and rejects a lock that timed out
rather than measuring through it. `reduce.py` exits non-zero unless every criterion holds, and
`report.py` prints the table below straight out of `analysis.json`, so no number here is retyped.

Controls:

```
python3 perf/c10_fixed_cost/test_controls.py                       # 50 CPU known-answer cases
python3 perf/c10_fixed_cost/negative_control.py runs/<name> 512    # break the real capture on purpose
```

`test_controls.py` covers only what this row adds to the baseline: clock coverage scored per arm in
both directions, a mid-fold clock move, gaps, read errors and thin sampling, the fit against an
exact synthetic answer, its refusal to print an uncertainty from two clocks alone, the analytic
`dF/dT` gains, the 10.0 s demand arithmetic by hand and the comment-only AST check with negative
controls. The baseline's own 13 controls are closed and are not redone. `negative_control.py` then
mutates one field of a copy of the finished capture and requires the targeted fold to be rejected
for the named reason while every other fold stays accepted.

## Measured

One benchlocked session, run `sweep1`, qb2 card 0. Cells are accepted warm folds only.

Reduced verdict: **GO**

### 512 aa

| clock | folds | median | min | max | stdev | adjacent \|delta\| | mean board W | host CPU s | elapsed Mcycles |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1350 MHz | 7 | 14.9306 s | 14.8828 s | 15.0958 s | 0.0781 s | 0.0781 s | 77.4 | 16.816 | 20156.3 |
| 1200 MHz | 7 | 16.1081 s | 16.0364 s | 16.1636 s | 0.0429 s | 0.0415 s | 65.7 | 17.981 | 19329.7 |
| 1000 MHz | 7 | 18.5518 s | 18.4907 s | 18.7602 s | 0.0952 s | 0.0843 s | 52.1 | 20.447 | 18551.8 |
| 800 MHz | 7 | 22.3574 s | 22.1914 s | 22.5835 s | 0.1230 s | 0.1257 s | 43.3 | 24.231 | 17885.9 |

- fit on all accepted folds (n=28, clocks [800, 1000, 1200, 1350]): **F = 3.9830 s +-0.1181**, C = 14665.0 Mcycles +-121.1, max |residual| 0.2692 s, rms 0.1195 s
- fit on cell medians (n=4, clocks [800, 1000, 1200, 1350]): **F = 3.9543 s +-0.2895**, C = 14678.2 Mcycles +-296.8, max |residual| 0.1035 s, rms 0.0812 s
- closed form on the 1350/800 MHz endpoints alone: **F = 4.1280 s +-0.0991**, C = 14583.5 Mcycles +-108.1; the endpoint arms enter F with gains 2.455 and -1.455
- inverse-clock model: A/A timing floor 0.1257 s, cell-median max |residual| 0.1035 s, per-fold rms 0.1195 s -> **the exact two-parameter form holds, F is point-identified**
- three estimators of F: all_folds 3.9830 s +-0.1181, cell_medians 3.9543 s +-0.2895, two_clock_endpoints 4.1280 s +-0.0991 -> F bounded to 3.6648 - 4.2439 s
- structure: 1 distinct CIF over 28 accepted folds, max pairwise domain RMSD 0.00e+00 A against the 0.6 A bar (byte-identical in every arm)
- demand for 10.0 s at 1350 MHz (now 14.8460 s):
  - F untouched: 8122.9 Mcycles allowed, a cut of 6542.1 Mcycles, **44.6 %** of the work term
  - F cut to 1.0 s: 12150.0 Mcycles allowed, a cut of 2515.0 Mcycles, **17.1 %**
  - cutting F to 1.0 s and touching no cycle at all: 11.8630 s

### 298 aa

| clock | folds | median | min | max | stdev | adjacent \|delta\| | mean board W | host CPU s | elapsed Mcycles |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1350 MHz | 7 | 9.6687 s | 9.6324 s | 9.8305 s | 0.0662 s | 0.0478 s | 70.2 | 11.018 | 13052.8 |
| 1200 MHz | 7 | 10.5973 s | 10.5833 s | 10.6229 s | 0.0145 s | 0.0198 s | 58.6 | 11.944 | 12716.7 |
| 1000 MHz | 7 | 12.3147 s | 12.2884 s | 12.3452 s | 0.0197 s | 0.0171 s | 47.4 | 13.658 | 12314.7 |
| 800 MHz | 7 | 14.9746 s | 14.9491 s | 15.0110 s | 0.0209 s | 0.0195 s | 38.0 | 16.327 | 11979.7 |

- fit on all accepted folds (n=28, clocks [800, 1000, 1200, 1350]): **F = 1.9500 s +-0.0438**, C = 10403.4 Mcycles +-44.9, max |residual| 0.1742 s, rms 0.0443 s
- fit on cell medians (n=4, clocks [800, 1000, 1200, 1350]): **F = 1.9154 s +-0.0844**, C = 10432.2 Mcycles +-86.5, max |residual| 0.0330 s, rms 0.0237 s
- closed form on the 1350/800 MHz endpoints alone: **F = 1.9511 s +-0.0625**, C = 10418.8 Mcycles +-51.5; the endpoint arms enter F with gains 2.455 and -1.455
- inverse-clock model: A/A timing floor 0.0478 s, cell-median max |residual| 0.0330 s, per-fold rms 0.0443 s -> **the exact two-parameter form holds, F is point-identified**
- three estimators of F: all_folds 1.9500 s +-0.0438, cell_medians 1.9154 s +-0.0844, two_clock_endpoints 1.9511 s +-0.0625 -> F bounded to 1.8310 - 2.0136 s
- structure: 1 distinct CIF over 28 accepted folds, max pairwise domain RMSD 0.00e+00 A against the 0.35 A bar (byte-identical in every arm)
- demand for 10.0 s at 1350 MHz (now 9.6563 s):
  - this size already folds in 9.6563 s, under the 10.0 s target, so the figures below are headroom rather than a demand
  - F untouched: 10867.5 Mcycles allowed, a cut of -464.0 Mcycles, **-4.5 %** of the work term
  - F cut to 1.0 s: 12150.0 Mcycles allowed, a cut of -1746.6 Mcycles, **-16.8 %**
  - cutting F to 1.0 s and touching no cycle at all: 8.7063 s

### across the two sizes

- F(512 aa) = 3.9830 s, F(298 aa) = 1.9500 s, difference 2.0330 s +-0.1260: **F is size DEPENDENT within 2 standard errors**
- work term 14665.0 Mcycles at 512 aa against 10403.4 at 298 aa: a work ratio of **1.410**
