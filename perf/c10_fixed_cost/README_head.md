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
