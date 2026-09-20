# of3t-traj20 — pass 1 notes, for the pass that writes the state doc

## What is running right now

Five full-scope arms, detached under `setsid nohup`, cwd `/home/ttuser/.coworker/wt/of3t-traj20`
(this slug's own worktree), logs `/home/ttuser/of3t_traj20/logs/<arm>.log`, results
`perf/of3t_traj20/traj20_<arm>.json`:

    shipped  warmup 1000, accum 1
    scaled   warmup 20,   accum 4
    wired    warmup 20,   accum 4, our side with weight_decay=0 and clip_and_accumulate called
    miswire  warmup 20,   accum 4, D11's off-by-one restored on our side (instrument-can-fail)
    stale    warmup 20,   accum 4, our step k fed step k-1's gradient (break control)

Measured pace with all five sharing 16 cores: 150-250 s per rung, so the last arm lands about
90 minutes after its launch: shipped/scaled/wired started 2026-09-20T15:56Z and
miswire/stale 2026-09-20T16:04Z. Each run prints one line per rung, so `tail -n 1` on the log
is the live position and a finished run ends with `wrote perf/of3t_traj20/traj20_<arm>.json`.

Collect all five with:

    /home/ttuser/ptxft-venv/bin/python3 perf/of3t_traj20/summarise.py perf/of3t_traj20/traj20_*.json

## What is already settled, in the repo

- `PREREGISTRATION.md`, pushed at `acd00795f` before any full-scope number existed.
- `schedule_family.json`: the fourth wiring gap, in closed form. 200,004 of 200,005 points
  exact against upstream's own scheduler; the single mismatch is at step 50000, ours 0.00171
  against their 0.0018, 5.0e-02 relative. `plateau_until=50000` makes it 200,005 of 200,005.
  Both families agree exactly on 0..1000, so the 20-step window is blind to it by construction.

## The loop-gain ladder, and what it already says about §7b's bar

`gainladder.sh` re-runs `scaled`, `stale` and `miswire` at rho in {0.005, 0.05, 0.2, 0.5} on a
40-tensor scope, to ask whether the growth law's power depends on how much of the drive is the
closed-loop feedback. Results at `/home/ttuser/of3t_traj20/gl_<arm>_<rho>.json`. The first two
rungs are in and they are the headline this row has to be honest about:

    arm        rho     exp    r2  share20     rel_d2    rel_d20    d1_ours
    miswire  0.005  -0.769 0.879   0.2510  2.166e+00  3.014e-01  1.553e-01
    miswire  0.050  -0.730 0.938   0.9269  2.070e+00  3.205e-01  1.553e-01
    scaled   0.005  +0.239 0.567   0.2510  1.048e-01  1.784e-01  0.000e+00
    scaled   0.050  +0.290 0.840   0.9269  1.048e-01  1.891e-01  0.000e+00
    stale    0.005  +0.251 0.421   0.2510  1.137e-01  2.001e-01  0.000e+00
    stale    0.050  +0.201 0.326   0.9269  1.137e-01  2.136e-01  0.000e+00

The deliberately mis-wired arm carries the largest divergence on the ladder (2.07 at k=2, 20x
the healthy arm) and its growth exponent is NEGATIVE. Raising the loop gain by 3.7x in feedback
share does not move any exponent. So at N=20 inside the warmup the divergence SATURATES rather
than compounds, and §7b's shape bar has not been shown capable of failing. What does
discriminate, on the same numbers, is `d_1` (0.155 against their exact 0) and the magnitude
against the fp32 differencing floor. That has to be written as the finding, not smoothed over:
§3e says a bar that cannot fail is not evidence, and the honest reading is that the shape is
the weakest of this instrument's three readings, not its bar.

## For the write-up

`_of3t_donecheck.py` wants, in `/home/moritz/.coworker/state/of3t-traj20.md`: DELTA, D1, SHAPE,
WIRING, CONTROL, PROVES, DOESNOT, a VERDICT line, a float64 mention, a `GRADIENTS:` section with
a worst case carrying a dotted parameter path, a `MODELS:` section with an `x of y` denominator,
and a DOESNOT of at least 100 characters naming the stability bound. 2000 B floor.
