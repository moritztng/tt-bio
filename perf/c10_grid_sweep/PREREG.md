# Pre-registration, written before the scored session

Committed before `grid_sweep.py` produced a scored row. The mechanism question, the kill criteria
and the predicted landing are all fixed here so nothing can be adjusted after the fact.

## The question

`percall_residual/` found the fold's matmul gap flat across a 1152x range in FLOPs per call, median
22.4 us, 13x the device's 1.70 us program-launch floor. A flat per-call cost has two candidate
producers and a core-count sweep separates them:

- **occupancy**: per-call time falls as cores rise and then saturates, with the saturation point
  moving with tile count;
- **per-program cost**: per-call time is roughly flat in core count and the sweep finds nothing.

A third producer exists and this harness tests it too, because it would look identical in a device
timing and is cheaper than either: **host dispatch**. A Python `ttnn.linear` call costs tens of
microseconds on the host, and `roof-launch-floor-per-device-program-not-python-call` is exactly
this confusion recorded once already.

## Predicted, per arm

1. **`cube` (control, dense 4096^3).** Per-call time falls close to 1/cores from 16 to 110 cores:
   at least **4x** between 16 and 110, and the 110-core arm reaches **80-110 TFLOP/s**.
   **Kill criterion: if the cube curve is flat, the instrument cannot see occupancy and no flat
   curve anywhere else in the session may be reported as evidence of anything.**
2. **768-family linears** (AI 219-307, straddling the 247 machine balance). Predicted to fall with
   cores but far less than 1/cores, and to be **within 15 % between 72 and 110 cores** — already
   saturated at the shipped grid. Predicted 110-core per-call time **28-56 us**, matching the
   modelled numbers the residual was computed against.
3. **16-head pair matmuls** (AI ~101, DRAM-bound, capped near 43 TFLOP/s). Predicted to saturate
   *earlier* than the 768 family, by roughly 48-64 cores, because bandwidth binds before cores run
   out. Predicted **within 10 % between 64 and 110**.
4. **`trimul` and `triatt`** (the shipped units). Predicted to **disagree with each other**, as the
   Wormhole 8x9 sweep's two classes did. No sign predicted for either; that is the measurement.
   Predicted 110-core unit times near the fold's own 6.348 ms (trimul) and 4.676 ms (triatt).
5. **Host issue.** Predicted `issue_us` of **15-45 us** for every matmul arm, independent of grid.
   If `issue_us` is at or above the measured per-call wall time for an arm, that arm is host-bound
   and its per-call cost is not a device cost at all.

## Predicted verdict

**The residual is not occupancy.** Predicted: the 110-to-72 core difference on the 768 family is
under 15 %, so at most ~5 us of the 22.4 us residual can be attributed to running out of cores at
the shipped grid, and the sweep therefore eliminates occupancy and points the campaign at per-call
setup — most likely host-side. Expected verdict **NO-GO** on per-class grid sizing as a lever.

## What would falsify it

A 768-family arm that is 1.5x or more slower at 110 cores than at some smaller grid, reproduced
across reps with the A/A floor beneath it, is a real lever and would flip the verdict to GO for
that class — subject to converting the per-call win through the call census into fold seconds and
Mcycles before anything is claimed.

## Protocol fixed in advance

Grid ladder 11x10, 9x8, 8x8, 8x6, 8x4, 6x4, 4x4, with 11x10 measured a second time late in the
ladder as the A/A floor. Arms interleave per rep, the ladder is the inner loop, two warmup reps are
discarded, and the statistic is the minimum over five scored reps. AICLK forced to 1350 MHz for the
session and sampled during every timed region; a region whose samples are not all 1350, or which
has fewer than 8 samples or a gap over 25 ms, is dropped and reported as dropped. Benchlock held
for the whole session. No production code changes: `TT_BIO_FORCE_GRID` is a shipped default-off
knob and the sweep restores the measured grid before exit.
