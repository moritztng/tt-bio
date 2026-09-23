# Speed bar above 1024 tokens

A structure model that folds 1536 tokens but takes an hour to do it has not gained a capability.
This is the bar a rung above 1024 must clear, written down before the size ladder measured any
such rung on the Wormhole Galaxy, so the result cannot choose it. An earlier capacity ladder on
the same Galaxy (2026-09-11, deep alignments, wall-clock including model load) did reach 1536 for
some models; it is a different configuration and was not read when the bar was set. `scripts/speed_bar.py` implements it and
`tests/test_speed_bar.py` checks that it fails what it says it fails.

## The rule

Fit log(runtime) against log(N) over the rungs the model already folds between 512 and 1024. A
new rung N may sit above the fitted line by at most

    (N / 1024) ^ max(0, order - k_fit)  x  max(1.25, 1 + 3 sigma)

- `order` is the highest-order work the model does: 3 for anything with a pair track (triangle
  multiplication and triangle attention are cubic), 2 for a sequence-only transformer.
- `k_fit` is the model's own fitted exponent. A model that scales as N^1.8 below 1024 is allowed
  to bend all the way up to N^3 above it, because that is where its own algorithm goes as the
  pair track takes over. It is not allowed to go past N^3.
- `sigma` is the model's measured single-rung noise, the same number the size ladder records.

Runtime is the fold's own `runtime_s` (no model load or process start), at the size ladder's fold
settings, all rungs of one comparison on one chip, one host and one commit. If the AICLK sampled
during the folds moves more than 3% across the rungs, or the host's 1-min load average passes
1.5x its core count during any rung (the release gate's own start ceiling), the comparison is
void and is reported as void, never as a pass or a fail. A capacity walk on a busy Galaxy still
says whether a size folds; it does not say how fast.

What it catches is the real hazard of chunking: a row-blocked path that swaps a memory wall for a
time wall by round-tripping to the host, recompiling per chunk, or spilling to host memory. Those
show as a step on top of the curve, not a smooth bend.

## What it does not cover

- **Absolute speed.** A model that is uniformly slow at every size passes. That is the job of the
  perf baselines and the GPU comparison.
- **Doing less work.** Fewer recycles or diffusion steps would pass this bar. They are forbidden
  by a separate rule.
- **Accuracy.** A fast wrong answer passes. Accuracy at the new rungs is judged separately against
  the model's own reference.
- **Anything outside `runtime_s`**: MSA search, model load, queueing, output writing.
- **Models with fewer than three rungs in 512..1024.** They are reported as ungated.
- **The fixed-cost term.** `runtime_s` carries a few seconds that do not grow with N. It pulls
  `k_fit` down, which widens the allowance. The bar errs lenient here, not strict.
- **Refusals and crashes.** A rung the model refuses or crashes on is a coverage result, not a
  speed result.

## Forecast from the recorded Galaxy ladder

Computed from `docs/size_ladder_baseline.d/*.json` (tt-galaxy-wh-l, recorded on the other
Wormhole Galaxy, 2026-09-07 to 09-20). The bar anchors on the fit measured in the same session
as the new rung, so these ceilings are a forecast of the order of magnitude, not the threshold
itself. Seconds, fold only.

| model | k_fit | 1280 fit / ceiling | 1536 fit / ceiling |
|---|---|---|---|
| boltz2 | 1.77 | 261 / 468 | 361 / 809 |
| esmfold2 | 1.86 | 493 / 796 | 692 / 1375 |
| esmfold2-fast | 1.84 | 283 / 492 | 396 / 850 |
| protenix-v1 | 1.75 | 154 / 255 | 212 / 441 |
| protenix-v2 | 1.55 | 591 / 1021 | 784 / 1764 |
| openfold3 | 2.77 | 776 / 1022 | 1286 / 1767 |
| openbind | 2.78 | 796 / 1044 | 1321 / 1804 |
| rf3 | 1.42 | 384 / 682 | 497 / 1178 |
| nesso1 | 1.50 | 61 / 106 | 80 / 182 |

opendde has no Wormhole Galaxy record. protenix-v2 and rf3 have no recorded sigma, so they get
the 1.25 floor. rf3's recorded 1088 rung already sits at 1.39x its fit against an allowance of
1.38x; with its real sigma measured it may pass, so it is flagged, not filed.
