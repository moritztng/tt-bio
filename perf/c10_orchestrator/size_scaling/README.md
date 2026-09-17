# The fold's device work grows slower than the target does

> **STATUS 2026-09-17, read this first: the finding below is very likely about to be withdrawn.**
> `c10-size-scaling` is measuring the discriminating leg right now and its in-flight data points at
> the artifact. From its own `runs/ladder1`, with the 512 aa anchor passed (14.9016 s at 1350 MHz,
> 22.3428 s at 800, W = 14,611.7 Mcycles against the 14,665.0 of record), the **512 → 768 aa leg
> gives p_work = 1.83** against 0.63 on the 298 → 512 leg. The pre-registered prediction was 0.6–1.0
> under this finding and 1.7–2.0 under the artifact. It landed at 1.83.
>
> That is **not yet the row's verdict**: 768 aa is incomplete (3 folds at 1350 MHz, 2 at 800), the
> 640 aa leg has not run, and a two-clock solve cannot check the functional form the way
> `c10-fixed-cost`'s four clocks did. The row owns the conclusion and will state it. But nothing
> downstream should be built on the numbers below in the meantime, and the reading to expect is
> that the size-independent term is 298 aa under-filling the grid, exactly as
> [`../floor_vs_measured/`](../floor_vs_measured/) predicted it would be.


`c10-fixed-cost` measured both terms of `T = F + W/f` at two sizes of the same `cdk2x2` fixture
family, same 35-row A3M, same 200 steps and 3 recycles, four pinned and during-sampled clocks:

| size | clock-immune `F` | clock-scaled work `W` |
|---|---|---|
| 512 aa | 3.9830 ± 0.1181 s | 14665.0 ± 121.1 Mcycles |
| 298 aa | 1.9500 ± 0.0438 s | 10403.4 ± 44.9 Mcycles |

Two terms at two sizes is a scaling measurement. Nobody had read it as one. Reading it gives two
results that point in opposite directions from what the campaign assumed.

## The work term scales at N^0.63, not N^1, N^2 or N^3

Work ratio 1.4096 ± 0.0131 against a token ratio of 1.718 (1.600 once 298 pads to 320). So
`p_work = 0.634 ± 0.017`, or 0.731 ± 0.020 on the padded axis. **The device's clock-scaled work
grows more slowly than the target itself.**

The fold provably contains work that scales at least linearly: per-token linears, the atom stream,
and pair tensors of shape `N×N×C` that the shape census names directly. Under a non-negative
mixture `W(N) = A + Σ B_k N^{p_k}`, evaluating at the *smallest* exponent present is a rigorous
lower bound on the size-independent term `A` (proved as a control, not asserted):

| assumed leading exponent | `A` | share of `W(512)` | seconds at 1350 MHz |
|---|---|---|---|
| N^1 — rigorous floor | 3,301 Mcycles | 22.5 % | 2.45 s |
| N^2 — pair-tensor reading | 8,220 Mcycles | 56.1 % | 6.09 s |
| N^3 — triangle reading | 9,357 Mcycles | 63.8 % | 6.93 s |

The campaign must delete **6,542 Mcycles** to reach 10.0 s with `F` untouched. The rigorous floor
alone is half of that; the pair-tensor reading is larger than the whole target.

## The clock-immune term is size-dependent, so it is not host dispatch

`F` ratio 2.0426 ± 0.0760, giving `p_fixed = 1.32 ± 0.07` on tokens, **19 σ away from zero and 4.7 σ
above 1**. Host dispatch is size-independent — the call count is the same at both sizes, only the
tensor dimensions change — so `F` cannot be dominated by it. That contradicts this campaign's own
PASS-18 reading, which had `F` as roughly 3.02 s of collapsible dispatch plus 0.61 s of host stages.
A term growing at N^1.3 looks like DRAM traffic and featurization, not like Python call overhead.

That prediction was written before `c10-trace-lever` reported, and the row then settled it the
hard way. Trace replay removes the diffusion loop's host dispatch entirely; it was predicted to
return +3.04 s and **measured -0.0214 s at 512 aa, inside its own 0.055 s A/A floor**, with a
byte-identical CIF in both arms. It removes nothing.

So two independent lines now say the same thing: `F` scales at N^1.3, and deleting the host
dispatch outright does not move the fold. **`F` is not collapsible host overhead.** A term that
grows at N^1.3 and survives the removal of its dispatch looks like DRAM-bandwidth-bound device
time, which on Blackhole runs in a clock domain AICLK does not drive and therefore lands in `F` by
construction. That has a consequence the campaign had written off: the byte axis is not a
kernel-time lever competing with arithmetic, it is **the only known lever against the 27 % of the
fold that the clock cannot touch.**

## What would make this wrong, and it is not a small worry

Grid under-fill at 298 aa would manufacture the whole thing. 298 tokens pad to 320 and spread over
110 cores; once a tensor is small enough that every core holds its minimum tile, the op costs the
same as a larger one. That inflates `W(298)`, shrinks the measured ratio, and inflates `A` — the
confound and the finding are indistinguishable from two sizes, and **both sizes here are at or
below 512.**

[`../floor_vs_measured/`](../floor_vs_measured/) has since put a number on that worry, and it is
not reassuring. If `W` is set by per-shape arithmetic roofs — and the floor artifact's
arithmetic half lands within 0.5 % of the measured `W/f` — then `W` should scale the way the FLOPs
do, and it does not. The two reconcile if 298 aa achieves only **48 % of 512 aa's rate** under an
N^2 FLOP model, which is about what 100 tiles on a 110-core grid would do. So there is now direct
counter-evidence, and the honest prior is that this finding may not survive.

The discriminator is a size *above* 512, where under-fill cannot apply. `c10-size-scaling` runs
384, 512, 640 and 768 aa from the committed `perf/size512/fixtures` ladder at two pinned clocks
each. If `W` still grows sublinearly from 512 to 768, the size-independent term is real at 512 and
is the largest single item this campaign has found. If it grows near N^2 there, the sublinearity
was a small-size artifact and there is nothing here to delete.

## What it would mean if it holds

Every lever in the corpus that landed at 1.02–1.05x made a kernel faster at a fixed call count, so
it attacked the size-*scaling* part — 6,400 to 11,400 Mcycles. The size-independent part, 3,300 to
8,200 Mcycles, has never been attacked, because until `F` was measured there was no way to see it.
A lever against it has to remove per-call cost, which is the same place the shape ranking (60
shapes, none over 3.6 %) and the dispatch arithmetic independently pointed.

At the pair-tensor reading, `A` spread over the fold's 465,664 top-level `ttnn` calls is 17,653
cycles or 13.1 µs per call. The device's own measured launch floor is about 1 µs per call, so **`A`
is an order of magnitude larger than kernel launch** and is not explained by it. Naming what it
actually is remains open and is `c10-fold-census`'s job.

## Status

Bounded, not measured. Nothing here is a lever, no cycle has been deleted, no accuracy spent, no
device was used to produce it.

    python3 size_scaling.py                      # writes size_scaling.json
    python3 -m pytest test_size_scaling.py -q    # 19 controls
