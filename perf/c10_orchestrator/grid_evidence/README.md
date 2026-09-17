# The grid-sizing lever's evidence is Wormhole, and Blackhole has no core sweep at all

The lever ledger ranks per-class grid sizing second, worth roughly 810 Mcycles, on the strength of
one measurement: the triangle product at **0.5932 ms on 32 cores against 1.2480 ms on 72**, a
2.10x win. That measurement is `perf/roof_tri_arith/tri_close_whglx_wh.json` — host `j10glx02`,
**Arch.WORMHOLE_B0, grid [8, 9] = 72 cores**, loadavg 9.85, no recorded clock.

Two things make it a weaker foundation than the ranking suggests.

**On the same sweep the other class moves the opposite way.** Cutting the SDPA to 32 cores makes
it **1.36x slower** (2.5851 ms against 1.9023 ms), while the product gets 2.10x faster. So the
lever is not "use fewer cores" — it is per class, and the sign is not predictable from one class.

**No Blackhole core-count sweep exists for either class.** Every Blackhole arm in
`perf/roof_triatt_rate/` ran the full [11, 10] = 110-core grid. Blackhole has 110 cores here and
130 on pc against Wormhole's 72, with different DRAM behaviour, so the 810 Mcycles is a
cross-architecture transfer. This project has been burned by that exact move before: a Blackhole
DRAM roof inverted an L1-residency lever's sign, and a Blackhole-fitted envelope capped Wormhole
Galaxy.

## What Blackhole does say

From the same session, same card, 512 aa:

| arm | rate |
|---|---|
| dense cube matmul | 67.59 TFLOP/s |
| triangle attention | 29.86 TFLOP/s, 2.26x off the cube |
| triangle multiplication | 11.47 TFLOP/s, **5.89x off the cube** |

That gap is measured on the right architecture, in one session, on one card, and it is the honest
motivation for looking at these two classes. It is not a prize: a trimul is not a dense cube, its
shapes and its bandwidth needs differ, and not all of the gap is recoverable. Both sessions ran at
loadavg 4.6 to 6.2 with no recorded clock, like everything else in this corpus.

> **2026-09-17: that session was throttled, and the absolute rates above are artifacts.**
> [`../frontier/`](../frontier/) found four independent Blackhole dense-cube measurements clustering
> at **104.93–114.20 TFLOP/s**. Against the cluster maximum, 67.59 implies a chip at **799 MHz** —
> the clock floor. **The ratios survive**, because both sides of each one were throttled together,
> so triangle multiplication really is 5.89x off its own session's cube. The absolute TFLOP/s do
> not, and at burst clock they would be roughly 1.69x higher. Do not carry 67.59, 29.86 or 11.47
> out of this table as Blackhole rates.

## So the row should be scoped differently

Not "recover a third of the 2.742 s the two classes hold above their roofs" — that number inherits
the Wormhole transfer. The honest scope is **measure a Blackhole core-count sweep for trimul and
triatt at a pinned, recorded clock, because none exists**, and let the sign fall out per class. It
is a microbenchmark, not a fold, so it is cheap. Wormhole's optimum for the product was 32 of 72
cores; Blackhole has 110, so there is real reason to look and no reason to assume.

    python3 grid_evidence.py                     # prints grid_evidence.json
    python3 -m pytest test_grid_evidence.py -q   # 7 controls
