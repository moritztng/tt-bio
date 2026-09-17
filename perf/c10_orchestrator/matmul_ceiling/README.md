# Bandwidth does not explain the fold's matmul rate

>  **REFUTED 2026-09-17 by measurement. It is bandwidth after all.**
>
> `c10-fold-census` measured the fold's real shapes on qb2: **68 of 70 priced keys have the BYTE
> roof binding**, and in seconds that is **10.2666 s of 10.5368 s — 97.4 %**. The matmul class runs
> at **21.38 TFLOP/s**, not the 78.24 ceiling computed below.
>
> **Why this was wrong, specifically.** The ceiling here divides FLOPs by *compulsory* traffic,
> `(M·K + K·N + M·N)` — the bytes a perfectly-reusing implementation would move. The fold's shapes
> do not achieve that reuse: they are thin and small-K, so the bytes actually moved are far higher
> and the real arithmetic intensity is far lower. "Not excluded by bandwidth" was true of the
> *compulsory* byte count and false of the real one.
>
> The limits below already said 78.24 is "not excluded by bandwidth, NOT achievable". That hedge was
> correct and it was not enough: a compulsory-traffic roofline is an upper bound loose enough to
> point a campaign at the wrong axis, which is what it did. The sensitivity analysis showing the
> result robust across every roof pair is also beside the point — the error was in the byte count,
> not the roof.

[`../frontier/`](../frontier/) established that the matmul class's achieved rate is the **only** axis
with enough headroom to reach 10.0 s — and that the number saying so is modelled from per-shape
rates measured on pc's 130-core firmware, not on qb2. Before spending a chip, one thing can be
settled from the shapes alone: **is that gap even allowed by the hardware?**

For each recorded matmul, arithmetic intensity is FLOPs over bytes, and the ceiling is
`min(dense cube, AI × DRAM bandwidth)` using the campaign's own measured roofs (cube 104.93 TFLOP/s,
stream 424.7 GB/s, machine balance 247.1 FLOP/byte).

| | |
|---|---|
| recorded matmul shapes | 94,616 calls, 83.68 TFLOP |
| structural ceiling, weights re-read each call | **78.24 TFLOP/s** |
| structural ceiling, weights resident | 80.11 TFLOP/s |
| modelled rate today | 19.88 TFLOP/s |
| what 10.0 s needs on this axis | 35.49 TFLOP/s — **45 % of the ceiling** |

**A 3.9× gap that roofline does not account for.** Had the ceiling landed near today's rate, the
headroom would have been illusory and 10.0 s settled as impossible without a fold. It didn't. So
whatever limits these matmuls is **implementation** — grid occupancy, tile quantisation, K-loop
efficiency, per-program cost — not DRAM.

Weight residency turns out not to decide it: cold and warm differ by under 3 %.

## Two shape families worth naming

The fold's matmuls sit close to the machine balance rather than far above it:

- the **16-head pair matmuls**, 8,448 calls each, AI ≈ 101 — genuinely DRAM-bound, capped near
  43 TFLOP/s
- the **768-family linears**, AI 219–307 against a balance of 247 — straddling the knee

## What this sharpens

`c10-fold-census` was asked to measure the achieved rate. The more useful question is **why it sits
~4× below what arithmetic intensity permits**, because that gap is where the campaign's only viable
axis lives.

## Does the conclusion depend on which roofs are right? No.

The ceiling above uses the floor artifact's roofs (cube 104.93, DRAM 424.7). Those carry an
unrecorded clock, so the honest check is whether the finding survives **every** Blackhole roof pair
the campaign has on record:

| cube | DRAM | ceiling | × today | target as % of ceiling |
|---|---|---|---|---|
| 104.93 | 424.7 GB/s | 78.24 | 3.94× | 45.4 % |
| 112.71 | 424.7 GB/s | 79.99 | 4.02× | 44.4 % |
| 122.28 | 442.9 GB/s | 83.94 | 4.22× | 42.3 % |
| 122.30 | 443.1 GB/s | 83.98 | 4.22× | 42.3 % |

**The ceiling moves 7.3 % across the whole range**, the gap to today stays 3.9–4.2×, and the target
stays 42–45 % of the ceiling. The conclusion does not depend on the roof choice, and the
better-measured roofs make it slightly *stronger*.

The bottom two pairs are `c10-fold-census`'s in-session roofs, read from its `sweep1` and `sweep2`
`replay.json` — **that row has not published them**, and they are used here only to bound a
sensitivity, never as a result. Its two sweeps agree to 0.01 %, which is why they are worth bounding
against. Controls enforce both the label and the fact that the finding holds at every pair.

## Limits

Roofline is an upper bound that ignores exactly what probably binds here. **78 TFLOP/s is "not
excluded by bandwidth", not "achievable"** — a control fails if this note ever says otherwise.

Only the 18 recorded matmul shapes, 86 % of the class's calls but **29 % of its modelled floor**;
the remainder is not extrapolated. Both roofs carry an unrecorded clock, so the absolute ceiling
moves with them and the ratio is the more robust reading. Bytes assume bf16 with no accumulator
traffic.

    python3 matmul_ceiling.py                      # writes matmul_ceiling.json
    python3 -m pytest test_matmul_ceiling.py -q    # 11 controls
