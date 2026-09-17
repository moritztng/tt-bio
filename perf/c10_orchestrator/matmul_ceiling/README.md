# Bandwidth does not explain the fold's matmul rate

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

## Limits

Roofline is an upper bound that ignores exactly what probably binds here. **78 TFLOP/s is "not
excluded by bandwidth", not "achievable"** — a control fails if this note ever says otherwise.

Only the 18 recorded matmul shapes, 86 % of the class's calls but **29 % of its modelled floor**;
the remainder is not extrapolated. Both roofs carry an unrecorded clock, so the absolute ceiling
moves with them and the ratio is the more robust reading. Bytes assume bf16 with no accumulator
traffic.

    python3 matmul_ceiling.py                      # writes matmul_ceiling.json
    python3 -m pytest test_matmul_ceiling.py -q    # 11 controls
