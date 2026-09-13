# BinaryNg: three rows are at the roof, two are not bandwidth rows at all

Measured on whglx card 28 (Wormhole B0, 8x9 grid), host thread cap 8, no model code changed.
Rig: `site_roof.py` (pass 1), `amort.py` (pass 2), `pass3.py`, `pass4.py`. Raw logs and JSON beside
them. Full write-up, with the Blackhole carry and the Phase 2 prices, in
`~/.coworker/state/k10-p1-binaryng-why.md`.

Measured DRAM roof on this part: **240.1 GB/s** read+write, 228.4 GB/s copy. Controls: a starved
`ttnn.add` lands at 99.7 % of it, a HiFi4 2048^3 matmul at 24.9 %.

| row | ms | DRAM MB | DRAM GB/s | % of roof | bound-by |
|---|---|---|---|---|---|
| residual `add_(P DRAM, P DRAM)` | 0.8464 | 201.33 | 237.9 | 99.1 % | bandwidth |
| residual `add_(P DRAM, P L1)` | 0.6204 | 134.22 | 216.3 | 90.1 % | bandwidth |
| trimul pair mask `mul_(CH DRAM, mask DRAM)` | 0.8565 | 201.33 corrected | 235.9 | 98.3 % | bandwidth |
| triatt out gate `mul_(HM DRAM, HM DRAM) sig` | 1.2418 | 201.33 | 162.1 | 67.5 % | SFPU sigmoid |
| trimul out gate `mul_(P L1, P DRAM) sig` | 1.2375 | 67.11 | 54.2 | 67.8 % on total memory | SFPU sigmoid |

Three things worth taking away from here and not re-deriving.

**A broadcast operand is read once per block of the axis it broadcasts over, not once.** The pair
mask is 0.524 MB and costs exactly what a full 67.11 MB operand costs. At C = 16/32/64/128 the
broadcast `mul_` and its non-broadcast twin run within 2.5 % of each other at every width. Put the
mask in L1 and the row goes 0.8566 -> 0.6238 ms, which is 0.524 MB of L1 buying back an operand's
worth of DRAM traffic.

**The fused `SIGMOID` is 0.3910 ms on 33.55 M elements; `RELU` is 0.0179 and `TANH` is 0.0842.**
Same bytes, same shape, gate on against gate off. The overhead is linear in elements to within 4 %,
so it is SFPU arithmetic. `sigmoid(x) = 0.5 + 0.5*tanh(x/2)` is the same function by a 4.6x cheaper
route; `HARDSIGMOID` is 22x cheaper but is a different function, not a cheaper route to the same
one. The fusion the model already ships is worth having: unfused, sigmoid as its own program then
`multiply_` costs 1.5830 ms against 1.2414 fused.

**Fusing an elementwise into its producer is worth nothing on a row that is compute-bound.** The
direct measurement of "this operand never crosses DRAM" — gate operand in L1 instead of DRAM —
moves the row from 1.2424 to 1.2404 ms.

Two method notes. Timing one call between two `synchronize_device`s carries a **0.2555 ms** host
floor on this box, enough that 12.6 MB and 50.3 MB both come out at 0.2555 ms; every number here
issues 16 calls between one pair of syncs instead. And every in-place row asserts
`out.buffer_address()` is operand `a`'s, so `read a + read b + write a` is three operand-sizes and
not four.
