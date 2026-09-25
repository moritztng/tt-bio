# BindCraft 2 design throughput, per box

`throughput.py` computes every number in `state/bcx-throughput.md` from banked measurements and
public list prices. No hardware is touched. Run it and it reprints the tables and rewrites
`out/throughput.json`:

    python3 perf/bcx_throughput/throughput.py

A cycle is one design trajectory against hPDL1 chain A (115 aa) with a 146 aa binder on the shipped
five-model `multimer_v3` pool: 125 gradient rounds, 15 mutate steps, ProteinMPNN redesign and the
validation refold ensemble.

| | per chip/GPU per hour | per box per hour |
|---|---|---|
| Blackhole p300c / TT-QuietBox 2 (4 chips) | 0.386-0.457 | 1.20-1.75 |
| H200 / DGX H200 (8 GPUs, 7 workers each) | 32.06 | 459.1 |

Box to box that is 263-384x on the whole cycle, 6.4-9.4x per dollar of list price and 33-49x per kW.
Fixing the gradient-phase software gap buys 3.79-4.19x end to end rather than the 6.7-10.6x it buys
on the phase, because the 1,253 s ProteinMPNN/validation tail runs on host JAX and does not move.

Every input, with its source, is the `IN` block at the top of `throughput.py`.
