# b2z2-arrival-skew-attack — PREDICTED, before the first device number exists

Pre-registered by `b2z2-arrival-skew-attack`, 2026-09-12, pushed before any measurement of mine
ran. Copy in `~/.coworker/state/b2z2-arrival-skew-attack.PREDICTION.md`.

## What is being predicted

`b2z2-tile-arrival-latency` split the trunk matmul reader's own time on WH: dependency
serialisation 67.6 % (CHAINWAIT 61.8 % + DOWNWAIT 5.8 %), the unicast forward 15.2 %,
circular-buffer room 13.9 %, the injector's DRAM read 3.2 %, bytes in flight 0.63 %.
`b2z2-mcast-operand-build` then measured a broadcast of the same bytes at 0.93837x on the block
and read the loss as **arrival skew**: a multicast cannot send until the LAST receiver credits it,
so it waits out the spread of N arrival times while the chain waits out one gap.

Nobody has measured the spread. Two accounts of the same 67.6 % are still open and they want
opposite levers.

* **H-RAMP — the chain generates the wait itself.** Every receiver asks at roughly the same time
  and then waits for the data to walk the chain, so CHAINWAIT is a staircase in hop index: core i
  waits i hops. The wait is the chain's own fill latency, i.e. hop cost.
* **H-SKEW — the axis is skewed and the chain absorbs it.** Receivers ask at very different times;
  each core's wait is mostly its predecessor not being ready, not the transfer.

## PREDICTION: H-RAMP, and it is arithmetic off the sibling's own numbers

Under H-RAMP with all cores asking together, core i waits i hops, so the axis-mean CHAINWAIT is
(N-1)/2 hops against one hop of FWD per forwarding core. The prior row measured
**CHAINWAIT / FWD = 61.8 / 15.2 = 4.07** on an 8x9 grid, where (N-1)/2 is **3.5 to 4.0**. Under
H-SKEW that ratio is set by demand jitter and has no reason to land on the grid's half-axis.
DOWNWAIT — a forwarder idle because its successor has not asked yet — is the direct signature of
demand skew and it is **5.8 %**, an order below CHAINWAIT. The receivers are already waiting.

So, pre-registered:

1. **Shape: a ramp, not a tail.** Mean CHAINWAIT per core rises monotonically with hop index.
   Spearman rho(hop index, mean CHAINWAIT) > **0.8**; a linear fit through hop index explains
   **R^2 > 0.7** of the between-core variance. Fewer than **20 %** of the summed CHAINWAIT sits in
   cores more than 2 sigma above the ramp's own prediction for their hop.
2. **Demand skew is small.** Per op instance and per axis, the spread of ASK times
   (CHAINWAIT zone begins, i.e. after cb_reserve_back returned) is under **25 %** of the spread of
   ARRIVAL times (CHAINWAIT zone ends). The cores arrive apart because the chain hands them the
   block in order, not because they wanted it at different times.
3. **Arrival spread is most of the wait.** The per-op arrival spread (last arrival minus first)
   accounts for **> 70 %** of the summed per-core CHAINWAIT on that axis.

## PREDICTED LEVER AND ITS RATIO: two injectors per axis

If it is a ramp, the lever is fewer hops, not a broadcast. Two injector cores per axis, each
serving half the axis, halves the worst hop index from N-1 to about (N-1)/2 and halves the mean.
It costs one extra DRAM read of the same block per axis, and DRAM read is 3.2 % of the reader's
time with 0.63 % of it on the wire, so the trade is strongly favourable. This is **not** the
fan-out arm that lost at 0.69918x: that one put all N transaction issues on the injector's single
RISC, while this keeps one issue per core and only shortens the walk.

* Isolated `generic_minimal_matmul`, M=8192 N=384 K=128, K_num_blocks=1, WH card 16:
  **1.05x to 1.25x, point 1.12x.**
* One 512 aa Pairformer block, WH card 16: **1.01x to 1.08x, point 1.035x.**
* Blackhole cell, 10-11 hops against 8-9: worth more, not less. Point **1.05x** on the block.

## FALSIFIERS, stated in advance

* **H-RAMP falsified** if rho < 0.5 or the linear fit explains under 40 %, or if the ask spread is
  over half the arrival spread. Then the multicast row's skew reading stands, the lever above is
  dead before it is built, and the right lever is a deeper credit window (its own named leftover).
* **The multicast NO-GO has to be re-read** if H-RAMP holds. Its mechanism — the injector waiting
  out the spread of N credit arrivals — requires the credits to be spread out. If the asks are
  measured together, the 0.93837x is not the credit barrier and needs another cause (multicast
  issue cost, or NOC multicast bandwidth against unicast), and **§4-H's "the multicast is done" is
  a conclusion resting on a mechanism this pass would have refuted**.
* **The lever is dead on its own terms** if the dual-injector arm is below 1.00x on the isolated
  matmul, whatever the skew distribution says. A ramp that is not removable by shortening it is a
  finding, not a lever, and it gets reported as NO-GO.
* Bit-exactness is not a prediction, it is a precondition. Changing which core delivers a tile
  cannot change a number: `torch.equal`, max abs 0.0, negative control must break the comparison,
  round-trip-only control must not.
