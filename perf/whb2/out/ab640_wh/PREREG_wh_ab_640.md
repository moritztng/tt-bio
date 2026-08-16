# Pre-registered prediction, WH fold-level A/B for K3 at 640 aa
Written 2026-08-16T00:35Z, BEFORE the run. Source numbers: perf/whb2/out/divk_wh.json (this pass).

Per-call, padded 640, Wormhole 8x9:
  stock fallback today  14.9193 ms   (fused declines, q_stock_fallback=320)
  fused at dividing k   8.0200 ms    (k=160)
  saving                6.8993 ms/call

Call count at 640 aa: 1680 triangle-attention calls per fold (BH census 12.4 measured
5040 declines = 3 ladder q x 1680 calls; the count is size-independent per recycle).

PREDICTED delta, upper bound: 1680 * 6.8993 ms = 11.59 s
PREDICTED delta, lower bound: 5.8 s. Isolated per-op timing over-syncs roughly 2x against
the same work timed batched (memory tt-bio-isolated-op-timing-oversync-inflates-cost), so
half the upper bound is the honest floor.

WH 640 aa baseline wall is NOT yet measured. Measured WH walls: 384 aa 31.919 s, 512 aa
48.744 s. 640 aa is on the chunked path (SEQ_LEN_MORE_CHUNKING = 608), so 75-110 s is the
expectation. Predicted share of wall: 6-13 %.

KILL GATE: if the A/A floor from arms AA1/AA2 is larger than the measured A-vs-B delta,
report INCONCLUSIVE and quote no ratio. This is the 12.5 discipline and it is not optional.
ACCURACY: not bit-exact. K3 changes the online-softmax reduction order. pLDDT is the arm.
CONTENTION: run outside benchlock, deliberately. The lock has starved this job for 100 min
behind another worker while two foreign folds ran outside it anyway. loadavg is recorded per
arm and the A/A pair is what judges the delta.
