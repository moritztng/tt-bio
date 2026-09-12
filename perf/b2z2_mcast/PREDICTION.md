# b2z2-mcast-operand-build — replace the operand daisy chain with a multicast

VERDICT: IN PROGRESS (prediction pre-registered, nothing measured)
BRANCH: wk/b2z2-mcast-operand-build
CARD: whglx card 16 (UMD logical 16), pinned, lease `state/leases/whglx-card16.json`
ARCH: WH for everything I can measure. No Blackhole chip is free, so the cell number is OWED,
  not quoted. What it would take: one p300c processor on qb2 under benchlock, the same paired
  block A/B, ~20 min.

## PREDICTED — written before the first device number exists

`b2z2-tile-arrival-latency` split the matmul reader's own time on WH: dependency serialisation
67.6 %, the unicast forward 15.2 %, CB room 13.9 %, the injector's DRAM read 3.2 %. The chain plus
the forward is 82.8 % of 239712.3 core-us per block. Multicast deletes the handshakes and collapses
the forward from N hops to one, so 82.8 % of that reader time is what stands in front of it.

**What that is worth, priced two ways, because the two readings differ by 3x and I am not going to
pretend they agree.**

* **Accounted-time reading (the one I predict).** The reader's 239712.3 core-us is 19.9 % of the
  math-thread input wait carried by the same 16 programs (16746.6 us/core x 72 cores =
  1,205,755 core-us), and those programs carry 35.2 % of the block's 47540.1 us/core wait. So
  0.828 x 239712.3 = 198.5k core-us against the block's 3.423M core-us of input wait = **5.8 % of
  the input wait, 3.2 % of the 85.24 ms block span**. That is **1.033x on the block**.
* **Latency-amplification reading (the brief's ~1.10x).** A serial chain's cost to the math thread
  is not only the reader core-us it bills; the last core on the chain starts N hop-latencies late
  and the math thread behind it idles for all of them, which is wait the reader never books. If
  that amplification is real the lever reaches the brief's ~1.10x.

**PREDICTED block ratio on WH, paired A/B in one process, n>=5: 1.02x - 1.10x, point 1.035x.**

FALSIFIERS, stated in advance:

1. **< 1.005x** (inside the A/A floor) kills the lever on Wormhole and says the chain was never the
   cost — the serialisation is real but removing it buys nothing because something else refills the
   wait.
2. **> 1.10x** says the accounted-time reading under-prices the chain and latency amplification is
   the right model. That is a result about the cost model, not just about this kernel.
3. **Slower than the chain** is the outcome the shipped kernel's own comment warns about
   (*"Critical to performance for sender to push data to compute before mcasting. This frees sender
   to start next read earlier"*). The mechanism would be the injector now blocking on the SLOWEST of
   N receivers where the chain only blocked on its immediate successor. I keep the push-to-compute
   ordering exactly as shipped so this is the only thing that changes.

**Bit-exactness: PREDICTED torch.equal, and it is scored not assumed.** The same bytes reach the
same cores at the same L1 addresses; no accumulation order changes. The fixed-config custom kernel
means the `b2z2-l1-sharded-residency` mechanism (ttnn reselecting a program config under a moved
operand) cannot apply. Negative control: perturb one input tile and confirm the check fails.

**Blackhole, PREDICTED before any BH run, so the confirmation is scored.** The chain is 8-9 hops on
whglx's 8x9 WH grid and 10-11 on the cell's 11x10 BH grid. If the term is the chain, its cost is
~linear in hop count, so the same lever is worth **1.235x more of a delta on Blackhole**
(10.5/8.5): a WH block ratio of R predicts a BH block ratio of about **1 + 1.235 x (R - 1)**. At
the predicted 1.035x on WH that is **1.043x on the cell's Pairformer block**. FALSIFIER: a BH
ratio at or below the WH ratio refutes the hop-count model.

## MEASURED

(nothing yet)

---

## PREDICTED, the fan-out arm — added 2026-09-12 ~21:0x UTC, before its block number exists

The multicast arm hung the chip twice and the root cause turned out not to be the multicast at all
(see the state doc). The second arm, `MM_BCAST_FANOUT`, has the injector write the block to each
receiver itself with only the two primitives the shipped kernel already issues.

**It deletes a different subset of the term.** The chain costs 67.6 % handshakes + 15.2 % forward.
Fan-out deletes the handshakes outright: no core waits on its predecessor, every core waits on the
injector. It does NOT delete the bytes — the chain already moves N x B in total, one hop at a time,
and the fan-out moves the same N x B — but it moves the **issue** of all N transactions onto the
injector's single RISC, which the chain spread over N cores. The injector was measured spending
6207.4 core-us issuing its DRAM read against 1516.3 waiting for it, so issue time is not free.

**PREDICTED block ratio on WH, paired A/B, n>=5: 1.02x - 1.08x, point 1.03x.** Lower than the
multicast's ceiling because the bytes stay; near the same point estimate because the bytes were
0.63 % of the reader's time and the handshakes were 67.6 %.

FALSIFIERS: **< 1.005x** says the serialisation is real but removing it buys nothing, which is the
same kill the multicast arm faced. **Slower than the chain** says the injector's serial issue of N
writes costs more than the N-hop chain it replaced, which would be a measurement of issue cost, not
of the chain. Bit-exactness is `torch.equal` with the same negative control.

(Written while the correctness smoke test was on the card; no block A/B number existed yet.)

---

## MEASURED — 2026-09-12, whglx card 16, WH 8x9. Both predictions falsified, in the same direction.

**On one settled Pairformer block at 512 aa: the multicast is 0.93837x, i.e. 6.6 % SLOWER than the
daisy chain.** `perf/b2z2_mcast/block_mcast_whglx_c16.json`. chain 85.692 ms, mcast 91.3202 ms,
7 mirrored A B B A reps spanning 0.93806-0.93937 (0.14 %), A/A floors 0.99971 and 1.00118, all 4
generic matmul program shapes in the block on the arm. Predicted 1.02x-1.10x; falsifier 3 fired.

**Parity holds and the check reads something.** `torch.equal` on both outputs, `max_abs_diff` 0.0
and 0.0. The negative control (one element of the 1x512x512x128 pair track moved) correctly breaks
the comparison, and the round-trip-only control correctly does not, so the control is not detecting
the torch round trip.

**The fan-out arm is 0.69918x, 1.43x slower, also bit-exact.**

**What the two arms together say. The chain is not paying for hops. It is buying skew tolerance.**

| where | what the axis looks like | chain vs multicast |
|---|---|---|
| one matmul, back to back, nothing else on the grid | every core arrives together | **1.01845x for the multicast** |
| inside a Pairformer block | cores arrive skewed by the ops in front of the matmul | **0.93837x, the chain wins** |

A multicast cannot send until the LAST receiver has posted its credit, so it is a barrier across the
grid axis. The chain never waits for the axis: core i sends as soon as core i-1 asks, so it is a
pipeline, and a pipeline absorbs per-core skew that a barrier converts into stall. The isolated
matmul removes the skew and the multicast wins there, which is the control that makes the mechanism
a measurement rather than a story. The fan-out arm says the same thing from the other side: it
deletes the chain and concentrates all N transaction issues on the injector's one RISC, and loses
by more.

**The Blackhole prediction inverts, and it is still the same model.** It was
`1 + 1.235 x (R - 1)` from the hop count, 10-11 hops on the cell against 8-9 on whglx. With
R = 0.93837 that now predicts **0.9239x on the cell** — a deeper chain is a deeper pipeline, so the
multicast should lose by MORE on Blackhole, not win there. The sign is what is being predicted; a
cell measurement at or above 1.0 would refute the skew reading.

**The one thing that could rescue it**, unbuilt and unpriced: the barrier only bites because a
receiver can run at most one block ahead (the in0/in1 circular buffers are two blocks deep). A
deeper credit window would let early cores post credits for later blocks and let the injector send
without waiting on the laggard. That is a different lever from `b2z2-cb-depth-prefetch`'s, which
swept the K ring where `K_num_blocks == 1` left nothing to prefetch; this is the M/N block loop,
which does iterate.
