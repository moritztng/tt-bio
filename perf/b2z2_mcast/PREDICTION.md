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
