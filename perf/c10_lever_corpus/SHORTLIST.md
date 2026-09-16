# Shortlist — what is worth a chip, ranked

The target, stated in the campaign's own currency: the work term must fall **15355 -> 9583.65 MHz*s**
under the planning fit, a gap of **5771 Mcycle-equivalents**, or 20.9 % of the work term plus cutting
the 2.901 s clock-immune term to 1.0 s.

Every predicted saving below is a PREDICTION made from the corpus, not a measurement. Where a
prediction converts a share of the cell into the work term it assumes that share is entirely
clock-scaling, which is the conservative direction for a device-class lever and the generous one for
a host-class lever. Both assumptions are named in the row that uses them.

Nothing here is approved. Nothing here is a speedup claim.

---

## 1. Decompose the 2.901 s clock-immune term — the biggest nameable item, and it is not a kernel

**Predicted:** cutting it 2.901 -> 1.0 s is 1.901 s at any clock, **2566 Mcycle-equivalents at
1350 MHz, 44.5 % of the whole gap.** No other single item in the corpus is close.

**Mechanism.** The fit says 20.3 % of a burst-clock fold does not scale with the clock at all
(13.1 % at base clock). The corpus names what is in there without ever having measured the split:
the ~200 per-step host syncs a fold blocks on, the per-op launch floor measured at **20.6 us fixed +
8.3 us per 512x768 block** on Wormhole, and whatever host residual survived the conditioning, z-init
and confidence ports.

**Why the prior measurements under-priced this class, and this is the one place the clock argument
holds.** A lever that deletes clock-immune seconds shrinks a term that is a *smaller share* of a
longer fold. The same 0.5 s saving reads **1.02315x at 800 MHz and 1.03630x at 1350** — 57 % larger
at burst. Every host-class lever in the ledger (`TT_BIO_DEVICE_CONDITIONING` 1.04642x,
`TT_BIO_DEVICE_ZINIT` 1.01090x, the confidence port 1.0128x) was measured at an unrecorded clock
that the campaign's own telemetry says sat near 800-850 MHz most of a 200-step fold. Their ratios at
burst are roughly 1.56x what was recorded.

**Class near a roof, named:** the **353,384 of 465,664 op calls** that `roof-true-floor` classed as
irreducible non-matmul work are all traffic-bound at 1.7040 TB and 0.108 TFLOP, and the 15.031 s
floor **omits the launch term entirely**. `roof-orchestrator`'s registered prediction is a true floor
of 15.5-17.0 s once it is included, which would put the fold at 90-100 % of its floor and leave the
fixed term as the only place left to look.

**Kill in one pass:** two pinned-clock arms of the same tree at two clocks, interleaved in one
process, fitting `F` directly instead of inferring it from two endpoints one of which was
co-tenanted. If `F` comes back under 1.5 s, this route is already mostly spent and the row retires.
The campaign has `c10-fixed-cost` queued for exactly this; do not duplicate it.

---

## 2. Per-class grid sizing for the triangle product and the fused SDPA

**Predicted:** the two classes carry **1.458 s + 1.284 s = 2.742 s** above their binding roofs at the
17.340 s cell, 15.8 % of it. Recovering a third is **5.3 % of the work term, ~810 Mcycles, 14.0 % of
the gap.** A third is a guess; the measured op-level headroom is larger and the transfer to a fold is
not.

**Mechanism.** `roof-tri-close` measured both open classes getting **hurt** by the wide grid, in wall
clock, doing identical work:

    arm                        72 cores    32 cores    16 cores
    triangle product, ms         1.2480     0.5932      0.8712
    fused SDPA, ms               1.9023     2.5851      5.0849
    fused SDPA, core-ms           137.0       82.7        81.4

The triangle product runs **2.10x** better on 32 cores than on 72; the SDPA holds its efficiency
between 16 and 32 cores and loses **1.66x** of it at 72.

**Why the prior measurement under-priced it — and it is not a clock argument.** Grid coverage was
closed early as *"103.8 of 110 cores, 94.4 %, prize 0.000 s"*. That measured whether cores were
**idle**. It did not measure whether using them was **counterproductive**, and on these two classes
it is. The standing lesson `core-coverage-ratio-not-recoverable-time` said a coverage ratio is
usually the op's own quantum floor; here the ratio is fine and the wide grid is costing time.

**Class near a roof, named:** `roof-shape-honest-roofs` put arithmetic as binding at 11.134 s of the
15.031 s floor, and TriangleMultiplication and TriangleAttention are the two largest rows in the
WASTE table after the pair Transition, which has already been worked.

**Kill in one pass:** one process on a pinned-clock, exclusively held Blackhole chip, sweeping core
count per class with its own A/A floor as the contention detector. If the best per-class pick sits
inside the A/A floor of today's grid, retire it. Carry the caveat the source row carries: its
absolute nanoseconds are one Wormhole card at roughly 53 % of the fleet's dense cube, and Blackhole's
110 cores are wider than the 72 that produced the reading, so the sign should transfer and the
magnitude should not be assumed.

---

## 3. Resolve the sign of `TT_BIO_SDPA_GRID_Q_CHUNK`, which is default-ON today

**Predicted:** between **-623 and +281 Mcycles** — the two readings on record disagree in sign. The
downside is 10.8 % of the gap in the wrong direction, on a flag users get by default right now.

**Mechanism.** `_grid_q_chunk` picks the smallest chunk that still fills one grid pass. The atom
SDPA's batch x heads term is large, so an occupancy lever's sign is path-dependent in a way a read
deletion's is not.

**Why the record is wrong, and it is not a clock argument.** The published **1.01831x** was a
marginal on a branch already carrying `TT_BIO_ATOM_L1`, not a standalone. Measured on `main`'s atom
path it made the **sampler stage 15.2 % slower**, on one clean rep from a session that was killed
when the host went from loadavg 14 to 176. The ledger refuses to price it; this is the cheapest chip
hour in the corpus and it is a correctness-of-record item, not a speedup hunt.

**Kill in one pass:** one interleaved isolated A/B on `main`'s atom path, pinned clock, own-session
A/A floor. Inside the floor in either direction, the flag is inert and the row closes either way.

---

## 4. `TT_BIO_DIT_FUSED_QKV` has never had a Blackhole number

**Predicted:** 1.04124x on the Wormhole step. The 200-step sampler is roughly 26.7 % of the cell, so
**~1.06 % of the fold, ~162 Mcycles, 2.8 % of the gap** if it transfers at full value. The one direct
piece of evidence on transfer in this corpus is favourable: both Wormhole step levers in the
five-lever stack carried onto the Blackhole fold (LN 1.03932x -> 1.02694x, LAY 1.02574x -> 1.02936x).

**Accuracy:** 0.346 A against the 0.60 A bar with a 1.84 A seed floor beside it. It **cannot ship
with `TT_BIO_HEAD_PAD_TAIL`** — together they read 0.713 A, over the bar. Approve one or the other,
never both, and never by adding their readings: this cell's perturbations are strongly sub-additive
in the other direction too (LN alone 0.70391 A, the five-lever stack containing it 0.49519 A).

**Kill in one pass:** one interleaved Blackhole step A/B at a pinned clock. Inside its own A/A floor,
retire it.

---

## 5. Route `_FP32_SOFTMAX_L1_GRID` — the largest measured win in the corpus, and not a C10 item

**1.2655x / 1.2623x on a whole 512 aa fold**, two interleaved sessions agreeing to 0.25 %, ~20.1 s
off a 96.8 s Wormhole Galaxy fold, **bit-exact**: all 20 folds of both arms of both sessions wrote
the same CIF sha256 and the same 0.84482 pLDDT, so there is no accuracy spend at all.

It does **not** help Boltz-2, which ships the fused SDPA. The path is taken by default by OpenFold3
(trunk, template, MSA embedder, confidence head) and AF2-IG, on the Galaxy JapanFold serves from. It
is on `wk/roof-bh-envelopes-on-wh` @ `b9a88cd27`, release-gated, not merged, and it needs
`scripts/lever_census.py`'s `stats-dict` reader fixed first — it takes `calls`/`blocked` off
`FP32_SOFTMAX_STATS`, which are the bias-hoist lever's counters and not the L1 grid's.

Listed here because it is the largest unbanked measured result this project holds, not because it
moves the 512 aa Boltz-2 cell. Never quote it as a Boltz-2 speedup.

---

## What the shortlist adds up to

Items 1 + 2 + 3(upside) + 4 come to roughly **3819 Mcycle-equivalents, 66 % of the 5771 gap**, and
**two thirds of that is item 1, which is not a kernel lever at all**. On this corpus there is no
stack of byte-deleting levers that reaches 10.0 s at 512 aa, and the campaign's opening premise —
that the byte levers were under-priced by a throttled clock — is arithmetically the wrong way round
for that class. What a throttled clock suppressed is the host, dispatch and launch class, and that
class is exactly where the fixed 2.901 s lives.
