# b2z2-trunk-fold-ab-bh — pre-registered, written before the first fold

Committed before any fold ran. Branch `wk/b2z2-trunk-fold-ab-bh`, whglx card 10 (Wormhole, 8x9),
512 aa, `perf/size512/fixtures/cdk2x2_512.yaml` + its fixed 35-row a3m, 200 sampling steps,
3 recycles, 1 sample, seed 0, templates off, timed at `predict_one`.

## What is being asked

`b2z2-trunk-byte-round2` measured **1.05106x on one PairformerLayer** from three bit-exact levers
(`TT_BIO_TRIMUL_FUSED_GOUT`, `TT_BIO_TRIATT_FUSED_QKVG`, `TT_BIO_TRIATT_FUSED_QKVGB`), against a
1.00064x A/A floor, and **located 0.497 s/fold**. No fold has run. This row runs it.

## The arithmetic the prediction comes from

The three levers remove **4.858 %** of the block's wall (4.173 ms of 85.905 ms, WH, medians of 7).
A fold's Pairformer blocks are the only place that saving can appear. On Blackhole the campaign
scores 10.22 s of PairformerLayer device span against a 20.188 s fold, i.e. **50.6 %**. WH's own
share is not measured, so the band below carries 45-60 % rather than a point.

    fold gain = 0.04858 x (Pairformer share of the fold)

## P1 — all three levers, fold ratio at 512 aa

**Point 1.0245x. Band 1.015x - 1.032x.** 1.015x is the 45 % share with a third of the block saving
lost to dispatch and host time the block A/B never had; 1.032x is the 60 % share realising in full.

## P2 — `qkvg` alone, the scale check

Block ratio 1.01459x = 1.44 % of the block wall. **Point 1.0073x. Band 1.003x - 1.011x.**
This arm is deliberately near the fold's measurability floor: if it lands inside its A/A floor
while P1 clears, that bounds where block-level numbers stop transferring, which is the useful half.

## P3 — linearity

Block gains are 0.05106 and 0.01459, a ratio of **3.50**. Predicted fold-gain ratio
(all3 gain)/(qkvg gain) = **3.5, band 2.0 - 5.0**. A fold gain that is not proportional to the
block gain means something between the block and the fold absorbs or amplifies it.

## P4 — parity, bit-exact

All three arms write the **identical CIF**: `sha256` equal across every rep of every arm, plddt
equal. The negative control (the fused qkvgb path's bias output scaled by 1+2^-10, one fold) must
produce a **different** sha. If the negative control does not break the check, no pass is reported.
This is the trap that bit this lineage: `tt_bio.reference` zero-initialises 23 of a PairformerLayer's
weights, so a block-output comparison against it cannot fail. This row compares folds of the real
checkpoint, and proves the comparison is sensitive before trusting it.

## P5 — eligibility census flat

`qkvg`, `qkvgb`, `trimul_gout` served/rejected counts identical in every rep of a given arm, and the
base arm serves zero of all three. A lever that silently drops a tuned kernel somewhere else is how
an earlier byte row's sign flipped.

## P6 — the A/A floor itself

Predicted **below 1.005x** on the median-of-reps estimator, computed as the 95 % band of that same
statistic resampled from the base arm's own reps. Quoting a per-position floor beside a
median-of-reps ratio is the error CONTEXT's A/A block calls out; the floor here is the floor of the
statistic actually reported.

## FALSIFIER (the brief's, pre-registered verbatim in effect)

**If the all3 fold ratio sits inside its own A/A floor while the block ratio reproduces at >= 1.04x
on this same card, then a 1.05x block lever does not reach the fold** — and that verdict applies to
every block-level number this campaign has quoted, not just to these three levers.

Secondary falsifier: if the census shows the fused path is never served during a real fold (the
block A/B built its layer directly and a fold may route around it), the number is 1.000x for a
structural reason and the block A/B was measuring a path the fold does not take.

## Blackhole

Owed, not predicted. The biggest of the three levers works because a one-tile-wide N sits on a grid
that wants seventeen; BH is 11x10 against whglx's 8x9 and redistributes those tiles differently.
Needs a qb2 card. No projection is recorded here on purpose.
