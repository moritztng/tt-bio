# C12: Boltz-2 512 aa, asked for 12.5 s minimum and 10.0 s preferred

**Answer: 0.5756 s is measured and banked, taking the fold from 14.881 s to 14.305 s. 12.5 s is
reachable but not in hand, and it turns on one unbuilt lever whose kernel already exists. 10.0 s is
excluded by a hardware floor.** Every number below is at a forced, during-sampled 1350 MHz on qb2
Blackhole p300c.

*Revised 2026-09-18: this document said 12.5 s was not reachable, short 0.2775 s, with the explicit
caveat that 1.5672 s of measured device time had never been screened and was counted at zero. That
screen landed and found 0.6355 s of bound in the block — 2.3x the gap. The caveat was load-bearing
and the old verdict did not survive it. 10.0 s is unmoved: it was never a shortfall of levers.*

## What is actually in hand

    silu + cond-hoist, measured at the fold   +0.5756 s   9.3 sigma, CI [+0.4539, +0.6974]
    fold of record                             14.881 s
    fold with it                               14.305 s   1.0402x

Pooled from **two independent sessions** (31 reps and 10 reps), not one run: inverse-variance
weighted, sessions statistically consistent (z = +0.48), and both A/A controls unresolved from zero
and falling on opposite sides of it, which is what a true null looks like. It clears its
pre-registered margin clause at **3.57x** the pooled A/A floor.

Accuracy is settled on the fixture the bar is written for. `cdk2x2_298`: worst deviation
**0.38302 A** against that fixture's own **0.80218 A** seed floor, so the lever moves the structure
**0.477x** as far as changing the random seed already does, with a **0.0000 A** A/A control and
identical digests. At 512 aa it sits inside the seed floor on both metrics at all five seeds. It
lands in the **HOLD** band (above the 0.35 A pass bar, below the 0.60 A reject bar), which is by
construction where a human decides rather than a gate.

**It is not shipped.** Both flags default `False` on `main`, and flipping them is
[ask 8879](../../../state/pending-input/8879.md), open. `silu` is a one-line default change with an
empty `tt_bio/` diff. `cond-hoist` needs a **28-line eager-build merge first**: on `main`
`_cond_weights()` builds lazily, so flipping the default alone makes the first fold of every process
pay 0.34-0.57 s to save 0.2415 s. Fine for a long-lived JapanFold worker, a net loss for a one-shot
CLI fold. See `landing/LANDING.md`.

## 12.5 s: reachable, and it turns on one lever

The fold is **88.76 % device time**: 13.2090 s measured in situ against 1.6489-1.6720 s of
everything else. So deleting **all** host time leaves 13.2090 s, which misses 12.5 s by 0.7090 s.
Both targets are device problems, and 12.5 s needs 17.3 % of measured device time.

Taking each device class at the most any C12 row measured it could ever give:

    class            in situ   max ever   why
    Matmul            3.9705     0.1352   class cap measured in situ
    GenericOp         3.3830     1.2137   all six sites traffic-bound, 2.1693 s traffic floor
    BinaryNg          2.3899     0.1321   already at the DRAM roof, byte deletion only
    LayerNorm         1.3808     0.2415   cond-hoist's share; no other measured mechanism
    Transpose         0.5174     0.2850   a dim0/dim1 mover that would have to be written
    SDPA + heads + 12 1.5672     0.6355   SCREENED 2026-09-18; was counted at zero
    device bound                2.6430
    host, generous              0.0960
    CEILING                     2.7390   -> fold 12.142 s

**This section said "not reachable, short 0.2775 s" for five passes, and the thing that changed it
was the caveat it carried.** The bottom row of that table was 1.5672 s counted at zero, flagged as
the largest unscreened block in the fold. `c12-tail-classes-screen` itemised all of it and found
**0.6355 s of bound, 2.3x the gap it had to close**, so the old headline does not survive its own
screen. Its signature sums reconcile to the three zeroed classes exactly (SDPA 0.43348 vs 0.4335,
`NlpCreateHeads` 0.30067 vs 0.3007, six leads 1.22312 s + 0.34414 s unaccounted = 1.56726 s).

Split by MATURITY, because "the most a class could give" was mixing a kernel that already ships
with one nobody has written (`perf/c12_orchestrator/book/ceiling.py`):

    tier                                gives  cumulative     fold   vs 12.5 s
    (device enumerated + host)                     2.1035   12.777   short 0.2775 s
    T1 built + measured elsewhere      0.3138      2.4173   12.464   CLEARS by 0.0363 s
    T2 accuracy reading needed         0.0770      2.4943   12.387   CLEARS by 0.1133 s
    T3 unbuilt design work             0.2448      2.7390   12.142   CLEARS by 0.3580 s

**T1 alone straddles the target**: fold 12.464 s at its full-delete bound, 12.543 s at its own
pre-registered 0.234 s discount — either side by under 0.05 s. T1 is the diffusion-side head-major
qkv/out deletion, and it is the only lever in the campaign whose kernel is **already written and
shipped default-on elsewhere in the same tree** (`tt_bio/triatt_qkv.py:39,131` on `origin/main`,
bit-exact by `torch.equal` at six sizes, measured in-fold at 1.1797x on the TriAtt body). The
diffusion side never got it. `c12-diffusion-head-major` owns it.

**Why T1 is not another decaying estimate.** C12's record is that every lever sized off a ROOF RATIO
collapsed on contact (0.3423 -> 0.035, 0.9454 -> 0.081, 0.4544 -> 0.0184) while both levers sized off
an EXECUTED GRAPH met or beat their predictions (silu by 5.3 %, cond-hoist against 0.1445 s). T1 is
neither an estimate nor a rate bet: it deletes ops priced at their own measured in-situ time, and its
precondition — head_dim a whole number of tiles in padded form, 64 = 2 tiles and 32 = 1 tile — was
checked at all four executed signatures. Its risk is transcription risk, not prize risk.

**So the honest statement is contingent, not triumphant.** 12.5 s is no longer forbidden by the
arithmetic; it requires T1 to land near its bound and the banked 0.5756 s to hold. Nothing has been
measured at the fold for T1 yet.

## Why 10.0 s is excluded rather than merely hard

10.0 s needs 4.8810 s against that 2.1035 s ceiling. The binding constraint is structural: **all six
`generic_op` sites are traffic-bound** against a measured 260.9 FLOP/byte machine balance, so its
3.3830 s cannot go below a **2.1693 s traffic floor** however fast the arithmetic becomes. Raising
arithmetic rate, which earlier passes built the 10.0 s case on, cannot touch it. That case compared
an op moving 402.9 MB per call against a dense cube and was a 4.4x phantom.

Reaching 10.0 s therefore needs a **different op set**, not a faster one: fewer or cheaper ops doing
the same model arithmetic. That is kernel and algorithm work outside what this campaign scoped, and
it is not the same as doing less of the model's own work, which is barred.

## What is still open

**The 1.5672 s screen is DONE and it is what moved the answer** — see the tier table above. What it
left behind, in descending order:

    T1  0.3138 s  head-major qkv/out on the diffusion side   GO, row open, kernel already written
    T3  0.1330 s  OuterProductMean layout round-trip         needs a custom outer-product kernel
    T3  0.1118 s  head-48 merge, net of +0.0236 s back       needs a 16-way batched-matmul absorb
    T2  0.0770 s  SDPA attention bias in bfp8                needs an Angstrom reading, no row
        0.3441 s  unaccounted inside the tail                counted at zero

Closed with reasons rather than left hanging: the pair Transition's 11-way chunk (0.1504 s) is
**byte-neutral in the chunk count**, so chunk-height tuning cannot move one byte of it and both its
ops already run at 92.8-103.3 % of their mix-matched roof; Embeddings (0.0571 s) at 5.5 % of roof is
the known per-element gather floor on Blackhole, proven not-bandwidth by a bfp8 arm at half the bytes
landing 0.1 % apart. That screen also settled the campaign's **47-vs-16 dispute from the fold's own
dispatch: the executed chunk height at 512 aa is 47**, so the source reading that disputed it was
evaluating a path the fold does not take.

Two byte models had to be fixed before any of it could be priced, and both are the same trap: a Slice
charged its whole input tensor rather than the extent it returns and read **1336.2 GB/s, 3.4x the
part's roof**, and SDPA's K/V re-read factor has to come from the executed `q_chunk_size` or the
token site reads 37.4 % of roof instead of 56.0 %. **A rate above the roof is a broken byte model,
not a fast op.**

Also pending: `c12-reblock-delete` (1.0062 s central, band 0.9342-1.2007) has never completed a
measurement, blocked by hardware rather than by analysis. Its number refines the route but does not
change the verdict, because the ceiling above already credits its class with the full 1.2137 s it
could ever give. Its optimistic 1.2007 s is in fact pinned as a physical maximum: added to
`c12-genop-triatt-slack`'s 0.0184 s it exceeds the class bound by 0.0054 s.

## Hardware, because it cost more than any lever

qb2 card 2 failed **four times in 2.5 hours** on the night the deciding measurements were due: twice
mid-session at 159 and 60 folds, twice at device open, with ARC answering throughout. A chip reset
restores a clean 3.5-3.7 s open which then degrades within minutes. Card 2 is blocked, the 410D board
pair is inverted so card 3 works and card 2 idles as its sibling, and card 3 is untested by design so
that the next run separates a bad chip from a bad board.
