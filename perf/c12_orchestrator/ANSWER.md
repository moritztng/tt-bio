# C12: Boltz-2 512 aa, asked for 12.5 s minimum and 10.0 s preferred

**Answer: 0.5756 s is measured and banked, taking the fold from 14.881 s to 14.305 s. 12.5 s is not
reachable on the op set the fold runs today, and 10.0 s is excluded by a hardware floor.** Every
number below is at a forced, during-sampled 1350 MHz on qb2 Blackhole p300c.

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

## Why 12.5 s is not reachable

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
    SDPA + heads + 12 1.5672     0        never screened, counted at zero
    device bound                2.0075
    host, generous              0.0960
    CEILING                     2.1035   -> fold 12.777 s

**12.5 s needs 2.3810 s, so it is short 0.2775 s even if every remaining lever lands at its
optimistic end simultaneously.** And that ceiling is generous twice over: it assumes a transpose
kernel nobody has written hits its full bound, and against a campaign record where every lever sized
off a roof ratio collapsed on contact (0.3423 -> 0.035, 0.9454 -> 0.081, 0.4544 -> 0.0184, a 40-96 %
decay each time) the expected value is well below it.

## Why 10.0 s is excluded rather than merely hard

10.0 s needs 4.8810 s against that 2.1035 s ceiling. The binding constraint is structural: **all six
`generic_op` sites are traffic-bound** against a measured 260.9 FLOP/byte machine balance, so its
3.3830 s cannot go below a **2.1693 s traffic floor** however fast the arithmetic becomes. Raising
arithmetic rate, which earlier passes built the 10.0 s case on, cannot touch it. That case compared
an op moving 402.9 MB per call against a dense cube and was a 4.4x phantom.

Reaching 10.0 s therefore needs a **different op set**, not a faster one: fewer or cheaper ops doing
the same model arithmetic. That is kernel and algorithm work outside what this campaign scoped, and
it is not the same as doing less of the model's own work, which is barred.

## The one thing still open that could move the answer

**1.5672 s of device time has never been screened** (SDPA 0.4335, NlpCreateHeads 0.3007, and a
`twelve smaller` 0.8330 s bucket that no document ever broke out). That is 11.9 % of device time and
**5.6x the 0.2775 s the ceiling falls short by**, and it is counted at zero above.
`c12-tail-classes-screen` is running on it. So the honest claim is not that the fold is exhausted,
but that **what this campaign enumerated is exhausted, and the largest unscreened block is several
times the remaining gap.**

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
