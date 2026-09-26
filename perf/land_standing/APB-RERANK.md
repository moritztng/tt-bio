# `TT_BIO_APB_CONCAT_HEADS` does not cost OpenFold3 1.254 A. It re-ranks five samples that both arms produce.

Measured 2026-09-26 10:37-10:53Z by `land-standing`, qb2 card 3, six folds interleaved
off/on/off/on/off/off, one subprocess per fold, cached a3m so the MSA is not a variable.
Harness `perf/land_standing/apb_seedfloor.py`, scorer `perf/land_standing/apb_score.py`,
artifacts under `perf/land_standing/out/apb_seedfloor/`.

**This overturns my own 10:28Z verdict in `state/notice-apb-lever-parked-on-an-expired-blocker.md`,
which read the gate's one-number-per-arm summary as a 1.254 A accuracy cost and held the lever on
it.** The summary line cannot distinguish a structural move from a re-ranking, and this was a
re-ranking.

## The instrument is the gate's own, and it is controlled

`off:s0` reproduces the release gate's openfold3 table to the digit: **1.658 / 2.929 / 1.652 /
1.700 / 3.620**. `on:s0` reproduces the on-arm's: **2.912 / 1.657 / 1.640 / 1.691 / 3.876**. So
this session reproduces the 10:28Z control exactly, on a host at loadavg 5.66 — which also says
the structure output is load-independent, as it should be.

## 1. The two arms produce the same five structures. Only the order changes.

Nearest-neighbour CA RMSD, every off sample against every on sample at the same seed:

    seed 0                                        seed 1
    off rank 0 -> on rank 1   0.0441 A            off rank 0 -> on rank 0   0.0919 A
    off rank 1 -> on rank 0   0.1036 A            off rank 1 -> on rank 2   0.0620 A
    off rank 2 -> on rank 2   0.0417 A            off rank 2 -> on rank 1   0.0285 A
    off rank 3 -> on rank 3   0.0837 A            off rank 3 -> on rank 4   0.0783 A
    off rank 4 -> on rank 4   2.1869 A            off rank 4 -> on rank 3   1.3633 A

The assignment is a permutation at both seeds, and it is not a marginal one: the four delivered-
class twins sit at **0.03-0.10 A** while the next-nearest candidate is **0.46-2.19 A** away, an
order of magnitude of separation. Rank 4 is the one sample that genuinely moves, and rank 4 is the
worst fold in the set (3.59-3.88 A) — nobody receives it.

**So at seed 0 the "1.254 A cost" is rank 0 and rank 1 swapping places.** The 1.658 A structure
and the 2.912 A structure both exist in both arms. The confidence head ordered them differently.

## 2. At seed 1 the lever is inert on the delivered structure, and shipping main is worse

    seed 1   delivered off 3.091 A   delivered on 3.109 A   the two structures 0.0919 A apart

Read that beside seed 0: **the lever off already delivers 3.091 A at seed 1**, worse than the
lever on delivers at seed 0 (2.912 A). The quantity the 10:28Z note attributed to the lever is
produced by re-seeding alone, in the arm that ships today.

## 3. The seed floor, measured here rather than borrowed

CA RMSD between the delivered structures of the lever-off arm at four seeds:

    off:s0 vs off:s1   2.2232        off:s1 vs off:s2   1.2139
    off:s0 vs off:s2   1.7353        off:s1 vs off:s3   1.5378
    off:s0 vs off:s3   1.7219        off:s2 vs off:s3   0.8494

    seed floor  n=6   0.8494 .. 2.2232 A   mean 1.5469 A
    delivered RMSD vs 7ROA across seeds, lever off:  1.658 .. 3.091 A

The lever's largest observed move of the delivered structure is **2.1344 A at seed 0**, inside the
floor's range and below its top (2.2232 A); at seed 1 it is **0.0919 A**, an order of magnitude
below the floor's minimum. Its delivered RMSD-vs-native, **2.912 A** and **3.109 A**, lands inside
the band the off arm covers by re-seeding, **1.658-3.091 A**.

Against the gate's own bar of **3.5 A / TM 0.70** the lever is not the binding constraint at either
seed: off:s1 clears by 0.409 A and on:s1 by 0.391 A, an 0.018 A difference.

## Verdict on the lever's accuracy leg: CLEARED

The accuracy objection I raised at 10:28Z does not survive its own measurement. The lever moves the
delivered structure by less than re-seeding does, and the mechanism is a confidence re-ranking of
an unchanged sample set rather than a degradation of the fold.

**What is still outstanding is the PERF number, and it is a different gap.** `c14-land-tail`'s
**+0.0551 s** was measured at `80d401428`, a tree that is not an ancestor of main and is now 5392
commits behind it, in `tt_bio/tenstorrent.py`, which has gained 2396 lines since. The accuracy case
is now re-established at the merge tree; the speed case is not, and re-taking it needs a quiet box.
A +0.0551 s claim also needs a floor tight enough to resolve it, which is a stricter requirement
than anything this lever has been measured against so far.

## A separate finding, larger than this lever, about shipped OpenFold3

Delivered against oracle-best, lever off, same four seeds:

    seed 0   delivered 1.658    best available 1.652 (rank 2)
    seed 1   delivered 3.091    best available 1.655 (rank 1)
    seed 2   delivered 2.638    best available 1.605 (rank 3)
    seed 3   delivered 2.584    best available 1.632 (rank 1)

Every seed has a ~1.6 A sample in it. **At three of four seeds the confidence head delivers a
2.58-3.09 A sample instead.** That is a property of shipped main, not of any lever here, and it is
worth about 1.0-1.4 A of accuracy on this target — an order of magnitude more than the flag this
file is about. It also means the gate's openfold3 arm clears its 3.5 A bar by 0.41 A at seed 1 and
is closer to red than its seed-0 number suggests. Filed for whoever owns OpenFold3 confidence.
