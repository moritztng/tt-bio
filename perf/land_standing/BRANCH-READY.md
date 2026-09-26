# `wk/land-standing` is ready to merge: one code line, +50.999 s

Rewritten 2026-09-26 06:5xZ. **The previous version of this file said the only shared-surface
change was "comment-only". That is no longer true and was the most important thing on the page** —
the branch now flips a shipped default. A merge handover that understates its own blast radius is
worse than none, so this is the corrected one.

    0 behind origin/main        merging is a fast-forward; main afterwards IS this tree
    tree                        a957974ce071145c22e8a12eed30558b3b646959
    109 files, +6221 / -46
    non-perf files              exactly one: tt_bio/tenstorrent.py
    NON-COMMENT lines changed   exactly one, in the whole tree:

        -_TRIATT_HIFI_DIVIDING_K_DEFAULT = False
        +_TRIATT_HIFI_DIVIDING_K_DEFAULT = True

Everything else is comment or lives under `perf/land_standing/`.

## What that one line buys

**+50.999 s, 1.6351x** on an OpenFold3 fold at 832 tokens. A/A floor 1.306 s, so the effect is
**39x its own floor**. AICLK sampled DURING every leg at 1350 MHz, arms interleaved in one process
on one device open, board-pair sibling verified idle.

**Reach is exactly one length.** Enumerated over all 48 tile-aligned lengths from 32 to 1536 with
a model validated against six device outcomes (`capreach.py`): ten lengths serve no fused pair, the
HiFi route is `openfold3.trunk` alone, and OpenFold3 pads to a multiple of 64 — so 832 is the only
one a user can present. The lever opens that one and **changes the pick at no length that serves
today**.

## Accuracy: clears on every instrument

CA RMSD, Kabsch, float64, over all 832 CA, on a fixture where OpenFold3 is confident (tiled CDK2
at MSA depth 513, pLDDT 0.806):

    control (same arm, rerun)    0.000000 A    the instrument has no floor
    this lever                   0.450148 A    against the 0.60 A bar
    seed floor (two seeds)       1.974757 A    4.4x the lever's move
    lever AND seed together      1.948630 A    no more than the seed alone

Both confidence heads move the favourable way (+0.000551 pLDDT, +0.000647 pTM).

Worth knowing why this took so long: every earlier arm was folded `--single_sequence` and read
pLDDT 0.37, near the confidence heads' floor, where the lever appeared to read the *other* way.
Depth was the whole problem — 0.503 single-sequence, 0.602 at depth 36, **0.882 at depth 513**.

## Gate: ten arms green at the tip, two re-green on the merge tree

Every arm scored against this worktree and asserted per arm from the gate's own scoring-tree line.
That guard exists because an earlier run silently imported `/home/ttuser/tt-bio-dev` and reported
two green arms for a tree without the flip.

    openfold3      1.658 / 0.902   <=3.5 / >=0.70    PASS      boltz2   1.827 / 0.902   PASS
    rf3            1.239 / 0.958   <=3.0 / >=0.75    PASS      opendde  1.395 / 0.940   PASS
    protenix-v2    1.999 / 0.867   <=6.0 / >=0.50    PASS      esmfold2 1.343 / 0.961   PASS
    esmfold2-fast  1.708 / 0.918   <=4.5 / >=0.60    PASS      capacity 7.23 / 7.00 GiB PASS
    batch-position PASS            l1-budget PASS, one digest across native / 8x8 / narrow

`l1-budget` is the one that matters for a kernel-routing change: its three grid classes return the
identical digest `c3073854d423570ae48cb8ce35ccb27e`, so the flip introduces **no card-dependence**
— the hard stop this class of change risks, and the arm that caught exactly that for Region T.
`rf3` lands on 1.239 / 0.958, to the digit the gate's own source comment records as its baseline.

Re-run on the merge tree after merging main forward: `openfold3` and `l1-budget` both green with
identical numbers, plus `tests/test_bindcraft2.py` (the tests main changed) at 9 passed / 14
skipped.

## One risk named and NOT closed

Main added `tests/test_bindcraft2_hw.py`, a device test that differentiates AF2's 48 Evoformer
blocks on card. That is the **taped** path, where this flip opens 288 — verified on this tree, not
quoted (`bc288.py`: `n=288 off=None -> on=(ladder,288,288)`, 256/320/384 inert, reproducing
`bcx-forward`'s own serves.json). **It cannot run on qb2: BindCraft2 is not installed.** So the
interaction is identified, argued favourable from that path's own pre-existing measurement
(0.021702 -> 0.018661), and not executed here.

## What is deliberately NOT on this branch

`TT_BIO_TRIATT_FUSED_HIFI`, worth a measured **+8.471 s** at 832 (1.1134x, 10.2x its A/A floor) —
held because it moves the structure **2.602 A**, over both the 0.60 A bar and the 1.974757 A seed
floor. Accuracy-failing levers are Moritz's call, so that one is handed up rather than landed.
