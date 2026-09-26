# `TT_BIO_TRIATT_FUSED_HIFI`: +8.471 s, and it fails the accuracy bar intra-domain

Handed up to Moritz 2026-09-26 by `land-standing`. Not landed, and this row will not land it: the
charter puts anything failing the accuracy bar with him.

## What it is

The process-wide choice of whether the fp32-softmax path takes the fused SDPA instead of the
materialised score tensor. `env_flag("TT_BIO_TRIATT_FUSED_HIFI", False)` in `tenstorrent.py`.
Default off, twenty comment lines, and **no recorded reason for being off** — it claims 1.88x more
accurate at the kernel and 20.2x faster at 512 aa.

## Reach, counted on a real model

On an OpenFold3 fold at 832 tokens, served calls go **384 -> 440**. The extra 56 are the sites
that pass `fused_hifi=None` and therefore follow the process-wide flag.

## Speed: it is real, and it is the first op claim this row tested that reached the fold

Six legs alternating, AICLK 1350 MHz sampled DURING every leg:

    off  83.356 / 83.197 / 82.528   median 83.197   A/A floor 0.828 s
    on   74.726 / 73.359 / 75.410   median 74.726

    +8.471 s, 1.1134x, effect 10.2x its A/A floor

`TT_BIO_SDPA_WIDE_K_UP` (1.1554x at the op) and `TT_BIO_TRIMUL_OUT_L1` (1.033-1.182x at the op)
both failed to transfer. This one transfers.

## Accuracy: it fails, and the failure is NOT a fixture artifact

Same fixture and instrument as the dividing-k work — tiled CDK2 at MSA depth 513, where OpenFold3
is confident (pLDDT 0.806). CA RMSD, Kabsch, float64:

    control off vs off     0.000000 A    the instrument has no floor
    LEVER   off vs on      2.602139 A    against the 0.60 A bar -- 4.3x over
    seed floor, measured   1.974757 A    the lever moves MORE than re-seeding does

**The obvious way this could have been wrong was tested and it was not wrong.** This fixture is a
tandem repeat, and this row has previously found tandem copies hinge freely and inflate a global
RMSD. Restricting the superposition and the score to ONE copy removes that degree of freedom
without refolding anything:

    window            within-arm    off vs on
    full 1-832          0.000         2.602
    copy1 1-298         0.000         2.390
    copy2 299-596       0.000         2.325
    core 1-150          0.000         1.077

**The move survives the hinge being removed.** A single 298-residue copy still moves 2.39 A and
even the 150-residue core moves 1.077 A, both far above the 0.60 A bar, against a within-arm
distance of exactly zero at every window. So this is real intra-domain structural change, not an
artifact of how the copies are arranged.

Both confidence heads fall as well: pLDDT 0.80679 -> 0.805988, pTM 0.576249 -> 0.569106.

## The one thing that could still rescue it

Its comment claims the fused kernel is **1.88x more accurate than the materialised softmax at the
kernel**. That and a 2.39 A trajectory move can both be true — this row documented exactly that
shape for dividing-k, where per-call float64 distance was favourable while the 48-block trajectory
still diverged. Nobody has scored this lever against a **ground-truth** structure, only against
the incumbent arm. If the fused route is closer to the experimental answer, the sign of this
verdict changes. That needs a confident target with a deposited structure, which this fixture is
not.

## What is being offered

**+8.471 s** on an OpenFold3 fold at 832 tokens, reproducible, against **2.39 A of intra-domain
structural change** on a bar of 0.60 A. Measured on `wk/land-standing`, i.e. on top of the
dividing-k flip, which is the tree this row would ship.

## The release gate cannot settle the ground-truth question — it is structurally blind to this lever

Tried 2026-09-26, because the section above names scoring against a **deposited** structure as the
one thing that could rescue this lever. The gate's `openfold3` arm folds 7ROA at 117 aa and scores
CA-RMSD / TM against the experimental answer, so running it with the flag off and on looked like
exactly the missing instrument.

Both arms return **1.658 A / TM 0.902, PASS** — identical. **That is not neutrality.** The firing
counters say why:

    off   TRIATT_FUSED_HIFI_STATS  served 0, declined 384, too_short 0
    on    TRIATT_FUSED_HIFI_STATS  served 0, declined 432, too_short 40

Turning the lever on routes **48 more calls into the fused-HiFi path and serves none of them** —
`served` is **0 in both arms**, and 40 of the newly-routed calls are below the route's own
`_TRIATT_FUSED_HIFI_MIN_S = 128` floor. At 117 aa there is nothing for this lever to change, so a
gate result here is a statement about the fixture, not about the lever.

**So the gate is the wrong instrument for this candidate and cannot be made into the right one by
running it again.** Its targets are small by design; the only arm large enough is `rf3-1024aa`,
and that is RoseTTAFold3, while `TriangleAttention.__init__` states that *"OpenFold3 is the only
model here on the fp32 route"* this lever gates.

**What the ground-truth score would actually need**: a deposited structure of roughly 700+
residues that OpenFold3 folds confidently. That is the same fixture problem the dividing-k work
solved at 832 tokens by extracting a deep MSA — one size up, and with the added requirement of an
experimental answer, which tiled CDK2 does not have.

Until that exists, the verdict stands on what is measured: **+8.471 s, and 2.390 A of intra-domain
movement against a 0.60 A bar**, scored against the incumbent arm because no ground truth is
reachable.
