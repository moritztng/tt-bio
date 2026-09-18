# RETRACTED: the 512 aa plDDT deficit is not real. It was one seed against one seed pair.

Pass 36 of `c12-orchestrator` concluded, from `acc512.json`'s two seeds, that the banked
silu+cond-hoist stack carried a **real plDDT deficit at 512 aa** — "silu −0.016657, **5.82x** that
fixture's plDDT seed floor, reproducible bit-exactly over 32 folds... a real signal, not scatter" —
and that **94 % of it was silu's alone**, so silu should not go default-on until adjudicated. It put
that in both live briefs and asked for more seeds.

`c12-compose-fold` ran five seeds (`acc512_seeds.json`, qb2 card 2, forced and during-sampled
1350 MHz, commit `163c1572a`, A/A control 0.0000 Å with identical digests). **The concern does not
survive it.**

## The error: I used the narrowest of ten seed pairs as "the seed floor"

    base-vs-base pair      all-atom Å     ΔplDDT
    s0 vs s1                  5.3502     −0.002860   <- pass 36 used THIS as the floor
    s0 vs s2                  9.8967     −0.011738
    s0 vs s3                 11.1043     +0.010391
    s0 vs s4                 15.7059     −0.010169
    s1 vs s2                  7.9541     −0.008878
    s1 vs s3                 10.9151     +0.013251
    s1 vs s4                 15.0138     −0.007309
    s2 vs s3                 17.4479     +0.022129   <- the actual widest
    s2 vs s4                  9.1618     +0.001569
    s3 vs s4                 21.9058     −0.020560   <- the actual widest, all-atom

The plDDT scatter the sampler produces by itself spans **|ΔplDDT| up to 0.022129**, not 0.00286. One
seed pair is not a floor; it is one sample of a floor.

## The stack, per seed, against that floor

    seed   all-atom Å     ΔplDDT      vs widest seed ΔplDDT
      0       12.4113    −0.017641           0.80x
      1        1.3703    +0.000711           0.03x
      2        0.4525    +0.001390           0.06x
      3        0.2589    −0.001212           0.05x
      4       10.7116    +0.004346           0.20x

**Every seed is below the seed scatter on plDDT**, worst case 0.80x. And four of the five sit at
−0.0012 to +0.0043, i.e. indistinguishable from zero — the −0.0176 that pass 36 built its case on is
**seed 0 alone**. It is bit-exactly reproducible across 32 folds *at that seed*, which is why it
looked solid: determinism within a seed says nothing about variation across seeds. That distinction
is the whole lesson.

## The JSON's own REJECT verdict has the same defect

`acc512_seeds.json` reports `verdict: REJECT`, `stack_over_seed_floor 2.32` — computed as
12.4113 / 5.3502, i.e. the worst stack reading over the **narrowest** seed pair. Against the widest
(21.9058 Å) it is **0.57x**. The stack's per-seed all-atom readings (12.41, 1.37, 0.45, 0.26,
10.71 Å) all sit inside a base-vs-base band that spans 5.35–21.91 Å. So that REJECT is an artifact of
the denominator, and the field should not be quoted as a verdict on this lever.

This is the same chimeric-hinge saturation the campaign already recorded: at seed 0 silu flips the
basin (12.69 Å) while cond-hoist does not (0.52 Å), and at seeds 1–3 the whole stack moves
0.26–1.37 Å. Basin choice is seed-dependent under any bf16 perturbation, which is why 512 aa
whole-molecule RMSD cannot score this and why plDDT — carrying no frame — was the right metric.

## Where that leaves the lever

The decision fixture is unchanged and is `cdk2x2_298`, which the 0.35/0.60 Å bar is written for:
**0.38302 Å worst, 0.477x its own seed floor, A/A 0.0000 Å, HOLD band not PASS band.** At 512 aa the
stack is now inside the seed floor on **both** metrics. So the accuracy objection pass 36 raised is
withdrawn, and the recommendation it implied — flip cond-hoist, hold silu — is **not** supported:
there is no measured basis for treating the two levers differently on accuracy.

**What does NOT change:** the 298 aa verdict is still HOLD rather than PASS, so the pair is a
Moritz-decides default flip and not an automatic one, and nothing here is a reason to flip anything
without him. The perf number is untouched: **+0.6118 s, CI [+0.4127, +0.8109]**.
