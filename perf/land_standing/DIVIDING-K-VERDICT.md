# TT_BIO_TRIATT_DIVIDING_K: what it is worth, and what is still unknown

Written 2026-09-26 by `land-standing` so the next reader does not re-derive any of it. Every
number here was measured on qb2 card 3, p300c; the artifacts are under
`perf/land_standing/out/`.

## What the lever does

`_tri_att_sdpa_hifi_inner` builds its k ladder from `_dividing_k_chunks` instead of the shipped
pick alone. Where the shipped k does not divide the padded length, the fused route today offers
only illegal rungs and declines, falling through to `_fp32_softmax_attention`.

## Reach: one length, and it is a hole rather than a frontier

OpenFold3's trunk is the only site with `tri_att_sdpa_hifi` on by default (`boltz2.trunk`,
`rf3.tri_att` and OpenFold3's confidence/msa/template sites are all `False`). It pads the pair
axis to a multiple of 64, so of the 20 lengths the lever changes only five can ever be presented.
All five measured, by reading `TRIATT_FUSED_HIFI_STATS` and `PICKS` out of real folds:

| tokens | shipped arm | with the lever | verdict |
|--------|-------------|----------------|---------|
| 704  | 384 served, `(704,704) q352 k704`   | identical, CIF byte-identical | inert |
| 832  | **0 served, 384 declined**          | 384 served, `(832,832) q416 k416` | **live** |
| 1088 | 384 served, `(1088,1088) q64 k1088` | not needed | inert |
| 1216 | 384 served, `(1216,1216) q64 k1216` | not needed | inert |
| 1472 | 384 served, `(1472,1472) q64 k1472` | not needed | inert |

At the four inert lengths `_tri_att_fused_large_s` serves before the k ladder is reached. So 832
is a hole between 704 and 1088, which both serve. It is not a size ceiling: 1472 is fine without
the lever.

A user folding 769-832 tokens is the only one who loses the fused route.

## Accuracy: favourable where it can be measured, unanswerable where it cannot

At the kernel, against a float64 reference, the route the lever unlocks is closer to the
reference than the path it replaces at every length checked — 8.7 % to 12.6 %, and 11.4 % at 832
(`perf/land_standing/out/khole_fixed/`).

At the structure, the question is open and **this box cannot close it**. Five folds at 832, three
with the lever on and two without, give

    window        within-arm (seed)   across-arm (seed+lever)   permutation p
    full 1-832        19.574 A            16.143 A                 0.30
    copy 1-298        15.882 A            12.975 A                 0.30
    core 1-150         9.937 A             8.407 A                 0.90

Across-arm distances are if anything smaller than within-arm ones, so there is no separation to
see — but the test has no power to see one either. Five folds give ten labellings, so p cannot
fall below 0.1, and a fold that disagrees with itself by ~10 A across seeds inside a 150-residue
core cannot resolve an effect orders below that.

The cause is the fixture, not the lever. 832 tokens of tiled CDK2 folded single-sequence is about
the least determined input available, and narrowing the window does not rescue it: the floor
falls from 19.6 to 9.9 A and stops. What is needed is a confident target at 832 tokens folded
with an MSA. There is no local ColabFold DB on this box and `~/.boltz/msa` is empty, so that
needs a cached a3m via `RELEASE_GATE_MSA_DIR` or the online server.

## Speed: PRICED, and it is large — +50.999 s, 1.635x at 832 tokens

Taken 2026-09-26 on qb2 card 3, p300c. Four legs alternating OFF, ON, OFF, ON so drift lands on
both arms; clock sampled every second DURING each leg from
`/sys/class/tenstorrent/tenstorrent!3/tt_aiclk`; board-pair sibling (card 2, BDF `0000:03:00.0`)
watched on the same cadence.

    leg  arm   wall_s     AICLK med (n)   sibling max   load med   CIF
     1   off   131.946    1350 (132)      800           10.47      dc66841144ce26c1
     2   on     80.044    1350  (80)      800           11.43      7fcb245cef064658
     3   off   130.640    1350 (130)      800           11.88      dc66841144ce26c1
     4   on     80.545    1350  (80)      800           12.47      7fcb245cef064658

    OFF median 131.293   ON median 80.294   delta +50.999 s   ratio 1.6351x
    A/A floor (the two OFF legs) 1.306 s = 0.99 % of the OFF median
    effect / A/A floor = 39.0x        ON-pair spread 0.501 s

The sibling never left 800 MHz, so the board pair was idle for all four legs, and each arm
reproduced its own CIF digest exactly.

**This retires what this document previously said.** It called the speed term "~1.5x and
unpriced" and repeated the branch comment's "1.009x-1.019x of a round" as if that described this
lever; that figure is about the fused-forward bypass inside a BindCraft 2 round, a different
measurement in a different loop. At the one length this lever reaches, it is worth **51 seconds
on an OpenFold3 fold**.

**One caveat that bounds the ratio, not the seconds.** These folds ran at
`--sampling_steps 20`; production and the release gate use 200. Triangle attention is trunk work
and runs per recycle, not per diffusion step, so the ~51 s absolute saving should carry to a
200-step fold while the RATIO falls, because the extra diffusion steps are time the lever does
not touch. Quote the seconds; re-measure before quoting 1.635x at production settings.

## Accuracy, re-measured on the TRUNK where there is no sampler and no floor

The structure-level test failed for want of resolution: the diffusion sampler sits between the
lever and the CA-RMSD, and on this fixture re-seeding moves the structure 19.6 A, so nothing
below that could be seen. The lever lives in the trunk, and the trunk takes no seed. Measured
there instead, three folds at 832 tokens, same input, same seed
(`perf/land_standing/trunkcmp.py`, capture hook `perf/land_standing/trunk_dump_sitecustomize.py`):

    tensor                OFF vs OFF (control)   ON vs OFF (the lever)
    s  [1,832,384]              0.000e+00              2.152e-01
    z  [1,832,832,128]          0.000e+00              8.724e-02

**The control is exactly zero** — two OFF runs agree bit-for-bit on both tensors and on all six
global scalars (rms, sum, absmax each). So the trunk is deterministic across runs and this
instrument has **no floor to clear at all**, which is the thing the structure-level test could
not give.

**And the effect is large: 21.5 % relative on the single representation, 8.7 % on the pair.**
That is not a rounding difference. It is what two genuinely different kernels accumulate over the
trunk's blocks — at 832 the shipped arm declines every call and runs `_fp32_softmax_attention`,
while the lever serves the fused route, so the arms are not near-neighbours.

**What this does and does not settle.** It settles that the change is *real and large* and that
the earlier "indistinguishable from re-seeding" reading was the fixture's noise floor hiding it,
not evidence of a small effect. It does not settle *direction*: there is no float64 trunk
reference here, so the trunk measurement cannot say whether ON is closer to correct. The only
directional evidence remains per-call, where the fused route is 8.7-12.6 % closer to float64 than
the fall-back it replaces.

Those two can both hold — each call slightly better, the 48-block trajectory still diverging far —
and deciding between them needs a reference, not another arm.

## Calibrating the trunk divergence: 21 % on `s` is the trunk, not the lever

The trunk numbers above say the lever moves things a lot. They do not say whether a lot is
unusual, so two more arms were run at 832 with the same instrument — both default-OFF flags that
change a kernel path without being considered harmful:

    arm                              s rel L2      z rel L2
    OFF vs OFF (control)             0.000e+00     0.000e+00
    TT_BIO_SOFTMAX_CKC=1             0.000e+00     0.000e+00   <- null: does not reach this path
    TT_BIO_APB_CONCAT_HEADS=1        2.118e-01     0.000e+00
    TT_BIO_TRIATT_DIVIDING_K=1       2.152e-01     8.724e-02

**`APB_CONCAT_HEADS` is a one-op head re-assembly of a path the shipped code reaches in four
launches — a reorganisation, not an accuracy trade — and it moves `s` by 21.2 %, against the
lever's 21.5 %.** So the single representation's divergence carries **no information about harm**:
it is what this trunk does to any small change in a kernel path, and last pass's reading of 21.5 %
as "large" was a number without a scale beside it.

**`SOFTMAX_CKC` is a null control and is reported as one.** It drops a known 2.9e-2 softmax error
by 10-60x, so it should have moved something; it returned bit-identical, which means it does not
reach OpenFold3's trunk at 832 — `_fp32_softmax_attention` already passes
`_SOFTMAX_PRECISE_CKC`. It calibrates nothing. Its value is that a third independent
configuration reproduced the baseline bit-for-bit, so the instrument is not inventing differences.

**What is still unmatched is the pair track.** The lever moves `z` by 8.7 % and the head refactor
does not touch `z` at all, so there is no benign control for that half yet. Finding one — a
default-off flag that perturbs the pair path and is agreed harmless — is the cheapest next step
on this question, and it needs one fold.

## Recommendation

Keep it off by default, but the trade has changed and the note should say so. The gain is no
longer "one length and an unpriced speed term": it is **51 seconds of an OpenFold3 fold**, 39x
its own A/A floor, at every input that pads to 832 tokens. What still blocks it is the one thing
this box cannot supply — a confident 832-token fixture on which the structural effect could be
detected if it were harmful.

So the cost of waiting is now known and it is not small. If someone wants this sooner, the
cheapest unblocking step is a cached a3m for one real ~800-residue target, not more device
time.

If someone wants it on sooner, the honest minimum is one confident 832-token fold per arm with an
MSA, showing the structures agree to within that fixture's own seed floor.
