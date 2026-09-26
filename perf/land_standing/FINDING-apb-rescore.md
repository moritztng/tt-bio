# APB re-scored at the merge tree: it FIRES, it discharges the grid hard stop — and it moves OpenFold3 by 1.254 A

2026-09-26 09:5xZ, `land-standing`. Gate re-scored at merge tree `317725511` (`origin/main`
`29bc7d00c` confirmed an ancestor) with `TT_BIO_APB_CONCAT_HEADS=1` in the environment, not a
second default flip. This is the re-score the 09-20 branch-tip green could not stand in for.

## It fires — counted, not assumed

Unlike `TT_BIO_TRIATT_FUSED_HIFI`, which served 0 of everything at the gate's targets, APB executes:

    pid 2107690 (openfold3 arm)   served 192    declined    0
    pid 2113076                   served 484    declined 4800
    pid 2115893                   served 484    declined 4800

So the arms below are a verdict ON the lever, not merely a regression check around it. Note the
ratio: **484 of 5284 is 9.2 % served** in the two later processes — the guard is
`_APB_CONCAT_HEADS and not atom_level and self.dtype != ttnn.float32`, so most head re-assembly
calls decline. A perf claim on this lever is a claim about that 9 %.

## The grid hard stop is discharged on THIS tree

`l1-budget`, the arm that catches card-dependent output, returns **one md5 across all three grid
classes** with the flag on:

    l1-budget:native   0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS
    l1-budget:8x8      0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS
    l1-budget:narrow   0 clashes   a2e7f667fb45e7c50f7a55f5336e0819   PASS

This matters because the sibling lever `TT_BIO_MM_SHORT_M_BW` was killed by exactly this arm.

## But OpenFold3 moves, and the lever's accuracy case never covered OpenFold3

Every arm still PASSES its bar. That is not the whole story:

    arm             APB=1 (this run)   baseline (APB off)   delta
    openfold3       2.912 / 0.767      1.658 / 0.902        +1.254 A , TM -0.135
    boltz2          1.878 / 0.864      1.827 / 0.902        +0.051 A , TM -0.038
    rf3             1.240 / 0.958      1.239 / 0.958        unchanged
    opendde         1.397 / 0.940      1.395 / 0.940        unchanged
    protenix-v2     1.997 / 0.867      1.999 / 0.867        unchanged
    esmfold2        1.343 / 0.961      1.343 / 0.961        unchanged
    esmfold2-fast   1.708 / 0.918      1.708 / 0.918        unchanged

**Why this is attributable to the lever and not to drift**, checked rather than asserted:

- the baseline appears **twice independently** — the dividing-k gate (174 s) and the fused-HiFi
  gate (138 s) both report openfold3 `1.658 / 0.902` and boltz2 `1.827 / 0.902`, identical to
  three decimals, so these arms are deterministic;
- `git diff --stat 48bf63c2a HEAD -- tt_bio/` is **empty** — `tt_bio` is byte-identical between the
  fused-HiFi gate tree and this one, so the only variable is the env var.

OpenFold3 keeps **0.588 A** of headroom against its 3.5 A bar where it had 1.842 A, and **0.067**
of TM headroom above the 0.70 floor where it had 0.202. It passes. It passes with a third of the
margin it had.

**And the lever's original accuracy evidence does not cover this model.** `c14-land-tail` cleared it
on Boltz-2 (0.2244 A all-atom vs a 0.35 A bar) and OpenDDE (0.1337 A) — the two arms that move
least here. OpenFold3, where it moves most, was not in that case.

## What I am NOT concluding yet

A single gate target at a single seed is not a seed floor. 1.254 A could still sit inside
OpenFold3's own seed spread on this target — this row's own dividing-k work measured a 1.974757 A
seed floor on a different fixture, which is larger than this delta. **So this is not yet "the lever
fails accuracy"; it is "the lever visibly changes the model its case never tested, and the case has
to be extended before it can land default-on."**

The control that settles attribution is an APB-off arm at THIS tree in the same session. The two
baselines above are that control in substance, but they ran at a different hour, so an interleaved
one is owed before any verdict.

---

# VERDICT after the re-score and an interleaved control: do NOT land this default-on yet — 2026-09-26 10:28Z

## The control, interleaved in one session at one tree on one card

    APB=0  round 1   served=0     1.658 / 0.902   PASS   167s
    APB=1  round 1   served=192   2.912 / 0.767   PASS   173s
    APB=0  round 2   served=0     1.658 / 0.902   PASS   148s
    APB=1  round 2   served=192   2.912 / 0.767   PASS   139s

Both arms reproduce exactly. **The control is shown to have moved** — `served=0` on the off legs
against `served=192` on the on legs — so this is not a control gated like its subject.

**The +1.254 A on OpenFold3 is caused by the lever.** Not drift, not the hour, not the tree.

## Full gate at the merge tree: ten arms, all PASS

    openfold3       2.912 / 0.767   <=3.5 / >=0.70   PASS   (baseline 1.658 / 0.902)
    boltz2          1.878 / 0.864   <=3.0 / >=0.75   PASS   (baseline 1.827 / 0.902)
    rf3             1.240 / 0.958                    PASS   unchanged
    opendde         1.397 / 0.940                    PASS   unchanged
    protenix-v2     1.997 / 0.867                    PASS   unchanged
    esmfold2        1.343 / 0.961                    PASS   unchanged
    esmfold2-fast   1.708 / 0.918                    PASS   unchanged
    batch-position                                   PASS
    l1-budget       one md5 across native/8x8/narrow PASS   grid hard stop discharged
    capacity        7.24 GiB / 7.03 GiB              PASS   (baseline 7.23 / 7.00)

**Correction to the earlier section of this file.** I wrote that the lever serves "9.2 %" of head
re-assembly calls. That was the first six counter lines, not the gate. Across all 26 dumped
processes the total is **137112 served / 20640 declined — 86.9 % served**, with 11 processes
recording work. The lever is heavily exercised, which makes these arms a real verdict on it.

Also small but real: capacity's peak DRAM rises with the flag on, 7.23 -> 7.24 GiB and
7.00 -> 7.03 GiB. Far inside the 10.5 and 12.0 GiB budgets, but it is not free.

## Why this is a hold and not a pass

Every arm passes its bar, so a gate-only reading says "ship it". That reading would be wrong here:

- OpenFold3's headroom against the 3.5 A bar falls from **1.842 A to 0.588 A**, and its TM headroom
  above the 0.70 floor from **0.202 to 0.067**. A lever that spends two thirds of a model's margin
  needs to have been aimed at that model.
- **It was not.** `c14-land-tail` cleared this lever on Boltz-2 (0.2244 A all-atom vs a 0.35 A bar)
  and OpenDDE (0.1337 A) — which are, precisely, the two arms that barely move here. The model it
  moves most is the model its accuracy case never tested.
- The gate's RMSD-against-reference is **not** the instrument that case used, and one target at one
  seed is not a seed floor. 1.254 A is smaller than seed floors this fleet has measured on other
  fixtures, so this is emphatically **not** "the lever is wrong".

**What it is:** the lever changes a model its evidence never covered, by an amount that eats most
of that model's gate margin, and nobody has measured OpenFold3's seed floor on this target to say
whether 1.254 A is inside it.

## What would clear it

An OpenFold3 accuracy arm on the same footing as the Boltz-2 and OpenDDE ones already in the case:
the seed floor for this target, and the lever's A/B against it. If 1.254 A sits inside that floor,
this lands. If it does not, the lever ships opt-in with a docs entry, exactly as
`TT_BIO_MM_SHORT_M_BW` did.

That measurement is a contained piece of work and it is the only thing standing between this lever
and a decision. It is NOT "waiting on a human" — the 2026-09-24 blanket grant already removed that.
