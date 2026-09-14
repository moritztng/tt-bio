# TT_BIO_MSA_LADDER — CA-lDDT against the experimental structure, both arms, both sizes

`roof-msa-ladder` shipped the lever OFF on one reading: paired same-seed all-atom RMSD between the
ladder arm and the shipped 1024-rung arm is 1.32206 A at 512 aa against a 0.60 A kill bar. RMSD
against last-shipped moves when a diffusion trajectory moves, whether or not the answer got worse.
The reading that decides it is distance to the experimental answer, and it says the opposite:

| size | pseudo-domain | CA-lDDT off | CA-lDDT on | on − off | CA RMSD off | CA RMSD on |
|---|---|---|---|---|---|---|
| 298 aa | whole (294 CA) | 0.96340 | 0.96226 | **−0.00114** | 0.79229 A | 0.81387 A |
| 512 aa | copy1, res 1-298 (294 CA) | 0.93585 | 0.93939 | **+0.00354** | 1.11487 A | 1.06215 A |
| 512 aa | copy2, res 299-512 (210 CA) | 0.91317 | 0.91847 | **+0.00530** | 1.23405 A | 1.15617 A |

**At 512 aa, the size that failed the RMSD bar, the ladder is closer to 1HCL than the shipped arm
on both pseudo-domains, by CA-lDDT and by CA RMSD.** At 298 aa it is 0.00114 lDDT worse, which is
6.1 % of the seed floor's width.

Numbers: `native_lddt.json`. Harness: `native_lddt.py`, which imports `native()` and `pair()` from
`perf/b2z2_fusebias/score.py` unmodified and adds no scoring maths of its own.

## The floors these are read against

| floor | 298 whole | 512 copy1 | 512 copy2 |
|---|---|---|---|
| seed floor, TT stack, 4 seeds (lDDT range) | 0.95950-0.97806 | 0.93585-0.94797 | 0.91317-0.92654 |
| width of that range | 0.01856 | 0.01212 | 0.01337 |
| the ladder's move | −0.00114 | +0.00354 | +0.00530 |
| arithmetic-only control (upstream fp32 vs bf16-mixed) | +0.00005 | +0.00012 | +0.00005 |
| a known 0.495 A lever, per-seed lDDT delta (n=4) | −0.00203..+0.00275 | −0.00230..+0.01564 | −0.00383..+0.01615 |

Both arms land inside the seed envelope at all three positions. The ladder's move is not inside the
arithmetic-only control, and it should not be expected to be: that control is a paired same-draw
comparison where the trajectory does not move, and the ladder's does, by 1.28 A of CA. The lever it
does belong beside is `ttbase` -> `ttmain` from `k10-p1-accuracy-anchor`, a real coordinate-moving
lever worth 0.49519 A whose 4-seed mean lDDT cost is 0.00001 and whose single-seed readings span
−0.00383 to +0.01615. The ladder's three readings sit inside that span.

Seed floors, the bf16 control and the reference class all come from
`perf/k10_anchor/out/score.json` on `wk/k10-p1-accuracy-anchor`; nothing is recomputed or retyped.

## Why the 1.32206 A is displacement and not error, measured rather than argued

Negative control, same scorer: take the ladder arm's own CA coordinates and displace them by an
incoherent random walk of exactly the arm-to-arm CA RMSD.

| size | displacement | lDDT after an incoherent move of that size | lDDT the lever actually costs |
|---|---|---|---|
| 512 aa copy1 | 1.282 A | 0.71450 (−0.22489) | +0.00354 |
| 512 aa copy2 | 1.282 A | 0.71996 (−0.19851) | +0.00530 |
| 298 aa | 0.119 A | 0.95861 (−0.00365) | −0.00114 |

A 1.28 A move that was error would cost ~0.22 lDDT. This one costs nothing and at 512 aa pays. The
coordinate move is a coherent trajectory displacement, which is what the standing lessons
`unpaired-cross-stack-rmsd-carries-full-seed-floor` and
`b2z2-seed-basin-flips-under-any-bf16-perturbation` predict and what the negative control now
demonstrates on this specific structure.

## Per-residue, because this leg holds one seed

`lddt_per_residue` already returns the per-residue vector; the global number throws it away. Mean
per-residue delta with a moving-block bootstrap (block 10, 20000 resamples — lDDT is a
neighbourhood metric on a chain, so an i.i.d. interval would be too narrow):

| size | pseudo-domain | mean per-residue delta | 95 % CI | residues improved |
|---|---|---|---|---|
| 298 aa | whole | −0.00132 | [−0.00445, +0.00111] | 19.7 % of 294 |
| 512 aa | copy1 | +0.00446 | [+0.00161, +0.00916] | 39.5 % of 294 |
| 512 aa | copy2 | +0.00610 | [+0.00171, +0.01231] | 46.2 % of 210 |

At 512 aa both intervals exclude zero on the positive side. At 298 aa the interval straddles zero,
so the −0.00114 is not distinguishable from no change even before the seed floor is applied.

## Controls

- **Per-arm bit-exactness.** All 14 folds at 512 aa and 6 at 298 aa hashed: each arm has exactly one
  SHA over every session and every repeat, so 0.00000 of the numbers above is run-to-run noise. pc
  card 0's known matmul nondeterminism did not fire.
- **Instrument, against a known answer.** Arm-to-arm all-atom RMSD through this scorer's text CIF
  parser and Kabsch reproduces `rmsd.py`'s gemmi figures exactly: 1.32206 A at 512 aa and 0.24539 A
  at 298 aa. Two independent parsers and superpositions, five decimals.
- **Cross-leg.** This leg's OFF arm is byte-for-byte the shipped stack at seed 0, and
  `k10-p1-accuracy-anchor` folded that independently as `ttmain-s0` in another session. Scored here,
  the difference is 0.000000 lDDT and 0.00000 A at all three positions, so the two legs are directly
  comparable and neither drifted.
- **Negative control.** The table above: the metric does see a move of this magnitude, by 0.2 lDDT,
  when the move is incoherent.

## What this does not cover

One seed. The 12 folds this leg kept are all seed 0, and no card was held here to add more, so the
per-seed reading is n=1 and the residues carry the statistics instead. The direction is what
survives that: at 512 aa the lever is ahead on both domains, by both metrics, with bootstrap
intervals clear of zero, and the failing reading was a distance to last-shipped that the negative
control shows is displacement.

No Blackhole p300c number is claimed. `roof-msa-ladder` classified this lever as eligibility/layout
kind, which does not transfer between parts, and its 0.795 fold-seconds is a p150a measurement. The
accuracy result here is a property of the arithmetic and does not need re-taking per part; the
**performance** number for qb2 does.

## Recommendation

**GO for default-on staging.** The accuracy objection that held `TT_BIO_MSA_LADDER` off is answered:
at the size that failed, the lever is closer to the experimental structure, not further. What is
still owed before it ships as a fleet default is the p300c performance number, not an accuracy one.

One caveat that is unchanged and matters more for the service than any of the above: the win is
0.000 s for any MSA deeper than 512 rows, and a real ColabFold search routinely returns hundreds to
thousands. The published 35-row fixture is the best case.
