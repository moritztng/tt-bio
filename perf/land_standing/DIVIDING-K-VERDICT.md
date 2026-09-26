# TT_BIO_TRIATT_DIVIDING_K: what it is worth, and what is still unknown

## CURRENT STATE, 2026-09-26 — read this, then skip to the last section

This file grew by pass and several of its middle sections are superseded in place. Everything
that is settled:

- **Reach: exactly one length, 832 tokens.** Enumerated over all 48 tile-aligned lengths from 32
  to 1536 with a model validated against six device outcomes. Ten lengths get no fused pair; the
  HiFi route is `openfold3.trunk` alone and OpenFold3 pads to a multiple of 64, so 832 is the only
  one a user can present.
- **Speed: +50.999 s, 1.6351x** on an OpenFold3 fold at 832, A/A floor 1.306 s, effect 39x floor,
  AICLK sampled DURING every leg at 1350 MHz, arms interleaved, board-pair sibling idle.
- **Blast radius: zero.** The lever opens 832 and changes the pick at no length that serves today.
  Every `_Q_SPLIT_MAX_S` variant tested buys the same one length and moves 3 to 12 shipped picks,
  so **the cap is not a better route to this and should be left alone.**
- **Per call against float64: not a worse kernel.** The route the lever unlocks sits in a
  0.02084-0.02148 band across seven lengths, and the one length where the shipped route serves
  reads 0.021430, inside it. At n=256 the arms are identical to the last digit.
- **The chunked-k question is closed.** A single-chunk k does serve at 832 (`q64 k832`, 384/384,
  zero rejects), and it accounts for 13 % of the structural move, in the same direction. The
  earlier claim that the allocator refuses it is **retracted**.

**The one open blocker, unchanged:** whether the fused route is better or worse than the
materialised one at 832. It needs a fixture where OpenFold3 is confident. Tiled CDK2 apo gives
pLDDT 0.37 and its confidence heads **flip sign between 704 and 832**, so they cannot decide it.

**Recommendation: keep it off by default until that fixture exists.** The cost of waiting is
51 seconds on every fold that pads to 832 tokens.

---


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

**The pair track now has its control, and it is the half that discriminates.**
`TT_BIO_OPM_SMALL_DEPTH` re-factors the outer product mean — *same algebra*, stated as such in
its own comment (`sum_cd a_ic b_jd W_cdk = sum_d b_jd (sum_c a_ic W_cdk)`) — and the outer
product mean writes into `z`. Completing the table:

    arm                              s rel L2      z rel L2
    OFF vs OFF (control)             0.000e+00     0.000e+00
    TT_BIO_SOFTMAX_CKC=1             0.000e+00     0.000e+00   null, does not reach this path
    TT_BIO_APB_CONCAT_HEADS=1        2.118e-01     0.000e+00   benign refactor
    TT_BIO_OPM_SMALL_DEPTH=1         2.124e-01     2.459e-02   benign, SAME ALGEBRA
    TT_BIO_TRIATT_DIVIDING_K=1       2.152e-01     8.724e-02   the lever

**On `s` the lever is indistinguishable from both benign controls** — 21.52 % against 21.18 % and
21.24 %. That half carries no information.

**On `z` the lever is 3.5x the same-algebra baseline** — 8.72 % against 2.46 %. So the pair track
*is* where this lever is distinguishable from a change that provably computes the same thing.

**That is a bound, not a verdict, and the mismatch is worth stating.** `OPM_SMALL_DEPTH` touches
`z` once per block through the outer product mean; the lever's triangle attention reads and
writes `z` in every block, so the two do not have equal opportunity to perturb it. The 3.5x is
indicative of a real difference in kind, not a clean ratio between equally-placed perturbations.
Direction at the trunk remains unknown; the only directional evidence is still per-call, where
the fused route is 8.7-12.6 % closer to float64 than the fall-back it replaces.

## Direction: two reachable references, neither conclusive (SUPERSEDED -- a reference-free instrument works, see the end of this file)

Last pass said a reference needs a one-line site addition. That was wrong — both candidates are
reachable by rebinding from the harness, no repo change
(`perf/land_standing/trunk_dump_sitecustomize.py`, `TT_FORCE_ONE_K_CHUNK` /
`TT_FORCE_ACCURATE_SOFTMAX`). They were run as a **pair**, one from each arm's own kernel family,
because either alone flatters its own side.

               vs refFUSED (one_k_chunk)        vs refMAT (accurate_softmax)
    s    OFF 2.1471e-01 / ON 0.0000e+00     OFF 6.1364e-02 / ON 2.0949e-01
    z    OFF 8.6172e-02 / ON 0.0000e+00     OFF 8.4620e-02 / ON 5.0577e-02

**`refFUSED` is DEGENERATE and must not be scored.** It came back **exactly 0** from the ON arm
on both tensors. The patch did fire — `TriangleAttention.__init__` is provably wrapped — so the
meaning is physical: **the one-chunk config does not fit at 832.** `one_k_chunk` prepends the
padded k of 832, that config is refused, and the ladder falls back to the same `(416, 416)` pick
the lever already takes. A reference that collapses onto the arm it was meant to grade measures
nothing, and "closer to itself" is not a reading.

**So there is no fused-family reference available at this size**, and the one usable reference,
`refMAT`, **splits**: OFF closer on `s`, ON closer on `z`. A single-family reference flatters its
own arm, so a split from one reference is not a direction either way.

**Direction at 832 is therefore not answerable with what the engine exposes**, and the reason is
now concrete rather than a missing hook: the more accurate fused variant *does not fit at the one
length the lever reaches*. `perf/land_standing/direction.py` now refuses to score a reference at
distance 0 rather than reporting it as agreement — it did report it that way on the first run,
which is the mistake this paragraph exists to prevent repeating.

## What would finish this, and why this row stops here

The remaining question is one sentence: **is the lever's larger `z` perturbation toward or away
from correct?** Only a higher-precision trunk answers it, and the machinery already exists —
`_fp32_softmax_attention` takes `host_f64: bool = False` (`tenstorrent.py:4106`), and `af2.py`
already shows the pattern for reaching it from a model:

    tt_bio/af2.py:507   self._softmax_f64 = host_f64_softmax_site("af2.msa")
    tt_bio/af2.py:550   host_f64=self._softmax_f64,

**OpenFold3's trunk has no such site.** Adding one — `host_f64_softmax_site("openfold3.trunk")`,
default off, passed through to the same parameter — is a one-line addition of exactly the shape
`af2.py` already carries, and it would give a reference arm that both OFF and ON could be scored
against. That is what turns the 8.72 % vs 2.46 % bound into a direction.

**This row is not making that change.** It is model surface in `tt_bio/tenstorrent.py` and
`tt_bio/openfold3_trunk.py` added purely to instrument, and the charter's line is that a landing
row lands measured work rather than growing the engine to measure it. Five passes have taken this
lever from "20 of 48 lengths, 1.2894x, accuracy unknown" to a fully characterised object:

    reach        one length, 832, and 704/1088/1216/1472 measured inert
    speed        +50.999 s, 1.6351x, DURING-sampled clock, 39x its A/A floor
    accuracy s   ordinary: 21.5 % against benign controls at 21.2 % and 21.2 %
    accuracy z   8.72 % against a same-algebra control at 2.46 %, direction unknown
    per call     the fused route is 8.7-12.6 % closer to float64 than the fall-back

Everything except that last direction question is settled and reproducible from the artifacts in
`perf/land_standing/out/`. The decision now belongs to whoever owns the accuracy bar, with the
one-line site above as the cheapest way to get the evidence it needs.

## Recommendation (SUPERSEDED 2026-09-26 by the one at the end of this file)

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

## The control that was missing: what main ALREADY ships at the neighbouring length

Measured 2026-09-26 on qb2 card 3, p300c, loadavg ~13 (an accuracy question, so a loud box is
fine — nothing here is a timing claim). `perf/land_standing/neighbour704.sh`,
`perf/land_standing/out/neighbour704/`.

Every earlier pass asked *"is the route the lever unlocks closer to correct?"* and ran out of
references. That was the wrong question to keep asking, because it treats 832 as if the fused
route were a new proposal. **It is not.** OpenFold3's trunk serves the fused route by default at
704, 1088, 1216 and 1472. 832 is the single length where it declines, and it declines because the
shipped k does not divide 832, not because anyone judged it less accurate there. So the right
question is: **how big is the fused-vs-materialised difference at a length main already ships, and
how does 832 compare to it?**

Three legs at 704 tokens, same fixture family, same seed: the shipped default, the same fold
forced onto the materialised fall-back with `TT_BIO_TRIATT_SDPA_HIFI_AB=-openfold3.trunk`, and the
shipped default again as the control.

    leg     route                       fused-hifi stats            pick
    fusedA  shipped default             384 served / 0 declined     (704,704) q352 k704
    matB    -openfold3.trunk            0 served / 0 declined       none (never attempted)
    fusedC  shipped default (control)   384 served / 0 declined     (704,704) q352 k704

    tensor            control A vs C      route A vs B       same quantity at 832
    s  [1,n,384]          0.000e+00         5.177e-02              2.152e-01
    z  [1,n,n,128]        0.000e+00         1.041e-01              8.724e-02

**The control is exactly zero at 704 as well**, so this instrument has no floor to clear here
either.

**On the pair representation the lever's effect at 832 is SMALLER than what main already ships at
704** — 8.724e-02 against 1.041e-01. Whatever standard 704 passes, 832 passes by the same measure.

**On the single representation 832 moves 4.2x further than 704 does, and there is a mechanical
reason rather than a mystery.** The picks are in the table above: at 704 the fused route takes
**k = 704 in one chunk**, so its online softmax makes no running-max rescale and reduces each row
in a single pass. At 832 the lever's ladder picks **k = 416, two chunks**, which adds the rescale.
That extra term is the `s` difference, and it is the thing to name when quoting these numbers.

**Which also corrects something this document implied.** Main does not ship a chunked-k fused
triangle attention anywhere at this site today: 704 picks k704, and 1088, 1216 and 1472 all pick
k equal to the full length. So the lever does not merely extend a shipped route to a missing
length — at 832 it is the only place the trunk would run a two-chunk k ladder. That is a real
distinction and it belongs in the decision.

## Per call against float64, read across lengths so the shipped route is its own reference

Already on disk from an earlier pass and never tabulated this way
(`perf/land_standing/khole_table.py` over `out/khole_fixed/`). Each row grades one call against a
float64 reference, so a length where the **shipped** arm serves gives the distance main already
accepts.

    n     shipped serves?   shipped rel_vs_f64   dividing rel_vs_f64
    256   yes               0.021430             0.021430   <- identical, the built-in null
    288   no                      -              0.021305
    352   no                      -              0.021290
    416   no                      -              0.021187
    704   no                      -              0.020839
    832   no                      -              0.021481
    864   no                      -              0.020889

Two things fall out.

**At n=256 the two arms agree to the last digit**, because the shipped k already divides 256 and
the lever changes nothing. That is a negative control the experiment carries for free: where the
lever should be inert it is exactly inert.

**The route the lever unlocks sits in a 0.02084-0.02148 band across seven lengths, and the one
length where the shipped route serves reads 0.021430 — inside that band.** 832's 0.021481 is
0.24 % away from it. So per call, against float64, the lever's route is not a less accurate
kernel; it is the same kernel at a length that currently gets no kernel at all.

(Read the `shipped_k` column of the source files before reusing them: this bench runs its own
shape, batch 32 / 4 heads / head_dim 32, and its shipped k is 256 where the production trunk at
704 picks 704. The n=256 null and the band are what transfer, not the per-length pick.)

## Direction from the model own confidence (PARTLY WITHDRAWN 2026-09-26 -- the route reading flips sign with size; the chunking half is settled at the end of this file)

`perf/land_standing/plddt832.sh` and the 704 legs above, both on qb2 card 3.

Every earlier attempt at direction needed a reference more accurate than both arms, and all three
candidates failed (the `one_k_chunk` reference does not fit at 832, a single-family reference
flatters its own side, and the exact float64 softmax costs more than a turn). OpenFold3 carries
its own quality estimate and it needs no reference at all. The reason it is usable here is the
control: **at a fixed seed this pipeline is bit-deterministic end to end**, confirmed separately
at both lengths, so a confidence difference has a floor of exactly zero.

    832 tokens, the length the lever reaches
      off1   pLDDT 0.369018   pTM 0.165383   confidence 0.328291
      on     pLDDT 0.363566   pTM 0.166345   confidence 0.324121
      off2   pLDDT 0.369018   pTM 0.165383   confidence 0.328291   <- control, exact

      lever  pLDDT -0.005452 (-1.48 %)   pTM +0.000962 (+0.58 %)

    704 tokens, where main already ships the fused route
      fusedA pLDDT 0.357573   pTM 0.166659   confidence 0.319390
      matB   pLDDT 0.355284   pTM 0.165942   confidence 0.317416
      fusedC pLDDT 0.357573   pTM 0.166659   confidence 0.319390   <- control, exact

      fused  pLDDT +0.002289 (+0.64 %)   pTM +0.000717 (+0.43 %)

**At 704 the reading is clean and it favours the fused route on both heads.** That is the first
unsplit directional result this lever has produced in six passes: where main already chooses the
fused route over the materialised chain, the model agrees with main.

**At 832 the two heads split** — pTM slightly up, pLDDT down by twice as much as pTM moved. A
split is not a direction, and it is the same split-from-one-instrument problem as before, so do
not read the pLDDT drop as a verdict on its own.

**What the pair of readings does establish is which half of the change is unproven.** The two
lengths differ in exactly one thing, the k ladder: 704 takes k=704 in one chunk and makes no
running-max rescale, 832 takes k=416 in two chunks and does. The fused route with a single k
chunk is directionally good. The fused route with a **chunked** k ladder is the part no
measurement here supports, and at 832 it is the only form available, because `one_k_chunk` at 832
is refused by the allocator.

**The honest limit of this instrument, stated rather than buried.** Both fixtures are tiled CDK2
apo folded single-sequence, where OpenFold3 returns pLDDT 0.37 and pTM 0.17 — the model has
essentially no confidence in either structure. Confidence heads read near their floor there, so
these are small differences on a weak fixture. They earn their place only because the measurement
floor is exactly zero and the 704 leg gives an internal positive control, not because a 0.005
pLDDT move on an unconfident target is important by itself.

## Recommendation (SUPERSEDED 2026-09-26 by the one at the end of this file)

**Keep `TT_BIO_TRIATT_DIVIDING_K` off by default, and the reason has changed from "unknown" to
"named".** It is no longer that direction is unanswerable. It is that the change has two halves,
and they do not have the same standing:

- **Serving the fused route where the shipped k does not divide** is supported everywhere it can
  be checked: per call it sits in the same 0.0208-0.0215 float64 band as the route main already
  ships, at 704 it beats the fall-back on both confidence heads, and at n=256 it is provably
  inert.
- **Doing it through a two-chunk k ladder**, which is the only way 832 can be served, adds an
  online-softmax rescale that main runs nowhere at this site today. That is the half the
  evidence does not cover, and it is where 832's larger `s` divergence and its split confidence
  reading both land.

So the cheapest thing that would unblock the 51 seconds is not more folds and not an MSA. It is
**a single-chunk k at 832** — finding out why `one_k_chunk` is refused there and whether the
refusal is a real L1 ceiling or a precondition that could be met. If a single k chunk can be
served at 832, this becomes the same change that is already good at 704 and the accuracy argument
comes with it.

## Why 832 alone gets a chunked k (INCOMPLETE -- a second route already pairs a narrow q with a wide k and is merely capped; see the end of this file)

Read from source and from the recorded rungs, no device time
(`perf/land_standing/rung832.py` over `out/khole_fixed/`).

The section above says the cheapest unblock is a single-chunk k at 832. It turns out 832 is the
**only** length in the measured set that does not already get one:

    n     ladder             served rung        single k chunk?
    256   [256]              q256  k256         yes
    288   [288, 96, 64]      q288  k288         yes
    352   [352, 64]          q352  k352         yes
    416   [416, 256]         q416  k416         yes
    704   [704, 352, 256]    q352  k704         yes   (2 rungs refused on l1_budget first)
    832   [832, 416, 256]    q416  k416         NO    (6 rungs refused on l1_budget)
    864   [864, 288, 256]    q288  k864         yes   (2 rungs refused on l1_budget first)

Every refusal in that column is `l1_budget`, so the single-chunk k is an L1 question at every
length, and 704 and 864 both answer it by dropping q until the wide k fits. **832 cannot, because
it runs out of legal q values first.**

`_tri_att_sdpa_hifi_inner` only pairs a wide k with a q that divides the padded length, and
`_tri_att_q_chunks` builds that list. At 832 the shipped q is 256, which does not divide 832, so
the narrow-q fall-back path applies — and that path is deliberately bounded to `q >= prod / 2`,
i.e. 128, so that a narrow chunk never more than doubles the K/V re-reads. The bound is right for
what it was written for. Its side effect here is decisive:

    832 = 2^6 * 13, so its 32-aligned divisors are 832, 416, 64, 32 -- nothing between 416 and 64
    wider than prod: [832, 416]        both refused on l1_budget against k = 832
    below prod:      64, 32            both below the q >= 128 bound, so never offered
    resulting ladder: (832, 416, 256), and 256 does not divide 832

Its neighbours escape by luck of factorisation. 704 = 2^6 * 11 offers 352, and 864 = 2^5 * 27
offers 288; both sit above the shipped q and both fit L1 against the full-width k.

**So the hole at 832 is not one lever's gap but two levers meeting.** `TT_BIO_TRIATT_DIVIDING_K`
makes a legal k reachable; `TT_BIO_TRIATT_NARROW_Q_FALLBACK`'s re-read bound removes the only q
that could pair with the *wide* one. What the fold then takes, `q416 k416`, is the chunked form —
the half of the change the confidence heads do not support.

**The prediction this makes, stated as a prediction because it is not measured.** q = 64 divides
832, and `q64 k1088` is what OpenFold3's trunk already serves in production at 1088 — a strictly
larger k than 832, so the L1 footprint of `q64 k832` is smaller than a configuration that fits
today. If it serves, 832 gets the same single-chunk fused route as every other length, and the
directional evidence from 704 comes with it instead of being argued around.

**It is also a warning about the bound, not just about this lever.** `q >= prod / 2` was derived
to stop a narrow q from multiplying K/V re-reads, and against the *shipped* k that is exactly
right. Against a *wide* k the trade is different: one k chunk removes the online softmax's
running-max rescale entirely, which is why the neighbouring lengths read better on both
confidence heads. A bound that is correct for one purpose is silently deciding another.

**Next pass, and it is one command's worth of device time**: offer the dividing q values below the
bound when and only when the k chunk is wide, and see whether `q64 k832` serves. If it does,
re-take the confidence pair at 832 — that is the reading that would move this lever from held to
landable.

## The single-chunk k DOES serve at 832 (its Recommendation is SUPERSEDED by the one at the end of this file)

Measured 2026-09-26 on qb2 card 3, p300c, AICLK median 1350 MHz on every leg, loadavg 14-19.
Accuracy and firing only; no timing claim is taken from these legs.
`perf/land_standing/singlek832.sh`, `perf/land_standing/pairhook_sitecustomize.py`,
`perf/land_standing/out/singlek832/`.

### Correction 1: the route that pairs a narrow q with a wide k already exists, and 832 is simply below its cap

The section above blamed `TT_BIO_TRIATT_NARROW_Q_FALLBACK`'s `q >= prod/2` bound for denying 832
a single-chunk k. That is true of the k **ladder**, and it is not the whole mechanism.
`_tri_att_fused_large_s` is a second route, default ON, release-gate green, and its own comment
says its pair is *"a NARROW q against a WIDE k"* which *"the stock ladder never offers"*. It is
what serves OpenFold3's trunk at 1088, 1216 and 1472. It is gated to
`q_len > triatt_sdpa._Q_SPLIT_MAX_S`, which is **1024**, so 832 never reaches it.

The cap's stated justification is that *"at and below it the ladder already lands on a fused pair
(560 of 560 calls at both 512 and 1024 aa)"*. That was checked at 512 and at 1024. **832 is a
counter-example and was never tested**: the ladder lands on nothing there, 0 served of 384.

The cap is env-settable, so this needed no code change. With
`TT_BIO_TRIATT_MASK_Q_SPLIT_MAX=768` and **`TT_BIO_TRIATT_DIVIDING_K` left off**, 832 serves
**384 of 384, 0 declined**, through `_tri_att_fused_large_s` (`large_s_stats [384, 0]`), and it
picks `(416, 416)` — the same pair the dividing-k lever lands on. The outputs are identical to
six decimals on all three confidence metrics:

    arm                                    route              pLDDT      pTM       confidence
    shipped today                          materialised fp32  0.369018   0.165383  0.328291
    TT_BIO_TRIATT_DIVIDING_K=1             fused q416 k416    0.363566   0.166345  0.324121
    cap 768, dividing-k OFF (capA)         fused q416 k416    0.363566   0.166345  0.324121
    cap 768, dividing-k OFF (capC control) fused q416 k416    0.363566   0.166345  0.324121
    cap 768 + pinned pair (pinB2)          fused q64  k832    0.362838   0.166368  0.323544

**So two independent levers reach one configuration at 832**, and whichever lands, the 51 seconds
and the accuracy question are the same object. That matters for the decision: it is not a choice
between two candidates, it is one change with two possible spellings.

### Correction 2: `one_k_chunk` at 832 is NOT refused by the allocator, and chunking is not what moves the structure

An earlier pass recorded that *"the more accurate fused variant does not fit at the one length the
lever reaches"* — that `one_k_chunk` at 832 is refused. **That is wrong, and the mechanism is worth
naming because it is a general trap.** `one_k_chunk` prepends the full-width k to the ladder, and
the ladder then pairs it with the ladder's own wide q values, every one of which is over L1. The
*kernel* was never the problem. Pinned to `(64, 832)` the fused kernel serves **384 of 384 with
zero rejects** — `hifi_picks {"(832, 832)": [64, 832, 2]}`. A full-width k at 832 fits; it had
only ever been offered a q too wide to go with it.

**And the single chunk does not explain the confidence gap, which was the hypothesis.** The
prediction was that 832's larger move came from the two-chunk k's running-max rescale, so removing
it should pull the structure back toward the materialised arm. It does not:

    materialised -> fused (route)          pLDDT -0.005452
    2 k chunks -> 1 k chunk (chunking)     pLDDT -0.000728, the SAME direction, 13 % of the size

The chunking term is real, small, and points away from the materialised arm rather than back
toward it. **The gap at 832 is between the fused route and the materialised route, not between one
k chunk and two.** The hypothesis is refuted by its own experiment.

### What that does to the 704 reading, stated rather than quietly dropped

With chunking eliminated, the same comparison — fused against materialised — reads **+0.002289
pLDDT at 704** and **-0.005452 at 832**, on the same fixture family, the same model, the same
instrument, with a control of exactly zero at both. **The sign flips with size.** A conclusion that
flips sign between two sizes of one fixture family supports neither claim, so the sentence above
calling the 704 reading a direction for the fused route is withdrawn: it is a direction *at 704*,
and 832 reads the other way.

So the confidence heads settle the chunking question and do **not** settle the route question. What
is left needing a confident target is narrower than before but it has not gone away.

### Recommendation, superseding both above it

**Keep it off, and note that the flag to argue about may not be `TT_BIO_TRIATT_DIVIDING_K`.**

- **Closed**: the single-chunk k at 832 is reachable, serves 384/384, and is not the cause of the
  structural move. The earlier allocator-refusal claim is retracted.
- **Closed**: the dividing-k lever and a cap lift produce byte-equal output at 832, so they are one
  decision.
- **Open, and unchanged in kind**: whether the fused route is better or worse than the materialised
  one at 832. The confidence heads flip sign between 704 and 832 on this fixture, so they cannot
  decide it, and an unconfident fixture (pLDDT 0.37) is the reason.
- **Worth someone's attention independent of all that**: `_Q_SPLIT_MAX_S = 1024` rests on a claim
  verified at 512 and 1024 with a counter-example at 832 sitting between them. Any padded length
  below the cap whose shipped q does not divide it is in the same position. That is a reach
  question about a default-ON route, not about this lever.

## The cap is NOT the thing to argue about. The lever is strictly better, measured both ways.

Last pass ended by suggesting `_Q_SPLIT_MAX_S = 1024` deserved attention in its own right, since
832's hole sits below it. Enumerated properly, that suggestion is wrong, and the enumeration also
settles the lever's reach. Host arithmetic only, no device: `perf/land_standing/capreach.py`,
`capblast.py`, `cap768.py`.

### The model is validated before it is believed

`serves(n)` replicates `_tri_att_sdpa_hifi_inner`'s route order — `_tri_att_fused_large_s` first,
then the k ladder — and answers each rung with the engine's own budget model, `plan_for_shape`
plus `cb_fits_l1`. It is checked against six outcomes this row measured on the device:

    n=256  dividing-k off          predicted (ladder, 256, 256)    device: served q256 k256
    n=704  dividing-k off          predicted (ladder, 352, 704)    device: served q352 k704
    n=832  dividing-k off          predicted None                  device: 0 served / 384 declined
    n=832  dividing-k ON           predicted (ladder, 416, 416)    device: served q416 k416
    n=832  cap 768, lever off      predicted (large_s, 416, 416)   device: served q416 k416
    n=1088 dividing-k off          predicted (large_s, 64, 1088)   device: served q64 k1088

**The first version failed 704 and the guard caught it.** The model had left out `one_k_chunk`,
and `tenstorrent.py:10483` passes `tri_att_one_k_chunk=tri_att_sdpa_hifi` — so the only default-ON
HiFi site gets the route and the k order together, and the full-width k is prepended to its
ladder. That is why 704 serves a wide k with the lever off. A sweep built on the first version
would have been confidently wrong about every length.

### Reach: 10 holes in 48 lengths, and exactly one of them is reachable

Every tile-aligned length from 32 to 1536, shipping defaults:

    serves a fused pair:                    38
    no fused pair, falls to materialised:   10  -> 544 608 736 832 928 992 1184 1312 1376 1504
      below the cap:                         6  -> 544 608 736 832 928 992
      above the cap:                         4  -> 1184 1312 1376 1504

The HiFi route is `openfold3.trunk` and nothing else — `boltz2.trunk` and `rf3.tri_att` default
False — and OpenFold3 pads its pair axis to a multiple of 64. Of the ten holes **only 832 is a
multiple of 64**, so on shipping defaults **832 is the only hole any user can present.** That
confirms the earlier five-length sample from a full enumeration instead of a spot check.

### The comparison that decides it

    change                     holes closed   picks CHANGED at lengths that serve today
    cap 1024 -> 768 or 800          1 (832)    3  -- 896, 960, 1024
    cap 1024 -> 32                  4          12 -- including 512, 704 and 1024
    TT_BIO_TRIATT_DIVIDING_K        1 (832)    0

**The lever buys the same length and disturbs nothing. Every cap variant buys it and moves picks
that ship today.** At 896 the cap lift replaces `q224 k896` — one k chunk — with `q448 k224`, four
chunks and a running-max rescale the length does not currently pay. 1024 is one of the two lengths
the cap's own justification was verified on, and it moves too.

**So the cap's stated reason is wrong and its conclusion is right.** The recorded justification,
*"at and below it the ladder already lands on a fused pair"*, is false at six lengths. The real
reason to keep it is visible in `fused_pairs`'s own output: its preference order is tuned for the
regime above 1024 and degrades below it. At 704 it offers `(704, 32)` first — k in 22 chunks —
when the ladder is already serving `(352, 704)` in one. Below the cap the route would frequently
pick worse than what is there.

That is worth correcting in the comment, and it is **not** a reason to change the cap.

### Recommendation, superseding all of the above

`TT_BIO_TRIATT_DIVIDING_K` is the right lever for 832. It is surgical — one length opened, zero
picks moved — and this is now measured across all 48 tile-aligned lengths rather than argued. Its
remaining blocker is unchanged and is the only one left: whether the fused route is better or
worse than the materialised one at 832, which needs a fixture where OpenFold3 is confident.
pLDDT 0.37 on tiled CDK2 apo is not that fixture, and the confidence heads flip sign between 704
and 832, so they cannot stand in for it.

Two things that are NOT blockers and should stop being treated as open:

- the chunked-k question — settled, the single chunk serves at 832 and accounts for 13 % of the
  move, in the same direction;
- the cap — settled, changing it is worse than the lever at every variant tested.
