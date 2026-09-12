# b2z2-trunk-fold-ab-bh — the 1.05x block lever DOES reach the fold. 1.02648x, 1.046 s, Wormhole.

TASK TYPE: VERIFY/BENCHMARK (a fold A/B of three built levers, both architectures) | PLAYBOOKS
loaded: VERIFY/BENCHMARK + ACCELERATE + ALWAYS-ON | memories read: `b2z2-radical-2x-wave2`,
`negative-control-must-break-what-check-reads`, `pc-card0-512aa-fold-nondeterminism`,
`whglx-cross-account-artifacts-strand-off-worktree`, `qb2-p300c-dispatch-granularity-is-a-board-pair`,
`qb2-tt-smi-r-resets-board-pair-not-chip`, `perf-gate-single-shot-legs-recurring-false-alarm`,
`merged-lever-defaults-off-is-not-a-landed-win`, `benchlock-one-shot-check-blind-to-mid-run-contention`,
`bf16-roundtrip-bit-exact`, `ssh-remote-background-launch-stdin-hang`

VERDICT: GO on Wormhole. The falsifier did NOT fire. `b2z2-trunk-byte-round2`'s 1.05106x block
  lever lands on the fold at **1.02648x**, which is what a ~53 % Pairformer share of the fold
  predicts from a 4.858 % block saving: the block-to-fold transfer coefficient is **1.0**, not the
  discount this wave has watched twice. **Blackhole is INCONCLUSIVE and no BH number should be
  quoted from this row.** Two of qb2's four chips will not open at all and the one healthy chip
  shares a board with another row's live job: 21 pooled base folds spread 10.97 % against
  Wormhole's 0.63 %, and at that noise the whole lever sits inside its own floor.
BRANCH: wk/b2z2-trunk-fold-ab-bh (pushed). `2a36953c1` the pre-registered prediction, `8c07511ef`
  the harness, `4a37f8bac` the negative-control fix, `8aa6e3a72` the results.
ARCH: WH+BH. Wormhole = whglx card 10, 8x9. Blackhole = qb2 card 1, 11x10 — the published cell's
  own card. Every number is labelled with the card it came from; nothing is transferred between them.
CARD: whglx card 10 pinned `TT_VISIBLE_DEVICES=10`, leased `worker:b2z2-trunk-fold-ab-bh`, trace
  region 512 MiB. qb2 card 1 same. No reset was run on either box.
  **qb2 cards 2 AND 3 refuse to open**: `Timed out while waiting for active ethernet core
  (x=29,y=25) to become active again`, core-dumped both times. That board pair is wedged and needs
  a reset somebody else has to authorise, because card 0 was serving a live job throughout.

PREDICTED: `perf/b2z2_fold_ab/PREDICTION.md`, committed at `2a36953c1` before the first fold.
  P1 all3 fold ratio 1.0245x, band 1.015-1.032x. P2 qkvg alone 1.0073x, band 1.003-1.011x.
  P3 gain ratio all3/qkvg = 3.50, band 2.0-5.0. P4 bit-exact with a control that breaks. P5 census
  flat. P6 A/A floor under 1.005x.
  Falsifier: the all3 fold ratio inside its own A/A floor while the block ratio reproduces.

MEASURED: **P1 HOLDS at 1.02648x, 0.8 % above its point and inside its band. P2 is REFUTED
  DOWNWARD: qkvg alone banks 1.00213x, below its band and inside the 1.00329x floor — it does not
  reach the fold at all. P3 is REFUTED by a factor of 3.5: the measured gain ratio is 12.42, not
  3.50, so fold gain is NOT proportional to block gain across these levers. P4, P5 and P6 hold.**

## The numbers

| | WH, whglx card 10, 8x9, n=7 | BH, qb2 card 1, 11x10, n=21 pooled |
|---|---|---|
| base fold | **40.5637 s** | 20.7022 s |
| all three levers | **39.5173 s** | 20.3993 s |
| ratio | **1.02648x** | 1.01485x |
| seconds saved | **+1.0464 s** | +0.3029 s |
| A/A floor, same estimator and n | **1.00329x** | 1.01838x |
| verdict | **CLEARS the floor 8x on the gain** | **inside the floor — not a result** |
| `qkvg` alone | 1.00213x, +0.086 s, inside the floor | 1.00025x, +0.005 s, inside the floor |
| base spread | **0.63 %** | **10.97 %** |
| loadavg during the run | 7.04-17.16 | 3.40-13.59 |

The Blackhole column pools two sessions on the same card, same commit, same protocol: n=7
(`fold_ab_512_bh_qb2c1.json`) and n=14 (`..._n14.json`). Read separately they are 1.03436x against a
1.03355x floor and 1.01441x against a 1.02039x floor — neither resolves, and the first one's
apparent clearance by 0.08 pp was noise, which is exactly why it was rerun rather than reported.

Paired reps per arm, arms interleaved base/all3/qkvg every rep in ONE process, one warmup fold
per arm discarded, `os.getloadavg()` read inside every rep. 512 aa `cdk2x2_512.yaml` + its fixed
35-row a3m, **200 sampling steps, 3 recycles**, 1 sample, seed 0, templates off, timed at
`predict_one`. `TT_METAL_DEVICE_PROFILER` absent, not zero. Draws:
`perf/b2z2_fold_ab/out/fold_ab_512_{wh_c10,bh_qb2c1}.json`, scored by `analyze.py`.

## The A/A floor this row had to fix before it could read its own Blackhole leg

The harness's recorded floor shuffles the base arm's reps and takes median-of-3 against
median-of-3. That is the floor of a median-of-3, and the ratio is quoted as a median-of-7, so it
runs ~1.5x too wide: on Blackhole it reads 1.04224x and would have "refuted" a 1.03436x that it
does not address. `analyze.py` bootstraps two independent size-n resamples from the base arm's own
draws instead, which is the null distribution of the statistic actually reported: 1.00329x on WH
and 1.03355x on BH. Same rule CONTEXT's A/A block already states, one level further in — **the
floor has to be the floor of the estimator, and n is part of the estimator.**

## The finding: block savings transfer at 1.0, and lever composition does not

Wormhole is the clean leg (0.63 % base spread over 7 reps) so read the mechanism off it.

* The block A/B removed **4.858 %** of one PairformerLayer's wall. The fold moved **2.58 %**
  (1.0464 of 40.5637 s). That implies a Pairformer share of **53.1 %** of the WH fold, against the
  **50.6 %** the campaign scores on Blackhole (10.22 s of 20.188 s). **The block saving arrived in
  full.** No dispatch tax, no host-time dilution, no overlap absorbing it. A block-level ratio is a
  usable fold prediction if you multiply it by the stage's honest share — that is the thing this
  wave did not know an hour ago, and it applies to every block-level number the campaign quotes.
* **The levers do not compose linearly.** `qkvg` alone is 1.01459x on the block and should have
  been 1.0076x on the fold; it measured **1.00213x**, inside the floor, and on Blackhole it is
  measurably NEGATIVE (0.99505x, -0.102 s). All three together realise ~100 % of their block gain.
  The mechanism is visible in the byte census the parent row published: `qkvg` deletes one of the
  normed pair tensor's THREE readers, so the allocation stays live for the other two and only its
  own read count drops. `qkvgb` deletes the third reader. Until the last reader goes, the tensor is
  still resident and still being re-read, and the intermediate state costs about what it saves.
  **A read-deleting lever pays when it deletes the LAST reader, not a proportional share of them.**
  That is a composition rule, and it is the opposite of the additivity the block A/B measured — the
  block ratios were additive to 0.1 pp and the fold ratios are not.
* The realization coefficient the parent row produced (1.3 to 2.9, "price a deleted read by what
  its reader costs") survives at the fold as a statement about the STACK. It does not survive as a
  statement about any single lever in it.

## Blackhole: the card could not answer the question, and that is the finding

Both BH sessions are bit-exact with a flat census, and both sit inside their floors. The reason is
not the lever, it is the card. **qb2 cards 2 and 3 will not open** — `Timed out while waiting for
active ethernet core (x=29,y=25) to become active again`, core dump, twice, that board pair is
wedged. The only healthy chip is card 1, whose board partner card 0 served another row's job for
the whole run: loadavg 3.40-13.59 and a base-arm spread of **10.97 % against Wormhole's 0.63 %**,
a 17x noise difference on the identical protocol. A lever worth ~2.6 % cannot be read through 11 %
of card noise at any n that fits in a pass — n=21 only buys a 1.018x floor.

What the draws do say, held loosely: the all3 arm is faster in both sessions and the pooled gain is
+0.303 s, which is **57 % of the 0.53 s the Wormhole transfer coefficient predicts at a 20.70 s
fold**. That is consistent with the brief's warning that Blackhole's 11x10 grid redistributes the
qkvgb lever's 17 N tiles differently, and it is equally consistent with noise. **It is not
evidence and must not be quoted.** No Blackhole number from these levers goes on the perf page
until the leg is rerun on an uncontended chip.

PARITY: **BIT-EXACT on both architectures, with a negative control that fires.** All three arms
write the identical CIF in every rep: sha256 `da476491dbb2a847` for all 21 WH folds, `a91aa44441f0d9c5`
for all 21 BH folds, plddt identical to every digit. The negative control — the fused `qkvgb` path's
bias output scaled by 1+2**-6 — moves the sha to `02e3e7c2400537ba` (WH) / `9930ea81758da4fb` (BH)
and the plddt with it. **The control's first version used 1+2**-10 and could not fail**: the bias is
bf16, whose mantissa is 8 bits, so a 2**-10 scale rounds straight back to the original tile. It
passed against a broken arm exactly the way this lineage's `tt_bio.reference` fixture did, and it
was caught before any result was read (`4a37f8bac`). Eligibility census flat in every rep of every
arm: all3 serves 560/560/560 `qkvg`/`qkvgb`/`trimul_gout` with zero rejects, base serves none of
the three, and `qkv_heads`, `tail` and `reblock_gated` are identical across all three arms.

DEFICIT-SECONDS: 1.046 s removed from the fold on Wormhole, measured end to end, n=7 paired,
against a 1.00329x floor. 0.0 s bankable on Blackhole: +0.303 s pooled over 21 reps sits inside a
1.01838x floor on a contended card, so that leg explains and removes nothing. The parent row
LOCATED 0.497 s from block arithmetic against Blackhole's 10.22 s trunk span; the Wormhole fold
delivered 2.1x that, so the located figure was a lower bound where it could be read. Still 0.0 s on
the SHIPPED path: all three defaults are OFF.

TILE-MOVEMENT-DELTA: -9.63 % of the block's input-tile wait, unchanged from the parent row's
measurement and now carried by a fold that moves the predicted amount. The three arms remove 4.858 %
of the block wall with bit-identical arithmetic, an unchanged MAC count apart from one extra N tile
per attention, the same output buffers and two fewer device programs, so the saving has to come out
of the movement term; 4.858 % of a 36.3438 ms block is 1.765 ms, which is 9.63 % of its 18.3366 ms
wait. What this row adds is that the term survives the trip to the fold: 2.58 % of the fold against
a 53.1 % Pairformer share is full transfer, so the movement saving is not being re-absorbed by
overlap somewhere else in the trunk.

CHEAT-CHECK: clean. 200 sampling steps, 3 recycles, full MSA depth, 1 sample, seed 0 — the shipped
protocol, asserted in the harness and recorded in every artifact's `env.protocol`. Every arm
computes the identical numbers from the identical weights and writes the identical CIF; the levers
change which buffer a read comes from, not what is computed. Nothing is skipped, memoized or
approximated.

## What this is now a candidate for, and what that needs

Three bit-exact levers, 1.02648x on a Wormhole fold, no accuracy argument to have. **This is a
default-on candidate and what it needs is the OOM/size ladder, not a parity case** — the levers
allocate one concatenated qkv+gate+bias weight per triangle attention and hold the trimul's gate
across the channel loop, so the question they raise is peak memory at 1024/1536 aa, not correctness.
The ladder has NOT been run and this row does not claim the flip.

## Next, in order

1. **Rerun the Blackhole leg on an uncontended chip.** It is the only open item on this row. qb2
   cards 2 and 3 need a board-pair reset before they can serve it, and that reset cannot run while
   card 0 is busy; card 1 alone cannot answer a 2.6 % question through 11 % of neighbour noise.
   Nothing about these levers goes on the perf page until it lands.
2. **The size ladder** at 640/1024/1536 aa with all three on, both arches. That is the whole
   remaining gate on a default flip.
3. **Retest composition, not levers.** The block A/B measured these three as additive to 0.1 pp and
   the fold says they are not. Any other stacked byte lever this campaign priced by adding block
   ratios needs the same check, `PairWeightedAveraging.proj_z` and the MSA layer first.
4. qb2 cards 2 and 3 are wedged on an ethernet-core timeout and need a board-pair reset, which
   cannot happen while card 0 is serving. That reset is the prerequisite for item 1.
