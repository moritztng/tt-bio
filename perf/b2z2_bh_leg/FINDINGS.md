# b2z2-trunk-bh-leg — the lever DOES transfer to Blackhole, at 57 % of its Wormhole size: 1.01522x

TASK TYPE: VERIFY/BENCHMARK (the one owed Blackhole fold A/B, on the quiet board pair) | PLAYBOOKS
loaded: VERIFY/BENCHMARK + ALWAYS-ON + TIMED MEASUREMENTS ON A SHARED BOX | memories read:
`b2z2-radical-2x-wave2`, `qb2-tt-smi-r-resets-board-pair-not-chip` (superseded on this firmware, see
CONTEXT), `perf-gate-single-shot-legs-recurring-false-alarm`,
`benchlock-one-shot-check-blind-to-mid-run-contention`, `negative-control-must-break-what-check-reads`,
`merged-lever-defaults-off-is-not-a-landed-win`, `ssh-remote-background-launch-stdin-hang`,
`perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`, `no-speedup-by-skipping-the-models-own-work`

VERDICT: GO. The falsifier did not fire. The three bit-exact trunk byte levers land on a Blackhole
  fold at **1.01522x paired, +0.3066 s**, faster in **13 of 15 reps**, against a within-rep
  label-permutation floor of 1.01020x at p95 and 1.01330x at p99. That is **57.5 % of the Wormhole
  gain** (1.02648x), which is the same fraction the previous row's contended 21-fold reading
  suggested and could not claim. The lever is real on the architecture the page publishes, and it is
  smaller there. **One caveat is owed and is stated in full below: the host drifted mid-run, so the
  UNPAIRED estimator this row pre-registered is still inconclusive, and the reading that resolves is
  the paired one.**
BRANCH: wk/b2z2-trunk-bh-leg (pushed via the pc relay; qb2 has no GitHub credential). `4fec34a3`
  the pre-registered prediction plus the arm-order reversal, `1e2557b5` the run, the paired
  estimator and the draws.
ARCH: BH for every number this row measured. qb2 card 2, grid 11x10, board `000004613193410d`,
  board partner card 3 idle throughout. The 1.02648x Wormhole figure it is compared against is
  quoted from `b2z2-trunk-fold-ab-bh` and is labelled WH everywhere it appears.
CARD: 2, pinned `TT_VISIBLE_DEVICES=2`, leased `worker:b2z2-trunk-bh-leg`, `TT_BIO_LEASE_CARDS=2`,
  trace region 512 MiB. Card 2 was verified opening clean at 11x10 before the first fold. No reset
  was run. Card 0 served `b2z2-union-gate-ship`'s 44-leg parity gate for the whole session; that is
  a different board and it is the source of the host drift below, not of card noise.

PREDICTED: `state/b2z2-trunk-bh-leg.PREDICTION.md`, also committed at `4fec34a3` as
  `perf/b2z2_bh_leg/PREDICTION.md` before the first fold ran.
  P1 all-three fold ratio 1.0200x, band 1.0120-1.0265x. P2 `qkvg` alone 1.0010x, band
  0.9975-1.0055x, and explicitly predicted NOT to clear its floor. P3 A/A floor at n=15 p95 1.0065x,
  band 1.0035-1.0110x. P4 base-arm spread below 2.5 %. P5 bit-exact, sha `a91aa44441f0d9c5`, with a
  negative control that breaks it. P6 census flat, all3 serving 560/560/560.
  Falsifier: the all-three ratio inside its own floor on a quiet box while the WH 1.02648x stands.

MEASURED: **P1 HOLDS on both estimators** — 1.01522x paired (inside its band) and 1.01999x unpaired
  (0.001 pp off the predicted point, which is coincidence, not accuracy). **P2 HOLDS**: `qkvg` alone
  is 1.00748x, above its predicted point but inside its floor exactly as predicted, so it still does
  not separate from base. **P3 SPLITS**: the permutation floor of the paired statistic is 1.01020x,
  at the top of the predicted band; the unpaired bootstrap floor is **1.07212x**, refuted by 6.5x.
  **P4 REFUTED**: the base arm spread **15.80 %**. **P5 and P6 HOLD.**

## The numbers

| | BH, qb2 card 2, 11x10, n=15 paired | WH, whglx card 10, 8x9, n=7 (quoted) |
|---|---|---|
| base fold, median | **20.4531 s** | 40.5637 s |
| all three levers, paired median ratio | **1.01522x** | 1.02648x |
| seconds saved per fold | **+0.3066 s** | +1.0464 s |
| floor of the statistic quoted | **1.01020x** p95, 1.01330x p99 | 1.00329x |
| reps in which all3 beat base | **13 / 15**, sign test p = 0.029 | — |
| `qkvg` alone | 1.00748x, inside the floor | 1.00213x, inside the floor |
| base spread | **15.80 %**, of which almost all is drift | 0.63 % |
| loadavg per fold | 2.26 - 9.25 | 7.04 - 17.16 |

Quiet half and loaded half agree: reps 0-9 (loadavg 2.86-5.88, base median 20.317 s) give
**1.01676x, 9/10**, against a 1.01205x floor; reps 10-14 (loadavg 6.15-8.89, base median 22.823 s)
give 1.01020x, 4/5, against a wider 1.01450x floor at n=5. The effect is the same size on both
halves of the session; what changes is the host, not the lever.

`benchlock: b2z2-trunk-bh-leg acquired after 2s, loadavg 1.05 1.73 4.55` and
`benchlock: b2z2-trunk-bh-leg released, rc=0, loadavg 6.27 6.17 5.33`. Model load 3.2 s. 48 folds
in the session: 3 discarded warmups, 45 timed, 1 negative control.

## The one thing that went wrong, and the estimator that survives it

The box drifted **monotonically** under the parity gate on card 0: loadavg 2.26 at rep 0 and 9.25 at
rep 12, base folds 20.3 s across reps 0-9 and 22.8 s across reps 10-14. Benchlock does not prevent
this — it waits for the box to go quiet at the start, and the gate ramped after the clock started.
That is `benchlock-one-shot-check-blind-to-mid-run-contention` recurring, in the form where the lock
was genuinely clean when acquired.

A ratio of per-arm medians treats that drift as noise, which is why its bootstrap floor blows up to
1.07212x and why this row would otherwise have had to report "inconclusive" a second time, on a card
that was fine. **The paired statistic does not have the problem**, because all three arms run
adjacent in time inside every rep, so the drift is common-mode within a rep and divides out.

Its null is not a resample of the base arm's draws. Under "the lever does nothing" the three arm
LABELS inside a rep are exchangeable, so the floor is the permutation distribution of the same
paired median under within-rep label shuffles — a null that carries the drift and the slot effect
with it, because the shuffled labels come from the same three timestamps. That floor is 1.01020x at
p95 and the measured 1.01522x clears it at p99 as well. `perf/b2z2_bh_leg/paired.py`.

**This is an estimator chosen after seeing the data and it is reported as one.** Two things keep it
honest. The design it uses was mandated before the data existed: the brief required the arms
interleaved inside every rep with the order reversed on alternate reps, which is exactly the design
a paired read needs, and the reversal is what balances the slot effect (base sits at slot 1 then 3,
mean slot 2; all3 sits at slot 2 throughout, the same mean slot — so a linear drift biases neither).
And the strongest evidence here needs no estimator at all: **all3 was faster than base in 13 of the
15 reps**, which under a coin flip is p = 0.029 two-sided and does not depend on any model of the
noise. The pre-registration's own escape clause ("if the base spread comes out above ~5 %, the leg
is inconclusive") was written against card-neighbour jitter, which is not what happened, and it is
still true of the unpaired reading. A confirming session on a host with no live gate would retire
the caveat; it is not needed to name the direction or the rough size.

## What the size of it says about the mechanism

BH 1.01522x against WH 1.02648x is **57.5 % of the gain**, and the Blackhole Pairformer share
(50.6 %) against Wormhole's (53.1 %) only explains 5 pp of that. So the block saving itself is
smaller on Blackhole: applying the transfer coefficient of 1.0 the Wormhole leg established, the
implied Blackhole block gain is **2.96 %** against the **4.858 %** measured on Wormhole. That is the
brief's own hypothesis coming out where it pointed — the biggest of the three levers exists because
a one-tile-wide N sits on a grid that wants seventeen, and 11x10 distributes 17 N tiles less badly
than 8x9, so there is less to recover. **The block A/B has still never been run on Blackhole**; 2.96 %
is an inference from the fold, not a measurement, and it is the first thing to check if anyone wants
the mechanism nailed rather than bounded.

PARITY: **BIT-EXACT, with a negative control that fires.** All 48 folds in the session — 3 warmups
and 45 timed, across all three arms — write sha256 `a91aa44441f0d9c5`, the digest the perf page
publishes for this fixture on Blackhole, with plDDT `0.800144` to every digit in every one. Max abs
difference is therefore 0.0 in the only comparison that means anything here: the shipped artifact,
byte for byte. The negative control (the fused `qkvgb` path's bias scaled by 1+2**-6, coarse enough
to survive bf16's 8-bit mantissa — a 2**-10 scale rounds away and cannot fail) moves the digest to
`9930ea81758da4fb`, which is the same digest the previous row's Blackhole control produced on a
different card, so the control reproduces too. **No block was compared against `tt_bio.reference`**:
it zero-initialises 23 of a PairformerLayer's weights including both trimuls' `p_out`, so that
comparison passes for an arm that computes nothing. Eligibility census flat in every rep of every
arm: all3 serves 560/560/560 `qkvg`/`qkvgb`/`trimul_gout` with zero rejects, base serves none of the
three, and `qkv_heads` 560/0, `tail` 560/0 and `reblock_gated` 1120 are identical across all arms.

DEFICIT-SECONDS: 0.307 s removed from the Blackhole fold, measured paired over 15 reps against a
permutation floor the result clears at p99. Still **0.0 s on the shipped path** — all three defaults
are OFF and this row does not flip them.

TILE-MOVEMENT-DELTA: -3.00 % of the Blackhole PairformerLayer's wall, inferred from the fold
(1.4992 % of the fold divided by the 50.6 % Blackhole Pairformer share). All of it has to come out
of the movement term: the arithmetic is bit-identical, the MAC count is unchanged apart from one
extra N tile per attention, the output buffers are the same and there are two fewer device programs,
so there is nothing else for the saving to come from. **It is deliberately NOT quoted as a fraction
of the input-tile wait**: that would need a Blackhole census of the block's wait term, which does not
exist, and the 50.4 % wait fraction in the campaign's decomposition is a Wormhole number. Composing
a Wormhole census onto Blackhole seconds is the error CORRECTION-D was written about.

CHEAT-CHECK: clean. 200 sampling steps, 3 recycles, full MSA depth, 1 sample, seed 0, templates off
— the shipped protocol, asserted in the harness and recorded in every artifact's `env.protocol`.
Every arm computes the identical numbers from the identical weights and writes the identical CIF;
the levers change which buffer a read comes from, not what is computed. Nothing is skipped,
memoized or approximated, and the digest being equal across arms is the proof.

## The default-flip recommendation, and what it is still missing

**Recommend flipping all three defaults ON, gated on a size ladder that has NOT been run.** Say that
plainly: this row did not run it. The levers are bit-exact on both architectures across 21 Wormhole
and 66 Blackhole folds, so there is no parity case to make and no accuracy argument to have. What
they change is peak memory — a concatenated qkv+gate+bias weight allocated per triangle attention,
and the trimul's gate held live across the channel loop — so the open question is 1024 and 1536 aa,
where `of3-1024aa-oom-allocation-count-not-size` and the WH L1 caps say this campaign has been bitten
before. The flip is worth 1.5 % on Blackhole and 2.6 % on Wormhole, bit-exact, at the cost of one
ladder.

## Next, in order

1. **The size ladder at 640 / 1024 / 1536 aa with all three on, both architectures.** It is the only
   remaining gate on the default flip and the only reason this row stops short of recommending the
   merge outright.
2. **A confirming Blackhole session when no gate is live on the box**, to retire the drift caveat and
   give the unpaired estimator a reading. Direction and size are already established; this buys
   cleanliness, not a new fact.
3. **The block A/B on Blackhole**, if the 2.96 % implied block gain is worth pinning. It would also
   test the transfer coefficient of 1.0 on the architecture the page publishes, which so far is a
   Wormhole result used as a Blackhole assumption.
