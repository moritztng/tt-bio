# b2z2-trunk-bh-leg — PREDICTION, written before the first fold

Pre-registered 2026-09-13, before any fold ran on qb2 card 2. Nothing below is adjusted after the
fact; the measured column goes in the state doc.

## What is being measured

The three bit-exact trunk byte levers on `wk/b2z2-trunk-byte-round2` @ `67c2e432b`
(`triatt_qkv.qkvg_heads` / `_QKVG_ENABLED`, `TT_BIO_TRIATT_FUSED_QKVGB`, `TT_BIO_TRIMUL_FUSED_GOUT`),
on a 512 aa Boltz-2 fold, **Blackhole, qb2 card 2, grid 11x10**, 200 sampling steps, 3 recycles,
full MSA depth, seed 0, 1 sample. Three arms interleaved inside every rep, order reversed on
alternate reps, n=15 paired, `benchlock` held, `os.getloadavg()` read inside every fold.

Card 2's board partner is card 3 and card 3 is idle. The 44-leg parity gate is on card 0, board
`0000046131934103`; card 2 is board `000004613193410d`. That is the whole reason this leg can be
read when `b2z2-trunk-fold-ab-bh`'s could not — it had only card 1, which shares the gate's board.

## Numbers the prediction is derived from

* Wormhole fold, n=7 paired, quiet card: **1.02648x**, +1.0464 s of 40.5637 s, A/A floor 1.00329x.
* Wormhole block A/B: the three levers remove **4.858 %** of one PairformerLayer's wall. The fold
  moved 2.58 %, implying a **53.1 % Pairformer share** of the WH fold. Transfer coefficient 1.0.
* Blackhole Pairformer share, campaign census: **50.6 %** (10.22 s of a 20.188 s fold).
* **The block ratio has never been measured on Blackhole.** Every block number in this lineage is
  Wormhole. That is the single largest uncertainty here, and it is the mechanism the falsifier is
  about: the biggest of the three levers exists because a one-tile-wide N sits on a grid that wants
  seventeen, and Blackhole's 11x10 distributes those 17 N tiles differently from Wormhole's 8x9.

## P1 — all-three fold ratio on Blackhole: **1.0200x**, band **1.0120-1.0265x**

Pure transfer of the Wormhole block saving onto Blackhole's stage share is 4.858 % x 50.6 % =
2.458 %, i.e. **1.0252x**, +0.509 s of a ~20.7 s fold. That is the top of the band, not the point,
because the block gain itself is a Wormhole measurement and the grid argument above says Blackhole
should realise less of it, not more. The pooled 21-fold contended reading from the previous row
(1.01485x, held loosely, inside its own floor) is the bottom half of the band. Point 1.0200x =
+0.408 s.

## P2 — `qkvg` alone: **1.0010x**, band **0.9975-1.0055x**, and it will NOT clear its floor

On Wormhole `qkvg` alone banked 1.00213x, inside the floor; on the contended Blackhole card it read
1.00025x pooled and 0.99505x in one session. The composition rule the parent row established says
why: `qkvg` deletes one of the normed pair tensor's three readers, the allocation stays live for the
other two, and the intermediate state costs about what it saves. **A single-lever arm that fails to
separate from base is the predicted outcome, not a defect.** It is in the run as the scale check the
brief asks for: if `qkvg` alone came out at half of all3, the all3 number would be suspect.

## P3 — A/A floor at n=15 on a quiet chip: p95 **1.0065x**, band **1.0035-1.0110x**

The previous row's Blackhole floors were 1.03355x (n=7) and 1.02039x (n=14) on a card whose board
partner was serving a gate. A floor is set by the base arm's own spread; the prediction that card
2+3 being quiet is what was missing is exactly the prediction that this number collapses by ~3x.

## P4 — base-arm spread across 15 folds: **< 2.5 %**

Wormhole quiet: 0.63 %. Blackhole contended: 10.97 %. A quiet Blackhole chip on a host that still
runs a parity gate on another board should land nearer the first.

## P5 — parity: bit-exact, every arm, every rep

All 45 timed folds write sha256 `a91aa44441f0d9c5` — the digest the perf page publishes for this
fixture on Blackhole — with identical plDDT. `torch.equal` / max abs 0.0 in the sense that the CIF
bytes are identical. The negative control (fused `qkvgb` bias scaled by 1+2**-6, which is coarse
enough to survive bf16's 8-bit mantissa) moves the sha to something that is not that digest.

## P6 — eligibility census flat

all3 serves 560/560/560 `qkvg` / `qkvgb` / `trimul_gout` with zero rejects in every rep; base serves
none of the three; `qkv_heads`, `tail` and `reblock_gated` identical across all three arms.

## FALSIFIER

**If the all-three fold ratio sits inside its own A/A floor on a quiet box while the Wormhole
1.02648x stands, the lever does not transfer to Blackhole.** That is a first-class result and it
gets reported as one: the published cell is Blackhole, so a lever that only pays on Wormhole is not
a lever this campaign can ship, and the mechanism would be named — 17 N tiles on an 11x10 grid.

The falsifier only fires on a quiet card. If the base-arm spread comes out above ~5 % (P4 badly
missed), the leg is inconclusive again and says so rather than reporting either verdict.
