# The screen-to-fold transfer factor, from six campaigns of record

`c13-transfer-audit`. CPU only on pc, no device opened, no card lease. Every number below is quoted
from a state document of record with its file and line. Nothing here is re-measured, and nothing in
the history is corrected.

## What is in the table

`table.md` (87 rows) and `transfer_table.json`. A row exists where a campaign wrote down a
**prediction before the measurement** and then **took the measurement**. 40 rows are scored against a
fold (35 of them enter the statistics); 47 against a step, block, layer, op or in-situ fold
second.

`R = measured / predicted` in the row's own pre-registered currency. Ratios are converted to
`ratio - 1` first: a 1.02x lever predicted at 1.01x is 2x wrong, not 1.01x wrong. Where a point and a
band are both registered, the point drives `R` and the band drives the `in band` column.

Six screen kinds, and the sixth is the one the brief did not name:

| kind | what produced the prediction |
|---|---|
| `replay gap` | a difference read off a capture between two replay configurations |
| `roof headroom` | an efficiency fraction against a roof, measured or asserted |
| `count/rate model` | programs, calls or bytes counted from source x a rate |
| `op/block A/B` | an on-device isolated op or block A/B, extrapolated by call count or by Amdahl |
| `fold ratio transferred` | a **measured fold ratio** carried to another arch, tree or stack |
| `novel mechanism` | a kernel or shard that does not exist yet, priced on how it ought to behave |

Six rows are recorded and not scored: three whose prediction was "this is dead" (near-zero
denominator, so `R` is meaningless and the decision is what gets scored), two one-sided floors
(`>= 1.15x`, `>= 1.3x`), and `b2z-sampler-steps`, which is a work-removal cheat under the standing
rule and appears for the record only.

## BY KIND

All 81 scored rows:

| screen kind | n | geomean R | 95 % CI | scatter | wrong sign or nulled | in its own band |
|---|--:|--:|---|--:|---|---|
| `replay gap` | 1 | 0.029 | one point | — | 0/1 | — |
| `roof headroom` | 9 | 0.085 | [0.025, 0.339] | 6.1x | **3/9 = 33 %** | 0/5 |
| `count/rate model` | 34 | 0.889 | [0.593, 1.306] | 2.9x | 6/34 = 18 % | 4/15 |
| `op/block A/B` | 16 | 1.469 | [0.954, 2.561] | 2.7x | 2/16 = 12 % | 3/10 |
| `fold ratio transferred` | 17 | 1.229 | [0.896, 1.652] | 1.9x | 1/17 = 6 % | 5/14 |
| `novel mechanism` | 4 | — | — | — | **4/4 = 100 %** | 0/3 |

The 35 rows scored against a fold, which is the brief's actual question:

| screen kind | n | geomean R | 95 % CI | scatter | wrong sign or nulled |
|---|--:|--:|---|--:|---|
| `replay gap` | 1 | 0.029 | n=1 | — | 0/1 |
| `roof headroom` | 1 | 0.000 | n=1 | — | 1/1 |
| `count/rate model` | 5 | 0.132 | [0.024, 0.723] | 11.1x | **3/5 = 60 %** |
| `op/block A/B` | 12 | 1.543 | [0.903, 3.067] | 3.0x | 1/12 = 8 % |
| `fold ratio transferred` | 15 | 1.131 | [0.841, 1.457] | 1.8x | **0/15 = 0 %** |
| `novel mechanism` | 1 | -2.741 | n=1 | — | 1/1 |

## The result that actually matters: it is not the kind, it is where the RATE came from

`count/rate model` has the largest n and the widest spread, so it is the only kind the corpus can cut
in two. Split it on one question — **was the rate you multiplied measured on the very kernel, link or
capture you multiplied it against, or was it borrowed from somewhere else?**

| `count/rate model` | n | geomean R | 95 % CI | scatter | wrong sign or nulled |
|---|--:|--:|---|--:|---|
| **rate measured on the same thing** | 20 | **0.984** | [0.832, 1.184] | **1.51x** | **0/20 = 0 %** |
| **rate borrowed from elsewhere** | 14 | 0.690 | [0.198, 2.413] | **7.05x** | **6/14 = 43 %** |

Twenty predictions built by multiplying a measured rate by a counted quantity are unbiased to 1.6 %
with 1.5x scatter and not one of them has the wrong sign. Fourteen built by multiplying a *borrowed
constant* — a per-program launch floor, a two-clock "clock-immune" term, an assumed ns/tile or us/call,
a published link fit taken on other shapes — scatter 7x and land the wrong side of zero 43 % of the
time. The borrowed-rate group contains, in order of badness: `b2z2-step-binaryng-fusion` (-0.365),
`b2z2-twopass-loop-bh` (-0.083), `b2z2-sharded-sampler` (-0.047), `b2z2-dual-chip-fold` (-0.014),
`c10-trace-lever` (-0.007), `b2z2-step-adaln-sdpa` (0.000), `c12-host-decomp` (0.024). Six of the
corpus's twelve sign flips are in that list of seven.

**This reframes the other five kinds rather than sitting beside them.** A replay gap is a borrowed
rate with extra steps: it is a cost difference between two configurations, neither of which is the
fold. A roof headroom fraction is a borrowed rate by construction — it is an efficiency taken from
another shape, or a peak nothing on this part reaches. An op/block A/B is a measured rate on the exact
kernel, and it behaves like the measured-rate group (geomean 1.47, 12 % bad). A transferred fold ratio
is a measured rate at the largest available scope, and it is the best in the corpus (geomean 1.23, 0
sign flips in 15 fold readings). A novel mechanism has no rate at all, and it is 4/4 wrong.

So the single rule the corpus supports is: **a screen is worth what its rate was measured on.**

## The rest of the distribution, in three observations

**1. The errors are not one-signed, so a flat discount is the wrong instrument for four of the six
kinds.** `c10-qchunk-sign` predicted +0.030 s at 298 aa and measured **+0.8288 s**, 28x in the
generous direction, on the same screen kind and the same session as levers that came in 3x short
(`c10-qchunk-sign.md:31-32,48`). `c12-compose-fold` exceeded its own pre-registered additive ceiling —
0.4977 s predicted against **+0.5756 s** measured — and named the reason: "an op-level bound is a
bound on the op, not on the fold" (`c12-compose-fold.md:918-934`). `b2z2-elision-bh-measure` applied a
deliberate one-third haircut to a Wormhole-to-Blackhole transfer and came out 2.5x low: "the discount
was backwards" (`b2z2-elision-bh-measure.md:20-31`).

**2. Every prediction in the corpus about a mechanism that did not yet exist as a KERNEL came back
with the wrong sign** — `c12-reblock-delete` (-1.0062 s predicted, **+2.7578 s** added, `:786,808,880`),
`b2z2-mcast-operand-build` (1.035x predicted, **0.93837x**, `:15-23`), `b2z2-arrival-skew-attack`
(1.12x predicted, **0.89157x / 0.79451x / 0.4523x**, `:18-28`), `b2z2-pairformer-megakernel-build`
(0.900 ms/call predicted, **-1.0255 ms/call**, `:44-46`). **A new DATAFLOW is a different story and the
record is clear about it**: `b2z2-trunk-shard-scale-wh` pre-registered block ratios of 1.35x / 1.70x /
1.90x at 2 / 4 / 8 chips for a shard that did not exist and measured **1.342x / 1.640x / 1.942x**
(`:35-47`), and `b2z2-shard-replication-attack` pre-registered its collective at 1.50 / 2.02 ms and
measured 1.70 / 1.97 — "the one number the prediction got nearly exactly right" (`:33-48`). Both had
fitted their own extensive-versus-constant split and their own link curve first. The shards that
flipped sign are the two that were priced onto a *fold* from a published link fit rather than from
their own measurement: `b2z2-sharded-sampler` 0.993x against a predicted 1.10-1.20x fold (`:22-35`)
and `b2z2-atom-shard-wire` 0.99380x against a predicted 1.10-1.16x (`:22-24`).

**3. A stack's composition predicts better than its members.** `b2z2-bh-stack-atom` predicted
**1.1204x** and measured **1.11982x**, 0.05 % apart on 73 folds under benchlock (`:88-96`).
`b2z2-bh-compose-landed` predicted 1.055-1.065x for its union and measured **1.05985x**, while being
4.5x wrong on K2 and sign-wrong on DST inside the same table (`:25-38`). `b2z2-everything-union-wh`
landed inside its P1 band while being wrong on all four stage terms in the same direction, because it
had the trunk at 50 % of the fold when it is 66.9 % — "two errors in opposite directions cancelled,
which is worth saying rather than hiding" (§7 P6). Composition errors partly cancel; that is real, and
it is not a reason to trust the members.

## Two of the four cases in the brief are not what the brief says

The brief's headline table is what this audit was sent to widen, and widening it corrected two of its
four rows.

* **`c12-reblock-delete` -1.0062 s -> +2.7578 s.** Confirmed exactly, `c12-reblock-delete.md:786,808,880`.
  The measurement is an op A/B (7.4369 ms/call against 1.2226 ms/call) multiplied by 560 calls, not a
  timed fold. The row says so.
* **`c10-trace-lever` 3.04 s -> -0.0214 s.** Confirmed, `c10-trace-lever.md:21-29,37`. 20 accepted warm
  folds per size, clock forced and sampled at about 1 kHz during every fold with min = max = 1350 MHz,
  -0.0214 s against a 0.055 s A/A floor.
* **`core_grid` 4.908 s -> 0.156 s is real, but it is not one row's error, and the corpus contains its
  own fix.** The 4.908 s was measured by `c10-fold-census` between two of its own replay configurations
  and that row disclaimed it in advance: "That 4.908 s is the gap between two of MY replay
  configurations. It is NOT a fold saving and must not be entered on the ladder as one"
  (`c10-fold-census.md:208-209`). `c10-core-grid` then declined to carry it and pre-registered
  **0.0 to 0.4 s** instead, from reading the source and finding 5 live bare sites rather than 18. It
  measured **0.1446 s** at 512 aa (`c10-core-grid.md:11-16,57-60`). The 31x miss and the correctly
  discounted prediction of the same lever are both on the record, and the discounted one landed inside
  its band. This is the most useful pair of rows in the audit.
* **`c12-genop-triatt-slack` 0.4544 s -> 0.0184 s is NOT a screen-to-fold transfer.** The row has no
  `MEASURED:` block and ran no arm: 0.4544 s was the inherited roof ceiling and 0.0184 s is the row's
  own, smaller, *pre-registered prediction* (`c12-genop-triatt-slack.md:111-123`, verdict STOP at
  `:145`). Both halves are screens. The 25x and the "95.9 % artifact" are the ratio of two estimates,
  not a measured error, and this row belongs in the survivorship count rather than in the table.

## Predictions built on figures that were later refuted

Flagged rather than averaged in, as the brief requires.

| row | the refuted input | what refuted it |
|---|---|---|
| `c10-trace-lever` 3.04 s | rescaled onto `c10-fixed-cost`'s F = 3.9830 s, read as deletable host time | `c12-profiled-fold` measured 13.2090 s of in-fold device time at 88.8 % coverage, leaving **1.6720 s** as the ceiling on ALL exposed host time, so F exceeds every non-device second in the fold by 2.3110 s (`c12-host-decomp.md:492-498`) |
| `c12-genop-triatt-slack` 0.4544 s | `headroom.py:33` roofed a 2R+1W key against the 1R+1W clone arm, the two roofs differing 1.108x; the lineage includes the 78.24 TFLOP/s ceiling | the ceiling "divided FLOPs by COMPULSORY traffic ... 'not excluded by bandwidth' was true of the compulsory byte count and false of the real one" (`c10-orchestrator.md:692`) |
| `b2z2-orchestrator` wave-2 1.45x-1.75x | the 71.3 ns/tile datum rate, and the 23.841 s cell as the denominator | the datum rate measured **20.82 ns/tile**, 3.42x off (`b2z2-datum-rate-floor.md:11-18`); the denominator was retired (`b2z2-orchestrator.md:18-31`) |
| anything sized off a 15.031 s floor | the floor itself | `roof-triatt-rate-fix.md:35-39` moves it to **12.706 s** and the 17.270 s fold's remaining band from 2.239 s to **4.564 s** |
| `c10-orchestrator`'s `arithmetic_free_traffic` bracket | a 22 % disagreement between two in-house roofs treated as the error bar | it "**underpriced by 1.64 to 1.99x**": bracketed 2.074-2.523 s against a measured **4.1310 s** (`c10-orchestrator.md:692`) |

That last one is the only place in the corpus where a roof screen erred in the generous direction, and
the row that found it says why: the spread between two of your own roofs measures your instruments'
agreement, not their accuracy.

## SURVIVORSHIP — the missing rows, and they are the majority

The table cannot contain a lever that was screened and then dropped. The record says that is the
common case.

* **104 of the 255 state documents across `b2z`, `b2z2`, `roof`, `ttx`, C10 and C12 carry a
  `DEFICIT-SECONDS:` line, and 66 of those 104 (63 %) book exactly 0.0 s.** Much of the remaining 38
  says "explained" or "located" rather than "removed": `b2z-kernel-cycle-census` 5.27 s located,
  `b2z2-msa-layer-census` 3.83 s explained, `b2z2-tile-shape-and-format` 2.06 s located with 0.0 s
  removed, `b2z2-orchestrator` 7.90 s which is the gap itself.
* **`ttx-deadend-catalogue` is the cleanest single reading, because it is a ranked catalogue rather
  than a set of rows: 21 rows, and "exactly one has a fold-level number attached (A4, and its measured
  value today is 1.15-1.17x *slower* because of the fallback)". The three largest entries — 2.85x,
  2.431x and 1.222x — "are op- and block-level"** (`ttx-deadend-catalogue.md:105-115`). One in
  twenty-one, and the one is negative.
* **`c10-lever-corpus` censused the whole corpus: 42 distinct `TT_BIO_*` flags carry a ratio somewhere
  across 1303 state markdown files; its ledger prices 23, carries 6 unpriced with a stated reason and
  refuses 2 as contested; 0 of its 29 ledger rows carries a clock, and no clocked 512 aa lever ratio
  existed anywhere in the corpus** (`c10-lever-corpus.md:39-47`).
* At least 14 rows say in their own words that a lever was priced and not built —
  `b2z2-step-matmul-group` 0.437 s off-fold, `b2z2-fused-activation-sweep` 1.381 s identified with
  0.0 s banked, `b2z2-l1-sharded-residency` 0.853 s as an upper bound on a bound,
  `c12-fused-eltwise-at-pin` 0.1321 s with its fold session discarded by its own 0.9203 A/A floor,
  `c12-genop-triatt-slack`, `roof-fuse-trimul-out`, and nine more.

**What the missing data does to each estimate, by kind, because it is not uniform.**

* `op/block A/B` and `fold ratio transferred` carry little bias. These rows got measured because the
  lever was cheap to arm, not because the screen was large. `b2z2-qchunk-isolated-bh` took a 28-fold
  reading on a lever it had itself pre-priced at 1.0033x (`:34-37`) — not a number anyone chases.
* `roof headroom` and borrowed-rate `count` carry severe, one-signed bias, and **0.085 is the
  optimistic end, not a conservative one.** The roof rows that are in the table are there because
  somebody re-derived them downward and wrote the correction down: `c12-genop-triatt-slack`
  0.4544 -> 0.0184 s, `c12-matmul-key-attribution` 0.7794 -> 0.0584 s on its biggest key and
  0.9200 -> 0.1352 s on the class (`:233-234`, priced down 13.35x and 6.80x),
  `b2z-diffusion-utilization`'s 0.5042 s -> 0.0054 s at 94x (`b2z2-sampler-ceiling-map.md:14-15,115`).
  The roof figures nobody re-derived never produced any number at all and sat in books at face value,
  which is exactly what the 4.908 s did for three campaigns. There is no reason to think the
  un-re-derived ones are better than 0.085.
* `novel mechanism` has n=4 and all four are sign flips, but the four are the ones somebody was willing
  to *build*. The novel mechanisms that were screened and then dropped are not in the table and cannot
  be worse than 4/4 wrong, so this is the one kind where survivorship cannot make the picture darker.

## DISCOUNT, quotable in a brief

| screen | quote this | uncertainty | and say this beside it |
|---|---|---|---|
| a rate measured on the same kernel x a counted quantity | **0.98** | 95 % CI [0.832, 1.184], 1.5x scatter, n=20 | Zero sign flips in 20. This is the only construction in the corpus that can be quoted as a number. |
| `op/block A/B` extrapolated | **1.47** | 95 % CI [0.954, 2.561], 2.7x scatter, n=16 | It UNDER-reads the fold. Do not discount it. 12 % bad, all from structural arms. |
| a measured fold ratio transferred | **1.23** | 95 % CI [0.896, 1.652], 1.9x scatter, n=17 | 0 sign flips in 15 fold readings. Best in the corpus. |
| a borrowed constant x a counted quantity | **0.69, and do not use it** | 95 % CI [0.198, 2.413], 7.05x scatter, n=14 | 43 % land the wrong side of zero. A point factor on a coin is not a factor. |
| `roof headroom` | **0.09** | 95 % CI [0.025, 0.339], 6.1x scatter, n=9 | 3 of 9 had no lever at all. An existence question, never seconds. Survivorship makes 0.09 the optimistic end. |
| `replay gap` | **0.03** | one point, 34x | Not a candidate price at all. `c10-fold-census.md:208` says so itself. |
| `novel mechanism` (a kernel) | **no factor exists** | 4 of 4 wrong sign | Build the cheapest correct version and A/B it, or drop it. |

Mechanical caveats. `replay gap`, `roof headroom` and `novel mechanism` each have n <= 1 **at the
fold**; the nine-point roof figure is carried by in-situ measurements, not by folds, and is labelled as
such in `table.md`. Only `op/block A/B` (n=12) and `fold ratio transferred` (n=15) support a per-kind
factor from the fold subset alone. Nothing here should be read as a distribution for the two
single-point kinds; they are quoted as what they are, one observation each, and the replay gap's 34x
is one number that happens to agree with the roof kind's 11.8x median in direction.

## RECOMMENDATION

**Screening as practised here does have predictive value, and it is entirely concentrated in screens
whose rate was measured on the thing being priced. The screens the campaigns actually spent their
route budgets on are the ones with borrowed rates, and those are worthless — 7x scatter and a 43 %
chance of the wrong sign.**

Four rules a brief can enforce mechanically.

1. **Every predicted second must name the rate it multiplied and where that rate was measured.** If
   the rate came from a different kernel, a different shape, a different replay configuration or a
   two-clock fit, the figure is written as a band with the borrowed-rate factor applied and it never
   enters a book. This one rule would have caught the 4.908 s, the 3.04 s, the 0.4544 s and the
   1.45x-1.75x wave-2 headline before any of them cost a row.
2. **A roof headroom fraction and a replay gap may never be entered as seconds.** Measured transfer
   0.03-0.09, a third with no lever at all. They answer "could there be a lever here", which is worth
   asking and worth nothing more.
3. **A novel kernel gets no number.** 4 of 4 flipped sign, and two of those cost a campaign its
   headline. A novel *dataflow* may be priced, but only off its own fitted extensive-versus-constant
   split and its own measured collective, which is what the two trunk-shard rows did and why they
   landed within 9 %.
4. **Do not discount an op-level A/B onto the fold.** It reads 1.47x geomean with a CI whose lower
   bound is 0.95. Two campaigns applied deliberate haircuts — `b2z2-elision-bh-measure` one-third,
   `b2z2-step-fusion-next-sites` 38 % — and both came out low by 2.4-2.5x. Carry the op number
   straight through and publish the 2.7x scatter beside it.

**The negative half, stated plainly, because the brief asked for it.** Sending every candidate
straight to a fold A/B is NOT the recommendation, and this corpus refutes it: a single fold session on
this fleet cannot resolve a lever below about 1 %. `c12-fused-eltwise-at-pin`'s fold session had an A/A
floor of 0.9203 against a 0.89 % lever — "the two identical arms differed by 9x the effect".
`b2z2-host-device-compose` read 1.06495x against a 5.07 % A/A floor and had to establish direction by
sign rather than by margin (`:33-49`). `b2z2-step-layernorm-fusion`'s fold leg "says 0.78 s and its own
base arm spreads 3.08 s" (`:37`). The op or block A/B is not a worse instrument than the fold; it is a
*more sensitive* one, and it is nearly unbiased. What has to change is not where the measurement is
taken. It is that a roof fraction, a replay gap, a borrowed per-program constant and an unbuilt kernel
stop being allowed to stand in for one.

## Reproducing this

    python3 table.py      # builds transfer_table.json from the quoted rows, prints every row
    python3 stats.py      # per-kind distribution, sign-flip and null rates, fold subset
    python3 discount.py   # bootstrap 95 % CI on the geometric mean of R per kind (20000 resamples)
    python3 ratesplit.py  # the measured-rate vs borrowed-rate cut inside `count`
    python3 render.py     # table.md

No device is opened and no model code is read. Every `pred` and `meas` in `table.py` carries the
`state/` file and line it was copied from; `table.md` reprints them in its last column.
