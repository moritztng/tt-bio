# Boltz-2 at 512 aa, radical-2x wave 2: the closing number and the ceiling under it

Re-derived from primaries by `closing.py` in this directory. Every rung is recomputed from the
committed artifact that measured it, and every rung is cross-checked against a second, independent
construction. Run it:

    python3 perf/b2z2_final/closing.py --json out/closing.json

**2x is not reachable at 200 sampling steps on one Blackhole processor, and on two it is arguable
only by assuming something about the diffusion sampler that has never been measured on any
architecture.**

---

## 1. The closing number

**Wave 2's measured contribution to the fold is 1.000x.** Nothing it built survives both the
session's own A/A floor and the 512 aa parity bar.

| what | measured | against | verdict |
|---|---|---|---|
| K2 (fused tri-att qkv+gate), bit-exact | 1.00448x | A/A floor 1.00229x | inside the noise |
| DST-resident gated move, bit-exact | 0.99748x | A/A floor 1.00229x | below 1.000x |
| K2+DST+M_block union, bit-exact | 1.01837x | A/A floor 1.02155x | inside the noise |
| MSA depth ladder | 1.04650x | 0.9765 A at 512 aa | dropped on parity |
| device conditioning | 1.06704x | 0.7714 A at 512 aa | dropped on parity |

MEASURED, BH, qb2 card 1, benchlock, cdk2x2_512 + its fixed 35-row a3m, 3 recycles, 200 steps,
1 sample, seed 0, `predict_one`, cold fold discarded (`b2z2-bh-compose-landed`, two sessions).

**Banked on main: nothing from either wave.** None of `TT_BIO_FUSE_BIAS_STACKS`,
`TT_BIO_MSA_DEPTH_LADDER`, `TT_BIO_DEVICE_CONDITIONING`, `TT_BIO_ATOM_SHIFT_GATHER` or
`TT_BIO_GATE_GRANULARITY` appears anywhere under `tt_bio/` at `b82a5f03`. The 20.079 s the page
publishes is `fc7fed56`'s two diffusion levers, which landed on 2026-09-11, the day before wave 1
opened.

**The campaign's largest measured fold-level factor is wave 1's 1.00814x**, and it is on a branch.
`b2z-levers-default-on` flipped three lever defaults and measured 20.054 -> 19.892 s on qb2 card 0
under benchlock. Of the three, only the fused bias stacks clears the fold's own noise, and it is the
one that is not bit-exact (the arms write different CIF digests). Its 512 aa structural cost has
never been scored. See section 5.

## 2. The denominator, settled by reading the file

`site/data/perf-512aa.json` has published five values for this cell and 23.841 s was never one of
them: 24.395, 24.924, 24.822, 23.504, and 20.079 s since `fc7fed56`. Walk them yourself with
`git log --format=%h -- site/data/perf-512aa.json` and read each blob. 23.841 s is
`b2x-baseline-attrib`'s internal re-measure.

The cell's own `ref` field is the best primary in the repo. It records the base arm it was measured
against (22.195 s, n=6, 1.18 % spread, same process, so the two diffusion levers are 1.1054x), the
card (qb2 card 0, 11x10), the protocol, and one sentence that binds everything after it:
*"read the cell as an upper bound on what the tree does today."*

**The base on that protocol has been measured five times and the sessions disagree by 4.60 %:**

| s | session |
|---|---|
| 19.865 | `b2z-orchestrator`, n=5, spread 1.56 % |
| 19.937 | `b2z2-diffusion-loop-attack`, qb2 c1, n=3, quiet box |
| 20.054 | `b2z-levers-default-on`, qb2 c0, n=3, benchlock at loadavg 1.37 |
| 20.188 | `b2z2-bh-compose-landed` s1, qb2 c1, n=10, A/A 1.00229x |
| 20.788 | `b2z2-bh-compose-landed` s2, qb2 c1, n=10, A/A 1.02155x |

Median 20.054 s. The cross-session spread is 20x the best within-session A/A floor and larger than
every bit-exact lever either wave produced. A paired interleaved A/B is the only instrument that can
see a lever this size, which is exactly what the compose row ran, and its answer was 1.000x.

## 3. The ceiling is 7.4 % lower than published, and the cause is a swapped denominator

`b2z2-redteam-ceiling` published a single-processor ceiling of **1.656x / 1.954x / 2.135x**. Its
arithmetic reproduces exactly: `closing.py` rebuilds all three rungs to the digit from the stall
identity, the phase split and the held host terms.

The defect is not in the arithmetic, it is in the quotation. **The floor was derived from the phase
split of the 18.594 s levered arm and then divided into the 20.079 s unlevered cell.** That books
the campaign's earned 1.0799x a second time, and it inflates every rung by exactly that factor.

| rung | published | self-consistent | move |
|---|---|---|---|
| trunk movement-free, host held | 1.656x | **1.534x** | -7.4 % |
| + sampler, conservative | 1.954x | **1.810x** | -7.4 % |
| + sampler, optimistic | 2.135x | **1.977x** | -7.4 % |

Cross-checked two ways. Building the same rungs from the 18.594 s arm's own split and from the
20.188 s measured base's own split gives ratios that agree to **0.28-0.83 %** on every rung. The
ceiling *as a multiplier* is robust; it is only the mixed quotation that is not.

**Consequence: 2x is not inside the single-processor bracket.** Redteam's headline sentence, "2x is
available on this part", was true only of the inflated reading.

## 4. The route table does not stack, and here is the arithmetic that proves it

A hostile reader's first question is whether the bounds in the campaign's route table can be added
up. They cannot, and the proof needs no new measurement.

Every trunk route acts inside one term: the math thread's input-tile wait, 18.3366 ms of a
36.3438 ms BH Pairformer block. Together with the 3.1066 ms of output-room wait that is 59.00 % of
the block, so a block with no movement at all runs at 0.41x, which is **2.439x**. That is the
envelope every trunk route lives inside.

| route | block ratio if perfect and free |
|---|---|
| byte axis, every delivered byte gone | 1.826x |
| work placement / reordering, three bounds added | 1.197x |
| datum rate to zero (236,118 arrivals x 20.82 ns) | 1.156x |
| CB ring topology, best arm | 1.0015x |
| dest capacity | 1.000x |
| tile-pass deletion | 0.971x (measured SLOWER) |
| **naive product** | **2.531x** |
| **movement-free envelope** | **2.439x** |

The product exceeds the envelope by 3.8 %, which is impossible for independent levers. **They are
not independent, they are overlapping decompositions of the same wait**, and the table is a set of
nested sub-bounds, not a menu. The byte axis alone is 75 % of the envelope; the reordering bound's
largest component is the same input wait; the datum rate is measured to be 26.8 % of it.

**What does stack** is the fold's three wall-clock brackets, because they are disjoint by
construction: trunk, sampler, everything else. That is the only legitimate stack and it is what the
rungs in section 3 do.

**Single-processor ceiling, with its assumptions named:**

* **1.534x** if the trunk's entire movement term goes to zero and nothing else moves. Assumes: the
  0.41 multiplier, measured on BH and confirmed on WH at 0.38, with an error bar of 50 of 232 ops
  whose per-core stall still exceeds their own residency.
* **1.611x** if the trunk goes movement-free and the sampler delivers its **measured** headroom
  (0.4-0.7 s across two named under-filled sites). **This is the best-supported point in the range**
  because nothing in it is transferred between blocks.
* **1.977x** if the sampler is also treated as movement-free. This requires transferring the
  Pairformer block's stall multiplier onto the diffusion step, where `DEVICE COMPUTE CB WAIT FRONT`
  is blank in all 1066 rows of every committed capture, on both architectures. The sampler's math
  thread is resident 57.6 % of the step wall against the block's 88.4 %, so the two kernel mixes are
  measurably different and the transfer is not supported.

## 5. Two parity questions that are open, not closed

**5a. The kill instrument and the published cell disagree about this fixture.**
`perf/b2z2_compose/score.py` kills an arm on a single whole-molecule Kabsch all-atom RMSD against
the base CIF, bar 0.60 A. The published cell's own parity note says of the same fixture:
*"The 512 aa fixture cannot score that -- it is CDK2 fused to a truncated copy of itself and its
free inter-domain hinge saturates whole-molecule RMSD for any reassociation -- so the reading there
is per pseudo-domain: 0.57 and 0.44 A with a 1.9 deg hinge between them."* It then records that the
shipped default moved **8.60 A** on this fixture between two shipped arms with both domains intact
at 1.31 and 1.00 A.

So the currently published cell would fail the bar the campaign uses to kill levers, by 14x. Either
the 0.60 A whole-molecule bar is right and the published cell is disqualified by it, or the cell's
per-pseudo-domain reading is right and the two killed arms have never been scored with it. **The
campaign cannot have both, and neither killed arm was scored per pseudo-domain.** The killed numbers
(0.9765 A and 0.7714 A) are well below the ~8 A saturation band, so this is not a claim that the
kills are artifacts. It is a claim that they rest on an instrument the page itself says cannot read
this fixture, and that resolving it costs no new folds if the CIFs were kept.

**5b. The campaign's only surviving fold-level factor has an unscored 512 aa parity leg.**
`TT_BIO_FUSE_BIAS_STACKS` is not bit-exact (2e-7 mean relative), scores 0.218 A at 298 aa and
passes, and has never been scored alone at 512 aa. The nearest thing to a 512 aa score on it is the
compose row's HOST arm, which contains `_fuse_bias_stack`'s collapse and reads **0.7714 A, a kill**.
`b2z2-host-residual-zero` says in its own words that the fused stack is the dominant numeric change
in that arm. **The experiment that settles it is one A/B: `TT_BIO_FUSE_BIAS_STACKS=1` against base
at 512 aa on the cell, scored both whole-molecule and per pseudo-domain.** Until it runs, 1.00814x
is a speed result with an open accuracy column, not a banked win.

## 6. The two-processor route, attacked rather than adopted

`b2z2-dual-chip-fold` projects 2.61-3.50 s off the trunk. Its measured parts are real: the on-card
link at 20.1 GB/s per direction fitting `t = 10.7 us + bytes / 20.1 GB/s` over a 32x size range, a
block-level mesh tax of **1.0044x** on a real `PairformerLayer`, shard speedups of **1.953x** and
**1.985x** on row-local pair-track ops, bit-exactness under `torch.equal` with a negative control
that rejects, and a shardable fraction q = 0.7925 derived from its own 256->512 scaling. The
projection recomputes from those inputs to the digit: block 36.702 -> 27.296 ms = 1.345x at three
gathers, 25.616 ms = 1.433x at two.

**What has to be true, in decreasing order of risk:**

1. **The mesh tax has to survive a sharded program.** The 1.0044x was measured with everything
   REPLICATED, so both chips ran the identical program on identical data. A sharded block runs a
   different per-core work split and adds a CCL. This is the load-bearing assumption and the row
   does not name it.
2. **The whole shardable fraction has to halve.** Measured for two eltwise ops. Not measured for
   triangle attention, which is 21 % of the block. The trimul contraction, 45 % of the block, reads
   1.213x raw and only halves after a 0.35 ms/op floor is subtracted.
3. **The gather count has to be three.** It is, and the walk in the row's section R is correct: the
   two row-local ops ride free on a half-fresh z, and gather 3 is load-bearing twice. But the walk
   covers the **pair track only**. The projection is carried onto "the trunk" and the trunk stage on
   the cell is 12.400 s, of which ~1.92 s is the MSA stack that was never priced, sharded or gathered.

**Does the ~0.35 ms/op host floor cap it? No, and the row's own answer holds up.** The floor is
present on one chip too, so it cancels in the ratio, and the block projection is built from a real
36.702 ms block measurement rather than from the floor-bearing microbenchmark. The block's own
average is 0.134 ms/program, below the floor. The one host cost the row does not price is the
gathers themselves: 3 x 280 = 840 extra dispatches per fold, which at the 0.0128-0.0343 ms/call
measured for a ttnn call is 11-29 ms. Negligible.

**Is a card-level number fair to publish? As its own row, yes. As this cell, no, and the page's own
economics argue against it either way.**

* The page's headline metric is throughput per dollar. **Two independent folds on the two processors
  is 2.000x the throughput; sharding one fold is 1.149-1.181x.** Sharding makes the metric the page
  leads with strictly worse. It buys latency and nothing else.
* The column is `p150a`, priced at one 300 W board. A p300c has no list price and draws ~600 W, so
  the dollar and watt cells have to be constructed rather than looked up.
* The page already says these workloads do not fill a B200 at 512 residues (68 % util, 379 W of
  1000 W). Doubling our silicon is an argument for letting the GPU have batch 2, which is worth more
  to them than 1.18x is to us.

**Two-processor ceiling.** The shard and the movement-free treatment are not independent: sharding
halves the same bytes the movement-free trunk deletes. At the trunk's own floor there is no movement
left to halve, so the shard halves the residual compute and the 1.411 s/fold of link time becomes
pure added cost: trunk floor 5.293 -> 4.174 s, a gain of 1.12 s rather than 2.6-3.5 s.

* **1.769x** with the sampler at its measured headroom.
* **2.200x** only if the sampler's stall is also assumed away.

## 7. What 2x would still take, in seconds

Against the measured base of 20.188 s, 2x against the published cell is 10.040 s, so **10.148 s has
to come out.** At the measured ceiling of every term:

| term | available |
|---|---|
| trunk, entire movement term deleted | 7.107 s |
| sampler, its two measured under-filled sites | 0.550 s |
| rest, against a dispatch lever already closed at 1.050x | 0.000 s |
| **total** | **7.657 s** |
| **deficit, one processor** | **2.492 s** |
| add the dual-chip shard on top of a movement-free trunk | 1.120 s |
| **deficit, two processors** | **1.372 s** |

The one-processor deficit of 2.49 s exists with the trunk's entire movement term already assumed to
be zero. There is no measured term left that contains it. The two-processor deficit of 1.37 s closes
only by granting the sampler the trunk's stall multiplier, which is the transfer section 3 rejects.

**So: not reachable on one processor at 200 sampling steps. Arguable on two, and only on an
assumption nobody has measured.**
