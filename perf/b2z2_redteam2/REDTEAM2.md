# b2z2-redteam-v2 — the four claims, re-derived from primaries

Reproduce everything here with `python3 perf/b2z2_redteam2/redteam2.py`. It opens no device, takes
no measurement, and reads only committed artifacts copied into `src/` from the branch that produced
each one. Ratios are quoted with the base they are a ratio of.

| claim | verdict | what it becomes |
|---|---|---|
| 1. single-processor ceiling 1.53x - 1.80x | **REFUTED** | **1.35x - 1.66x**, best-supported point **1.40x - 1.49x** (was 1.611x) |
| 2. the trunk is well explored | **QUALIFIED** | true of re-timing, false of the byte axis, which is now the trunk's ceiling |
| 3. the shard is 1.191x block / 1.0983x - 1.1138x fold | **QUALIFIED** | block is a construction; fold is **1.0857x**, and the low end wins |
| 4. the union 1.09858x, projections 1.2507x | **UPHELD** / **QUALIFIED** | 1.09858x stands (1.09675x position-matched); the stack is **1.206x - 1.251x** |

---

## 1. The ceiling. REFUTED, and the replacement is lower.

**The identity is fine.** `wait_in + wait_out + compute + non-resident` closes on wave 1's capture
to 0.0001 ms, and it closes on a second, independent Blackhole capture of the same block
(`b2z2-bh-tile-census`) that no rung was built from. The two agree term by term to 0.18-0.87 %.
Anyone attacking the ceiling by attacking the identity will fail; I tried and it held.

**What fails is the regime built on it.** Movement-free sets the two wait terms to zero and holds
compute and non-residency, giving 14.9006 ms and 2.4391x on the block. It holds one more thing
constant without saying so: the bytes. The block moves **4,710.8 MB of DRAM reads and 3,338.5 MB of
DRAM writes**, 8.0493 GB, counted per program out of the profiler's own operand shapes. Over today's
36.3438 ms that is 221.5 GB/s. Over 14.9006 ms it is **540.2 GB/s**.

The roofs measured on this part, all committed:

| roof | GB/s | what it is |
|---|---|---|
| 444.9 | the number the campaign quotes everywhere | a **fitted** asymptote out of `t = t_fixed + bytes/BW_eff` (`b2z-arch-deficit`). Above every measured roof below |
| 429.9 | `ttnn.add`, 2 reads + 1 write, 128 MiB | measured, `roofs_p300c_qb2_card2.json` |
| 396.2 / 390.7 | `ttnn.clone`, read N + write N, 192 / 128 MiB | measured, same file. Same currency as the block's count: total interface bytes |
| 390.0 / 269.6 | directional read / write, 64 MB | measured, `roofs_qb2c0_raw.json` |

Reads and writes share one budget rather than overlapping: a concurrent read+write clone reaches
390.7 GB/s of **total** traffic against 390.0 GB/s for reads alone, so concurrency is worth 0.2 %,
not 2x. Total-traffic accounting is the supported reading.

**So the DRAM interface binds before movement-free does, under every roof including the campaign's
own fitted one.**

| roof | block floor | block ratio | vs movement-free 2.4391x |
|---|---|---|---|
| 444.9 fitted | 18.0924 ms | **2.0088x** | binds |
| 429.9 measured | 18.7237 ms | 1.9411x | binds |
| 390.7 measured | 20.6023 ms | **1.7641x** | binds |

The count is conservative in the direction that matters: `census_tiles.py::traffic` counts each
operand tensor once per program, so a multicast operand is counted once rather than once per core.
Real interface traffic is at least this. The constraint binds at least this hard.

**The diffusion step is not DRAM-bound and this is the useful asymmetry.** 3.5947 GB over a
26.400 ms wall is 136.2 GB/s; movement-free at 16.0749 ms asks for 223.6 GB/s, 57 % of the measured
clone roof. The step's 1.6423x survives untouched.

### The corrected rungs

Same construction as `perf/b2z2_orch/ceiling_v2.py`, same phase split (measured base 20.188 s:
trunk 12.400, sampler 5.365, rest 2.423), same held host dispatch (trunk 0.355 s, sampler 0.330 s),
same sampler multiplier. Only the trunk multiplier changes.

| trunk treatment | trunk only | best-supported | + sampler likewise |
|---|---|---|---|
| movement-free (published) | 1.5433x | **1.6110x** | 1.8168x |
| DRAM roof, the campaign's own fitted 444.9 | 1.4278x | 1.4856x | **1.6588x** |
| DRAM roof, measured 429.9 | 1.4070x | 1.4631x | 1.6308x |
| DRAM roof, measured 390.7 | **1.3485x** | 1.3999x | 1.5527x |

The published row reproduces to the digit, including the 1.611x the campaign calls its
best-supported point, which is how I know the reconstruction is faithful.

**Single-processor bracket: 1.35x - 1.66x, best-supported 1.40x - 1.49x.**

### Two consequences

**The constraint is per processor.** Shard the trunk across the p300c pair and each chip moves
4.0247 GB, a 10.3011 ms floor against a 14.9006 ms movement-free block. DRAM stops binding. This
correction lowers the one-processor ceiling and leaves the two-processor one where it was, so it
widens the gap between them.

**On the trunk, the byte axis is not a lever inside the ceiling. It is the lever that raises the
ceiling.** bfp8_b at 0.531x the bytes takes the DRAM floor to 10.94 ms, below movement-free, which
restores 2.4391x as the binding term. Wave 1 dismissed bfp8 on a tile-count model that
`§2-CORRECTION-B` has since refuted, and again at 512 aa accuracy on a scorer `ceiling_v2.py`'s own
comment says sits under the fixture's noise.

**What would have changed my mind.** A DRAM roof above 540.2 GB/s measured on this part, or a byte
count materially below 8.0493 GB per block. The second is the one to go looking for: if a large
share of those bytes is one buffer re-read by many programs, an L1-residency pass removes the
constraint without touching numerics, and `b2z2-byte-axis-reopened` already prices that at 1.188x on
the block from fitted bandwidths.

---

## 2. "The trunk is well explored." QUALIFIED.

The 1.1968x reproduces: REBALANCE 495.5 + OVERLAP 401.5 + DECHAIN 113.0 = 1010.0 core-ms of 6141.137.

It is a bound on **re-timing a fixed dependency graph**. REBALANCE re-places producers near
consumers, OVERLAP fills inter-kernel idle, DECHAIN multicasts instead of forwarding. None of the
three changes what bytes are moved, and the row that measured it says so in its own words: *"That
does not put the fold at its floor generally. It says the remaining levers have to make the tile
cheaper or delete it, not re-time it."*

`CONTEXT §2-CORRECTION-C` turns that into *"So the trunk is well explored and its remaining levers
are small and hard"* and the wave's row allocation, one trunk row against five sampler rows, was set
on it. The "so" does not follow. Sized by the same campaign on the same block and not covered by the
bound: L1-interleaving every DRAM-interleaved read at 1.188x (DERIVED, fitted L1 bandwidth), the
whole byte envelope at 1.826x, and bfp8_b at 0.531x the bytes.

The megakernel's 2.9 % loss is evidence about **tile-pass deletion**, which is a third thing again,
and `b2z2-sampler-stall-split` already explained it: it was built for a block whose wait is 7.0 %
per-program.

Read with §1, this is sharper than a scoping quibble. The trunk's unexplored lever class and the
term that sets the trunk's ceiling are the same thing.

**What would have changed my mind.** A byte-axis measurement on the trunk that came back inside the
1.1968x bound, or evidence that the 8.0493 GB cannot be reduced.

---

## 3. The p300c shard. QUALIFIED, and the provenance question settles low.

Every fold arm on the branch, medians of n=5, all writing one CIF (`dd1c2a12f97772fb`):

| arm | median s | chips | trace | benchlocked | trunk stage s |
|---|---|---|---|---|---|
| stages | **19.7896** | 1 | no | **yes** | 12.2647 |
| single | 19.7677 | 1 | — | **no** | — |
| meshctl | 19.9660 | 2 | yes | yes | 12.3485 |
| meshtrace | 20.0223 | 2 | yes | yes | — |
| mesh_notrace | 21.0692 | 2 | no | yes | 12.3591 |
| sharded | 19.7091 | 2 | yes | yes | 11.7366 |

**What is measured:** one op sharded (`transition_z`), 1.01303x on the fold and **1.05214x on the
trunk stage** against its matched mesh+trace control. Bit-identical.

**1.191x on the block is a construction**, not a measurement: 22.754 ms of replayed per-op slab
timings + 3.011 ms single-track + 5.040 ms of gathers against a 36.702 ms one-chip block. No sharded
block has run. `ceiling_v2.py` labels it MEASURED. It should not.

**The fold range is a denominator question and the answer is the low end.** Both published figures
divide by the published cell:

| figure | what it divides |
|---|---|
| 1.11419x | block-scaling projection ÷ the published 20.113 s cell |
| 1.10340x | measured-anchored projection ÷ the published cell |
| 1.13013x | ÷ the **non-benchlocked** single-chip arm, mesh tax omitted |
| **1.09627x** | block-scaling projection ÷ this row's own benchlocked single-chip base |
| **1.08566x** | measured-anchored projection ÷ the same base |

The published cell is 1.01634x this row's own base. That factor is cross-session tree drift and the
first three rows book it as shard win. The defensible number is **1.0857x**, PROJECTED, mesh tax
charged, against a base measured in the same session on the same card.

The row's own `closing.py` reads committed JSON only and reports 1.0855x labelled PROJECTED, with
the cell quote beside it. The row is clean. The error is in `ceiling_v2.py`, which picks the
non-benchlocked arm as its single-chip base and then quotes against the cell.

**What would have changed my mind.** A sharded fold with all five pair-track ops running, measured
against a benchlocked single-chip base in the same session. Then it is a measurement and not a
projection, and the number is whatever it is.

---

## 4. The union and the projections. UPHELD, then QUALIFIED.

**1.09858x survives everything I could throw at it.** `union_provenance.py` re-runs clean: both the
levers-default-on commit and the published re-cell are ancestors of the branch, the base fold writes
`a91aa44441f0d9c5` (the digest the page publishes), and only the three new flags are toggled.
`union_recheck.py` reproduces the ratio from the 51 raw folds.

My own read of those folds adds one refinement the row's script does not make. The arms sit at
different positions in each rep and the base drifts +0.47 % across positions, so a single global
base median mis-prices each arm slightly. Matched to each fold's own two neighbouring base folds:

| arm | n | median s | vs global base | position-matched |
|---|---|---|---|---|
| HOST | 5 | 18.7860 | 1.04642x | 1.04330x |
| SILU | 5 | 19.1930 | 1.02423x | 1.02489x |
| AKW | 5 | 19.1330 | 1.02744x | 1.02426x |
| **UNION** | 10 | 17.8940 | **1.09858x** | **1.09675x** |

0.17 % apart. The design is properly interleaved, loadavg stayed 1.26-3.32, and every UNION fold is
faster than every base fold. Quote 1.0986x; 1.0968x if you want the strictest construction.

**The 1.2507x projection reproduces and rests on three soft assumptions.** Priced one at a time:

* **The three step levers are multiplied.** 1.03932 x 1.02574 x 1.07444 = 1.1454x. All three delete
  programs from the same 62.9 % per-program constant, so they are overlapping decompositions of one
  term, which is the error `§1-CORRECTION-B` named for the trunk route table. The floor is the
  largest alone, 1.0744x.
* **The mesh tax is not charged.** The same file's `shard()` charges it; `union()` does not.
  Measured, traced, both arms benchlocked: 1.00891x.
* **The shard enters at the constructed 1.1914x block ratio** rather than at the row's own
  measured-anchored 1.08566x fold saving.

| | fold x vs the union's own base |
|---|---|
| as published | 1.2507x |
| all three corrections applied | **1.2058x** |

**The stack is more robust than it looks: 1.206x - 1.251x, a 3.7 % spread.** And the one assumption
I expected to be worst is not. The atom key window is the only step lever measured on both
architectures: its WH step ratio of 1.07444x predicts 0.372 s off the Blackhole fold, and Blackhole
measured 0.525 s. For the lever where the transfer can be checked, WH **under**-predicts by 41 %.

**What would have changed my mind.** A base arm with the wave-1 levers off, a digest mismatch, or
two step levers measured together coming in at their product.

---

## The paragraph for Moritz

Boltz-2 at 512 residues takes **20.113 s** on one Blackhole processor today, and that is what the
perf page publishes. Wave 2 has measured **1.0986x** on top of it, unmerged, which would put the
fold near **18.3 s**. Every lever the campaign has named but not yet built, the three diffusion-step
fusions, adds up to about **1.12x - 1.14x** on one processor. Sharding the trunk across both
processors of the p300c card is worth a further **1.0857x**, projected, and that is a two-processor
result that cannot be quoted in a one-processor column. The ceiling on one processor, with the trunk
running as fast as its own DRAM interface permits and the sampler's math thread never stalling, is
**1.35x - 1.66x**, and the top of that range uses a bandwidth number more generous than anything
measured on the part. **2x on one processor at 200
sampling steps is not reachable**, and the reason is not that we have run out of ideas: it is that
the Pairformer trunk moves 8.05 GB across the DRAM interface every block, which takes 20.6 ms at
the fastest rate this card has ever been measured at, against a 36.3 ms block. Removing every stall
would leave the interface asking for 540 GB/s from a part that delivers 391. The one thing that
changes that answer is moving fewer bytes, by narrowing the trunk's operands or keeping them in L1,
and that is the work that would raise the ceiling rather than climb inside it. The second is the
card: two processors have two DRAM interfaces, the constraint stops binding, and 2x comes back onto
the table as a p300c number rather than a p150a one.

Line by line, the parts of that a skeptic can check: 20.113 s is `site/data/perf-512aa.json` at
`84da2a49b`. 1.0986x is 51 interleaved folds in one process on qb2 card 1, every union fold faster
than every base fold. 8.0493 GB is counted per program out of the profiler's own operand shapes and
is a lower bound. 390.7 GB/s is a `ttnn.clone` on a p300c processor. The 1.35x - 1.66x is the
campaign's own ceiling construction with that roof substituted for an assumption.
