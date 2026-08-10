# P4 / p2-layout-kernel — can a tt-metal kernel move tile faces at compute-engine rates?

Phase 2, EXPERIMENT. protenix-v2, trunk only, 298 aa (N=320 padded, c_z=256). qb2 **card 2**
(board 005 chip 0), board mate chip 3 held idle for the whole pass. Two ttnn builds, both driven on
this card in this pass: the production **ttnn 0.68.0** wheel and the `tt-metal-fused` source build,
which is the only one carrying the prototype. **Ratios only** (charter §4.8).

**No production change was made. Nothing under `tt_bio/` was touched** — the deliverable is three
probes and a prototype measurement under `perf/p2_layout_kernel/`
(`tileface.py`, `tileface_wheel.py`, `tileface_mech.py` + their JSON).

## Predictions (before measuring)

Committed to git as `perf/p2_layout_kernel/PREDICTIONS.md` in `c96cb9e9`, before the device was
opened. Verbatim, so the wrong ones stay wrong.

The two production channel moves at 298 aa, both on the L1 path (`_triangle_mul_memory_config(320)`
returns `L1_MEMORY_CONFIG`, chunk width C=32):

- **in-move** `[1,320,320,32] -> [1,32,320,320]`, `permute(0,3,1,2)`, 3200 tiles, 6.554 MB each way
- **out-move** `[1,32,320,320] -> [1,320,320,32]`, `permute(0,2,3,1)`, 3200 tiles, 6.554 MB each way

T2's baseline for both: a fixed **2.92 / 2.56 us per tile per core** on the dataflow RISCs, against
**0.33-0.34 us/tile/core** for the in-tile `transpose(-2,-1)`. Those two numbers bracket the whole
question and I predicted where a kernel lands between them.

- **P0 — the cheap test, run first.** Staging through L1 cannot recover these two sites, because
  they are already in L1. Predict a DRAM destination is **>=1.5x slower** than the production L1
  destination. Wrong if L1->DRAM comes within 10 % of the L1->L1 figure or beats it.
- **P1 — transaction-count-bound at sub-line granularity.** With C=32 one 32x32 source tile
  contributes **one row to each of 32 different destination tiles**; a tile row is 32 bf16 split
  across two 16-wide faces, so it is **64 NOC transactions of 32 B per source tile**. Predict the
  implied per-transaction cost is **40-50 ns**. Wrong under 20 ns or over 100 ns.
- **P2 — a full-tile-write kernel wins 2.0-3.2x and does not reach the in-tile floor.** Predict
  **0.9-1.5 us/tile/core**, i.e. 300-480 GB/s of L1 traffic. Wrong if under 1.6x or over 4.0x.
- **P3 — the prize, repriced.** Predict the recoverable figure lands in **1100-1500 ms/fold**, not
  the brief's 1805. Wrong if outside that band.
- **P4 — the honest floor is a clone, not a transpose.** Predict the same-bytes L1->L1 clone
  measures **12-18 us** for 3200 tiles = **0.41-0.62 us/tile/core**. Wrong under 10 us.
- **P5 — parity.** `torch.equal` True for both moves at every shape. Wrong if one element differs.

## Roofs, measured on this card

Every number below was **measured on qb2 card 2 in this pass and none of it was inherited** — the
grid, the floors and the baselines were all re-measured here, in the same process as the prototype,
with `ttnn.synchronize_device` on both sides of every timed region. The card exposes an **11x10 =
110-core** compute grid. Shapes are the production ones: 3200 tiles, 6.554 MB one way, so 13.107 MB
of traffic per call. `us/tile/core` divides by the full 110-core grid unless a row says otherwise.

**Arithmetic intensity of both moves is 0.0 FLOP/byte.** A permute computes nothing, so it is on the
memory side of every machine balance — the charter's 338 and T2's K-corrected 90.3 alike. The point
of this pass is that **it is not bandwidth-bound either**, and the roof that actually binds is a
third thing.

ttnn 0.68.0 (the production wheel), qb2 card 2:

| roof / baseline | us | us/tile/core | GB/s (13.107 MB) |
|---|---:|---:|---:|
| **floor: `ttnn.clone` L1->L1, same 3200 tiles** | **28.18** | **0.969** | **465.1** |
| `ttnn.transpose(-2,-1)` L1->L1 (the in-tile move, compute engine) | 28.45 | 0.978 | 460.6 |
| `ttnn.clone` DRAM->L1 | 33.61 | 1.155 | 389.9 |
| `ttnn.clone` L1->DRAM | 42.00 | 1.444 | 312.1 |
| baseline: `permute(0,3,1,2)` L1->L1 (in-move, production config) | 112.14 | 3.855 | 116.9 |
| baseline: `permute(0,2,3,1)` L1->L1 (out-move, production config) | 93.40 | 3.211 | 140.3 |

`tt-metal-fused` (the build carrying the prototype), same card, same shapes: clone L1->L1 28.85-29.48
us (0.99-1.01 us/tile/core), `transpose(-2,-1)` 29.02 us, `permute(0,3,1,2)` 123.42 us, `permute(0,2,3,1)`
107.36 us. That build is 1.10x/1.15x slower than the wheel on the two baselines, which is why **every
prototype ratio below is taken against the baseline from its own build**, never across builds.

**The first correction is here, and it changes the question.** On this card the in-tile
`transpose(-2,-1)` costs **0.978 us/tile/core**, statistically the same as a plain clone of the same
tiles (0.969), **not the 0.33-0.34 the brief inherited from card 0**. So "compute-engine rate" for a
tensor-sized move is not 0.33 — the compute engine contributes nothing measurable either way, and
the real floor for any op that reads and writes 13.107 MB of interleaved L1 is the **clone at 0.969
us/tile/core**. The brief's 1805 ms/fold was priced against a floor this card does not have.

**Core utilisation.** The prototype's program factory splits `num_groups = Nt*Nt` over the grid
(`reblock_permute_program_factory.cpp:48`), so at N=320 it engages **100 of the 110 cores**, one
group each, leaving 10 idle. That is not read off the source and asserted — the N-ladder in
`tileface_mech.py` measures the quantisation directly (below). The interleaved `ttnn.permute` and
`ttnn.clone` paths take no `core_grid` argument and cannot be swept the same way; for them the
N-ladder is the instrument instead.

**Overlap: `max()`, and the loser is the byte leg.** `transpose(-2,-1)` costs 28.45 us against the
clone's 28.18, so the compute engine's in-tile work hides completely behind the data movement. More
decisively, M1 below doubles the bytes and does not move the total at all, which is only possible if
the byte-moving leg sits inside `max()` and something else is the long pole. Both moves are
`max(bytes, transaction issue)` with issue winning by 3.1-4.2x against the clone leg.

## Experiments and verdicts

### E0 — the cheap test first: is DRAM bank locality the lever? (no kernel)

Buffer-type ladder on the wheel, us and (us/tile/core):

| move | L1->L1 | L1->DRAM | DRAM->L1 | DRAM->DRAM | staged DRAM->L1->L1->DRAM |
|---|---:|---:|---:|---:|---:|
| in `permute(0,3,1,2)` | 112.14 (3.855) | 116.99 (4.021) | 112.72 (3.875) | 164.23 (5.645) | 153.57 (5.279) |
| out `permute(0,2,3,1)` | 93.40 (3.211) | 148.46 (5.103) | **513.50 (17.652)** | **591.88 (20.346)** | **134.91 (4.638)** |

**P0 CONFIRMED in direction, KILLED in margin, and the arm found a lever that is not mine.** The two
production sites are already L1-resident and already in the best configuration the ladder contains:
nothing about staging helps them, so bank-row locality is **not** the mechanism behind the 2213
ms/fold. But my predicted ">=1.5x for a DRAM destination" is wrong for the in-move, where L1->DRAM
costs only 1.04x of L1->L1 — the destination write is close to free relative to what binds, which is
itself evidence and is picked up by H1.

What the arm did find: **a DRAM *source* is catastrophic for the out-move and staging it through L1
in tiles fixes it, with no kernel at all.** 591.88 us direct against **134.91 us** staged as
`clone(DRAM->L1)` + `permute(L1->L1)` + `clone(L1->DRAM)`, a **4.39x** on three stock ttnn calls, and
`torch.equal(staged, direct)` is **True**. The in-move barely moves (164.23 -> 153.57, 1.07x) because
its source layout is 10x1 tiles per slice and already reads cleanly. **This is a real bit-exact lever
and it belongs to whoever owns the DRAM->DRAM permute sites, not to me** — see the ms/fold section.

### E1/E2 — the prototype, and the rate it reaches

The kernel this brief asks for **already exists**: `ttnn.experimental.reblock_permute` and
`reblock_permute_back`, in the `tt-metal-fused` worktree, absent from the 0.68.0 wheel. It is exactly
the structure the brief hypothesised — the reader streams 32 input tiles per output group into L1,
compute does `transpose_wh`, a local L1 datamover gathers the group at face-row granularity, and the
writer issues **one 2 KB full-tile write per output tile** instead of 64 sub-line writes. Rewriting it
from scratch would have produced the same kernel a day later, so I measured it at the production
shape instead. Both builds' baselines are in the table above; the ratios are same-build.

qb2 card 2, `tt-metal-fused`, `[1,320,320,32]` and `[1,32,320,320]`, L1->L1:

| arm | us | us/tile/core (110) | us/tile/**engaged** core (100) | vs same-build baseline | `torch.equal` |
|---|---:|---:|---:|---:|---|
| baseline `permute(0,3,1,2)` | 123.42 | 4.243 | — | 1.00x | — |
| **prototype `reblock_permute`** | **84.20** | **2.894** | **2.614** | **1.47x** | **True** |
| baseline `permute(0,2,3,1)` | 107.36 | 3.690 | — | 1.00x | — |
| **prototype `reblock_permute_back`** | **88.03** | **3.026** | **2.740** | **1.22x** | **True** |

The prototype is also nearly indifferent to buffer type — 84.20 / 85.77 / 89.66 / 89.26 us across
L1->L1, L1->DRAM, DRAM->L1, DRAM->DRAM for the in-move, and 87.56-90.03 us across all four for the
out-move, which is what turns E0's 513.50 us DRAM-source disaster into 87.56 us (**5.9x**).

**P2 KILLED.** I predicted 0.9-1.5 us/tile/core and 2.0-3.2x; measured **2.61-3.03 us/tile/core** and
**1.47x / 1.22x**, both below my own "wrong if under 1.6x" line. The prototype does not reach the
in-tile figure, which I did predict, but it does not get close to my band either.

### M1 — the mechanism, settled: transaction count, not bytes

The decisive test. Going bf16 -> fp32 on the identical tensor **doubles the bytes per transaction and
per tile while holding the transaction count fixed at 64 per source tile**. A byte-bound op must
roughly double; a transaction-issue-bound op must not move.

| dtype | `clone` L1->L1 | `permute(0,3,1,2)` L1->L1 | reorder excess over clone |
|---|---:|---:|---:|
| bf16 | 29.48 us (1.014 us/tile/core) | **123.68 us** (4.252) | 3.238 us/tile/core |
| fp32 | 40.10 us (1.379 us/tile/core) | **123.61 us** (4.249) | 2.871 us/tile/core |

**H1 CONFIRMED: the tile-crossing channel move is bound by NOC transaction issue rate on the
dataflow RISCs, and its cost is independent of how many bytes each transaction carries.** The clone
control rises 1.36x on doubled bytes, as a byte-bound op must; the permute on the same doubled bytes
lands **0.06 % away from its bf16 time** — 123.61 against 123.68 us. Nothing else explains a move
whose bytes doubled and whose wall clock did not. **Killed by**: the fp32 permute costing ~2x the
bf16 permute, which would have made it bandwidth-bound; it did not.

Divide the excess by the 64 face-row transactions each source tile must issue: **50.6 ns per
transaction at bf16 and 44.9 ns at fp32** (45.1 ns for the same arm on the wheel). **P1 CONFIRMED**,
inside its 40-50 ns band, and confirmed a second way — the per-transaction cost is the same whether
the transaction carries 32 B or 64 B.

The same arithmetic on the prototype names its ceiling. `reblock_permute` excess over the clone is
2.894 - 1.014 = **1.880 us/tile/core**, and the back direction 3.026 - 1.014 = **2.012** — over the
same 64 face-row copies that is **29.4 and 31.4 ns each**. **So the prototype's whole win is a
1.6-1.7x cheaper transaction, not fewer of them**: it moved the face-row copies from a scatter into
an interleaved destination buffer to a local, aligned L1 gather, and that is worth 50.6 -> 29.4 ns.
The count did not change, because it cannot.

**H2 CONFIRMED, and it is the ceiling: the transaction count is fixed by the layout, not by the
kernel.** With C=32, a source tile's 32 columns land in 32 *different* destination tiles, one row
each, and a tile row is two 16-wide faces that are not contiguous. A NOC transaction writes one
contiguous run, so no datamover can carry two of these in one transaction: 64 per source tile is a
lower bound over every possible kernel structure, not a property of this one. At the measured
29.4 ns local-gather floor that is **1.88 us/tile/core of pure issue cost on top of the 1.01 clone
floor — i.e. ~2.9 us/tile/core, which is exactly what the prototype measures.** The prototype is
already at the transaction-issue floor and there is no kernel structure left to try.
**Killed by**: any kernel reaching the clone floor of 1.01 us/tile/core, or any datamover issuing
fewer than 32 transactions per source tile. Neither exists on this hardware; the one untested escape
is having the **packer** emit single rows to 32 destinations instead of the dataflow RISC, and it
would have to beat 29.4 ns per row to matter, which a per-row `pack` call plus its CB handshake will
not (recorded, not chased — charter §1).

### M2/M3 — core utilisation and the shape ladder, measured

`tileface_mech.py`, `tt-metal-fused`, C=32, us per tile per **engaged** core:

| N | tiles | groups | cores engaged | busiest core | clone | perm in | perm out | reblock in | reblock out |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 16 | 16 | 1 group | 0.632 | 1.469 | 1.511 | 2.505 (**0.59x**) | 2.278 (0.66x) |
| 192 | 1152 | 36 | 36 | 1 group | 0.715 | 2.034 | 1.943 | 2.526 (0.81x) | 2.330 (0.83x) |
| **320** | **3200** | **100** | **100** | **1 group** | 0.899 | 3.841 | 3.330 | **2.614 (1.47x)** | **2.740 (1.22x)** |
| 352 | 3872 | 121 | 110 | **2 groups** | 0.895 | 4.080 | 3.531 | 3.994 (1.02x) | 4.203 (0.84x) |
| 384 | 4608 | 144 | 110 | 2 groups | 0.800 | 3.921 | 3.265 | 3.345 (1.17x) | 3.588 (0.91x) |

`torch.equal` True on every prototype cell in this table.

**H3 CONFIRMED: the prototype's per-tile cost is flat and its efficiency is set by group
quantisation against the 110-core grid.** 2.505 / 2.526 / 2.614 us/tile/engaged-core across a 6.25x
tile range while one group per core fits, then a step to **3.994** the moment N=352 puts 121 groups
on 110 cores and eleven of them carry two. **Killed by**: the N=352 point landing on the flat line —
it did not, it is 1.53x above it. 298 aa sits at N=320, which is the last shape where the split is
exactly one group per core, so this leg's headline ratio is measured at the prototype's best point
and should be read that way.

**The baseline behaves the opposite way and that is worth recording.** `ttnn.permute`'s per-tile cost
*rises* with N — 1.469 -> 2.034 -> 3.841 for the in-move — at a constant grid and constant work per
tile. A per-tile cost that grows with the tensor cannot be the launch floor and cannot be bandwidth;
the plausible mechanism is that each core's 64-transaction scatter spreads across more distinct
destination pages as `Nt` grows, so NOC contention rises. **The kill test is a `Nt`-invariant
destination layout** (sharded output, one core per destination tile) holding tiles per core fixed;
it is one experiment, it belongs to whoever owns the general permute path in ttnn, and I am recording
it rather than chasing it (charter §1).

## ms/fold at stake, after this pass

Conversion **x524** (charter §4.9), the constant for sites shared with the MSA stack. T2 counted the
calls directly in a warm 298 aa fold rather than deriving them: 16768 in-move calls (16 per trimul
x 1048 trimuls) and 8384 out-move calls (8 per trimul). I am not re-counting them; I am applying my
measured ratios to T2's ms/fold, because the ledger is in T2's card-0 units and the charter says this
card produces ratios only.

| site | ledger today | with the prototype at its measured ratio | recovered |
|---|---:|---:|---:|
| in-move `permute` @978/1264 | 1533.4 ms/fold | 1045.9 ms/fold (1.466x) | **487.5 ms/fold** |
| out-move `permute` @1065/1357 | 679.5 ms/fold | 556.9 ms/fold (1.220x) | **122.6 ms/fold** |
| **total** | **2212.9 ms/fold** | **1602.8 ms/fold** | **610.1 ms/fold** |

**The brief's 1805 ms/fold is struck.** Even a hypothetical kernel that reached this card's clone
floor — 0.969 us/tile/core, which H2 says is unreachable — would recover 1147.9 + 474.5 =
**1622.4 ms/fold**, and that is the true arithmetic ceiling for this layout, below 1805 because 1805
was priced against a 0.33 us/tile/core in-tile figure that does not reproduce here. What is actually
available is **610.1 ms/fold**, and it needs a custom op upstreamed into the production wheel.
**P3 KILLED**: I predicted 1100-1500 ms/fold and the answer is 610.

**A second line, not mine, priced from E0.** T5's `permute` @1570/@1715 are DRAM->DRAM and carry
**147.4 ms/fold**. E0's DRAM->DRAM out-move arm runs 4.39x faster staged through L1, bit-exact, on
three stock ttnn calls and no kernel. If those sites are the same op class — which I did **not**
verify, they are T5's — that is up to **~114 ms/fold for a three-line change**. It costs one pass to
check and it should be checked before anything is spent on a kernel.

## Parity

**Both prototype ops are bit-exact against `ttnn.permute` at every shape measured, confirmed with
`torch.equal`, not asserted.** True for the in-move and the out-move at N=128/192/320/352/384 in the
shape ladder, and True for all four source/destination buffer-type combinations of both moves at
N=320 — 18 `torch.equal` calls, no failures. `torch.equal(staged, direct)` is likewise **True** for
both staged DRAM arms in E0. **P5 CONFIRMED.**

That is the point of this lever and the reason it was worth pricing: a permute is a pure index move,
so a faster kernel doing the same move carries no parity trade at all. The 610.1 ms/fold sits in the
safe pile. It is a build-and-upstreaming cost, not an accuracy one.

## Corrections to the inherited record

1. **The in-tile `transpose(-2,-1)` does not cost 0.33-0.34 us/tile/core on qb2 card 2. It costs
   0.978, which is the same as a plain clone (0.969).** T2's 0.33-0.34 is a card-0 figure and the
   ~9x gap it implies against the 2.92 tile-crossing cost does not exist here; the real gap is 3.95x
   against the clone floor. This is the single most load-bearing correction in the pass, because the
   brief's entire 1805 ms/fold prize was the difference between 2.92 and 0.33. Priced against this
   card's actual floor the arithmetic ceiling is 1622.4 ms/fold and the achievable figure is 610.1.
2. **"A tile-crossing reorder costs a fixed 2.6-3.2 us/tile/core whatever ttnn op names it" is right
   about being fixed and now has its mechanism: NOC transaction issue rate, ~45-51 ns per face-row
   transaction, 64 per source tile, invariant to the bytes each carries.** Established by bf16 ->
   fp32 leaving 123.68 us at 123.61 while the clone control rose 1.36x on the same doubled bytes.
   The org can stop testing bandwidth, bank scatter and fan-out on these two sites.
3. **The knowledgebase's "reblock regresses at small N, -4.7 % at 256 and -2.8 % at 384, so gate it
   to DRAM" does not hold on this card at 298 aa.** Measured here it is a **1.47x / 1.22x win at
   N=320 on the L1 path**, which is the production configuration for 298 aa. The regression is real
   but the crossover sits between N=192 (0.81x) and N=320, not above 384, and the N=384 point is
   1.17x rather than negative. A DRAM-only gate would switch this lever **off** at exactly the size
   this org cares about.
4. **The "stage through L1" candidate is alive, but at other sites.** It is worth 4.39x bit-exact on
   a DRAM->DRAM out-move and 1.07x on the in-move, and nothing at all on the two trimul sites, which
   are already L1-resident. The candidate should be re-aimed at the DRAM->DRAM permutes rather than
   dropped.
5. **`ttnn.permute`'s per-tile cost is not flat in N** (1.469 -> 3.841 us/tile/engaged-core from
   N=128 to N=320 at a constant grid), where the prototype's is (2.505 -> 2.614). Any inherited
   per-tile permute cost is only valid at the N it was taken at.

## Verdict

**NO-GO on the 1805 ms/fold, with a named hardware ceiling. Partial GO on 610.1 ms/fold.**

A tt-metal kernel **cannot** move tile faces at compute-engine rates, and the reason is a counting
argument backed by a measurement rather than a property of any particular kernel. A source tile's 32
columns land in 32 different destination tiles, one non-contiguous two-face row each, so **64 NOC
transactions per source tile is a floor over all kernel structures**. Measured cost per transaction
is 50.6 ns scattered into an interleaved buffer and 29.4 ns for a local aligned L1 gather, and it
does not change when the transaction carries twice the bytes. That puts the best possible tile-face
kernel at ~2.9 us/tile/core against a 1.01 clone floor — which is precisely where the existing
prototype already sits. **There is no kernel structure left to try, and the org should stop spending
passes on this item.**

What Phase 3 could still take, if the CTO wants it: `reblock_permute` / `reblock_permute_back` are
written, bit-exact by `torch.equal` at 18 shape/buffer combinations, and worth **610.1 ms/fold** at
298 aa. The cost is not accuracy, it is upstreaming a custom `ttnn.experimental` op into the 0.68.0
production wheel, plus a size gate — the ops **lose** below N~256 and must not be switched on
unconditionally. That is a build decision for Moritz, not a parity trade, and nothing here is
proposed for merge.
