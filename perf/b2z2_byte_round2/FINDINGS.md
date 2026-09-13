# The trunk's redundant reads are gone: 416.3 MB down to 13.7, and the block carries 1.05106x

`b2z2-trunk-byte-floor` published a 416.3 MB read-redundancy surface for one PairformerLayer, built
half of rank 1, and measured that its deleted bytes bought **1.76x** what the DRAM-byte ledger
priced them at. This row builds the other two sites. Both are bit-exact, both are worth more than
their byte line, and **neither is worth 1.76** — the coefficient is not a constant and the reason it
is not is the useful part.

## The ledger reproduces, to the megabyte

`perf/b2z2_byte_floor/census.py` on the committed trace and on a fresh one taken this pass, WH card
10: DRAM read **3,672.0 MB**, write **3,744.9 MB**, total **7.4169 GB**, redundant reads
**416.3 MB = 5.17 %** of the campaign's 8.0493 GB. Byte for byte the parent's table.

## What got built

| | lever | what it deletes | block ratio | A/A floor | realization |
|---|---|---|---|---|---|
| rank 1a | `TT_BIO_TRIATT_FUSED_QKVG` (parent) | q/k/v and the gate share one pass | 1.01459x | 0.99983x | 1.76 |
| rank 2 | `TT_BIO_TRIMUL_FUSED_GOUT` | the trimul's output gate rides its in-projection | **1.01080x** | 1.00021x | **1.31** |
| rank 1b | `TT_BIO_TRIATT_FUSED_QKVGB` | the pair-bias projection rides the qkv+gate pass | **1.02491x** | 1.00074x | **2.93** |
| all three | | | **1.05106x** | 1.00064x | 1.98 |

Every ratio is a median of 7, paired and interleaved in one process on one card, with an A/A leg
measured the same way in the same run and `os.getloadavg()` recorded inside every rep. Drop-first
and min/min agree with every median to within 0.1 pp. Draws in `out/ab_{gout,qkvgb,all}_512_wh_c10.json`.
Each arm's first draw or two is a program-cache miss of several hundred ms and is why the raw
spread column is large; the medians are unaffected and the A/A spreads are 0.16-0.24 %.

The three are additive: 1.01459 x 1.01080 x 1.02491 = 1.0518 against 1.05106 measured.

## The bytes, re-counted with the levers on

| | base | all three |
|---|---|---|
| DRAM read | 3,672.0 MB | **3,269.4 MB** (-402.6, exactly 3 x 134.2) |
| DRAM write | 3,744.9 MB | 3,744.9 MB, unchanged |
| L1 read / write | 943.5 / 1,347.3 MB | unchanged |
| redundant reads | 416.3 MB, 5.17 % | **13.7 MB, 0.17 %** |
| device programs | 417 | 415 |

**96.7 % of the surface is gone**, which is exactly the fraction the parent's census said was one
defect shape. What is left is the transition's `swiglu` weights — 0.1 MB tensors re-read once per
row chunk, 32 times — and four sub-MB shared masks. Deleting those means unchunking the transition,
which is a different lever with a different cost.

## BIT-EXACT, both sites, with controls that fire

`torch.equal` True at max abs **0.0** on both of the block's outputs for each lever, and the block
digest is identical in every rep of every arm (101156.188 / 340218.562 / 80569.953 for the three
harness seeds). Negative controls move the block by 1.74 and 0.49 respectively.

Both levers change an op's CLASS — `g_out` and the pair bias are `ttnn.linear` today and become
chunks of a `minimal_matmul` — and `_trimul_out_proj`'s own docstring warns that the two kernels
block the contraction differently. **That was measured before anything was built**
(`probe_opclass.py`): at c_z=128 both take the whole 4-tile contraction in one block and are
`torch.equal` at max abs 0.0, and a matmul's N width does not change its own arithmetic.
`probe_split.py` proves the new split writer is column-sensitive in the right chunk and only there.

**ELIGIBILITY CENSUS, flat in every arm.** `qkv_heads` 28/28 served, head-major tail 28/28,
`reblock_gated` 56, in base and lever arms of all three A/Bs. `trimul_tail_f1` declines at the same
per-call rate in both. No tuned kernel is dropped.

## The finding: the currency is wrong about OP SHAPE, not just about multicast

The parent explained its 1.76 by multicast: a matmul's `in0` is delivered to a whole grid row, so a
DRAM-byte ledger under-prices it. That explanation predicted this row's two sites would behave
alike, and it predicted the one-tile-wide bias projection would realize LESS. **Pre-registered P2
said 0.5-1.4, point 1.0. It measured 2.93.** The prediction was wrong and so was the reasoning
behind it.

**The real driver is how efficient the deleted reader was**, and it is measurable rather than
arguable (`probe_bias_cost.py`, medians of 11 on the same card):

| | |
|---|---|
| the stock bias `ttnn.linear`, `[512,512,128] x [128,4]` | **1.136 ms** |
| its own bytes at the measured 390.7 GB/s roof (67.1 read + 16.8 write) | 0.215 ms |
| the fused qkv+gate matmul, 16 N tiles | 2.582 ms |
| the same matmul at 17 N tiles — what the fusion actually pays | 2.654 ms |
| **the 17th N tile** | **0.072 ms** |
| so the fusion returns 1.063 ms per attention, **2.127 ms per block** | measured 2.057 |

97 % of the win is accounted for. The bias projection costs **5.3x its byte cost** because its N is
a single tile: the grid wants seventeen and gets one, so the op is occupancy-bound and its bytes
never described it. Folding it into a matmul that was already running buys the same arithmetic for
0.072 ms.

**So: price a deleted read by what its READER costs, not by what its bytes cost.** The realization
coefficient runs 1.3 (a well-shaped reader merged into another well-shaped reader) to 2.9 (a skinny
reader deleted outright), and a DRAM-byte ledger is a lower bound at every site this campaign has
measured — never an over-estimate. The multicast story was not wrong, it was incomplete.

## What the block floor becomes

**`perf/b2z2_orch/ceiling_v3.py` needs no edit**, and that is worth saying because the brief asked
for one. `REMOVABLE_BYTE_FRACTION = 0.0517` is a census of what EXISTS to remove, not of what has
been removed; building a lever does not move it. The floor it computes — 19.54 ms, **1.8603x** on
the block — is still the floor.

What moved is the distance to it, at the same arithmetic and the fraction actually built:

| built | bytes | byte-bound floor | block ratio at the floor |
|---|---|---|---|
| nothing | 8.0493 GB | 20.602 ms | 1.7641x |
| qkvg (parent) | 7.9151 GB | 20.259 ms | 1.7940x |
| qkvg + gout | 7.7809 GB | 19.915 ms | 1.8249x |
| **all three (now)** | **7.6467 GB** | **19.572 ms** | **1.8569x** |
| the whole surface | 7.6330 GB | 19.537 ms | 1.8603x |

**The ranking is built out.** 1.8569x of the available 1.8603x, and the remaining 0.0034x is the
transition's re-read weights.

## The harness bug this row had to fix first, which is the reusable part

The block-level bit-exact check came back BIT-EXACT with a negative control that **could not fail**,
and the reason was the fixture. `tt_bio.reference` zero-initialises every sub-unit's output
projection so the residual starts as identity, and with `p_out.weight` zero the trimul's whole
contribution is `0 * sigmoid(g_out)` — zero, whatever the gate is. **23 of the layer's weights are
all-zero**, including both trimuls' `p_out`/`g_out` and both attentions' `linear_o`/`linear_g`. A
block-output comparison against that fixture passes for a correct arm, for a wrong arm, and for an
arm that computes nothing. Every harness here unzeroes them first; shapes and ops are untouched, so
the timing is the timing either way.

## The kernel change both sites needed

`minimal_matmul`'s split writer divided N into `N_chunks` EQUAL chunks. That is the structural
reason these two sites sat unbuilt after the parent row: a 512-wide projection cannot share a pass
with a 128-wide one, and a 384-wide one cannot share with a 32-wide one. `MM_SPLIT_LAST_TILES` gives
the final chunk its own width, in both of tt-bio's forks of the kernel
(`kernels/mm_split/patch_mm_split.py`, regenerated from the wheel's own sources, and the
hand-maintained `kernels/triatt/`). Host side is `mm_generic.build(..., n_widths=)`.

Two details worth keeping. The `mm_split` fork's split path had no `MM_DUAL_NOC`, so a two-chunk
call would have silently lost the drain lever by taking a different branch; it has it now. And the
head-major tile-id transform reduces to the plain one at a chunk width of a single tile, so the
one-tile bias chunk lands correctly in an ordinary `[batch, seq, 32]` buffer while its four
head-major siblings keep theirs — no second define, no second kernel.
