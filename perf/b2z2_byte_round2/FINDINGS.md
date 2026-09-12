# 134.2 MB more, deleted bit-exactly, buys 1.01080x — and the realization coefficient is 1.31, not 1.76

`b2z2-trunk-byte-floor` published a 416.3 MB redundancy surface for one PairformerLayer and built
half of rank 1, measuring that its deleted bytes bought **1.76x** what the DRAM-byte ledger priced
them at. This row builds rank 2 and asks whether that 1.76 is a rule.

## The ledger reproduces, to the megabyte

`perf/b2z2_byte_floor/census.py` on the committed trace and on a fresh one taken this pass, WH card
10: DRAM read **3,672.0 MB**, write **3,744.9 MB**, total **7.4169 GB**, redundant reads
**416.3 MB = 5.17 %** of the campaign's 8.0493 GB. Byte for byte the parent's table.

## R2 — the trimul's output gate rides its in-projection

The trimul reads its own normed input twice: the fused four-way in-projection reads all 67.1 MB of
it, and so does `g_out` at the tail. `TT_BIO_TRIMUL_FUSED_GOUT` concatenates `g_out`'s weight onto
the in-projection's, so the gate is a second destination of one pass and the second read never
happens.

**The split writer had to learn an unequal chunk.** `minimal_matmul`'s split path divides N into
`N_chunks` EQUAL chunks, which is exactly why two projections of different widths over one
activation could not share a pass. `MM_SPLIT_LAST_TILES` gives the final chunk its own width
(`tt_bio/kernels/mm_split/patch_mm_split.py`, regenerated from the wheel's own kernels). Addressing
only: same tiles, same order, two buffers instead of one. The same edit gives the split path
`MM_DUAL_NOC`, which it did not have, so a two-chunk call does not lose the drain lever purely by
taking a different branch.

| | measured |
|---|---|
| block ratio | **1.01080x** (median of 7, paired interleaved, one process, one card; drop-first 1.01069x) |
| A/A floor, same run | **1.00021x**, spread 0.22 / 0.24 % |
| loadavg, recorded per rep | 6.6 throughout |
| DRAM reads | 3,672.0 -> **3,537.8 MB**, exactly **-134.2 MB** |
| DRAM writes / L1 | 3,744.9 MB / 943.5 / 1,347.3 MB, **unchanged** |
| redundancy surface | 416.3 -> **282.1 MB**, 5.17 % -> 3.50 % of 8.0493 GB |
| block output digest | 101156.188 in every rep of every arm |

`perf/b2z2_byte_round2/out/ab_gout_512_wh_c10.json` carries every draw.

**BIT-EXACT.** `torch.equal` True and max abs **0.0** on both of the block's outputs, negative
control (one column of the fused gate weight perturbed) fires at max abs 1.74
(`bitexact_gout.py`). The op class of `g_out` does change — it moves off `ttnn.linear` onto
`minimal_matmul` — and that was measured before anything was built rather than assumed:
`probe_opclass.py` is `torch.equal` at max abs 0.0 at both remaining sites, because at c_z=128 both
kernels block the whole 4-tile contraction at once. `probe_split.py` proves the split writer is
column-sensitive in the right half and only the right half.

**ELIGIBILITY CENSUS, flat.** Per arm, AB leg: `qkv_heads` 28/28 served, head-major tail 28/28,
`reblock_gated` 56, `qkvg` 0, in both arms. `trimul_gout` 14/0 in the lever arm and 0 in the base;
`trimul_tail_f1` declines 14 in the lever arm against 28 in the A/A leg, which is the same
per-call decline rate — the lever does not ask F1 and F1 was already declining every call at
c_z=128. No tuned kernel is dropped.

## The finding: 1.31, not 1.76

1.809 % of this trace's bytes x the byte model's 45.2 % byte-proportional share predicts **0.702 ms
of 85.884**. Measured **0.918 ms. Realization 1.31.**

The falsifier this row pre-registered was "below 1.0x of the DRAM-byte prediction" and it did **not**
fire: a deleted multicast `in0` is still worth more than its DRAM line. But it is worth **1.31x, not
1.76x**, on an operand of the same class at the same activation width, in the same block, measured
the same way. So **the coefficient is not a constant**, and the campaign should carry a band —
1.3-1.8 — rather than the parent's point value. The two sites differ in what the freed matmul is
doing with the delivered bytes: the parent's deletion merged two matmuls that were both already
`minimal_matmul` at the same block config, while this one also moves a `ttnn.linear` with its own
tuned program config into a chunk of another matmul, and that op pays its own dispatch and drain.

## What the block floor becomes

**Nothing in `perf/b2z2_orch/ceiling_v3.py` should change**, and that is worth saying because the
brief asked for it. `REMOVABLE_BYTE_FRACTION = 0.0517` is a census of what EXISTS to remove, not of
what has been removed; building a lever does not move it. The floor it computes — 19.54 ms,
**1.8602x** on the block — is still the floor.

What moved is the distance to it. Taking the same arithmetic at the fraction actually built:

| built | bytes | byte-bound floor | block ratio at the floor |
|---|---|---|---|
| nothing | 8.0493 GB | 20.602 ms | 1.7641x |
| qkvg (parent) | 7.9149 GB | 20.258 ms | 1.7940x |
| **qkvg + gout (now)** | **7.7805 GB** | **19.914 ms** | **1.8250x** |
| the whole surface | 7.6332 GB | 19.537 ms | 1.8602x |

Two thirds of the ranking is built. The last 1.83 pp — the tri-attention bias projection at 1.67 pp
plus the transition weights at 0.15 — is worth the final 1.8250x -> 1.8602x and **is not built**.

## The harness bug this row had to fix first, which is the reusable part

The block-level bit-exact check came back BIT-EXACT with a negative control that **could not fail**,
and the reason was the fixture, not the lever. `tt_bio.reference` zero-initialises every sub-unit's
output projection so the residual starts as identity, and with `p_out.weight` zero the trimul's
whole contribution is `0 * sigmoid(g_out)` — zero, whatever the gate is. **23 of the layer's
weights are all-zero**, including both trimuls' `p_out` and `g_out` and both attentions' `linear_o`
and `linear_g`. A block-output comparison against that fixture passes for a correct arm, for an
incorrect arm, and for an arm that computes nothing at all. Every harness here unzeroes them first;
shapes and ops are untouched, so the timing is the timing either way.
