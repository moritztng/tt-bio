# Tuning flags

tt-bio ships its device optimizations on by default. This page says what each one changes and what it
was measured against, so you can decide whether to turn one off. Every flag takes `0`, `false` or
`off` to disable.

Reference numbers are Boltz-2 on one Blackhole processor of a p300c (Tenstorrent QuietBox), 512 aa,
200 sampling steps, 3 recycles, `perf/size512/fixtures/cdk2x2_512.yaml`. Fold ratios are paired: both
arms are interleaved inside one process, so a ratio is never read across two sessions.

## `TT_BIO_ATOM_SHIFT_GATHER` — on

The atom transformer attends within a sliding window. Upstream assembles each window's key set by
multiplying the atom sequence with a one-hot selection matrix. That selection is a contiguous run of
atoms, so tt-bio shifts the sequence once and slices the runs straight out of it.

**Accuracy: identical.** Fifty-six timed folds across two trees wrote one structure per tree, byte
for byte, in both arms, pLDDT equal to six places. Off the device, the slice reproduces the reference
gather under `torch.equal` at every window count the fold produces, padded and unpadded, and refuses
four negative controls (`perf/b2z2_elision/test_shift_equiv.py`).

**Speed: 1.01985x on the fold** (19.8545 s to 19.4680 s, 20 folds per arm, all 10 paired reps
positive), 1.07562x on the sampling stage, which is where the whole gain is. Re-measured on the tree
the published benchmark cell comes from: 19.7275 s to 19.324 s, eight folds per arm interleaved
ABBA, every pair positive, median 1.02138x against an A/A floor of 1.00886x worst case.

tt-bio decides on the selection matrix, not on the shape of it: it reconstructs the matrix a centred
sliding window would produce and compares entry by entry, once per fold, for 2.6 ms of host time at
512 residues. A model whose atom attention selects a different key set, or pads its matrix
differently, fails that comparison and keeps the matrix multiply. No model name appears in the
condition.

Reading the matrix rather than its shape is the whole safety argument. The atom axis is padded out
to a bucket, and the selection matrix is empty in the windows past the real atom count, so that
matrix and a same-shaped one that selects real atoms there are two different transforms with
identical dimensions. An earlier version of this optimization looked at the shapes only, got those
windows wrong, and still wrote the identical structure, because the attention mask is built from the
same matrix and discards exactly the entries the selection got wrong. Nothing the model outputs
distinguishes the two, at any size. The matrix comparison does.

## `TT_BIO_PWA_BATCH_HEAD_WEIGHTS` — on

The MSA track weights each row of the alignment by a softmax over the token axis, one softmax per
attention head. Upstream builds each head's weights separately, which means projecting the whole
pair tensor through one column of a weight matrix, once per head. tt-bio projects through the whole
matrix once and slices the heads out of the result.

**Accuracy: identical.** The columns pad into the same tile either way, so the batched projection is
the same arithmetic in the same order, not an approximation of it. Off the device it matches the
per-head loop under `torch.equal` on both outputs, max absolute difference 0, and a negative control
fails the same comparison. On the device the three models that share this code all fold to the same
structure byte for byte at 512 residues: Boltz-2 `a91aa44441f0d9c5`, Protenix-v2 `15772214c5b9e990`,
OpenFold3 `6ee6ac7a3e730688`, each in both arms, pLDDT equal to six places.

BoltzGen reaches the same code on its design path and takes the batched projection 192 times in one four-binder design. Its designs are drawn unseeded, so two runs of the same build do not produce the same binders and there is no structure to compare between arms; what is comparable is the designability gate, which passes in every run of either arm.

**Speed: 1.01606x on the MSA track, which is too small to read on the fold.** On one Blackhole
processor the track goes 1.8678 s to 1.8371 s, median of twelve folds, all six paired reps positive
and the two arms fully rank-separated, against an A/A floor of 1.00859 from the same run. That is
about 0.03 s of a Boltz-2 fold, so this page does not quote a fold ratio for it: the fold wall
cannot resolve a win that size and a number that survives only in one session's noise is not a
measurement. On Wormhole the same track costs twice as much and the saving is correspondingly
bigger, 3.8337 s to 3.7054 s, or 1.03672x on a single MSA layer.

tt-bio batches the heads only while they all fit the one 32-wide tile a single head's projection
already pays for, which is a property of the shape and holds at every head count these models use.
Above 32 heads the batched output would be wider than the per-head one and the saving would have to
be re-measured, so it keeps the per-head loop there. No model name appears in the condition.

That condition is also why the structures above are quoted with a call count beside them. A declined
call and an engaged one write the same structure, so an identical digest on its own is equally
consistent with the optimization never having run. Every fold is recorded with the number of batched
and per-head projections it actually made: 16 on Boltz-2, 30 on Protenix-v2, 12 on OpenFold3, and
zero in every arm that had the flag off.

## `TT_BIO_TRIATT_FUSED_QKVG` — on

A triangle attention's query, key, value and gate projections all read the same normed pair tensor,
and the two matrix multiplies that produce them each read all of it: 67.1 MB at 512 residues, twice
per attention. Concatenating the gate's weight onto the qkv weight makes the four results four
output chunks of one pass, and the second read never happens.

**Accuracy: identical.** Every output tile is its own contraction over the whole contraction axis in
both forms, so which buffer a tile lands in cannot change its value. Measured rather than argued:
`torch.equal` at max abs 0.0 on the block outputs (`perf/b2z2_byte_round2/probe_opclass.py`), with
negative controls that move the block by 1.74 and 0.49. The 512, 640, 1024 and 1536 residue folds of
`perf/b2z2_size_ladder/` write byte-identical structures with the flag on and off.

**Speed:** the three flags in this group together are **1.01573x on the Blackhole benchmark cell**
(19.336 s to 19.0375 s, eight folds per arm interleaved ABBA, all eight pairs positive, worst-case
A/A floor 1.00805x, `perf/b2z2_trunk_ship/cell_512_qb2_c0.json`) and 1.02648x on a Wormhole fold. A
second session on the same box and fixture read 1.01947x over twelve folds against a floor of
1.01075x (`perf/b2z2_trunk_ship/cell_512_guard_benchlock_clean.json`); the number quoted here is the
lower of the two.
On the pairformer block alone, where the reads are, they are 1.05106x. This flag is the smallest of
the three on its own and does not separate from the floor of a single fold.

tt-bio asks for the fused pass only where it declines to exactly what the separate calls would have
declined to: one tile per head, no zero padding in the head channels, the same weight dtype, and no
bias on the gate or the output projection. A model that biases either keeps the separate
projections. No model name appears in the condition.

## `TT_BIO_TRIATT_FUSED_QKVGB` — on

The pair-bias projection is the third reader of that same normed tensor, one tile wide against the
other four's four. This flag puts it in the pass as well, so the tensor is read once per triangle
attention instead of three times. It needs `TT_BIO_TRIATT_FUSED_QKVG`; with that off it does
nothing.

**Accuracy: identical.** One detail matters for reproducing the shipped numbers: the bias
weight is taken back off the device rather than rebuilt, because it has already been scaled there in
bfloat16, and scaling in float32 and converting afterwards rounds differently. bfloat16 to float and
back is exact, so the fused form carries the same bits.

**One shape is declined to keep it that way: chains that fit in a single tile.** The tile-level
argument the other two flags rest on does not carry this one all the way down. The bias projection is
one tile wide against the other four's four, so adding it widens the fused result, and on a small
enough target that changes how the multiply is split across cores and therefore the order its partial
sums are added. Measured, not argued. Synthetic chains folded with this flag as the only difference
(`perf/b2z2_trunk_ship/qkvgb_boundary.json`) are byte-identical at 48, 64, 96 and 112 residues and
differ at 32; `trpcage_no_msa` at 20 residues differs too, and
`perf/b2z2_trunk_ship/trpcage_pairs.json` puts that change on this flag alone, the other two
reproducing the flags-off structure exactly. `prot_no_msa` at 117, `hsa_no_msa` at 585 and the 512 to
1536 residue ladder are all clean. A chain of 32 residues or fewer occupies one tile on the token
axis and 48 pads to two, so the fused pass declines whenever either axis of the pair tensor is a
single tile, and the bias projection runs where it always ran.

With that guard the flag is bit-identical at every length: 20, 32, 48 and 64 residues all write
byte-identical structures with the three flags on and off
(`perf/b2z2_trunk_ship/qkvgb_guard_verify.json`). It costs nothing, because a chain that short is not
a performance case. It declines nothing at the sizes that matter: 560 fused calls served per fold at
512, 1024 and 1536 residues, with the same structure digests the tree wrote before the guard existed
(`perf/b2z2_trunk_ship/cell_512_guard_qb2_c0.json`,
`perf/b2z2_size_ladder/out/ladder_guard_bh_c0.json`).

**Speed:** the largest of the three. 1.02491x on the pairformer block by itself.

## `TT_BIO_TRIMUL_FUSED_GOUT` — on

A triangle multiplication reads its normed input twice: once for the four-way input projection and
once for the output gate at the tail, with no write in between. Concatenating the gate's weight onto
the input projection's makes the gate a second destination of one pass.

**Accuracy: identical.** Same tile-level argument as above, and the same `torch.equal` at max abs 0.0
at the production shape. The op class does change, so it was measured and not assumed.

**Speed:** 1.01080x on the pairformer block by itself.

**It switches itself off on large targets, and that is not a failure.** Above roughly 1024 residues
the input projection runs a multi-iteration channel loop, which would recompute the gate once per
iteration, so the fused form declines and the tail runs the projection it always ran. The structure
is byte-identical either way, verified at 1536 residues where the gate declines all 560 calls. It
also declines under `--fast`, on row-blocked norms, on an L1 channel path, and where the output gate
carries a bias.

## What the three cost in memory

A fused pass holds a wider weight and a wider result, so peak device memory rises. Measured with all
three on against all three off, one Blackhole processor of a p300c, `cdk2x2` at four sizes
(`perf/b2z2_size_ladder/out/ladder_bh_c2.json`):

| residues | peak, off | peak, on | delta |
|---|---|---|---|
| 512 | 1.678 GiB | 1.875 GiB | +11.7 % |
| 640 | 2.090 GiB | 2.264 GiB | +8.3 % |
| 1024 | 3.473 GiB | 3.918 GiB | +12.8 % |
| 1536 | 5.694 GiB | 7.259 GiB | +27.5 % |

The 1536 row is the two triangle-attention flags on their own, because the trimul gate has already
declined every call by then. The 1024 and 1536 sizes were re-run after the single-tile guard landed
(`perf/b2z2_size_ladder/out/ladder_guard_bh_c0.json`): same digests, same engagement, peak within
0.8 percentage points of the table. No size refused an allocation: 7.26 GiB is 22.8 % of the 31.87 GiB
part, and the smallest largest-contiguous free block per bank at the high-water mark is 3013 MiB
with the flags on against 3157 MiB without them.
