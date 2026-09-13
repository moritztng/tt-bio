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
positive), 1.07562x on the sampling stage, which is where the whole gain is. Re-measured at
2f5072d8, the tree the benchmark cell was published from then: 19.7275 s to 19.324 s, eight folds
per arm interleaved ABBA, every pair positive, median 1.02138x against an A/A floor of 1.00886x
worst case.

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

## `TT_BIO_DEVICE_CONDITIONING` — on, Boltz-2 only

Boltz-2's diffusion conditioning reads the trunk's pair tensor three times, and upstream does all
three on the host: the pairwise conditioner, the 24-layer token bias stack, and the atom encoder's
`z_to_p_trans`. Each one is a channel map applied independently at every (i, j), so this flag runs
them on the card, where the trunk has just left the tensor. Only the `[n, n, 16]` projection the
atom encoder actually consumes comes back, and the token bias never leaves the device at all: it
goes straight into the diffusion cache, which used to upload the host's copy of it.

**Accuracy: not identical.** The device does this in bf16 where the host did it in fp32, and it
uses the fused bias stack, so the structure moves. It is scored against the experimental structure
1HCL rather than against the previous coordinates, four seeds per arm, both arms in one process.
Native CA-lDDT goes up on both pseudo-domains at 512 residues, 0.93732 to 0.94036 and 0.91573 to
0.91860 as a mean of four seeds, and is flat at 298 residues, 0.96742 against 0.96741. Native
CA-RMSD moves the same way. Three of the four seeds move 0.15 to 0.29 Å per pseudo-domain at 512
residues and the fourth moves 1.26 to 1.51 Å, against a seed floor of 0.97 to 1.87 Å over the same
pairs. That fourth seed is a basin, not a loss: it is the worst fold either arm produced, native
CA-lDDT 0.92021 and 0.89702 on the host path where every other host seed is above 0.9397 and
0.9136, and the device arm puts it back with the rest at 0.93585 and 0.91317. At 298 residues the
move is 0.19 to 0.39 Å against a 0.79 to 1.25 Å floor.

**Speed: 17.989 s against 18.773 s, a 512-residue fold.** Both arms in one process on one card,
interleaved as base / device / base inside every rep so drift cannot land on one arm, cold fold
discarded, 8 device folds against 16 host folds, under benchlock on an idle box. qb2, one Blackhole
processor of a p300c board, physical card 0, ttnn 0.68.0, 3 recycles, 200 sampling steps, one
sample, seed 0, templates off. Median ratio 1.04358x against an A/A floor of 0.99824x drawn from
the two host folds that bracket each device fold, and every device fold in the run is faster than
every host fold in it. Spread is 0.90 % on the device arm
(`perf/b2z2_cond/out/timing_qb2c0.json`).

`TT_BIO_DEVICE_CONDITIONING=0` restores the host path and the previous coordinates.

BoltzGen shares Boltz-2's `TrunkModule` but does not get this flag: it never asks the trunk to keep
the pair tensor on the device, so it takes the same deallocate it always did. No other model
reaches the pair track at all.

## `TT_BIO_FUSE_BIAS_STACKS` — on, Boltz-2 only

Boltz-2's diffusion conditioning builds a per-layer bias stack with one call per layer. This flag
builds all of them in one pass instead.

**Accuracy: not identical.** Unlike every other flag on this page, this one ships on a measured
control rather than on an algebraic argument. `cdk2x2_298` moves 0.218 Å all-atom against a 0.000 Å
A/A floor and a 0.35 Å bar, and pLDDT moves 0.0026. `TT_BIO_FUSE_BIAS_STACKS=0` restores the
per-layer stack and the previous coordinates.

**Speed: 1.0085x on the fold.**

The one thing that would have made this Boltz-2-only claim wrong is BoltzGen, which shares Boltz-2's
diffusion module. It does not call this path: BoltzGen builds its conditioning from
`tt_bio/boltzgen/model/modules/diffusion_conditioning.py`, which inlines the per-layer loop rather
than reusing `_fuse_bias_stack`, so a whole design run makes zero calls to it against a Boltz-2
fold's three. A model that reuses the shared stack builder gets this flag too; one that inlines its
own loop, like BoltzGen, does not.

## `TT_BIO_GATE_GRANULARITY` — 2

The reblock-permute kernel that Blackhole's channel-gating path uses (mirroring `binary_ng`'s own
structure: a sigmoid then a multiply, done in one gated kernel instead of two ops) acquires its
destination register tile by tile. This flag sets how many tiles it acquires per DST acquire, from 1
(the previous behaviour) to 4.

**Accuracy: identical at every value.** The three stages run in the same order through the same two
bf16 circular buffers regardless of granularity, so no rounding point moves; pinned by `torch.equal`
against the two-op sequence, per shape, on both architectures.

**Speed: 2 is the value that ships, not the fastest one measured.** Wormhole reads 1.0420x at
granularity 2 and 1.0759x at 4; Blackhole reads 1.0149x at 2 and 1.0011x at 4 — 4 is a wash on the
architecture the published cell is measured on, so 2 is the setting that wins on one architecture
without losing much on the other. Worth 0.014 s on a 512 aa Blackhole fold, under the fold's own A/A
floor — it ships because it is free and bit-exact, not because the fold moves. Capped at 4: above
that the kernel's multiply stage would need more DST slots than a 16-bit DST has to give it.

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
OpenFold3 `6ee6ac7a3e730688`, each in both arms, pLDDT equal to six places. Boltz-2's digest is
the one it wrote then. `TT_BIO_DEVICE_CONDITIONING` shipped after this was measured and the default
now writes `2bc758a1fb24ef30`; the other two are unchanged, because neither model reaches that
flag.

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

## `TT_BIO_SDPA_ADD_GRANULARITY` — auto

The fused SDPA kernel folds three additions into its main loop: the running-sum/max update and the
mask add. Both do one tile at a time by default; this sets how many tiles they batch per pass,
sized automatically from the query chunk unless you override it.

**Accuracy: identical at every granularity.** Bit-exact against the per-tile loop, because the same
adds happen in the same order — only how many run per pass changes.

**Speed: 1.0317x on the fused SDPA on Blackhole** (2.8299 to 2.7430 ms at 512x512) and **1.0267x on
Wormhole**. `TT_BIO_SDPA_ADD_GRANULARITY=1` restores the per-tile loop.

## `TT_BIO_SDPA_GRID_Q_CHUNK` — on

Scaled dot-product attention is computed in chunks of query rows, and ttnn hands one chunk to one
core. The chunk size tt-bio shipped was a fixed cap with no term for the card: an attention with few
heads produced fewer chunks than the card has cores and left most of the grid idle for the whole op.
This picks the widest chunk whose work still fills a single pass of the compute grid. The core count
comes from the device and the head count from the tensor, so no card and no model is named.

**Accuracy: identical.** The query axis partitions independent rows — each chunk computes its own
rows and nothing is combined across chunks. The softmax reduction order lives in the key axis, which
this does not touch. `torch.equal` and max abs 0.0 at every call site, and one CIF digest across all
84 timed folds of the three sessions below, at equal pLDDT.

**Speed: 1.0048x on the 512 aa Blackhole fold.** Two independent sessions on the shipped tree read
1.00496x (95 % CI [1.00048, 1.00635], same-session A/A floor [1.00075, 1.00393]) and 1.00479x (95 %
CI [1.00322, 1.00636], floor [0.99845, 1.00262]), 28 folds each, paired median over ABBA reps. At
the one call site it moves the op is **1.13x** (118 to 104 us, 20 paired reps a session). An earlier
session measured 1.00355x on a tree without the triangle fusions, where the same fold took 19.324 s
instead of 18.645 s.

**It moves one call site of two, and that is a property of the shapes.** On the 512 aa reference the
diffusion step's token attention goes from 256 query rows to 128 — 64 work units on 110 cores where
the fixed cap gave 32 — on all 4800 of its calls a fold. The atom attention's 32 query rows are a
single tile, so there is nothing to split, and all 1200 of its calls keep the shipped chunk. The
ratio is a Blackhole number and does not transport: the same rule is 1.3058x at the op on Wormhole,
because the win is occupancy and occupancy depends on the grid. Narrower chunks also re-read the
keys and values once more per chunk, 20.1 GB more traffic over the fold, which the idle cores more
than pay for here but would not on every card.

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

## `TT_BIO_TRIMUL_MM_TRANSPOSE` — on

A triangle multiplication moves its channels to the batch axis before the per-channel matmul, and
exactly one of the two operands wants the sequence axes the other way round. That cost a separate
transpose of a whole moved chunk, 67.1 MB at 512 residues, once per call. `ttnn.matmul` takes the
transpose itself through `transpose_a` / `transpose_b`, so the flag hands it over and the separate
op disappears.

**Accuracy: identical.** The structure digest is unchanged at 298 and 512 residues, and the matmul
is `torch.equal` at max abs 0.0 against transpose-then-matmul at the production shape on both
operands.

**Speed:** 1.00613x on the fold at 512 residues. The matmul itself gets slightly slower, because it
re-reads the operand tile-transposed; what goes is the whole extra program and 134.7 MB of
allocation per pairformer block.

**Smaller targets take it too, and there it is an op-level win.** A target small enough to keep the
moved chunk in L1 never had a separate transpose to delete: the channel move and the sequence swap
were one permute. Handing the swap to the matmul leaves only the channel move, which the
hand-written reblock kernel can do. At the 298-residue shape that is 0.3824 to 0.3486 ms with a
transposed second operand and 0.3684 to 0.3463 ms with a transposed first one, median of seven warm
calls, `torch.equal` at max abs 0.0 on both (`perf/util_op_deletes/mm_transpose_l1.json`). Across
the 2240 calls a 298-residue fold makes that is about 0.063 s, under what a fold A/B can resolve, so
it is quoted at the op and not at the wall. The structure does not move: `0cf1b879dca3c0d5` at
pLDDT 0.913091 in both arms.

**It switches itself off under `--fast`, and that is not a failure.** `--fast` hands the matmul
bfloat8_b operands, and a block-float tile shares one exponent per face, so transposing inside the
matmul re-quantises where transposing beforehand does not. It is worth 1.0868x there, but it stops
being an exact relayout, so it is declined rather than taken.

## `TT_BIO_TRIMUL_GP_BANK_SPLIT` — on

A triangle multiplication projects its gates and its values in one matmul, four column quarters
wide. The channel move that consumes that projection reads a value slice and its own gate slice back
to back and waits on both, and the slice offset is what picks the DRAM bank a read lands on.
Ordering the quarters by role, both gates and then both values, puts the two reads of every such
pair 8 tiles apart at the production width, and on Blackhole's 8 DRAM banks 8 tiles apart is the
same bank twice, so the pair serialises. This flag interleaves the roles instead and the pair lands
4 banks apart.

**Accuracy: identical.** The flag permutes which column a value is written to and read from and
changes no arithmetic, so the bar here is equality and not a band. `perf/k10_b1_permute/b1_equiv.py`
gets `torch.equal` on both triangle multiplications at 298, 320, 512 and 640 residues and at both
slice widths, 8 cells of 8, with a negative control that reads the new layout at the old offsets and
is required to differ. At the fold, one CIF digest across all sixteen folds of both arms at 512
residues (`2bc758a1fb24ef30`, pLDDT 0.864509) and one across all sixteen at 298 residues
(`de5d77b220e32a7a`, pLDDT 0.90916).

**Speed: 1.02287x on the fold** at 512 residues (17.7365 s to 17.3400 s, eight folds per arm
interleaved ABBA, all eight pairs positive), 1.5146x on the channel move itself.

It costs nothing at runtime: the weight is laid out once at load, and the reader takes the slice
offsets as arguments it was already taking.

**The gain is Blackhole's.** Wormhole has 12 DRAM banks, so the pair was never congruent there and
there is nothing to recover; the reorder is free on both. The same is true wherever the projection
runs a narrow slice, at 298 residues among others: the pair is two tiles apart in either order, and
those sizes read flat.

## Idle host threads when a box is full

Not a flag of ours. `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and `KMP_BLOCKTIME` are OpenMP's own, and
tt-bio fills them in for the per-card workers it spawns when, and only when, each worker is down to
two host threads. Set any of the three yourself and tt-bio leaves all three alone: they are one
setting spelled three ways, and `GOMP_SPINCOUNT=0` beside your `OMP_WAIT_POLICY=ACTIVE` would undo
it through the back door.

A fold's host work is small but constant, roughly 1.85 cores at 512 aa whether one fold is running
or thirty-two. Those threads spend most of their time waiting on the device, and OpenMP waits by
spinning. Spinning costs nothing while cores are spare and costs a core once they are not, so the
rule keys on the share each worker gets rather than on how many cards are busy: a small host with
four folds on eight threads is in the same regime as a full server with thirty-two on sixty-four.

Measured on a 32-chip Wormhole Galaxy (AMD EPYC 9354P, 32 cores / 64 threads), Boltz-2 512 aa, 200
sampling steps, 3 recycles, full MSA, one independent fold per chip, three timed folds per chip
after a barrier. Both arms of a row run back to back in one session, and each row is at the thread
share tt-bio itself hands that width:

| concurrent folds | threads each | spinning | parked | |
|---|---|---|---|---|
| 16 | 4 | 1131.6 folds/h | 1123.7 folds/h | 0.993x |
| 20 | 3 | 1344.8 folds/h | 1331.6 folds/h | 0.990x |
| 24 | 2 | 1534.7 folds/h | 1541.0 folds/h | 1.004x |
| 32 | 2 | 1789.4 folds/h | 1813.9 folds/h | **1.014x** |

Repeating the 32-fold row reads 1783.2 against 1811.4, so the same-configuration floor is 0.35 %
and the win at that width is real but small. The rows above the line are why it is a rule and not a
default: with three threads a worker still has a spare core to absorb the spinning, and parking
costs more in wake latency than it saves. Every row is bit-exact, both arms writing one CIF digest.

A much larger number is easy to measure here and it belongs to the line above, not to this one. Give
each of 32 workers a fixed 8 threads on the same box and spinning collapses to 1311.2 folds/h while
parking holds 1839.5, a 1.40x. That is 256 threads of demand on 64, and the fix for it is the thread
cap tt-bio already applies by default, which on its own takes that configuration from 1311.2 to
1789.4. Parking is the last 1.4 %.

## Off by default: `TT_BIO_DIT_FUSED_QKV`, `TT_BIO_HEAD_PAD_TAIL`

Two optimizations of Boltz-2's diffusion attention ship present but disabled. Setting either to `1`
turns it on; tt-bio's default fold does not use them.

`TT_BIO_DIT_FUSED_QKV` lets the q/k/v projection write the per-head layout directly, instead of
projecting and then reordering with a separate op. `TT_BIO_HEAD_PAD_TAIL` lets the gate and output
projections carry the head padding that the attention output already contains, instead of stripping
it in four ops first. Both are worth about 1.03-1.04x on the diffusion step alone; neither has a
measured whole-fold number on Blackhole.

They are off because they change the result, and the scoring says different things about them.
Against the experimental structure 1HCL at 512 residues, four seeds per arm, both arms in one
process, mean native CA-lDDT moves by -0.0071 and -0.0091 per pseudo-domain with the fused
projection on, and by -0.0017 and -0.0021 with the padded tail on. The fused projection is down on
seven of eight paired seed-domain readings; the padded tail is up on five of eight and is flat at
298 residues. Coordinates move 0.26-0.48 Å and 0.25-0.31 Å per pseudo-domain against a seed floor
of 1.07-1.39 Å, except on one seed at 512 residues where the sampler lands in a different basin and
any arithmetic change moves it about 1.5 Å.

Both are on the same code as the shipped path when they are off: a fold with neither set writes the
same structure as a tree without them, byte for byte, at 298 and 512 residues.
