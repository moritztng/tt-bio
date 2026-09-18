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

## `TT_BIO_DEVICE_CONFIDENCE`, `TT_BIO_DEVICE_CONF_HEADS` — both on, Boltz-2 only

Boltz-2 scores the structure it just predicted with a confidence head, and upstream builds that
head's input on the host: it normalises the trunk's pair tensor, adds the relative-position
encoding, the token bonds and the contact conditioning, broadcasts the single representation into
it and embeds the distogram. Every one of those is a channel map at each (i, j), and the trunk has
just left the pair tensor on the card, so `TT_BIO_DEVICE_CONFIDENCE` does the assembly there. It
also deletes the upload that used to feed the head's own pairformer: 134 MB of fp32 at 512
residues, replaced by an index map.

`TT_BIO_DEVICE_CONF_HEADS` continues the same idea past the pairformer. The pae and pde
projections and the bin contractions behind them run on the card, and only the three aggregated
numbers per token pair come down instead of a tile's worth of bin logits: 67.1 MB to 2.097 MB at
512 residues, 32.0x fewer bytes. It is built on top of `TT_BIO_DEVICE_CONFIDENCE` and is measured
with it, so set them together or not at all.

The win is the deleted host work, not the smaller download, and on Blackhole the download is
deliberately not the smaller one. Narrowing the readback needs a row-major layout, and a row-major
readback on Blackhole runs at a flat 159 MB/s: 0.524 MB through 8.389 MB, 256 to 1024 residues,
every rung within 2 MB/s of that, against 1.8 to 3.9 GB/s for the tiled read. An 8x byte saving
does not cover an 11 to 23x rate penalty, so the narrow shape loses at every rung of the size
ladder, 3.72 ms against 1.13 at 256 residues, 14.40 against 4.88 at 512 and 57.50 against 42.29 at
1024 (`perf/b2z2_confptm/download_shape_*_qb2_c0.json`, 20 interleaved reps per rung with the
three variants asserted bit-identical first). Wormhole ranks them the other way round, 17.500 ms
narrow against 32.644 tiled, and that is the machine the choice was originally fitted on, so the
head picks its download shape by `is_wormhole()`: slice on the card there, slice on the host here.
Scored inside a real fold with both arms interleaved in one process, picking the right one is
worth 9.61 ms of download plus the 0.64 ms untilize it no longer does at 512 residues, and 3.15
plus 0.56 ms at 298, with the head's output and the written CIF asserted identical on every fold
(`perf/b2z2_confptm/readback_ab_298_qb2_c0.json`, `readback_ab_512_qb2_c0.json`). It is a tenth of
a percent of the fold either way; it is not why the flag is on.

**Accuracy: the coordinates cannot move, and they do not.** The confidence head runs after the
sampler and its outputs are scores, so at one diffusion sample nothing it produces feeds back into
a coordinate. The claim is therefore an equality rather than an Ångström bar, and it holds: every
atom is bit-identical at 298 and 512 residues, max 0.000000 Å, against a same-arm control that is
also exactly zero. What does move is the confidence itself, in bf16 where the host used fp32 —
per-atom pLDDT by at most 0.362 at 512 residues and 0.185 at 298, on a 0–100 scale, mean 0.032 and
0.022.

The scores in `results.json` move too, and by less than the model moves them itself. Taking
`TT_BIO_DEVICE_CONF_HEADS` on its own at 512 residues on Blackhole, pTM shifts 0.0031 of its
0.6334 and `complex_pde` 0.0028, pAE by 0.149 Å mean absolute on a 14.76 Å mean and pDE by
0.055 Å on 5.77 Å, against a same-arm control that is exactly 0.000000 on all eight scalars. Fold
the same target with four diffusion seeds instead and pTM spans 0.0758, `complex_pde` 0.0538, pAE
1.04 Å and pDE 1.72 Å — 7x to 31x more than the flag moves them
(`perf/b2z2_confptm/seedscatter512_qb2_c0.json`). Per-residue pLDDT is untouched by this half of
the pair, exactly 0.000000, because `to_plddt_logits` reads the single representation and stays in
torch.

A CIF sha256 is the wrong instrument for this flag, and the run reports both readings for that
reason: `write_result` puts pLDDT in the B-factor column, so the file hash changes with every
coordinate identical. `perf/b2z2_confhead/score_conf.py` reports the coordinate delta and the
pLDDT delta separately.

**Speed: 16.537 s against 17.285 s, a 512-residue fold.** Both arms in one process on one card,
`base device device base` inside every rep so the order reverses within the rep, one cold fold per
arm discarded, 8 folds per arm under benchlock. qb2, one Blackhole processor of a p300c board,
physical card 0, ttnn 0.68.0, 3 recycles, 200 sampling steps, one sample, seed 0, templates off.
Median of 8 paired ratios **1.04743x**, all 8 positive, against an A/A floor of 1.00104x median
drawn from this session's own same-arm adjacent pairs
(`perf/b2z2_confship/cell_512_qb2_c0.json`). The earlier Wormhole reading for the first of the two
flags was 1.0128x; the Blackhole fold is less than half as long, so the same block of deleted host
work is a larger fraction of it.

Setting either to `0` restores the host path for that half. No other model reaches the Boltz-2
confidence head.

## `TT_BIO_DEVICE_ZINIT` — on, Boltz-2 only

Boltz-2 starts the trunk from `z_init`, a `[1, n, n, token_z]` pair tensor built by summing five
per-`(i, j)` terms: two broadcasts of a `[1, n, c]` projection, the relative-position tables,
`token_bonds`, the bond-type embedding and `ContactConditioning`. Upstream builds all of it in
torch and then uploads the result to the trunk. At 512 tokens that tensor is 134 MB and every
intermediate sum is another one. This flag builds the same sum on the card, where the resident
trunk consumes it, so the adds run in the trunk's own memory and the upload never happens. What
crosses the bus instead is an index map and one packed feature tensor.

It is the same construction the confidence head does with different weights, so both call sites go
through one `PairAssemblyDevice` and there is no second copy of the assembly.

The flag declines rather than diverging: the device path needs the resident trunk to be the
consumer, and it needs nothing else on the host to want the relative-position encoding it skips.
BoltzGen's token-distance recycle does want it, so BoltzGen takes the host path and says so on
stdout. Boltz-2 has no such module and templates do not read it.

**Accuracy: not identical.** The device sums in bf16 where the host summed in fp32, and it sums in
a different order. `cdk2x2_298` moves 0.264577 Å all-atom (CA 0.107924) against a 0.000000 Å A/A
floor and a 0.35 Å bar, both arms in one process. At 512 residues it is scored against the
experimental structure 1HCL rather than against the previous coordinates, eight seeds per arm:
native CA-lDDT goes 0.93357 to 0.93238 on one pseudo-domain and 0.91382 to 0.91288 on the other,
paired differences of -0.00119 and -0.00093 with t of -0.49 and -0.34, against a per-arm spread of
0.0096 to 0.0116. Four and five of the eight seeds move up. That is flat, not a loss.

Per-pseudo-domain RMSD cannot carry a 0.60 Å bar at 512 residues on this fixture, and the reason
is measured rather than argued: `cdk2x2_512` is a chimera whose two halves hinge, so re-running the
same build with nothing changed but the seed moves the worst pseudo-domain up to 2.79626 Å all-atom
across eight seeds. The bar sits below the column's own noise. Against that floor the flag moves
1.6372 Å, and every other column reads the same way: domain 2 1.26886 Å against a 1.74577 Å floor,
hinge-free 1.49445 Å against 2.38987 Å, CA-only 1.32167 Å against 2.37177 Å, whole molecule
12.0147 Å against 21.8416 Å. Every column moves less than a seed change does, which is why the
reading above is taken against the experimental structure instead.

**Speed: 1.01955x on a 512-residue Blackhole fold, ten of ten paired reps positive.** One p300c
processor, ten `base,on,base` brackets in one process under the box benchlock, 17.523 s against
17.195 s. The A/A floor taken from the same brackets is 1.00361x and its widest pair is 1.00992x,
below the slowest of the ten lever pairs (1.01566x), so every pair clears the floor. Wormhole read
1.01090x on an n300, six of six positive. `TT_BIO_DEVICE_ZINIT=0` restores the host path and the
previous coordinates.

## `TT_BIO_DIT_COND_HOIST` — on

Every layer of the token diffusion transformer reads the same conditioning vector, each through
six projections of its own, and the layer norm in front of those projections is a per-layer scale
over a shared normalised input. This flag folds each layer's scale into its own weight block, so
one parameter-free layer norm and two concatenated matmuls replace 144 matmuls and 48 layer norms
per sampling step. The dot products and the FLOP count do not change. What changes is how the
launches are grouped, and the order of one bf16 rounding.

The folded weights are built when the model loads rather than on the first fold, so a process that
folds once is not left paying for them.

**Speed: +0.2052 s at 512 aa** (14.588 s to 14.392 s at a forced 1350 MHz, five interleaved reps,
95 % CI [+0.1561, +0.2543], against the same session's paired A/A floor of +0.0324 s +/- 0.1087).
That is this flag on its own, which is what ships: the same session also carried
`TT_BIO_UNFUSED_SILU` and read +0.4798 s for the pair, but that flag is off by default and the
stack figure describes no shipped configuration. Timed at the block rather than at the fold, this
lever reads 1.071x and 1.085x across two sessions on this card. The win decays with size: +0.2809 s
at 298 aa, +0.2415 s at 512 aa and +0.1811 s at 768 aa, each above its own interleaved A/A floor.

**Accuracy: not bit-identical, and scored on this flag alone.** At 298 aa, the size the
0.35/0.60 A band is written against, it moves the structure 0.24726 A all-atom (worst of two
seeds). That is inside the pass band and 0.309x that fixture's own base-against-base seed floor of
0.79984 A, with a same-seed A/A control at 0.00000 A and identical digests.

At 512 aa whole-molecule RMSD reads 11.81553 A, and that figure is a hinge rotation rather than a
larger error. `cdk2x2_512`'s hinge is unconstrained, its own base-against-base seed floor runs
1.3735 to 17.5024 A over ten seed pairs, and RMSD superposes the whole molecule, so it cannot tell
a rotated lobe from a worse fold. CA-lDDT against the experimental structure is the instrument that
can: it is superposition-free and local, so a rigid inter-lobe rotation barely moves it. Against
1HCL domain 1 (252 CA) it reads 0.91585 with the flag off and 0.92767 with it on, +0.01182 in the
flag's favour, with the A/A control at exactly 0.00000. pLDDT moves the same way, +0.017089. Two
limits worth knowing: that reading is one seed, and domain 2 did not resolve against 1HCL. RF3's
token DiT inherits this default and has not been scored for it.

Scope: RF3's token DiT builds this same block, so the default applies to RF3 as well as Boltz-2.
The atom-level transformers take a different path and do not read the flag.

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

## `TT_BIO_MSA_LADDER` — on, Boltz-2 and BoltzGen

The MSA depth axis used to pad to a single 1024, so a 35-row alignment cost exactly what a 1000-row
one did. This flag pads it instead to the smallest rung of (64, 128, 256, 512, 1024) that holds the
depth, and to multiples of 1024 above that, so nothing deeper than 1024 rows changes shape. Three
units read the depth axis and between them they move 68.4 % of the MSA block's bytes. The reference
512 aa fixture carries 35 real rows, which is 29.3x padding.

**Accuracy: not identical, and the difference is displacement rather than error.** Poison the padded
rows with garbage instead of zeros and the MSA module's output is bit-identical
(`perf/roof_msa_ladder/z_parity_512.json`), so the padded region provably cannot reach a real token
and a shorter rung only reassociates the same terms. It reassociates them at the front of the trunk
though, ahead of 3 recycles, 64 pairformer blocks and 200 sampling steps, so the structure moves:
0.295 Å and 0.267 Å CA per pseudo-domain at 512 residues, 0.475 Å and 0.420 Å all-atom, against an
A/A floor of 0.000 Å over twenty-two folds. The whole-molecule figure is 0.904 Å CA, and the extra
comes from a 6.6° hinge between the two copies of this chimeric fixture rather than from either
copy.

Scored against the experimental structure 1HCL instead of against the other arm, the ladder is
closer on both pseudo-domains: native CA-lDDT 0.91822 to 0.92249 and 0.89503 to 0.90295, per-residue
paired means +0.00430 and +0.00743 with bootstrap CI95 [+0.00139, +0.00642] and [+0.00374,
+0.01063], both clear of zero. Native CA-RMSD moves the same way, 1.700 Å to 1.565 Å and 1.528 Å to
1.413 Å. Displacing the same arm's own CA atoms incoherently by that same 0.904 Å costs 0.22 lDDT,
dropping it to 0.776, so the metric can see a move this big and this move is not one.

At 298 residues, where the fixture is a single copy and the arms sit 0.111 Å apart in CA, the ladder
arm reads 0.00344 CA-lDDT lower on seed 0. The per-residue bootstrap excludes zero, but that
bootstrap resamples residues inside one structure, not seeds: the figure is 5.5x smaller than the
0.019-wide seed spread the same scorer measured across seeds, and smaller than the 0.00516 an
incoherent displacement of that same 0.111 Å costs. It is a one-seed reading below the floor that
would make it readable, not a resolved loss, and 298 residues is a size where the lever's own speed
win is smallest.

**Speed: 15.469 s against 16.361 s, a 512-residue fold.** 1.0577x, a paired median of 0.865 s over
ten pairs, arms alternating inside a pair with the pair order flipped every block, one process and
one card. All ten pairs favour the ladder and the two arms' ranges do not overlap: the slowest
ladder fold, 15.932 s, beats the fastest 1024 fold, 16.076 s. The A/A null on the same harness and
host reads 0.9981x. qb2, one Blackhole processor of a p300c, ttnn 0.68.0. A p150a reads 0.795 s on
the same fixture. Rung 64 costs 0.131 s to compile, once
(`perf/roof_msa_ladder/ab_512_qb2_rebased.json`).

The win tracks how shallow the alignment is: 0.433 s with 300 rows, and exactly nothing above 512
rows, where both settings pad to the same 1024. A real ColabFold search usually lands above that, so
this flag is worth more on the reference fixture than on a deep-MSA target.

`TT_BIO_MSA_LADDER=0` restores the single 1024 rung and the previous coordinates.

Boltz-2's MSA module and trunk read the ladder, and BoltzGen reaches it through the trunk it shares.
Protenix-v2, OpenFold3 and RF3 have their own MSA modules and do not read it.

## `TT_BIO_PAIR_FFN_L1_FC1` — on, ESMFold2 only

ESMFold2's trunk runs its pair transition in 32-row blocks. Inside a block the first matmul is
split into two halves whose product the SiLU multiply consumes immediately, and both halves used
to write their result to DRAM for that multiply to read straight back. This flag gives them an L1
destination instead, so 2.15 GB per call never leaves the chip.

The matmul had to be told how to drain. Its default schedule spends 1,212,416 B of a 1,461,760 B
bank on the two input buffers, which leaves no room for an L1 output, so the allocator refused one
at every row height and both halves fell back to DRAM without a word. Naming the output block
width at 16 fits the destination into what is left. The contraction block stays at 1, because that
is the accumulation order the DRAM path used: of 80 configs swept at this shape, all 20 with a
contraction block of 1 are bit-exact against the shipped call and all 60 above it differ by one
bf16 ULP.

**Accuracy: identical, bit for bit.** Only a destination moves, so this is not a precision trade.
Both arms in one process on one card, arms alternating, three rounds at 512 residues: the same
CIF, byte for byte, and the same pLDDT to four places
(`perf/ttx_b3/fold_ab_esm512_c0.json`). A second card says the same at 298, 512, 768 and 1024
residues (`perf/esm3p4close/fold_ab_*_c1.json`).

**Speed: 27.780 s against 30.222 s, a 512-residue fold.** 1.0879x, three folds per arm after a
discarded cold fold, arms alternating, every fold in the fast arm ahead of every fold in the slow
one on a 0.056 s A/A spread. qb2, one Blackhole processor of a p300c, ttnn 0.68.0, 11x10 grid, 10
recycles and 100 sampling steps, one fold at a time under the bench lock.

Where it pays depends on the size, so the four sizes were folded end to end rather than inferred:

| residues | fold, flag off | fold, flag on | |
|---|---|---|---|
| 298 | 18.613 s | 18.188 s | 1.0234x |
| 512 | 30.222 s | 27.780 s | 1.0879x |
| 768 | 66.601 s | 66.615 s | 0.9998x, inside a 0.126 s floor |
| 1024 | 147.230 s | 147.241 s | 0.9999x, inside a 0.108 s floor |

298 and 1024 are card 1, 512 and 768 card 0, so read each row against itself and not across rows.
At 512 residues the destination is not merely requested but served: the device's own latch census
counts 17216 served, 0 refused, 0 blocked and 0 declined with the flag on, against nothing at all
with it off (`perf/ttx_b3/fold_ab_esm512_latch_c1.json`). That is the number to read, because the
per-call counter next to it counts requests and is incremented before the call that can still
fall back.

The token axis pads to a multiple of 32, which is why 298 residues gets the lever at all: it runs
10 row blocks, 10/16 of what 512 runs, and the gated-call census is exactly 10/16 of it. Below
that nothing is row-blocked. At 768 and 1024 the block's other L1 residents leave too little room,
the device refuses the first call and the class retires to DRAM for the rest of the fold, so the
flag neither costs nor saves anything there.

No other model reaches it, by construction rather than by luck: the gated path needs a
`SwiGLUFFN` built with `fuse_swiglu=True`, and ESMFold2's `PairUpdateBlock`
(`tt_bio/esmfold2.py::PairUpdateBlock`) is the only place in the engine that builds one. Boltz-2, BoltzGen,
Protenix-v2, OpenDDE, RF3 and OpenFold3's MSA stack use the shared `Transition`, whose two
matmuls already write to L1, so the round trip this removes does not exist for them. OpenFold3's
diffusion track and AF2-IG have their own transitions again. Measured rather than assumed for the
three that were folded: Boltz-2, Protenix-v2 and OpenDDE at 512 residues all report zero gated
calls in both arms (`perf/ttx_b3/fold_ab_b2_512_c1.json`, `fold_ab_px2_512_c0.json`,
`perf/esm3p4close/fold_ab_odde512_c1.json`).

`TT_BIO_PAIR_FFN_L1_FC1=0` restores the DRAM output and the same coordinates.

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

## `TT_BIO_RESIDUAL_L1` — on

A Pairformer layer adds each sub-layer's output back into the pair tensor. Two of those updates
used to be written to DRAM and read straight back by the very next op: the starting triangle
attention's output projection, and the pair transition's assembled result. This flag has both
producers write into L1 instead, so the update never crosses DRAM in either direction.

It is a gate, not a placement. Each site asks whether the update fits across the grid's banks at
the live grid size, with the consuming matmul's per-core buffers reserved underneath it, and
leaves the result in DRAM when it does not. At 512 aa and below it fits; at 768 aa and above it
does not, and the site falls back. The large-target outcome is "no win", never "no fold".

**Accuracy: identical.** A memory config decides which banks a tile lands in, not what is in it.
Off the device the projection and the concat reproduce their DRAM output under `torch.equal`, max
|delta| exactly 0.0, at every size where the gate engages, and the same comparison refuses a
perturbed control (`perf/k10_binaryng_land/test_l1_equal.py`). At the fold, sixteen timed folds at
512 aa wrote one CIF sha256 and one plDDT, `0.864509`, in both arms, and 298, 512, 768 and 1024 aa
each wrote a single digest across both arms.

**Speed: measured with `TT_BIO_TRIMUL_MASK_L1`, not separately.** The two flags touch three
different sites in the same block and the pair was measured as a pair. See the next section for
the number.

## `TT_BIO_SDPA_ADD_GRANULARITY` — auto

The fused SDPA kernel folds three additions into its main loop: the running-sum/max update and the
mask add. Both do one tile at a time by default; this sets how many tiles they batch per pass,
sized automatically from the query chunk unless you override it.

**Accuracy: identical at every granularity.** Bit-exact against the per-tile loop, because the same
adds happen in the same order — only how many run per pass changes.

**Speed: 1.0317x on the fused SDPA on Blackhole** (2.8299 to 2.7430 ms at 512x512) and **1.0267x on
Wormhole**. `TT_BIO_SDPA_ADD_GRANULARITY=1` restores the per-tile loop.

## `TT_BIO_SDPA_FUSED_LARGE_S` — on

Triangle attention re-reads the same pair bias once per row of the pair tensor. The fused kernel
reads it once per head instead and holds it, and it needs a narrow query chunk against a wide key
chunk to fit. The chunk ladder that picks those two numbers walks query chunks widest first and
takes the first pair that runs at all, so above 1024 tokens it settled on a wide query the fused
kernel cannot take and handed the call to the stock attention. Every length from 1024 to 2592
tokens fell off the fast path this way. This flag lets the ladder try the fused pair first above
1024, ordered by the work each pair puts on a core, and serves 36 of the 50 lengths in that range
on an 11x10 grid at 4 heads. The other 14 have no 32-aligned divisor between 32 and themselves,
which the kernel declines by construction; that is a property of the token count, not of L1.

Nothing at or below 1024 tokens changes. There the ladder already lands on a fused pair, 560 of 560
calls at both 512 and 1024 residues, and those digests are bit-exact and shipped.

**Accuracy: not bit-exact above 1024 tokens, and there is no shipped digest to break** — no length
above 1024 served this kernel before. The key chunk sets the online-softmax reduction order, so a
wider key means fewer rescales of the accumulator. Against an fp32 evaluation of the same bf16
operands at 1536 tokens the fused pair is marginally closer than the ladder it replaces, 0.402555
against 0.402814. At the fold, a 1536-residue structure moves 1.007 Å all-atom and pLDDT goes from
0.786016 to 0.789493. That is above the 0.60 Å bar, which was set at 512 residues where re-running
with a different seed moves the structure 1.84 Å; at 1536 residues the same seed change moves it
36.6 Å, so the flag sits 36x inside the variation this size already carries, and it moves pLDDT the
favourable way. Two folds of the same arm in two processes are byte-identical.

An L1 refusal costs speed and not correctness. With every fused pair refused the call falls back to
the same stock attention the flag-off arm takes, bit-identical, max absolute difference 0.0.

**Speed: 4.23x on the attention op at 1536 tokens** (136.143 to 32.184 ms on a Blackhole
p300c), 2.678x at 1920 and 1.865x at 2208. The win survives changing operands: timed per call with
a fresh bias every call it is still 2.71x, and rotating whole operand sets 4.06x. At the fold it
saves **27.3 s of trunk time** at 1536 residues, which at the shipped 200 sampling steps is
**1.1856x** (174.178 to 146.915 s, off/on/off interleaved, one process per fold, against a 1.21 %
same-session A/A floor). The saving is a fixed trunk saving, so it barely dilutes with step count:
a 20-step fold of the same target read 1.1973x. `TT_BIO_SDPA_FUSED_LARGE_S=0` restores the stock
ladder.

**It is on by default because the win is a shipped-configuration number and the cap bounds the
risk.** The earlier 1.1973x came off a 20-step fold, which gives the trunk a larger share than the
default 200 steps does, so it could not carry the default on its own. At 200 steps the ratio is
1.1856x, 15.3x the same-session A/A floor, and the 27.3 s is trunk time rather than per-step time.
Below the cap the route is unreachable by construction, and that is measured and not just argued:
off/on/off in one process per fold gives one CIF digest per size across all three arms at 298, 512
and 1024 residues -- including 1024, the cap boundary itself, which is the last length where the
flip has to change nothing. The chunk pick is also unchanged in every arm at every one of those
sizes, so the stock ladder still serves the call.

**Reach depends on the head count and the grid, not on the model.** The fused pair needs one query
chunk per core, so a card with fewer cores, or a model with more heads on the same card, serves
fewer lengths: 36 of 50 at 4 heads on 110 cores, 25 at 8 heads, 18 at 12. Boltz-2, BoltzGen and
Nesso-1 run their trunk at 4 heads and get the full reach. Sites that run triangle attention in
fp32 (`Fp32TriangleAttention`, and the `fp32_softmax` branch that reaches `_tri_att_sdpa_hifi`)
never consult this flag.

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

## `TT_BIO_TRANSITION_L1_ROWS` — on, Blackhole only

Every transition block splits its input into row blocks so the SwiGLU's intermediates fit in L1.
The height of that block was 16 rows everywhere, a number fitted on Wormhole. Blackhole has more L1
per core and more cores, and nothing ever revisited the height for it, so a 512-residue pair tensor
was cut into 32 blocks when its own budget allows 11.

The flag makes the height a function of the shape and the card instead of a constant: the tallest
block whose live bytes (the normalized input plus the SwiGLU's two halves, tile-padded) fit the
card's measured per-core L1 budget, floored at the height that ships today so nothing gets shorter.
On Blackhole the budget is 514,756 B per core. At the pair channel that works out to
`height x width = 24576`, which is 48 rows at 512 residues, 32 at 768 and 24 at 1024, and 16 at
1536, where the old constant already sat at the budget. Above that the flag does nothing at all.

It is one expression, not a table: the MSA track has a quarter of the pair track's channel and gets
its own taller block from the same budget, and a model with a wider channel gets a shorter one.
A fixed height cannot do this. 768 residues refuses 48 rows and 1024 refuses 28, so every constant
between 24 and 48 clashes on some size.

**Accuracy: identical on the shapes it was measured on, and inside the noise elsewhere.**
Byte-identical CIF at 298, 512, 768 and 1024 residues, both arms, on an 11x10 Blackhole p300c
(`perf/roof_transition_chunk_bh_ship/`). Digests `a8c6fd65f70f4418`, `45781db716ebf020`,
`9e1a1fdd392e0b4c`, `9a049ee58a5a0f7e`. Also byte-identical with the MSA depth axis at its full
1024 rows, where the rule gives the MSA track a 96-row block instead of 16.

Bit-exactness is not a property of every shape, though, and the page should not claim it is. On the
release gate's no-MSA prot leg the two arms differ: 7.2161 Å from the fp32 reference with the flag
on against 7.0511 Å with it off, a 0.165 Å move on a target whose arms both already sit 7 Å out.
That leg is a documented bf16 floor and its verdict does not change with the flag.

**Speed: 1.023-1.035x at 512 residues, 1.030x at 768, 1.013x at 1024.** Four paired reps per
measurement, the off arm run on both sides of the on arm so the session's own A/A floor comes out of
the same folds, one process and one card:

| size | off | on | saved | A/A floor |
|---|---|---|---|---|
| 512 | 15.270 s | 14.750 s | 0.519 s | 0.005 s |
| 512, second session | 15.217 s | 14.875 s | 0.342 s | 0.030 s |
| 768 | 31.733 s | 30.806 s | 0.927 s | 0.120 s |
| 1024 | 56.573 s | 55.825 s | 0.748 s | 0.078 s |

512 is quoted as a range on purpose. Both sessions clear their own floor by an order of magnitude
and they still disagree by 0.177 s, because an A/A floor bounds contention inside a session, not
where the session itself sits.

The saving does not shrink as the win per block does: 512 residues goes from 32 row blocks to 11 and
1024 from 64 to 43, but 1024 has four times as many blocks to remove.

**Which models it reaches.** Boltz-2, BoltzGen and OpenFold3, whose pair track is 128 channels
wide and whose MSA track is 64. Protenix-v2 (256) and OpenDDE (384) share the same transition block
but keep the height they ship today, because the budget was fitted at 128 and over-predicts above
it: unbounded, it kills every seed of OpenDDE's structure and abag legs and the release gate's
capacity leg with an L1 circular-buffer clash, all of which pass with the flag off. Getting the
wider channels in needs a budget measured at that channel, not this one extrapolated.

AF2-IG's transition is a different block (ReLU, not SwiGLU) and OpenFold3's diffusion conditioning
has its own unchunked copy; neither changes. Wormhole is untouched: there the same budget only ever
shortens the block, which is what it already did.

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

## `TT_BIO_TRIATT_GATE_EPILOGUE` — off

Triangle attention ends with `o * sigmoid(g)`, and that multiply is its own op: it re-reads the
attention output and the gate from DRAM and writes the product back, 268.4 MB a Pairformer block,
3.34 % of the block's 8046.8 MB. Both operands first exist together inside the fused SDPA kernel, so
this flag folds the sigmoid and the multiply into that kernel's pack stage and deletes the op. The
gate's own producer cannot host it: `TT_BIO_TRIATT_FUSED_QKVG` writes q, k, v and the gate out of one
matmul, so an epilogue there would multiply by a tensor its own output is an input to.

**Accuracy: bit-exact either way.** `torch.equal` at max absolute difference 0.0 at the op, and over
a whole Pairformer block with its zero-initialised output projections refilled
(`perf/roof_gate_epilogue/block_ab_gated_512_qb2_c2.json`). At the fold, Boltz-2 at 298 and 512
residues writes byte-identical mmCIF with the flag on and off, 560 of 560 triangle-attention calls
served gated at each size with zero declines, and all-atom Kabsch reads 0.000000 Å against the 0.60 Å
bar. A configuration the gated kernel's CB set does not fit is declined and falls back to the
separate multiply, which is the same arithmetic in the same order.

**Speed: 0.99737x on the Pairformer block on Blackhole, which is why it is off.** One Blackhole
processor of a p300c, 512 residues, 11x10 grid, arms interleaved, 7 reps: 50.634 ms base against
50.768 ms gated, on a same-arm floor of 0.110 %. The op-level pair reads the other way and does not
carry: SDPA plus the separate multiply is 3.0202 ms against 3.0130 ms fused, a 0.24 % gain on a
0.089 % floor (`op_ab_gated_512_qb2_c2.json`). The block is the number that decides, and it is a
small loss.

**The bytes were not the cost.** The multiply this deletes already runs at 348.0 GB/s, 82 % of the
424.7 GB/s measured Blackhole DRAM roof, so there is little bandwidth left to recover. The sigmoid is
not free wherever it goes: bolted onto the multiply it costs 0.286 ms, and moved into the SDPA
kernel's reader it costs 0.851 ms against the 0.906 ms the deleted multiply was worth
(`sig_cost_512_qb2_c2.json`). A byte-count prediction scaled from the Wormhole ablation said 1.0132x
on Blackhole; the measurement came back below 1.0.

**Wormhole is not measured, and the arithmetic points the other way there.** Eliding the same two
multiplies on a Wormhole Galaxy card is worth 1.03200x a block, 1.0211x once the host kernel is
charged the 67.1 MB read it gains (`block_ablate_512_whglx_c2.json`). Wormhole's measured DRAM roof
is 227.5 GB/s against Blackhole's 424.7, so the deleted bytes are worth roughly 1.9x more there. The
block A/B has not been run on Wormhole with the kernel built, so the flag ships off on every card.
`perf/roof_gate_epilogue/FINDINGS.md` has the full record.

## `TT_BIO_TRIATT_B8` — off

Triangle attention's interior in `bfloat8_b`. The fused qkv+gate(+bias) matmul writes its five
destinations in the block format, the fused SDPA's destination follows `q.dtype`, and the gate
multiply runs on block-float operands. A bfp8 tile is 1088 bytes against bf16's 2048, and the two
concatenated q/k/v/gate buffers are 268.4 MB each, the largest pair-scale tensors a Pairformer
block moves.

Three things deliberately stay bf16. The pair representation `z` and every update written back into
it, because this model fails when block float quantises an accumulation rather than an operand: the
whole-track arm cost 1.4965 Å and a single residual site cost 13.21 Å. The stored weights. And the
normed pair tensor, because `ttnn.layer_norm` returns its input's format and narrowing there would
need a cast that costs more than it saves. The out projection keeps the model dtype, so the region
rounds once on the way out and the residual never sees block float.

**Speed: 1.01211x, +0.1750 s at 512 residues.** One Blackhole processor of a p300c, `cdk2x2_512`,
11x10 grid, production protocol (3 recycles, 200 sampling steps), AICLK held at 1350 MHz and sampled
during every fold at min = max = 1350 over 1840 samples, arms interleaved block by block, on a host
and board pair checked quiet before every arm: 14.630 s off against 14.455 s on. Every on-arm fold
is faster than every off-arm fold, by a gap of 0.080 s, against an A/A floor of 1.00432 that the
effect clears 2.80x. A second quiet session agrees to 0.15 % at 1.01358x. All 560 triangle-attention
calls per fold still serve on the fused SDPA and the fused qkv path with zero declines, so this is
the same kernels on narrower operands and not a fallback (`perf/c14_bfp8/regiont_quiet_ab2.json`).

An earlier reading of this flag was **1.04588x, +0.6640 s**, and it is refuted rather than
superseded. It is the same lever measured against a contended base arm: the on arm reproduces, the
off arm does not, and 15.135 s was a base fold taken while the board pair was busy. A p300c's two
chips share a power budget, and benchlock excludes other benchlock callers rather than other device
users, so a sibling that nobody leased can inflate the arm you are dividing by. Book +0.1750 s.

**Accuracy: 0.42886 Å worst at 512 residues against the 0.60 Å bar**, per pseudo-domain and
hinge-free, four seeds, paired same seed, card-independent, with a negative control
(`bfp8-accuracy-envelope`). Whole-molecule single-seed all-atom Kabsch between the two arms of the
speed run reads 0.523934 Å with the same-arm floor at 0.000000 Å exactly; re-folding with a different
seed moves the structure 1.02436-1.42336 Å, so the flag's effect is roughly half the variation the
sampler already produces. plDDT moves by 0.000017. Bit-exactness is lost by construction, which is
why it is a flag.

**Why it is off.** The default has not been flipped and the number above is a single size with a
single flag; 298 residues and any combination with `TT_BIO_TRIATT_BIAS_B8` are unmeasured, and block
float composes worse on accuracy than on speed. Turn it on per run if you want the second and can
accept a structure that is not bit-identical to the default.

## `TT_BIO_TRIMUL_MASK_L1` — on

The triangle multiplication masks its pair input before the contraction. The mask is `[1, 1, L, L]`
against a `[1, C, L, L]` chunk, so the multiply broadcasts it along the channel axis, and a
broadcast operand is read once per channel block rather than once. This flag puts the mask in L1,
where those re-reads cost no DRAM traffic.

It is the best ratio of the two: the mask is 0.52 MB, and moving that much on chip removes a whole
pair tensor's worth of DRAM reads. It fits at every size tested, 298 through 1024 aa, so unlike
`TT_BIO_RESIDUAL_L1` it does not go dark on large targets.

**Accuracy: identical**, on the same evidence as the section above: `torch.equal` at the op with a
control that fires, and one digest per size at the fold.

**Speed: 1.01492x on the trunk, taken with `TT_BIO_RESIDUAL_L1`.** The two flags place three
tensors in the same block, so they were measured together rather than multiplied together. Eight
folds an arm at 512 aa, interleaved ABBA in one process on an idle box: the Pairformer block wall
goes 9.5479 s to 9.4087 s, all eight paired ratios positive, against a same-arm floor of 1.0042x.

On the whole fold that is **17.736 s to 17.6325 s, 0.1035 s, 1.00748x**. Read the block number
rather than this one. The levers act only in the trunk, the fold wall is dominated by 200
diffusion steps they never touch, and the fold wall's own same-arm floor at this size is 1.01483x,
which is wider than the effect. All eight paired folds still came out positive, so the direction
is not in doubt; the size of it is better read where it happens.

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

## `TT_BIO_UNFUSED_SILU` — off

`Transition` is the engine's shared SwiGLU block. Its first matmul can apply silu as a fused
activation, and on Blackhole that costs more than running silu as a separate op afterwards: 174.0
us per call against 83.7 us at the 298 aa pair shape. The fused path runs silu at half the rate the
standalone op reaches, so unfusing pays a full extra round trip through L1 and still wins. The
penalty is specific to silu. A fused relu costs 2.4 us more than its standalone form and a fused
gelu 141.3 us more, and the gap holds across eight matmul program configs.

**This one is off, and it is held off on accuracy rather than on speed.** Turning it on costs
Protenix-v2 0.05088 and 0.07210 CA-lDDT per domain against the experimental 1HCL on cdk2x2_512,
with the two arms fully rank-separated over four seeds: the worst flag-off fold scores 0.03905 and
0.05107 above the best flag-on one. Protenix-v2 also loses 1.11 A and 1.19 A of CA-RMSD against
1HCL. Boltz-2 (-0.00091 / -0.00265 over four seeds) and OpenFold3 (-0.01238 / -0.00251 over two)
are clean on the same fixture, so a Boltz-2-only screen clears this lever and Protenix-v2 does not.
Nor does a structural seed-floor reading see it: Protenix-v2 scatters widely on that fixture while
landing at the same quality every time, so only the comparison against the experimental answer
separates the arms. Do not reopen without a Protenix-v2 CA-lDDT-vs-1HCL re-score on the
configuration you want to ship.

**Speed, if you turn it on anyway: +0.2336 s at 512 aa** (95 % CI [+0.1024, +0.3648]) paired over
five interleaved reps at a forced 1350 MHz, against the same session's paired A/A floor of
+0.0324 s +/- 0.1087.

**It is also not bit-identical.** Unfusing applies silu to the bf16-packed matmul output instead of
to the fp32 accumulator, so a structure moves. On Boltz-2 that movement is small: with
`TT_BIO_DIT_COND_HOIST` the pair deviates 0.25705 A all-atom at 298 aa, inside that fixture's
0.35 A band and 0.321x its own seed floor. That reading is exactly the screen the Protenix-v2
result above proves blind, which is why it does not clear the flag.

Scope, because the flag's name does not say it: every model that builds `Transition` inherits this
default. That is Boltz-2, Protenix, OpenFold3's MSA embedder, and the pairformer and MSA stacks.
OpenFold3's diffusion stack has its own SwiGLU transition and AF2 its own ReLU transition, and
neither reads this flag.

## Idle host threads when a box is full

Not a flag of ours. `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and `KMP_BLOCKTIME` are OpenMP's own, and
tt-bio fills them in for the per-card workers it spawns when, and only when, each worker is down to
two host threads. Set any of the three yourself and tt-bio leaves all three alone: they are one
setting spelled three ways, and `GOMP_SPINCOUNT=0` beside your `OMP_WAIT_POLICY=ACTIVE` would undo
it through the back door.

This covers every path that spawns one worker per card: `predict` across queued targets, ESMC
embeddings, and a BoltzGen design fanned out over a box of chips. They all take their worker
environment from one place, so the rule and the thread cap arrive together or not at all.

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
