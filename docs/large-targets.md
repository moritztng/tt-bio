# Large targets on Wormhole

The pair representation every AlphaFold3-family model carries is an N×N tensor, so its memory
grows with the square of the token count. On a 12 GiB Wormhole chip the naive implementation
crosses the ceiling at roughly 850-1100 residues, depending on the model's pair channel width.
OpenDDE is the strictest case: its structural-token expander runs the refiner on about 1.9x the
residue count, and its pair tensor is 3.7x the residue-scale one.

The pair-track ops (triangle multiplication, triangle attention, the pair transition) are all
row-local along the token axis, so past a size threshold they run in row blocks and free every
intermediate that is not the input or the output. The peak then scales as three live pair
tensors instead of four. The blocks are joined on the chip, and on the host only when the chip refuses the join. A deep alignment's MSA representation lives on the host and passes
through the chip one depth chunk at a time instead of being held whole beside the pair
tensors. ESMFold2's pair initialisation is row-tiled the same way.

Measured on the WH Galaxy (12 GiB chips) on the four targets the AbAg-XM campaign had to
exclude:

| target | residues | boltz2 | esmfold2 | protenix-v2 | opendde-abag |
|---|---|---|---|---|---|
| 9q7y | 853 | OK | OK | OK | OK |
| 9ivj | 891 | OK | OK | OK | OK |
| 9i3p | 980 | OK | OK | OK | OK |
| 9j4c | 1095 | OK | OK | OK | OK |

The last cell to fall, OpenDDE on 9j4c, was a lifetime bug rather than capacity: the structural
pair tensor and the sampler's pair bias stayed resident through the residue-axis confidence
stage that never reads them, so the confidence pair track started with under 2 GiB free. Both
are freed at the diffusion boundary now, which drops the confidence entry from 10.04 to
5.72 GiB.

RoseTTAFold3 was stuck at 627 residues on the same pool until 2026-09-07, and neither wall was
memory the model needs. Its template embedder and MSA module were the last two triangle-attention
sites still materialising the whole score tensor over the raw token axis, so 656 residues asked
for a single 2.37 GB buffer at the first trunk step; both now take the same fused attention the
rest of the model uses. Separately, the confidence head's global layer norm flattened the pair
tensor to a single row, and a tile-layout row pads to 32, so normalising a 0.10 GB tensor at
640 residues asked the allocator for 3.36 GB. Folding that flatten into rows pads nothing. RF3
then folds 630, 656, 716, 796, 891, 980 and 1095 residues with real alignments, at 80 pLDDT and
zero backbone breaks. None of the three changes is bit-exact with what it replaced;
`TT_BIO_RF3_TEMPLATE_FUSED_SDPA=0`, `TT_BIO_RF3_MSA_FUSED_SDPA=0` and
`TT_BIO_RF3_GLN_ROW_FOLD=0` restore the old routes and the old ceiling.

RFdiffusion3 hit the same shape of wall on the design side and it went from 490 to 704 to 1024
residues over two passes. First the token initializer's atom-pair section, which was building its
product over the whole token axis, so 640 residues asked for a single 2.11 GB buffer; it is
row-blocked against the part's own DRAM now, which is both lighter and faster (71.2 s against
124.5 s, and 8.2 GB host RSS against 16.1 GB, at 512 residues on the same card with the budget the
only difference). Designs up to about 310 residues stay in a single block and run the old path byte
for byte, and the five tensors the initializer hands the sampler are md5-identical at 128, 256, 480
and 512. `TT_BIO_ATOM_PAIR_BUDGET_BYTES=0` restores the unblocked path and the 640-residue wall
with it.

That left 704, and the thing holding it was a budget that had never applied. The pair transition
priced the L1 residency of its SwiGLU against a fixed 138 MB, and a Wormhole chip has 100 MB of L1,
so nothing was ever declined and at 768 residues the op asked the allocator for two buffers that
together miss a bank by 5408 bytes. 0.4 %, which is why 768 used to fold two times in three instead
of failing cleanly. The budget reads the part it is running on now, so above 704 the wider pair
transition writes to DRAM instead, and RFD3 designs 640, 704, 768, 832, 896, 960 and 1024 residues
on a 12 GiB Wormhole card, in 141 to 286 seconds, twice each on two different chips with
byte-identical output. Declining costs about 4 % of wall where it fires and buys the design.
Nothing below 768 changes path, and at 768 itself the old unconditional-L1 route returns the same
structure to the byte, so this is a capacity change and not an accuracy one.

Both ceilings hold at the 100 diffusion steps a real run uses, not at a short one. A 2-step run
never passes `D_II_self`, so it never enters self-conditioning, and the same SwiGLU then asks for
half the bytes. A ladder walked at 2 steps reported 992 and that number is withdrawn.

## 1536 tokens on one Wormhole chip

ESMFold2 (both checkpoints), Protenix-v1, Protenix-v2, Boltz-2 and RoseTTAFold3 fold 1536-token
targets on a 12 GiB Wormhole chip. On 3ABQ (1518 residues, real MSA) every one of them lands
0.6 to 1.6 Å CA-RMSD from the crystal, with CA lDDT at or above 0.95.

Before this, the first failures were between 1056 (ESMFold2) and 1184 (Protenix-v2) tokens, and
none of them was one tensor too big for the chip. Each was an allocation that DRAM refused
because too many pair-sized buffers were live at once or the free space was fragmented. Three
changes fix that class in the shared pair-track code:

- When DRAM refuses a whole-tensor pair op, the op re-runs in row blocks instead of failing, and
  the shape is remembered so later calls go straight to the blocked path.
- When the row blocks fit but their device concatenation does not, the blocks are assembled on
  the host and uploaded once.
- Tensors that a later stage never reads are freed when the stage ends. ESMFold2 releases its
  ESMC-6B language model after its single forward on Wormhole targets above 1088 tokens, which
  frees about half of the chip; a server folding many such targets reloads it once per fold.
  Protenix and RoseTTAFold3 free the diffusion conditioning before the confidence head.

The fallbacks only fire on a refusal, so a target that fits keeps its single pass. At 1024
tokens the output matches the previous code to within 0.41 Å CA-RMSD, and bit for bit on most
models, against a seed-to-seed spread of 1.7 to 3.3 Å on the same targets. The walls that remain
above 1536 are listed [below](#what-stops-each-model-above-1024-on-a-galaxy-chip).

**A cell that folds does not license the sizes below it.** These four targets fold, and OpenDDE
still throws at 576 residues on the same pool. The throw is an L1 static circular-buffer clash:
the layout follows the padded tile shape and the core-grid split, neither of which is monotonic
in residue count, so OpenDDE folds 544, throws at 576, and folds 608 again. The clashing address
also depends on the process's own allocator history, so one passing fold never establishes a size
as safe. What tt-bio enforces is therefore the largest size below each model's *first* measured
failure, not the largest that has been seen to work; the table of those limits is in the README
and the measured rows are in `tt_bio/size_limits.py`.

Per-cell fold times and peak DRAM are in the release notes of the version that landed this.
Normal-size targets never enter the blocked path: it is gated on a token-count threshold that
a 300-residue target does not reach, and the before/after timings on the standard 117/298-residue
benchmarks are unchanged within noise. The threshold is 1536 on Blackhole. On Wormhole it is
derived from the part's DRAM, so it varies by machine: on the Galaxy above it is 1088.

A token count alone does not decide it, though. The outer product mean builds an
`N x (C*D) x N` product and then permutes it, and a permute is out-of-place, so two of that
tensor are live at once. Between about 887 and 1088 tokens that pair is 3-4 GiB on a 12 GiB
part whose address space the trunk has already churned, and the allocator can refuse it with
half of DRAM free because no single hole is big enough — 992 residues on OpenDDE was refused by
704 bytes per bank. So the same block size that bounds the blocked path also bounds the whole
one, and that band takes row blocks whatever the token threshold says. Anything below it keeps
the byte-identical unblocked path, and on a 32 GiB Blackhole part the bound is above every size
the models reach, so Blackhole never changes path.

## What stops each model above 1024 on a Galaxy chip

Walked on one j10glx02 chip up to 2048 with the alignment depth each model reads (16384 rows;
Boltz-2 at its 8192 default, RF3 drawing 1024 per recycle), every first failure above 1536 is a
single pair-sized allocation that no free block on the chip can hold:

- `boltz2` folds 1920 and fails at 2048 in the diffusion cache, one 3.0 GiB tensor with 55 % of
  the chip free but no block large enough.
- `protenix-v1` folds 2048, the top of the ladder, since the diffusion transformer's pair bias
  waits on the host. Before that it failed at 2048 on that bias, with enough memory free but no
  block large enough.
- `protenix-v2` folds 1920 since the pair path row-blocks an allocation the chip refuses and a
  refused outer product mean re-runs in alignment-depth chunks. Above 1792 the trunk joins its
  pair blocks on the host, so 1920 takes about two hours. Before the depth chunks it failed at
  1920 in the outer product mean, and before the row blocking at 1792 in the diffusion pair
  conditioning.
- `rf3` folds 1600 and fails at 1664 in the diffusion atom encoder's pair permute, with enough
  memory free but no block large enough.
- `opendde` and `opendde-abag` fold 1536. At 1664 the residue pair no longer fits as one
  allocation, so every pair operation assembles its blocks on the host and a trunk recycle takes
  about 20 minutes instead of under 9. Both runs were past the 107 minutes the platform allows a
  1664-residue fold with three or four of their ten recycles still to run, so the limit stays at
  1536 because of run time, not a crash. At 1920 the fold fails outright on the trunk's pair.

`esmfold2` and `esmfold2-fast` both fold 1664 and fail at 1792 in the pair feed-forward, whose
1.5 GiB output needs 130.7 MiB in every DRAM bank. The largest free block is about 122 MiB by then,
with 86 % of the chip in use. `esmfold2` lands in the same place with its alignment
and single-sequence, because its MSA encoder sees at most 1024 rows per trunk loop.

`openfold3` and `openbind` fold 1536 residues at 14190 alignment rows now that the MSA
representation streams through the chip a depth chunk at a time. Both fold 1664 and fail at 1792
in the diffusion transformer's attention scores, one 206 MB tensor on a chip 96 % full whose
largest free block is 640 bytes per bank too small. The measured rows, with commits, wall times and allocation sizes, are in
`tt_bio/size_limits.py`.
