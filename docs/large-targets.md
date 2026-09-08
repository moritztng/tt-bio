# Large targets on Wormhole

The pair representation every AlphaFold3-family model carries is an N×N tensor, so its memory
grows with the square of the token count. On a 12 GiB Wormhole chip the naive implementation
crosses the ceiling at roughly 850-1100 residues, depending on the model's pair channel width.
OpenDDE is the strictest case: its structural-token expander runs the refiner on about 1.9x the
residue count, and its pair tensor is 3.7x the residue-scale one.

The pair-track ops (triangle multiplication, triangle attention, the pair transition) are all
row-local along the token axis, so past a size threshold they run in row blocks and free every
intermediate that is not the input or the output. The peak then scales as three live pair
tensors instead of four. The MSA features of very deep MSAs are streamed from the host between
recycling cycles instead of staying resident. ESMFold2's pair initialisation is row-tiled the
same way.

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
