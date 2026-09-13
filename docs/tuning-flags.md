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

## `TT_BIO_SDPA_GRID_Q_CHUNK` — on

Scaled dot-product attention is computed in chunks of query rows, and ttnn hands one chunk to one
core. The chunk size tt-bio shipped was a fixed cap with no term for the card: an attention with few
heads produced fewer chunks than the card has cores and left most of the grid idle for the whole op.
This picks the widest chunk whose work still fills a single pass of the compute grid. The core count
comes from the device and the head count from the tensor, so no card and no model is named.

**Accuracy: identical.** The query axis partitions independent rows — each chunk computes its own
rows and nothing is combined across chunks. The softmax reduction order lives in the key axis, which
this does not touch. `torch.equal` and max abs 0.0 at every call site, and one structure across 28
timed folds at equal pLDDT.

**Speed: 1.00355x on the fold**, 95 % CI [1.00180, 1.00506] over 28 folds against the same session's
A/A floor of [0.99934, 1.00147], and **1.1269x at the one call site it moves** (117.5 to 103.9 us,
50 paired reps).

**It moves one call site of two, and that is a property of the shapes.** On the 512 aa reference the
diffusion step's token attention goes from 256 query rows to 128 — 64 work units on 110 cores where
the fixed cap gave 32 — on all 4800 of its calls a fold. The atom attention's 32 query rows are a
single tile, so there is nothing to split, and all 1200 of its calls keep the shipped chunk. The
ratio is a Blackhole number and does not transport: the same rule is 1.3058x at the op on Wormhole,
because the win is occupancy and occupancy depends on the grid. Narrower chunks also re-read the
keys and values once more per chunk, 20.1 GB more traffic over the fold, which the idle cores more
than pay for here but would not on every card.
