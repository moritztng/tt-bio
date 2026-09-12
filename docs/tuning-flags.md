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

**Accuracy: identical.** 40 timed folds across both arms wrote the same structure byte for byte, one
sha256, pLDDT equal to six places. Off the device, the slice reproduces the reference gather under
`torch.equal` at every window count the fold produces, padded and unpadded, and refuses four negative
controls (`perf/b2z2_elision/test_shift_equiv.py`).

**Speed: 1.01985x on the fold** (19.8545 s to 19.4680 s, 20 folds per arm, all 10 paired reps
positive), 1.07562x on the sampling stage, which is where the whole gain is.

tt-bio decides on the selection matrix, not on the shape of it: it reconstructs the matrix a centred
sliding window would produce and compares entry by entry. A model whose atom attention selects a
different key set, or pads its matrix differently, fails that comparison and keeps the matrix
multiply. No model name appears in the condition.

Reading the matrix rather than its shape is the whole safety argument. The atom axis is padded out
to a bucket, and the selection matrix is empty in the windows past the real atom count, so that
matrix and a same-shaped one that selects real atoms there are two different transforms with
identical dimensions. An earlier version of this optimization looked at the shapes only, got those
windows wrong, and still wrote the identical structure, because the attention mask is built from the
same matrix and discards exactly the entries the selection got wrong. Nothing the model outputs
distinguishes the two, at any size. The matrix comparison does.
