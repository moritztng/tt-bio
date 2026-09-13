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
