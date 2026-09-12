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

## `TT_BIO_UNFUSED_SILU` — on

Transition blocks apply SiLU to the output of a matrix multiply. TT-NN can fold that activation into
the matrix multiply, but it then runs at half the rate the standalone operation reaches on this
hardware, so tt-bio runs it separately.

**Accuracy: not identical.** The unfused form applies SiLU to the rounded matrix-multiply output
rather than to the full-precision accumulator, so the structure moves. On `cdk2x2_512`, four seeds,
measured per domain: 0.244-0.404 A and 0.241-0.418 A. For scale, the same model at a different seed
moves 0.969-1.869 A, so the flag's largest move is well inside the sampler's own spread. Whole
molecule the fixture reads 0.8384 A, but that number is dominated by a hinge between its two domains
rather than by either domain's fold; see the note at the bottom of this page. The 298 aa monomer
control moves 0.190-0.429 A. Accuracy against the experimental structure (CA-lDDT vs 1HCL) is flat:
-0.0009 and -0.0027 on the four-seed mean, inside seed spreads of 0.029 and 0.030, with no consistent
sign.

**Speed: 1.02423x on the fold** (19.658 s to 19.193 s), 1.04445x on the pairformer block, which is
where all of it comes from.

## `TT_BIO_DEVICE_CONDITIONING` — on

Boltz-2's diffusion conditioning reads the pair tensor the trunk just produced. That tensor is
already on the card, so tt-bio runs the conditioning's pair track there too. The 201 MB download the
host path did, and the upload of the token bias that followed it, both disappear.

**Accuracy: not identical.** Device math is bf16 where the host path was fp32. On `cdk2x2_512`,
per domain: 0.3947 A and 0.3304 A. The 298 aa monomer control moves 0.3938 A, whole molecule. CA-lDDT
against the flag-off structure is 0.99341, and accuracy against the experimental structure goes up
rather than down on both copies.

**Speed: 1.04642x on the fold** (19.658 s to 18.786 s), none of it inside the pairformer block: it
comes out of the conditioning itself and out of host time with no device work under it.

## All three together

The measured union is **1.09858x** on the fold, 19.658 s to 17.894 s, n=10 against a 25-fold
interleaved control, and it moves each domain 0.4351 A and 0.4598 A at 512 aa, 0.3589 A at 298 aa.

That arm ran an earlier implementation of the key-window gather, which has since been replaced by
`TT_BIO_ATOM_SHIFT_GATHER`. The substitution is worth **1.09046x projected** (1.09858 x 1.01985 /
1.02744) and leaves the accuracy figures where they are, because both gathers write the identical
structure at this size. The projection is arithmetic, not a measurement; a union arm on this tree is
owed before the number goes anywhere a user reads it.

## A note on the RMSD figures

`cdk2x2_512` is two copies of the same 256 aa protein joined end to end. Any small change to the
prediction can rotate one copy against the other, and a whole-molecule RMSD reports that rotation
rather than any change in the folds themselves. So the figures above superpose each copy on its own.
The whole-molecule numbers are larger and we do not hide them: SiLU 0.8384 A, device conditioning
0.9292 A, the union 0.7534 A. The reference point is that flags tt-bio already shipped before this
page existed move the same fixture 8.60 A whole-molecule while leaving each copy's fold intact.
