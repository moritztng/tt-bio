# Tuning flags

tt-bio ships its device optimizations on by default. This page says what each one changes and what
it was measured against, so you can decide whether to turn one off. Every flag takes `0`, `false` or
`off` to disable.

The reference numbers below are Boltz-2 on one Blackhole processor of a p300c (Tenstorrent
QuietBox), 512 aa, 200 sampling steps, 3 recycles, `perf/size512/fixtures/cdk2x2_512.yaml`. Fold
ratios are paired: each arm is interleaved against a flags-off control in the same process, and the
A/A repeat floor of that session was 1.00899x, so anything below about 1.01x is not distinguishable
from noise.

## `TT_BIO_ATOM_KEY_WINDOW` — on

The atom transformer attends within a sliding window. Upstream builds each window's key set by
multiplying the atom sequence with a one-hot selection matrix; the window it selects is a contiguous
run of atoms, so tt-bio slices that run out of the sequence directly.

**Accuracy: identical.** The fold is bit-for-bit the same as the matrix-multiply path at both sizes
we score (512 aa sha256 `a91aa44441f0d9c5`, 298 aa sha256 `0cf1b879dca3c0d5`, both equal to the
control), all-atom RMSD 0.00000 A, CA-lDDT 1.00000. Off-fold the slice reproduces the exact gather
by `torch.equal` while the matrix multiply misses it by 0.03125, so the slice is the more faithful
of the two as well as the faster one.

**Speed: 1.02744x on the fold** (19.658 s to 19.133 s), 6.5x on the operation itself.

The path is selected on geometry alone: the query window, the key window and the shape of the
selection matrix the caller was handed. Any model whose atom attention has that geometry takes it;
no model name appears in the condition.

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

1.09858x on the fold, 19.658 s to 17.894 s, n=10 against a 25-fold interleaved control. The three act
on disjoint parts of the fold, so they compose: the product of the individual ratios is 1.10118x and
the measured union is 1.09858x, a 0.24 % gap inside a 0.90 % floor. Per domain the union moves
0.4351 A and 0.4598 A at 512 aa and 0.3589 A at 298 aa.

## A note on the RMSD figures

`cdk2x2_512` is two copies of the same 256 aa protein joined end to end. Any small change to the
prediction can rotate one copy against the other, and a whole-molecule RMSD reports that rotation
rather than any change in the folds themselves. So the figures above superpose each copy on its own.
The whole-molecule numbers are larger and we do not hide them: SiLU 0.8384 A, device conditioning
0.9292 A, all three 0.7534 A. The reference point is that flags tt-bio already shipped before this
page existed move the same fixture 8.60 A whole-molecule while leaving each copy's fold intact.
