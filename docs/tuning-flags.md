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

The path is selected on geometry alone — the query window, the key window and the shape of the
selection matrix the caller was handed. Any model whose atom attention has that geometry takes it;
no model name appears in the condition.
