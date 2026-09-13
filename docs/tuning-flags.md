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
