# 640 aa old-gate vs new-gate divergence

The brief for this fix asked for a bit-exact A/B at 256 / 512 / 640 aa. 256 and 512 are bit-exact
(structure digests `56fda066ae75d81e41e1937d6b611cab` and `0d24697b4c57b92743003191e59f6ef9`,
identical between a pristine pre-fix build and the fix), and `perf/wchunk_equiv.py` shows why that
is structural rather than lucky: zero rungs at or below the old 608-token gate change their
(chunked, w_eff, h) triple, on any of the four shipped channels.

640 aa cannot be bit-exact and never could. The threshold resolves to 608 on this grid, not 640,
so 640 sits INSIDE the old chunked band: the fix changes the path there by design
(old `(chunked, w_eff=480, h=8)` -> new `(unchunked, w_eff=640, h=6)`), and a different row
blocking gives the matmuls a different accumulation order.

`cmp640.py` measures how far apart the two arms land, with the atom ordering asserted equal and a
shuffled-atom negative control so the superposition cannot flatten a real difference:

    atoms compared: 5141
    as-written (no superposition): rmsd 2.2711 A, max 15.7782 A
    superposed:                    rmsd 2.2656 A, max 15.7651 A
    neg control (shuffled atoms):  rmsd 144.0683 A

    old640 (WTHR=608, chunked)        pLDDT 0.775901  pTM 0.109952  5141 atoms  224.9 s
    new640 (derived gate, unchunked)  pLDDT 0.775874  pTM 0.109041  5141 atoms  224.0 s

Confidence is identical to 3e-5 in pLDDT while the coordinates differ by 2.27 A, on a target the
model itself scores pTM 0.11. cdk2x2 is a chimeric fixture, not a deposited structure, so it cannot
score parity in the first place: two equally-unconfident folds of a chimera are free to sit 2 A
apart. Reproduce with:

    python3 perf/wchunk-p2/cmp640.py old640.cif new640.cif
