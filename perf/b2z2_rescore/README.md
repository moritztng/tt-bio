# b2z2-killed-levers-rescore

Three levers were killed at 512 aa on one whole-molecule Kabsch all-atom RMSD against a 0.60 A bar
(`perf/b2z2_compose/score.py`). `b2z2-fusebias-512-parity` then measured what that fixture does on
its own: same arm, different seed, per-domain all-atom 0.967 - 1.906 A and whole-molecule
6.80 - 17.36 A. The bar sits an order of magnitude below the fixture's own noise, so this row
re-reads the kills with the instrument the fixture supports.

`perf/b2z2_fusebias/score.py` is vendored here byte-identical to `wk/b2z2-fusebias-512-parity`
(md5 b17b3cf3112640a1a4f03fc912e8059e). It reproduces the committed `perf/b2x-integrate/split512.json`
numbers to five decimals. No fourth scorer was written.

    prepare_existing.py   renames an already-folded arm into the scorer's layout. The HOST and MSA
                          CIFs survived on qb2, so re-reading them costs no folds.
    fold_levers.py        the four-seed re-fold, for arms whose CIFs are gone.

The bar is the fusebias row's, unchanged: KILL above 0.60 A all-atom on either pseudo-domain
(split at `label_seq_id` 298, each superposed alone); KILL if the lever's worst per-domain move
exceeds the seed floor's worst; KILL if native CA-lDDT against 1HCL drops with one sign across all
four seeds by more than the seed spread.
