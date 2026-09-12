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
    cif_bh/               those CIFs, committed, so the next appeal does not depend on qb2's /home.
    out/                  the runs JSONs, the per-domain scores, and the rewritten compose scorer's
                          output over the two runs it originally killed.

The bar is the fusebias row's, unchanged: KILL above 0.60 A all-atom on either pseudo-domain
(split at `label_seq_id` 298, each superposed alone); KILL if the lever's worst per-domain move
exceeds the seed floor's worst; KILL if native CA-lDDT against 1HCL drops with one sign across all
four seeds by more than the seed spread.

## What it found

BH, qb2 card 1, seed 0, each arm against its own base folded in the same process:

| arm | domain 1 | domain 2 | hinge | whole molecule | old verdict | new |
|---|---|---|---|---|---|---|
| HOST | 0.580 A | 0.403 A | 4.37 deg | 0.771 A | kill | **pass** |
| MSA ladder | 0.709 A | 0.565 A | 4.48 deg | 0.977 A | kill | kill |

Against the fixture's own seed floor (WH, twelve same-arm pairs): per-domain 0.967 - 1.906 A,
whole-molecule 6.80 - 17.36 A, hinge 52.6 - 157.5 deg. Both levers move the hinge 12 - 36x less
than a seed change does, which is what the whole-molecule bar was reading and calling a loss.

`transition`, the third killed lever, could not be re-read: `wk/b2z-bfp8-narrow` committed its
JSONs but not its structures and nothing matching `512_transition*` survives on whglx, qb2 or qb1.
A kill that does not commit its CIFs cannot be appealed.

Still owed: four seeds per arm on one architecture, so the floor and native-lDDT bars apply without
crossing WH and BH. Seed 0 puts HOST 0.02 A under the bar on domain 1, so that leg is what would
settle it.
