# Upstream references at 1024, 1280 and 1536 tokens

Each structure model in `PREDICT_MODELS`, folded by its own upstream code on a GPU, on shared
fixtures, two seeds per cell. A TT fold at these sizes is scored against it, and the deviation is
read beside the reference's own seed-to-seed spread (the floor), never on its own.

## Score a TT fold

Fold the fixture yaml as it is, from the repo root, so the pinned alignments are used:

    tt-bio predict perf/mgx/ref/fixtures/3abq_1536.yaml --model boltz2 --out_dir /tmp/o
    python3 perf/mgx/ref/score.py --model boltz2 --fixture 3abq_1536 /tmp/o/.../3abq_1536_model_0.cif

It prints CA-RMSD, CA-lDDT and the worst single-chain CA-RMSD against each reference seed, and
the floor beside them. Chains are matched by sequence, so chain letters do not have to agree.
`--json` gives the full record, `--floors` the floor of every cell. A cell the upstream could not
fold prints its error instead. On a fixture with a crystal (7AQX, 2AD6, 3ABQ) a second line gives
the same numbers over only the residues the crystal resolves, floor included.

    boltz2 3abq_1536: vs ref s0 0.565 A / lDDT 0.9874 / chain 0.562 A | floor: single reference seed | n_ca 1406

(That line scores the 3ABQ crystal, so it also says the reference itself is right.)

Which number to read depends on the fixture. On 2AD6 and 3ABQ the seeds agree to about half an
angstrom and the whole-complex RMSD is the bar. On 7AQX the seeds disagree on where the nanobodies
dock, so read lDDT and the per-chain RMSD. Its nanobodies also carry a 19-residue HA and His6 tag
the crystal does not resolve, which folds sit anywhere up to 50 A apart; read the resolved line
there. The tiled CDK2 is one chain of repeated copies with no
defined arrangement between them, and some models fold it differently on every seed; read lDDT
there, and expect the floor to be wide.

## Fixtures

| fixture | what | tokens |
|---|---|---|
| `7aqx_1024` | VSG2 dimer with two nanobodies (PDB 7AQX) | 1012 |
| `2ad6_1280` | methanol dehydrogenase a2b2 (PDB 2AD6) | 1280 |
| `3abq_1536` | ethanolamine ammonia-lyase a2b2 (PDB 3ABQ) | 1518 |
| `cdk2x2_1024/1280/1536` | the size ladder's tiled CDK2 | 1024/1280/1536 |

Every chain carries one unpaired a3m, committed under `fixtures/msa/` and named in the yaml, so
the TT fold and the GPU fold read the same rows. Nothing is paired on either side: each upstream
pairs differently, and pairing on one side only would measure the pairing. Crystal structures for
the three PDB entries are in `fixtures/gt/`. The ladder's own 1280 a3m has 8 rows whose match
columns are not 1280; the esmfold2 and rf3 upstreams reject the file and tt-bio's protenix path
skips those rows, so `cdk2x2_1280` pins a copy without them.

## How the references were made

`make_plan.py` reads the model list and each model's recycles and sampling steps from
`tt_bio.main`, so the GPU folds at the settings tt-bio ships. `ref_setup.sh` builds each
upstream's own environment on a rented box and `ref_campaign.sh` runs `ref_fold.py` for every
cell, fp32 where the upstream has a switch for it. `collect.py` turns the box's output into `refs/` and
`manifest.json`, which records per seed the upstream package versions, checkpoint, GPU, dtype,
time and peak memory, and for a cell that could not be folded, the error.
