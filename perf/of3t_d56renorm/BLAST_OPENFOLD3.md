# Fold digests, what landed and what the card cost

Card qb1 physical 3, Blackhole p150a, AICLK 1350 MHz. Fixture
`perf/size512/fixtures/cdk2x2_128.yaml`, seed 0, 6 sampling steps, 1 diffusion sample.
`renorm_blast_radius.py` was launched three times. No single launch completed all arms, so
there is no machine-readable `BLAST_RADIUS_renorm.json` yet and this file records what each
launch actually produced rather than a summary that outruns it.

## Launch 1 -- refuted the harness, not the lever

Every arm exited 99. The harness ran `tt_bio.main` through `runpy` in its own process so it
could read the counters afterwards, and that breaks `multiprocessing` spawn. The deeper
problem it exposed is the one worth keeping: a `predict` run does its device work in SPAWNED
workers, so counters read in the launcher are blind to the processes that ran the model. The
counter dump moved into `tt_bio.taped_ttnn` under `TT_BIO_RENORM_STATS_DIR`, one file per pid,
summed by the reader.

## Launch 2 -- OpenFold3, the result

    arm    rc   renorm counters        cif sha256
    off     0   applied 0 declined 0   35d36fb149583ae5627085ce60db8e7127726a793e9dbf5be7b779f5c25af1b1
    on      0   applied 0 declined 0   35d36fb149583ae5627085ce60db8e7127726a793e9dbf5be7b779f5c25af1b1
    f64     0   applied 0 declined 0   768b47cfd9a7683e4a9f0d34aa479de1fe0089393310151de477b288db38c666

`off` and `on` are byte-identical. `f64` is the sensitivity control and it MOVES the digest, so
the identity above is a reading rather than a blind instrument. The counters are 0/0 and not
merely `applied 0`: `declined` counts the branch being evaluated, so 0/0 says the backward
closure never ran at all. `renorm_tape_control.py` drives the same counters non-zero on the
same tree, which is what makes those zeros informative.

Launch 2 was killed after these three arms because the tree was edited mid-run (the default
flip landed while it was going), which makes its later arms unrepeatable by construction.

## Launch 3 -- lost every arm to a co-tenant

Relaunched on the final tree with a fourth `default` arm, the one that scores the shipped
default rather than a forced flag. `openfold3/off` completed and reproduced
`35d36fb1...`; every later arm exited 75 after waiting the full 120 s lease timeout:

    DeviceInUseError: physical card 3 on tt-quietbox is in use by
    worker:of3t-d1-pairbias (pid 4104595); waited 120s

`state/leases/qb1-card3.json` records card 3 as held by `worker:of3t-d56-renorm`. The sibling
row was running `fold_targets.py --arm fix --targets 1ubq,1shg --seeds 1,11,21,31,41,51` on it,
and a second sweep of its own on card 2, which `state/leases/qb1-card2.json` assigns to
`of3t-d10d24-unify`. All four qb1 cards had open fds, so there was no sibling to fan out to.
The lease code refused the open rather than letting two processes collide, which is the
mechanism working.

## Owed

`default`, `on` and `f64` on OpenFold3 re-run in one launch on the final tree, and all four arms
on Protenix-v2 and OpenDDE. Both read the same two lines of shared code the AST proof covers.
