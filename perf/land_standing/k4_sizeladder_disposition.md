# Why the size-ladder arm does not hold K4

K4 (`TT_BIO_SDPA_BAND_DIV_K`) took 12 of 13 release-gate arms at `0cea91cda` with zero FAIL. The
thirteenth, size-ladder, is a ~2h45m single-verdict run that killed this host twice on 2026-09-22.
This file is why it is not worth a third attempt, and it is two independent facts, either of which
is sufficient.

## 1. The arm is red on main, and it is red for reasons that predate K4

Every size-ladder verdict recorded anywhere in this repo is FAIL. Five distinct commits, both
board classes, three rows, and **all five commits are ancestors of `origin/main`**:

    commit      card    host           dirty  members                      journal
    8bede8dfd   p300c   tt-quietbox2   False  6 of 9 models, individually  perf/pvx_gate_land/journal_gateland_8bede8dfd.jsonl
    8bede8dfd   p300c   tt-quietbox2   False  boltz2 (quiet re-run)        perf/pvx_gate_land/journal_boltz2_quiet.jsonl
    702aa961c   p300c   tt-quietbox2   -      openfold3                    perf/land_standing/out/ladder_of3_journal.json
    56ad6c0e0   p300c   tt-quietbox2   True   openfold3                    perf/land_standing/out/ladder_probe_journal.json
    0c0b4248e   p150a   tt-quietbox    -      openfold3                    perf/land_standing/out/ladder_of3_p150a_journal.json

The strongest of these is `8bede8dfd`: **p300c, this host, clean tree, scored from another row's
own worktree, six of the nine ladder models failing individually** -- taken 2026-09-20, two days
before K4 flipped on. `9dcde64f3` adds the p150a half from a detached `origin/main` worktree at
`e1d887a54`, where protenix-v1 fails 24 per-rung checks whose md5 matches the branch run's md5
exactly (`444db3e22175380daa205ca5a534e4be`, 24/24) -- the drift is baseline-vs-reality, not
branch-vs-main.

`perf/pvx_gate_land/sizeladder_drift_attribution.txt` attributes it: SDPA_WIDE_K, REBLOCK_PERMUTE,
PAIR_PROJ_MINIMAL_MATMUL, QKV_MM_CONFIG and TRIATT_HEAD_MAJOR_QKV all landed on main and were never
carried into `docs/size_ladder_baseline.d/`. The arm is measuring the age of its own baseline.

## 2. K4 cannot move the arm, and this is executed rather than argued

K4's guard is a literal `256 < q_len <= 384 and 256 < k_len <= 384` on the padded length. The
ladder's rungs are `SIZE_LADDER_RUNGS = (256,512,640,768,896,1024)` plus `SIZE_LADDER_CARD_RUNGS`
`{"p150a": (1152,1280,1408,1536)}`. Running `_sdpa_chunks_shipped` at every one of those eleven
rungs with the flag off and on, on the Blackhole grid:

    rung   1024 1088 1152 1280 1408 1536  256  512  640  768  896   -> moved: none
    moved at any ladder rung: []

The pick differs at exactly 298, 320 and 384 (`perf/land_standing/k4_band_negcontrol_out.txt`),
and 256 is excluded by the strict `>`. So no ladder rung enters the branch K4 changes. A rung is
byte-identical with the flag on, which is why the l1-budget arm -- the one Region T failed on --
returned the same CIF md5 `c3073854d423570ae48cb8ce35ccb27e` across three core grids with K4 on.

## What is actually owed, and it is not owed by K4

The baseline corpus needs re-recording against current main (`--size-ladder-record`, an explicit
human action by the arm's own design). Until someone does, the arm is red for every candidate and
gates nothing. Five candidates have now had to reason around it. That is a defect in the arm, and
filing it against the next perf lever that happens to arrive is how a campaign ships 0.0000 s.
