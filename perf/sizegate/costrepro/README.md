# protenix-v1 256 aa cost repro

Settles the +56% gap the size-ladder baseline flagged at the 256 rung: recorded 4.5 s, checked
7.0 s, same card. It is host CPU contention. Verdict and full write-up in
`~/.coworker/state/protenix-v1-256aa-cost-repro.md`.

    quiet.log     loadavg 0-2.6, 8 reps      4.4 s median, 2.3% spread   the recorded number
    loaded.log    loadavg 8-10, 5 warm reps  7.1 s median                the checked number
    loadctl.log   loadavg 16, 5 reps         9.9 s median                positive control
    b2load.log    boltz2 under the same load 15.5 s vs a recorded 5.1 s

`loaded.log` rep 0 (78.4 s) and `b2load.log` rep 0 (40.4 s) are cold-kernel-cache first folds.
`runtime_s` excludes model load but not first-touch JIT compile.

`repro_driver.py` drives `release_gate._run_census_fold` directly, so the fold config is the arm's
own. Run it from a checkout with `PYTHONPATH=$PWD`, pinned to one card:

    REPRO_REPS=8 REPRO_TAG=quiet TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python repro_driver.py

`REPRO_MODEL` and `REPRO_RUNG` pick a different cell. Every rep is a fresh process; the driver
records the loadavg it started each rep at, which is the variable that turned out to matter.
