# Release gate at the B2P integration tree

`gate_models_9af9e4152.txt` is `scripts/release_gate.py --model openfold3 --model boltz2 --keep`
run on qb1 UMD card 1 (a Blackhole p150a) at `9af9e4152`, which is `origin/main` at v0.10.0 plus
every B2P branch: b2p-wh, b2p-ceiling, b2p-ship, b2p-padup, b2p-soak. The pad-up
(`TT_BIO_TRIATT_HIFI_PAD_UP`) is on by default in that tree, so this is the gate reading a tree
where OpenFold3 takes the padded triangle-attention route at every 32 x p length.

    openfold3   RMSD 2.659 A   TM 0.777   floor <=3.5 / >=0.7    in band 1.0000   0 gaps   1 clash
    boltz2      RMSD 1.827 A   TM 0.902   floor <=3.0 / >=0.75   in band 1.0000   0 gaps   2 clashes
    GATE PASS, rc=0

Line 2 of the log records the tree it scored, `/home/ttuser/b2porch/tt_bio (commit 9af9e4152)`,
because a gate launched without `PYTHONPATH` scores the shared checkout instead.

`gate_models_9af9e4152_aiclk.txt` is the card's AICLK read from sysfs every 5 s for the length of
the run: 1350 MHz in every sample but the first, which caught the card still at its 800 MHz idle
floor. The gate's own wall column is not used as a perf number here.

`suites_9af9e4152.txt` is the card-free suite run at the same tree under `~/bcx_e2e_venv` with
`PYTHONPATH=~/b2porch:~/bcx_e2e/bc2`: 318 passed, 0 failed, 0 skipped, 156 s, `lsof` empty on all
four nodes throughout.
