# K4 (`TT_BIO_SDPA_BAND_DIV_K`) release gate — arm status at `0cea91cda`

Journal key: `{card_type: p300c, commit: 0cea91cda, dirty: false, fast: false,
diffusion_trace: false, host: tt-quietbox2, package: <this worktree>/tt_bio}`.
The journal itself (`perf/gate_journal/journal.jsonl`) is deliberately not committed — it is a
property of one host's runs. This file is the committed transcription of it.

**12 of 13 arms PASS. Zero FAIL. Outstanding: `size-ladder`.**

| arm | verdict | taken |
|---|---|---|
| fold-models | PASS | 2026-09-22 ~14:00Z |
| rf3-1024aa | PASS | 2026-09-22 ~14:00Z |
| rfd3 | PASS | 2026-09-22 ~14:00Z |
| rfd3-fusion | PASS | 2026-09-22 ~14:00Z |
| boltzgen | PASS | 2026-09-22 ~14:00Z |
| pxdesign | PASS | 2026-09-22 ~14:00Z |
| opendde-abag | PASS | 2026-09-22 15:12Z |
| nesso1 | PASS | 2026-09-22 15:12Z |
| capacity | PASS | 2026-09-22 15:12Z |
| l1-budget | PASS | 2026-09-22 18:27Z |
| batch-position | PASS | 2026-09-22 18:27Z |
| esmc (300m + 600m) | PASS | 2026-09-22 18:27Z |
| **size-ladder** | **outstanding** | killed by a host hard-reset at 17:20Z |

## The three arms taken 18:27Z

**l1-budget** — the arm Region T failed on card dependence, so it is the one that matters most
for a lever that changes a chunk size. With K4 on, all three grids agree byte-for-byte:

    leg                     clashes                             CIF md5     wall  result
    l1-budget:arith               -                165 checks / 4 parts       0s  PASS
    l1-budget:native              0    c3073854d423570ae48cb8ce35ccb27e      97s  PASS
    l1-budget:8x8                 0    c3073854d423570ae48cb8ce35ccb27e      64s  PASS
    l1-budget:narrow              0    c3073854d423570ae48cb8ce35ccb27e      59s  PASS

**batch-position** — three identical 256 aa targets, one 200 aa control, one process:

    pos  target                  coords   affinity   p(bind)     wall
    1    t1.yaml       62d70c2c60b1ed75   0.643811  0.287798     210s
    2    t2.yaml       62d70c2c60b1ed75   0.643811  0.287798     117s
    3    t3.yaml       62d70c2c60b1ed75   0.643811  0.287798     104s
    4    x200.yaml     7cf1215e1d6da4e1   0.530447  0.275993      96s

**esmc** — embedding parity against reference esm, PCC floor 0.99:

    model         per-res PCC   pooled   logits   argmax     wall  result
    esmc-300m         0.99963  0.99992  0.99989   1.0000      13s  PASS
    esmc-600m         0.99964  0.99989  0.99996   1.0000      13s  PASS

## Why size-ladder is outstanding, and why it is blind to K4 anyway

It is blind by construction: K4's guard is `256 < q_len <= 384 and 256 < k_len <= 384` on the
padded token length (`tt_bio/tenstorrent.py:1342`), and `k4_band_negcontrol_out.txt` shows the
pick moving at exactly 298/320/384 and byte-identical at every ladder rung
(256/512/640/768/896/1024/1088). Its flakes also predate K4: `openfold3-256-warmup` and
`rf3-896-warmup` failed on Sept 21, before K4 flipped on at Sept 22 13:49.

That is a reason to expect it green, not a reason to skip it. It is still owed.

It cannot be banked a model at a time. `gate_journal.resumable()` keeps only the LATEST record
per arm, so a `--size-ladder-models <one>` run replaces a partial record instead of accumulating
with it. `--size-ladder-fragment` and the rung-at-a-time carry-forward are record-mode only. So
it is one ~2h45m process that journals a single verdict at the end.

## How the gate must be launched here

The only python with `ttnn` is `/home/ttuser/tt-bio-dev/env/bin/python3`, and it has `tt_bio`
installed against the SHARED checkout. Without `PYTHONPATH` the gate scores `tt-bio-dev`, not
this worktree — it did, at commit `56ad6c0e0`, until it was killed and relaunched as:

    PYTHONPATH=/home/ttuser/.coworker/wt/land-standing \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:land-standing \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --resume \
      --journal perf/gate_journal/journal.jsonl
