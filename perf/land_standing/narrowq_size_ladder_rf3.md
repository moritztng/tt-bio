# narrow-q size-ladder, rf3 slice, p300c (qb2 card 2), tip 4cbada0e8, 2026-09-23 04:49-05:15Z

The full-ladder run of 01:50Z wedged on opendde 1024 rep0 (trunk 1/10, 03:54Z; its warmup at the
same size had passed) and left card 0 dirty. Its reset would take down card 1, held by another
row. The arm is re-run in slices on card 2 with their own journals
(`perf/land_standing/gate_ladder_card2_slices.sh`, `gate_ladder_card2_tail.sh`).

rf3 verdict: FAIL, rc=1. Drift attributed cell by cell against the baseline (recorded 2026-09-17 at
679b9ab4c) and `git diff origin/main...HEAD -- tt_bio`, which touches only `_tri_att_q_chunks`:

| drift | cells | source |
|---|---|---|
| TRIATT_PERSISTENT_MASK frac 0.000 -> 0.333 | rf3/896 only | this candidate, the lever's purpose |
| SDPA_Q_CHUNK_FITS overflow set 1 -> 0 | rf3/896 only | this candidate: 224 fits where 256 overflowed |
| SDPA_WIDE_K resolved False -> True | all 7 rungs | main, 65b1c356e (2026-09-19) default flip |
| QKV_MM_CONFIG (2,8) leaves the decline set | all 7 rungs | main, ce7467175 (2026-09-20) derived fused key |
| REBLOCK_PERMUTE new reject reasons | rf3/256 | main, 121cb8a2a (2026-09-18) trimul L1 path |
| TRIATT_SDPA_HIFI not in baseline | all 7 rungs | main, census lever newer than the baseline |

The candidate's own drift is exactly the pre-registered set: two lever counters at rf3/896 and
nothing at any other rung, including 640 and 1088, where the q >= prod/2 bound makes it a no-op.
The other four drifts come from main and would redden main's own run. The arm is red on main, so
it cannot hold this candidate by itself.

Runtimes (s) were taken at loadavg 4.5-5.8 on a 16-core host with card 1 folding, so they are not a
measurement. The fold A/B is the measurement.

    rung       256   512   640   768   896    1024   1088
    baseline   23.9  42.4  58.0  74.2  106.9  130.5  171.7
    this run   29.8  48.3  64.9  81.3  100.7  126.7  180.0

896 is the only mid rung faster than its baseline while its neighbours run 10-25 % slow, which is
the direction the +9.50 s fold A/B predicts.
