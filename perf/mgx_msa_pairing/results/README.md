# Crystal scores, `collect.py` output

All on whglx at `--host_threads 2`, AICLK sampled every 20 s during every call: 1000 MHz on 550
of 562 samples, 500 on 8 (a call's first sample), 892-967 on 4. Both arms read the same pinned alignments.

- `c.txt`: every model, seeds 0 and 1. BEFORE = main 8906d35a0, AFTER = 302ae0f02. Its
  openfold3 AFTER is the paired arm this branch then turned off (06b1ed313).
- `c8wt4.txt`: 8WT4 seeds 2-5 for openfold3 (paired) and openbind, same two trees.
- `d.txt`: openfold3 at 06b1ed313 (unpaired, as upstream 0.4.x) against the same BEFORE:
  bit-identical on every target and seed.
