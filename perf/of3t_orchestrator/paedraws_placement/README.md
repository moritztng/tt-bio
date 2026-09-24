# of3t-paedraws reference placement, pass 435 (2026-09-24 13:20-13:55Z)

qb2 ran all ten CPU references for draws 1-5 at 2 threads, nice 5, at load 72 on 16 cores: 0.31
core each, so float64 was about a day out. Changes:

- float64 draws 1-5 moved to qb1 (`run_f64_qb1.sh`, 3 threads, disk checkpoints in `/dev/shm`
  since qb1's disk has 1.1 GB free). float64 is host-independent (R166, cross-host floor
  9.14e-14); torch 2.8.0+cpu and the upstream 0.4.3 package tree hash match qb2; numpy differs
  (1.26.4 vs 2.5.2). Rate 1.42 core each, ~65 s per trunk block forward.
- bf16 stays on qb2 because a bf16 reference is host-dependent; restarted at 6 threads (draw 0's
  count) under `run_bf16_qb2.sh`, reniced 5 -> 0. It runs mostly serial: 0.46 core, ~60 s/block.
- the device step (draw 1, host-bound, 3 h timeout) reniced 0 -> -5: 0.51 -> 0.97 core.
- `run_f64_qb1.sh`'s own relaunch of chain_pd.sh was removed by killing its outer shell, so
  `run_bf16_qb2.sh` is the single scoring trigger.

Each disk-checkpointed float64 reference peaks around 15 GB (draw 0: 7.29 GB outer + 48 x 152 MB
trunk inputs); five fit in qb1's 213 GB free tmpfs.
