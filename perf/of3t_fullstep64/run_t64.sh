#!/bin/bash
# of3t-fullstep64 padding discriminator: the float64 full step on the pinned batch cut to 64 tokens (56 real),
# replaying the 384-wide run's draws (the atom axis is unchanged, 422 real atoms).
cd /tmp/of3t/of3t-fullstep64
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
exec nice -n 5 ~/of3-upstream-venv/bin/python perf/of3t_fullstep64/ref_step.py --mode f64 --replay-draws out/draws.pt \
  --batch t64/batch_step003_t64.pt --batch-sha256 b88fcf517a530cb5a06cfe26feab4c6cbdc65cf9dad06230a9ce4345415d12ee \
  --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir t64/out --threads 10 --rss-cap-gb 21
