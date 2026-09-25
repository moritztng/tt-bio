#!/bin/bash
cd /tmp/of3t/of3t-fullstep64
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
exec nice -n 5 ~/of3-upstream-venv/bin/python perf/of3t_fullstep64/ref_step.py --mode f64 --fd \
  --batch batch_step003.pt --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --disk-checkpoint /tmp/of3t/of3t-fullstep64/ckpt --threads 10 --rss-cap-gb 21 --chunk-size 32
