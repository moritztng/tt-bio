#!/bin/bash
# of3t-fullstep64: regenerate the 384-wide float64 reference and the upstream bf16 bar after
# disk_guard removed /tmp/of3t (2026-09-23 10:40Z). Same batch, same arguments as run_f64.sh, the
# banked draws replayed. The float64 gradient counts as the reference only if its loss reproduces
# 0.13882195489029983 and |g|^2 2.0386874813540397 (A40 passed on that forward).
cd /tmp/of3t-fullstep64
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
for mode in f64 bf16; do
  nice -n 5 ~/of3-upstream-venv/bin/python perf/of3t_fullstep64/ref_step.py --mode $mode --replay-draws out/draws.pt \
    --batch batch_step003.pt --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
    --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --disk-checkpoint ckpt/$mode \
    --threads 10 --rss-cap-gb 21 --chunk-size 32 > out/ref_$mode.log 2>&1
  echo "=== $mode exit $? $(date -u +%FT%TZ)" >> out/regen.log
done
