#!/bin/bash
# of3t-confpfe qb2: the 384-wide float64 reference and upstream's bf16 bar, rebuilt with every
# confidence head our step returns (D266). Same batch, checkpoint, draws and denoise draw as
# of3t-denoise's ref384 (perf/of3t_denoise/run_ref.sh); pc lacks the memory, so qb2 CPU, both
# modes at once. Scratch /home/ttuser/of3t_confpfe/ref384c.
W=/home/ttuser/.coworker/wt/of3t-confpfe; S=/home/ttuser/of3t_confpfe; O=$S/ref384c
R=/home/ttuser/of3t-campaign-refs
mkdir -p $O
cd $W
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
export PYTHONPATH=$R/of3pkg043:/home/ttuser/of3t_refprec/deps OMP_NUM_THREADS=6
PY=/home/ttuser/tt-bio-dev/env/bin/python
ARGS=(--denoise --batch $R/bundle_min_043/batch_step003.pt
      --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
      --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt --replay-draws /home/ttuser/of3t_fullstep64/draws.pt
      --threads 6 --rss-cap-gb 40 --chunk-size 32)
for mode in f64 bf16; do
  ( echo "=== $mode start $(date -u +%FT%TZ)" >> $O/chain.log
    nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode $mode "${ARGS[@]}" --out-dir $O/$mode \
      --disk-checkpoint $S/ckpt/$mode > $O/ref_$mode.log 2>&1
    echo "=== $mode exit $? $(date -u +%FT%TZ)" >> $O/chain.log
    rm -rf $S/ckpt/$mode ) &
done
wait
