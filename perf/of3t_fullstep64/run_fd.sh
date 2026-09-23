#!/bin/bash
# of3t-fullstep64 pc: the A40 control against the banked float64 gradient, after the bf16 arm.
# The first f64 run banked grads_f64.pt and was SIGKILLed during its FD (2026-09-23 11:04 CEST).
cd /tmp/of3t/of3t-fullstep64
while kill -0 2856031 2>/dev/null; do sleep 30; done
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
echo "=== FD start $(date -u +%FT%TZ)"
nice -n 5 ~/of3-upstream-venv/bin/python perf/of3t_fullstep64/ref_step.py --mode f64 --fd \
  --fd-from out/grads_f64.pt --expect-loss 0.13882195489029983 \
  --batch batch_step003.pt --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --threads 10 --rss-cap-gb 21 --chunk-size 32 \
  --disk-checkpoint $PWD/ckpt/fd > out/fd.log 2> out/fd.err
echo "=== FD exit $? $(date -u +%FT%TZ)"
rm -rf ckpt/fd
