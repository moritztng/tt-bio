#!/bin/bash
# of3t-inproj pc: the float64 reference and upstream's bf16 bar through the F2-fixed objective,
# then the A40 control on the new float64 gradient. fullstep64's batch, checkpoint and banked
# draws, replayed. Scratch /tmp/of3t-inproj only; outputs are copied under the worktree after.
cd /tmp/of3t-inproj
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
PY=~/of3-upstream-venv/bin/python
ARGS=(--batch batch_step003.pt --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
      --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --threads 10 --rss-cap-gb 21 --chunk-size 32)
for mode in f64 bf16; do
  echo "=== $mode start $(date -u +%FT%TZ)" >> out/chain.log
  nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode $mode --replay-draws out/draws.pt "${ARGS[@]}" \
    --disk-checkpoint ckpt/$mode > out/ref_$mode.log 2>&1
  echo "=== $mode exit $? $(date -u +%FT%TZ)" >> out/chain.log
  rm -rf ckpt/$mode
done
LOSS=$(python3 -c "import json;print(repr(json.load(open('out/REF_F64.json'))['loss']))")
echo "=== FD start $(date -u +%FT%TZ) expect-loss $LOSS" >> out/chain.log
nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode f64 --fd --fd-from out/grads_f64.pt \
  --expect-loss "$LOSS" "${ARGS[@]}" --disk-checkpoint $PWD/ckpt/fd > out/fd.log 2> out/fd.err
echo "=== FD exit $? $(date -u +%FT%TZ)" >> out/chain.log
rm -rf ckpt/fd
