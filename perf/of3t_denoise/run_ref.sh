#!/bin/bash
# of3t-denoise pc: the float64 denoise reference, upstream's bf16 bar through the same step, then
# the A40 control on the new float64 gradient. fullstep64's batch, checkpoint and banked rollout
# draws, replayed; the denoise draw is tt_bio.train.openfold3.denoise_draw(20260922, 422).
# Waits for of3t-inproj's pc chain (pc memory). Scratch /tmp/of3t-denoise; copied to the worktree.
cd /tmp/of3t-denoise
until grep -q "FD exit" /tmp/of3t-inproj/out/chain.log; do sleep 60; done
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
PY=~/of3-upstream-venv/bin/python
ARGS=(--denoise --batch batch_step003.pt --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
      --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --threads 10 --rss-cap-gb 21 --chunk-size 32)
cp /tmp/of3t-inproj/out/draws.pt out/draws_fullstep64.pt
echo "=== smoke64 start $(date -u +%FT%TZ)" >> out/chain.log
nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode f64 --replay-draws out/draws_fullstep64.pt \
  --denoise --batch batch_t64.pt --batch-sha256 $(cat batch_t64.sha) \
  --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out_t64 --threads 10 --rss-cap-gb 16 \
  > out/smoke64.log 2>&1
rc=$?; echo "=== smoke64 exit $rc $(date -u +%FT%TZ)" >> out/chain.log
[ $rc = 0 ] || exit $rc
for mode in f64 bf16; do
  echo "=== $mode start $(date -u +%FT%TZ)" >> out/chain.log
  nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode $mode --replay-draws out/draws_fullstep64.pt "${ARGS[@]}" \
    --disk-checkpoint ckpt/$mode > out/ref_$mode.log 2>&1
  echo "=== $mode exit $? $(date -u +%FT%TZ)" >> out/chain.log
  rm -rf ckpt/$mode
done
LOSS=$(python3 -c "import json;print(repr(json.load(open('out/REF_F64.json'))['loss']))")
echo "=== FD start $(date -u +%FT%TZ) expect-loss $LOSS" >> out/chain.log
nice -n 5 $PY perf/of3t_fullstep64/ref_step.py --mode f64 --fd --fd-from out/grads_f64.pt \
  --replay-draws out/draws_fullstep64.pt --expect-loss "$LOSS" "${ARGS[@]}" --disk-checkpoint $PWD/ckpt/fd > out/fd.log 2> out/fd.err
echo "=== FD exit $? $(date -u +%FT%TZ)" >> out/chain.log
rm -rf ckpt/fd
