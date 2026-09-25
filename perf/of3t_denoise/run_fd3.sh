#!/bin/bash
# of3t-denoise: FD with mse's stop-gradient alignment frozen at the base forward. FD2 (unfrozen)
# is kept as REF_F64_FD2_unfrozen.json: its floor is the stop-gradient, not the gradient.
cd /tmp/of3t-denoise
while kill -0 3764649 2>/dev/null; do sleep 15; done
mv out/REF_F64_FD.json out/REF_F64_FD2_unfrozen.json
cp -al out/REF_F64_FD2_unfrozen.json out/fd2.log out/fd2.err /home/moritz/of3t_keep/of3t-denoise/out/ 2>/dev/null
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 PYTHONPATH=$PWD:$PWD/up/of3pkg043:$PWD/up/deps OMP_NUM_THREADS=10
LOSS=$(python3 -c "import json;print(repr(json.load(open('out/REF_F64.json'))['loss']))")
echo "=== FD3 start $(date -u +%FT%TZ) expect-loss $LOSS (alignment frozen)" >> out/chain.log
nice -n 5 ~/of3-upstream-venv/bin/python perf/of3t_fullstep64/ref_step.py --mode f64 --fd --fd-from out/grads_f64.pt \
  --replay-draws out/draws_fullstep64.pt --expect-loss "$LOSS" --denoise --batch batch_step003.pt \
  --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/moritz/.boltz/of3-p2-155k.pt --out-dir out --threads 10 --rss-cap-gb 21 --chunk-size 32 \
  --fd-h 1e-4,1e-5 > out/fd3.log 2> out/fd3.err
echo "=== FD3 exit $? $(date -u +%FT%TZ)" >> out/chain.log
cp -al out/REF_F64_FD.json out/fd3.log out/fd3.err /home/moritz/of3t_keep/of3t-denoise/out/ 2>/dev/null
cp -a perf/of3t_fullstep64/ref_step.py run_fd3.sh /home/moritz/of3t_keep/of3t-denoise/
