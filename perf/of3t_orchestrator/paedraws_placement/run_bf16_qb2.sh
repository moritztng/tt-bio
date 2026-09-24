#!/usr/bin/env bash
# of3t-paedraws draws 1-5, bf16 references at 6 threads (draw 0's count; chain_pd.sh started them
# at 2 and they got 0.31 core each at qb2 load 72/16). of3t-orchestrator pass 435. float64 runs on
# qb1 (/dev/shm/of3t_paedraws_f64/run_f64.sh) and copies into $S/ref_sK/f64/. This script is the one
# scoring trigger: once every draw has f64, bf16 and device gradients and chain_pd.sh has exited, it
# runs chain_pd.sh again, which skips every existing output and only scores.
W=/home/ttuser/.coworker/wt/of3t-paedraws; S=/home/ttuser/of3t_paedraws
R=/home/ttuser/of3t-campaign-refs; CK=/home/ttuser/of3-weights/of3-p2-155k.pt
DRAWS=/home/ttuser/of3t_fullstep64/draws.pt; L=$S/chain.log
log() { echo "=== $* $(date -u +%FT%TZ)" >> $L; }
cd $W
source /home/ttuser/tt-bio-dev/env/bin/activate
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
for k in 1 2 3 4 5; do
  O=$S/ref_s$k/bf16
  [ -s $O/grads_bf16.pt ] && continue
  ( mkdir -p $O; log "ref s$k bf16 start (6 threads, run_bf16.sh)"
    PYTHONPATH=$R/of3pkg043:/home/ttuser/of3t_refprec/deps OMP_NUM_THREADS=6 nice -n 5 \
      python3 perf/of3t_fullstep64/ref_step.py --mode bf16 --seed $k --denoise \
      --batch $R/bundle_min_043/batch_step003.pt \
      --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
      --checkpoint $CK --replay-draws $DRAWS --threads 6 --rss-cap-gb 40 --chunk-size 32 \
      --out-dir $O --disk-checkpoint $S/ckpt/s$k/bf16 > $S/ref_s${k}_bf16.log 2>&1
    log "ref s$k bf16 exit $?"; rm -rf $S/ckpt/s$k/bf16 ) &
done
wait
for i in $(seq 288); do   # up to 24 h
  n=0
  for k in 1 2 3 4 5; do
    [ -s $S/ref_s$k/f64/grads_f64.pt ] && [ -s $S/ref_s$k/bf16/grads_bf16.pt ] && [ -s $S/grad_PD384_s$k.pt ] && n=$((n+1))
  done
  [ $n = 5 ] && ! pgrep -f "[c]hain_pd.sh" >/dev/null && break
  sleep 300
done
log "run_bf16.sh: $n of 5 draws complete, relaunching chain_pd.sh to score"
bash perf/of3t_paedraws/chain_pd.sh > $S/chain.rescore.out 2>&1 < /dev/null
python3 perf/of3t_paedraws/aggregate.py >> $L 2>&1
log "run_bf16.sh done, aggregate exit $?"
