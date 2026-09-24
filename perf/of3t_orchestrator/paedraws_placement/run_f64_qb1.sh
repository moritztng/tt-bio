#!/usr/bin/env bash
# of3t-paedraws draws 1-5, float64 references only, moved off qb2 (load 72/16, 0.31 core each) by
# of3t-orchestrator pass 435. float64 is host-independent (cross-host floor 9.14e-14, R166); bf16 is
# not, so the bf16 references stay on qb2 beside draw 0's. Same arguments as chain_pd.sh except
# --threads 3 and paths. Each result is copied to qb2's $S/ref_sK/f64/, where chain_pd.sh scores it.
D=/dev/shm/of3t_paedraws_f64; W=$D/wt; R=/home/ttuser/of3t-campaign-refs
CK=/home/ttuser/of3-weights/of3-p2-155k.pt; Q=tt-quietbox2; S=/home/ttuser/of3t_paedraws
L=$D/run.log
log() { echo "=== $* $(date -u +%FT%TZ)" >> $L; ssh -o BatchMode=yes $Q "echo '=== qb1: $* $(date -u +%FT%TZ)' >> $S/chain.log"; }
cd $W
source /home/ttuser/tt-bio-dev/env/bin/activate
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
for k in 1 2 3 4 5; do
  O=$D/ref_s$k/f64
  ( ssh -o BatchMode=yes $Q "test -s $S/ref_s$k/f64/grads_f64.pt" && exit 0
    mkdir -p $O; log "ref s$k f64 start $(hostname)"
    PYTHONPATH=$R/of3pkg043:$D/deps OMP_NUM_THREADS=3 nice -n 5 \
      python3 perf/of3t_fullstep64/ref_step.py --mode f64 --seed $k --denoise \
      --batch $R/bundle_min_043/batch_step003.pt \
      --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
      --checkpoint $CK --replay-draws $D/draws.pt --threads 3 --rss-cap-gb 40 --chunk-size 32 \
      --out-dir $O --disk-checkpoint $D/ckpt/s$k > $D/ref_s${k}_f64.log 2>&1
    rc=$?; rm -rf $D/ckpt/s$k; log "ref s$k f64 exit $rc"
    [ "$rc" = 0 ] && [ -s $O/grads_f64.pt ] || exit 1
    ssh -o BatchMode=yes $Q "mkdir -p $S/ref_s$k/f64.qb1" && scp -q $O/* $Q:$S/ref_s$k/f64.qb1/ \
      && ssh -o BatchMode=yes $Q "rm -rf $S/ref_s$k/f64 && mv $S/ref_s$k/f64.qb1 $S/ref_s$k/f64" \
      && log "ref s$k f64 copied to qb2" ) &
done
wait
# chain_pd.sh scores after its own wait; if it has already ended, relaunch it: every step skips
# an existing output, so it only scores.
if ! ssh -o BatchMode=yes $Q "pgrep -f [c]hain_pd.sh >/dev/null"; then
  ssh -o BatchMode=yes $Q "cd /home/ttuser/.coworker/wt/of3t-paedraws && setsid nohup bash perf/of3t_paedraws/chain_pd.sh > $S/chain.rescore.out 2>&1 < /dev/null &"
  log "chain_pd.sh relaunched on qb2 to score"
fi
log "qb1 f64 done"
