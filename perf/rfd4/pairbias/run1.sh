#!/bin/bash
# pairbias run1, under chip<card>.lock: per-call bench (chain vs rms_pair_bias, error vs float64), then the paired
# traced in-trunk A/B (stack_ab.py, 30 reps) at 256 and 512: base = land6's CAND set, arm = CAND + pair_bias_rms.
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-pairbias; O=$W/artifacts/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairbias PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
stamp "start $(git log --oneline -1 | cut -c1-70)"
timeout 900 $PY perf/rfd4/pairbias/bias.py --weights ~/rfd4-bundle --out $O/bias.json > $O/bias.log 2>&1
rc=$?; stamp "bias rc=$rc"
[ $rc = 0 ] || { stamp end; exit 1; }
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
for n in 256 512; do
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights ~/rfd4-bundle --out $O/$n-ab.json --n $n --reps 30 \
      --base $CAND --levers $CAND,pair_bias_rms > $O/$n-ab.log 2>&1
  stamp "ab $n rc=$?"
done
stamp end
