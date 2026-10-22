#!/bin/bash
# pairbias run2, under chip<card>.lock: outer_residual per-call (write.py), then paired traced in-trunk A/B
# (stack_ab.py, 30 reps) at 256 and 512: CAND vs CAND + pair_write_res, and CAND vs CAND + both levers.
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-pairbias; O=$W/artifacts/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairbias PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
stamp "start $(git log --oneline -1 | cut -c1-70)"
timeout 900 $PY perf/rfd4/pairbias/write.py --out $O/write.json > $O/write.log 2>&1
rc=$?; stamp "write rc=$rc"
[ $rc = 0 ] || { stamp end; exit 1; }
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
for n in 256 512; do
  for arm in pair_write_res pair_write_res,pair_bias_rms; do
    tag=$n-${arm//,/+}
    timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights ~/rfd4-bundle --out $O/$tag.json --n $n --reps 30 \
        --base $CAND --levers $CAND,$arm > $O/$tag.log 2>&1
    stamp "ab $tag rc=$?"
  done
done
stamp end
