#!/bin/bash
# pairbias run5: lever A again after the EPS rename (bias.py: error vs float64 and per-call ms), then if it runs
# the paired traced A/B of CAND+A and CAND+B+A against CAND at 256 and 512 (30 reps).
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-pairbias; O=$W/artifacts/run5; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairbias PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70)"
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
timeout 900 $PY perf/rfd4/pairbias/bias.py --weights ~/rfd4-bundle --out $O/bias.json > $O/bias.log 2>&1
A=$?; stamp "bias rc=$A"
[ $A = 0 ] || exit 1
for n in 256 512; do
  for arm in pair_bias_rms pair_write_res,pair_bias_rms; do
    tag=$n-${arm//,/+}
    timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights ~/rfd4-bundle --out $O/$tag.json --n $n --reps 30 \
        --base $CAND --levers $CAND,$arm > $O/$tag.log 2>&1
    stamp "ab $tag rc=$?"
  done
done
stamp end
