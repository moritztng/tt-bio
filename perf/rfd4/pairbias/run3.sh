#!/bin/bash
# pairbias run3, one chain under chip<card>.lock: lever A per-call bench, the A and A+B paired traced A/B at 256
# and 512 (30 reps, base = land6 CAND), census v3 of CAND at 256/512, then the 512 design grade (8 seeds,
# rng upstream, CA RMSD vs normal = trimul_t1): normal, cand, cand+B, cand+A+B if A runs.
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-pairbias; O=$W/artifacts/run3; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairbias PYTHONPATH=$W
mkdir -p $O/design; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70)"
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
timeout 900 $PY perf/rfd4/pairbias/bias.py --weights ~/rfd4-bundle --out $O/bias.json > $O/bias.log 2>&1
A=$?; stamp "bias rc=$A"
if [ $A = 0 ]; then
  for n in 256 512; do
    for arm in pair_bias_rms pair_write_res,pair_bias_rms; do
      tag=$n-${arm//,/+}
      timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights ~/rfd4-bundle --out $O/$tag.json --n $n --reps 30 \
          --base $CAND --levers $CAND,$arm > $O/$tag.log 2>&1
      stamp "ab $tag rc=$?"
    done
  done
fi
for n in 256 512; do
  TT_BIO_LEVERS=$CAND bash perf/rfd4/e2e/elt_prof.sh $O/census-$n $C --n $n > $O/census-$n.log 2>&1
  $PY perf/rfd4/e2e/census_v2.py $O/census-$n $O/e2e-$n-v3.json >> $O/census-$n.log 2>&1
  stamp "census $n rc=$?"
done
leg() {  # leg <name> <levers>
  TT_BIO_LEVERS="$2" timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
      --weights $HOME/rfd4-bundle --out $O/design/$1 --rng upstream --seeds $(seq 0 7) > $O/design/$1.log 2>&1
  stamp "design $1 rc=$? warm=$(grep -c "\"kind\": \"warm\"" $O/design/$1/runs.jsonl 2>/dev/null)"
}
leg normal trimul_t1
leg cand $CAND
leg candB $CAND,pair_write_res
[ $A = 0 ] && leg candAB $CAND,pair_write_res,pair_bias_rms
stamp end
