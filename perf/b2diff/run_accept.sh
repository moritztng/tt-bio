#!/bin/bash
# The two acceptance checks the plan mandates besides the fold A/B:
#   1. multiplicity -- one B=4 fold per arm, digest compared. The batched sampling path is merged
#      and shipping (3.56x at B=4); a diffusion change that helps at B=1 and hurts it is a net loss.
#   2. shared lineage -- protenix-v2, OpenDDE and openfold3 at 512 aa with the flags OFF. L6 touches
#      AdaLN.s_terms, which openfold3 calls through its own s_terms= kwarg, so "structurally
#      exclusive" is an argument and this is the measurement.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=900
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "########## 1. multiplicity, B=4 ##########"
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
  $PY -u perf/b2diff/fold_ab_b2diff.py --sizes 512 --samples 4 --no-timers \
    --arms on,l7l6,l7l6s6,on --out perf/b2diff/ab512_b4.json
echo "RC_B4=$?"

for M in openfold3 protenix-v2 opendde; do
  echo "########## 2. neutrality: $M 512 aa, flags OFF ##########"
  /home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
    $PY -u perf/other512/fold_ab_multi.py --model $M --sizes 512 --arms on \
      --out perf/b2diff/neutral_$M.json
  echo "RC_$M=$?"
done
echo ALLDONE
