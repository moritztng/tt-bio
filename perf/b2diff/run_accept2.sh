#!/bin/bash
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=900
PY=/home/ttuser/tt-bio-dev/env/bin/python3

# B=4 with the region timers ON, so the call census proves the batch ran rather than the argument
# echoing itself back. Sample-dim batching keeps the call COUNT at B=1's and grows the shapes.
echo "########## B=4, timers on ##########"
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
  $PY -u perf/b2diff/fold_ab_b2diff.py --sizes 512 --samples 4 \
    --arms on,l7l6,l7l6s6 --out perf/b2diff/ab512_b4_timed.json
echo "RC_B4T=$?"

for M in openfold3 protenix-v2 opendde; do
  echo "########## neutrality: $M 512 aa, flags OFF ##########"
  /home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
    $PY -u perf/other512/fold_ab_multi.py --model $M --sizes 512 --arms on \
      --out perf/b2diff/neutral_$M.json
  echo "RC_$M=$?"
done
echo ALLDONE
