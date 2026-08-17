#!/bin/bash
# Neutrality for the new defaults, measured on qb2 card 0. The predecessor's neutrality folds are
# qb1 card 1 (grid 13x10), so they are not a reference for this card. Here the OFF arm is the same
# tree with the two flags forced back to main's value by env, so the pair differs only by the
# defaults this task flips -- no second worktree, no cross-host comparison.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf-p2-land
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf-p2-land PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=900
PY=/home/ttuser/tt-bio-dev/env/bin/python3

for M in openfold3 protenix-v2 opendde; do
  echo "########## $M 512 aa, flags forced OFF (main's defaults) ##########"
  BOLTZ2_BIAS_SLICE_HOIST=0 BOLTZ2_ADALN_S_MEMO=0 \
  /home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf-p2-land -- \
    $PY -u perf/other512/fold_ab_multi.py --model $M --sizes 512 --arms on \
      --out perf/b2diff/p2_neutral_${M}_off.json
  echo "RC_${M}_off=$?"
  echo "########## $M 512 aa, new defaults (both ON) ##########"
  /home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf-p2-land -- \
    $PY -u perf/other512/fold_ab_multi.py --model $M --sizes 512 --arms on \
      --out perf/b2diff/p2_neutral_${M}_on.json
  echo "RC_${M}_on=$?"
done
echo ALLDONE
