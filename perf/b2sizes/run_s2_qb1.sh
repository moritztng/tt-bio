#!/bin/bash
# S2 execution chain, qb1 card 0. qb2 (where every earlier number in this task was measured) is
# unreachable -- ssh times out -- so this pass re-anchors on qb1. qb1 card 0 is a Blackhole p150a
# with a 13x10 = 130-core grid against qb2 card 1s 11x10 = 110, so absolute times do NOT transfer
# and every arm below is compared only against an on arm measured in the same process.
WT=/home/ttuser/.coworker/wt/boltz2-sizes-perf
cd $WT || exit 70
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=1800
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
STEP="${1:-screen}"

case "$STEP" in
screen)
  echo "=== S2b screen $(date -Is) ==="
  $BL boltz2-sizes-perf -- $PY -u perf/b2sizes/s2b_mask_q_parallel.py \
      --sizes 768,1024 --reps 7 --out perf/b2sizes/s2b_screen_qb1.json
  echo "screen RC=$?"
  ;;
ab768)
  echo "=== S2 fold A/B 768 $(date -Is) arms=$2 ==="
  $BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 768 \
      --arms "$2" --out perf/b2sizes/s2_ab_768_qb1.json
  echo "ab768 RC=$?"
  ;;
ab1024)
  echo "=== S2 fold A/B 1024 $(date -Is) arms=$2 ==="
  $BL boltz2-sizes-perf -- $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 1024 \
      --arms "$2" --out perf/b2sizes/s2_ab_1024_qb1.json
  echo "ab1024 RC=$?"
  ;;
esac
echo "=== step $STEP done $(date -Is) ==="
