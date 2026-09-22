#!/usr/bin/env bash
# The three pin legs at padded 384, restricted to blocks 45 and 44, fanned across the three
# idle p150a cards on qb1, plus their padded-64 denominators at the same restriction on card 2.
# Serialising them would cost ~36 min of device time for the same answer.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-blk4544
O=/home/ttuser/of3t_blk4544
cd "$W"

BLK_CARD=1 bash perf/of3t_blk4544/arm.sh P384 384 sm4         45,44 > "$O/log_P384.txt" 2>&1 &
p1=$!
sleep 75
BLK_CARD=3 bash perf/of3t_blk4544/arm.sh C384 384 paircontract 45,44 > "$O/log_C384.txt" 2>&1 &
p3=$!
sleep 75
( for a in "S64B sm16" "P64B sm4" "C64B paircontract"; do
    set -- $a
    BLK_CARD=2 bash perf/of3t_blk4544/arm.sh "$1" 64 "$2" 45,44 > "$O/log_$1.txt" 2>&1
  done
  BLK_CARD=2 bash perf/of3t_blk4544/arm.sh S384 384 sm16 45,44 > "$O/log_S384.txt" 2>&1 ) &
p2=$!

wait $p1; echo "P384 rc=$?"
wait $p3; echo "C384 rc=$?"
wait $p2; echo "card2 chain rc=$?"
for f in S64B P64B C64B P384 C384 S384; do
  echo "--- $f ---"
  tail -4 "$O/log_$f.txt" 2>/dev/null
done
