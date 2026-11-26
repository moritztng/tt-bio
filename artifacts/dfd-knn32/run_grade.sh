#!/usr/bin/env bash
# dfd-knn32 production grade, in order:
#   1. re-grade the control npz already on disk (its dump is complete for 0,5,10,15)
#   2. dump + grade 7F8T_FAD, the one PoseBusters ligand whose degree reaches upstream's cap
# Steps are 0,5,10,15,18: sampling calls forward 19 times, so index 18 is the last one.
set -u
cd /home/moritz/.coworker/wt/dfd-knn32
G=artifacts/dfd-knn32/grade
PORT=/home/moritz/tt-bio/env/bin/python3.10

echo "=== control re-grade $(date -Is)"
TT_VISIBLE_DEVICES= PYTHONPATH="$PWD" nice -n 5 "$PORT" \
  scripts/diffdock_port/score_net_grade.py --npz "$G/7BJ6_TVK.npz" \
  --out "$G/7BJ6_TVK.grade.json" > "$G/7BJ6_TVK.grade.log" 2>&1
echo "  rc=$?"; grep -E "^step" "$G/7BJ6_TVK.grade.log" || tail -3 "$G/7BJ6_TVK.grade.log"

echo "=== 7F8T_FAD $(date -Is)"
STEPS=0,5,10,15,18 OUT="$PWD/$G" ./artifacts/dfd-knn32/grade_knn32.sh 7F8T_FAD
echo "=== GRADE CHAIN DONE $(date -Is)"
