#!/bin/bash
# Runs on pc (artifacts/mol1-quality/tick.d, every ~10 min): once qb2 holds every seed-42/66 TT-BH fold of tt-bh-main
# (qb1's come back through sync_qb1.sh) it starts full_grade.sh final-bh once, so the BH verdict does not wait on WH.
#   bash bh_grade.sh [OUT [N]]   OUT default final-bh, N = folds expected (167 cases x 2 seeds)
set -uo pipefail
OUT=${1:-final-bh} N=${2:-334} Q=ttuser@tt-quietbox2 S="ssh -o BatchMode=yes -o ConnectTimeout=15"
n=$(timeout 120 $S $Q "test -e ~/mol1-quality/$OUT/LAUNCHED && echo L ||
  { cd ~/mol1-quality/tt-bh-main && grep -l rc=0 .runs/*/seed_42/rc .runs/*/seed_66/rc 2>/dev/null | wc -l; }") || exit 1
[ "$n" = L ] && exit 0
echo "$(date -u +%FT%TZ) BH folds (42,66): qb2 has $n/$N"
if [ "$n" -ge "$N" ]; then
  $S -f $Q "cd ~/mol1-quality && mkdir -p $OUT && date -u +%FT%TZ > $OUT/LAUNCHED && setsid nohup bash ~/.coworker/wt/mol1-quality/perf/mol1_quality/full_grade.sh $OUT tt-bh-main 42,66 > $OUT/launch.log 2>&1 < /dev/null &"
  echo "$(date -u +%FT%TZ) all $N BH folds on qb2: full_grade.sh $OUT launched (qb2:~/mol1-quality/$OUT/grade.log)"
fi
