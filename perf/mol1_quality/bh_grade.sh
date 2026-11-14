#!/bin/bash
# Runs on pc (artifacts/mol1-quality/tick.d, every ~10 min): once qb2 holds every TT-BH fold of BH_DIR for SEEDS
# (qb1's come back through sync_qb1.sh) it starts full_grade.sh OUT once, so the BH verdict does not wait on WH.
#   bash bh_grade.sh OUT N SEEDS BH_DIR [WH_SEEDS WH_DIR]   N = folds expected (167 cases x seeds), BH_DIR the arm under
#   ~/mol1-quality (bh_arm.sh OUT); WH_SEEDS and WH_DIR go to full_grade.sh
set -uo pipefail
OUT=${1:?OUT} N=${2:?N} SEEDS=${3:?SEEDS} BH=${4:?BH_DIR}; shift 4
Q=ttuser@tt-quietbox2 S="ssh -o BatchMode=yes -o ConnectTimeout=15" G="${SEEDS//,/ }"
n=$(timeout 120 $S $Q "test -e ~/mol1-quality/$OUT/LAUNCHED && echo L ||
  { cd ~/mol1-quality/$BH && for s in $G; do grep -l rc=0 .runs/*/seed_\$s/rc 2>/dev/null; done | wc -l; }") || exit 1
[ "$n" = L ] && exit 0
echo "$(date -u +%FT%TZ) BH folds ($SEEDS): qb2 has $n/$N"
if [ "$n" -ge "$N" ]; then
  $S -f $Q "cd ~/mol1-quality && mkdir -p $OUT && date -u +%FT%TZ > $OUT/LAUNCHED && setsid nohup bash ~/.coworker/wt/mol1-quality/perf/mol1_quality/full_grade.sh $OUT $BH $SEEDS $* > $OUT/launch.log 2>&1 < /dev/null &"
  echo "$(date -u +%FT%TZ) all $N BH folds on qb2: full_grade.sh $OUT launched (qb2:~/mol1-quality/$OUT/grade.log)"
fi
