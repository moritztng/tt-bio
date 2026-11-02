#!/bin/bash
# Runs on .107 (pc's tick calls it over ssh): moves Mol-1 WH cases onto a fix tree. With no NAME it takes every case
# whose fold in OUT refused device DRAM on another tree (harness.log "out of device DRAM"; D-orch-53/54: 7374eb5a6's
# per-cycle MSA depth fragments WH DRAM, the fix tree folds them). Per case: its queued old-tree jobs go to
# pool/withdrawn, its old folds to OUT-oom-<old tree>, and one prio-1 job folds seeds 42+66 on tree-FIX into OUT.
# With OUT other than tt-wh nothing is withdrawn or moved and only seed 42 folds: that is the cross-tree check,
# succeeded cases refolded beside their first fold. A case whose job is running is skipped and named. Needs wh_jobs.py beside it.
#   bash wh_requeue.sh FIX_SHA [OUT [NAME...]]
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd); R=~/spd/mol1-quality; Q=~/spd/pool
FIX=$1 OUT=${2:-tt-wh}; shift $(( $# < 2 ? $# : 2 ))
cd $R && [ -f tree-$FIX/TREE_SHA ] || { echo "no $R/tree-$FIX"; exit 1; }
NAMES=("$@") SEEDS="42 66"; [ $OUT = tt-wh ] || SEEDS=42
if [ ${#NAMES[@]} = 0 ]; then
  mapfile -t NAMES < <(for h in $OUT/.runs/*/seed_*/harness.log; do d=${h%/harness.log}
    grep -qs rc=0 $d/rc || ! grep -q 'out of device DRAM' $h || grep -q "tree $FIX " $d/run.log || echo ${d%/seed_*}
  done | sed 's#.*/##' | sort -u)
fi
[ ${#NAMES[@]} = 0 ] && exit 0
rm -rf /tmp/mq_jobs && python3 $P/wh_jobs.py $R $R/tree-$FIX /home/ubuntu/spd/mol1-port/cache /tmp/mq_jobs 1 "$SEEDS" $OUT > /dev/null
for n in "${NAMES[@]}"; do
  f=/tmp/mq_jobs/1-mol1-quality-$n-s42.sh; [ -f $f ] || { echo "no job $n"; continue; }
  for r in $Q/running/*-mol1-quality-$n-*.sh; do [ -e "$r" ] && [ ! -e "$r.end" ] && { echo "LIVE $n, skipped"; continue 2; }; done
  if [ $OUT = tt-wh ]; then
    for j in $Q/queue/*-mol1-quality-$n-s*.sh; do [ -e "$j" ] && mv "$j" $Q/withdrawn/${j##*/}.fixtree-$FIX; done
    old=$(grep -ho 'tree [0-9a-f]*' tt-wh/.runs/$n/seed_*/run.log 2>/dev/null | tail -1 | cut -d' ' -f2)
    mkdir -p tt-wh-oom-${old:-x}/.runs
    for x in .runs/$n mol1_results_${n%%.json*}; do [ -e tt-wh/$x ] && mv tt-wh/$x tt-wh-oom-${old:-x}/$x; done
  fi
  j=1-mol1-quality-$n-$OUT-$FIX.sh; cp $f jobs/$j.new && mv jobs/$j.new $Q/queue/$j
  echo "$(date -u +%FT%TZ) queued $j (seeds $SEEDS, tree-$FIX)"
done
