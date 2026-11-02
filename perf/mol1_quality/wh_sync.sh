#!/bin/bash
# Runs on pc (artifacts/mol1-quality/tick.d, every ~10 min): WH folds that finished on .107 go to qb2's grader as they
# land, only what grade.py reads (predictions/ + the run record, gzipped: pc reaches .107 through the ~200 KB/s tunnel).
# Once every paired case has every seed of SEEDS on qb2 it starts full_grade.sh OUT once (D-orch-36: WH graded on 42+66;
# SEEDS=42 with N=167 is the early one-seed grade, run first because the .107 queue folds seed 42 ahead of 66).
# Only folds whose run.log names a tree in SHA count, and arms/tt_wh/ENGINE_SHA carries them joined by + (grade.py keys
# its cache by it). SHA is a comma list: cases that refused DRAM on 7374eb5a6 refold on the fix tree (wh_requeue.sh).
#   bash wh_sync.sh [OUT [N [SHA [SEEDS]]]]   OUT default final-wh, N = folds expected (167 cases x 2 seeds), SHA the WH trees
set -uo pipefail
OUT=${1:-final-wh} N=${2:-334} SHA=${3:-7374eb5a6,0b2e93ecd} SEEDS=${4:-42,66}
rcs=$(sed 's#[^,]*#.runs/*/seed_&/rc#g; s#,# #g' <<< "$SEEDS")
G=ubuntu@172.16.102.107 Q=ttuser@tt-quietbox2 R=spd/mol1-quality/tt-wh A=mol1-quality/arms/tt_wh
S="ssh -o BatchMode=yes -o ConnectTimeout=15"
say() { echo "$(date -u +%FT%TZ) $*"; }
exec 9> /tmp/mol1-wh-sync.lock; flock -n 9 || exit 0
$S $Q "test -e ~/mol1-quality/$OUT/QUALITY.json -o -e ~/mol1-quality/$OUT/LAUNCHED" && exit 0
src=$(timeout 120 $S $G "cd $R && for f in \$(grep -l rc=0 .runs/*/seed_42/rc .runs/*/seed_66/rc 2>/dev/null); do
  grep -qE 'tree (${SHA//,/|}) ' \${f%/rc}/run.log && echo \$f; done") || { say ".107 unreachable"; exit 1; }
have=$(timeout 120 $S $Q "mkdir -p $A && cd $A && echo ${SHA//,/+} > ENGINE_SHA && grep -l rc=0 .runs/*/seed_*/rc 2>/dev/null")
new=$(comm -23 <(sort <<< "$src") <(sort <<< "$have") | grep .)
if [ -n "$new" ]; then
  # a multi-entry JSON file's results dir drops the .json and entry from the case name (same rule as sync_qb1.sh);
  # rc goes last so a transfer cut short leaves the fold looking unsynced, and it is resent whole next tick
  list=$(while read -r rc; do d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}
           echo "mol1_results_${c%%.json*}/$c/$s/predictions"; echo "$d/run.log"; done <<< "$new"; echo "$new")
  timeout 3000 $S $G "cd $R && tar czf - --ignore-failed-read -T -" <<< "$list" | timeout 3000 $S $Q "cd $A && tar xzf -" \
    || say "transfer cut short"
fi
n=$(timeout 120 $S $Q "cd $A && grep -l rc=0 $rcs 2>/dev/null | wc -l")
say "WH folds ($SEEDS): .107 $(grep -c . <<< "$src") done, qb2 has ${n:-?}/$N (+$(grep -c . <<< "$new"))"
if [ "${n:-0}" -ge "$N" ]; then
  $S -f $Q "cd ~/mol1-quality && mkdir -p $OUT && date -u +%FT%TZ > $OUT/LAUNCHED && setsid nohup bash ~/.coworker/wt/mol1-quality/perf/mol1_quality/full_grade.sh $OUT tt-bh-main 42,66 $SEEDS > $OUT/launch.log 2>&1 < /dev/null &"
  say "all $N WH folds on qb2: full_grade.sh $OUT launched (qb2:~/mol1-quality/$OUT/grade.log)"
fi
