#!/bin/bash
# Runs on qb2: grades a BH stack A/B (ab_arm.sh) on the 33 cases of fixb/CASES.json: arm A and arm B each against gpu,
# B against A, and the matched pose RMSD of both against gpu beside the GPU's own seed floor. A and B are fold dirs
# under ~/mol1-quality; a B folded on qb1 is pulled first (finished folds only, rc=0, as sync_qb1.sh does). Re-runnable,
# grading is cached per fold.
#   bash ab_grade.sh NAME A_DIR B_DIR [qb1]   -> ~/mol1-quality/NAME/QUALITY.json, log NAME/grade.log, GRADE-EXIT at the end
#   Fix B (D-orch-25): ab_grade.sh fixb tt-bh arms/tt_bh_b qb1;  final main (D-orch-97): ab_grade.sh ab-1edf66f5f tt-bh-main tt-bh-ab-1edf66f5f
set -uo pipefail
cd ~/mol1-quality; N=${1:?name}; A=${2:?A dir}; B=${3:?B dir}; mkdir -p "$N" "$B"
if [ "${4:-}" = qb1 ]; then
  Q=ttuser@tt-quietbox; RB=fixb/tt-bh-b; [ "$N" = fixb ] || RB=$(basename "$B")
  timeout 120 ssh -o BatchMode=yes $Q "cd ~/mol1-quality/$RB && find .runs -name rc | xargs grep -l rc=0" |
  while read -r rc; do
    d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}; r=mol1_results_${c%%.json*}/$c/$s
    [ -d "$B/$r" ] || timeout 1800 rsync -a --exclude claim $Q:mol1-quality/$RB/$r/ "$B/$r/" --mkpath < /dev/null
  done
  rsync -a $Q:mol1-quality/$RB/aiclk.log "$B/aiclk.log"
fi
P=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality; G=$P/grade.py
python3 "$P/arms.py" ~/mol1-fair-gpu-ref arms
{ PATH=$PWD/env/fb/bin:$PATH nice -n 19 env/fb/bin/python "$G" --inputs inputs --cases fixb/CASES.json --work "$N/work" \
    --out "$N/QUALITY.json" --arm gpu=arms/gpu --arm gpu_self=arms/gpu_self --arm tt_bh="$A" --arm tt_bh_b="$B" \
    --ab tt_bh_b=tt_bh --ref-split "$P/ref_split.json"
  echo "GRADE-EXIT rc=$?"; } > "$N/grade.log" 2>&1
