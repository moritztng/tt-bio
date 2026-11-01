#!/bin/bash
# Runs on qb2: pulls qb1's Fix B folds (qb1_fixb.sh) into ~/mol1-quality/arms/tt_bh_b and grades the A/B (D-orch-25)
# on the partial grade's 33 cases: tt_bh (main 90cef828f) and tt_bh_b each against gpu, tt_bh_b against tt_bh, and
# the matched pose RMSD of both against gpu beside the GPU's own seed floor. Re-runnable; grading is cached per fold.
#   bash fixb_grade.sh      -> ~/mol1-quality/fixb/QUALITY.json, log fixb/grade.log, GRADE-EXIT at the end
set -uo pipefail
cd ~/mol1-quality; mkdir -p fixb arms/tt_bh_b
Q=ttuser@tt-quietbox  # finished folds only (rc=0), as sync_qb1.sh pulls them
timeout 120 ssh -o BatchMode=yes $Q 'cd ~/mol1-quality/fixb/tt-bh-b && find .runs -name rc | xargs grep -l rc=0' |
while read -r rc; do
  d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}; r=mol1_results_${c%%.json*}/$c/$s
  [ -d arms/tt_bh_b/$r ] || timeout 1800 rsync -a --exclude claim $Q:mol1-quality/fixb/tt-bh-b/$r/ arms/tt_bh_b/$r/ --mkpath < /dev/null
done
rsync -a $Q:mol1-quality/fixb/tt-bh-b/aiclk.log arms/tt_bh_b/aiclk.log
rsync -a $Q:mol1-quality/fixb/CASES.json fixb/CASES.json
P=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality; G=$P/grade.py
python3 "$P/arms.py" ~/mol1-fair-gpu-ref arms
{ PATH=$PWD/env/fb/bin:$PATH nice -n 19 env/fb/bin/python "$G" --inputs inputs --cases fixb/CASES.json --work fixb/work \
    --out fixb/QUALITY.json --arm gpu=arms/gpu --arm gpu_self=arms/gpu_self --arm tt_bh=tt-bh --arm tt_bh_b=arms/tt_bh_b \
    --ab tt_bh_b=tt_bh
  echo "GRADE-EXIT rc=$?"; } > fixb/grade.log 2>&1
