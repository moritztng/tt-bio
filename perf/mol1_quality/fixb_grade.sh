#!/bin/bash
# Runs on qb2: pulls qb1's Fix B folds (qb1_fixb.sh) into ~/mol1-quality/arms/tt_bh_b and grades the A/B (D-orch-25)
# on the partial grade's 33 cases: tt_bh (main 90cef828f) and tt_bh_b each against gpu, tt_bh_b against tt_bh, and
# the matched pose RMSD of both against gpu beside the GPU's own seed floor. Re-runnable; grading is cached per fold.
#   bash fixb_grade.sh      -> ~/mol1-quality/fixb/QUALITY.json, log fixb/grade.log, GRADE-EXIT at the end
set -uo pipefail
cd ~/mol1-quality; mkdir -p fixb arms/tt_bh_b
rsync -a --exclude .runs --exclude 'claim' ttuser@tt-quietbox:mol1-quality/fixb/tt-bh-b/ arms/tt_bh_b/
rsync -a ttuser@tt-quietbox:mol1-quality/fixb/CASES.json fixb/CASES.json
G=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality/grade.py
{ PATH=$PWD/env/fb/bin:$PATH nice -n 19 env/fb/bin/python "$G" --inputs inputs --cases fixb/CASES.json --work fixb/work \
    --out fixb/QUALITY.json --arm gpu=arms/gpu --arm gpu_self=arms/gpu_self --arm tt_bh=tt-bh --arm tt_bh_b=arms/tt_bh_b \
    --ab tt_bh_b=tt_bh
  echo "GRADE-EXIT rc=$?"; } > fixb/grade.log 2>&1
