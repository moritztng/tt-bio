#!/bin/bash
# Runs on qb2: the full paired grade (every case of CASES.json) plus the FoldBench-AB headline. The verdict run once
# every arm is complete; before that an interim whose missing cases read as FAIL per arch but whose per-fold grading
# lands in the shared cache (work/), so the final run only grades what arrived since.
#   bash full_grade.sh OUT_DIR [BH_DIR [BH_SEEDS [WH_SEEDS]]]  -> OUT_DIR/QUALITY.json, OUT_DIR/grade.log, GRADE-EXIT at the end
# BH_DIR: the tt_bh arm (default tt-bh-main, current main with Fix B); BH_SEEDS e.g. 42,66 while seeds 101+ are folding.
# WH_SEEDS default 42,66 (D-orch-36); 42 alone gives the early one-seed WH grade.
set -uo pipefail
cd ~/mol1-quality; out=${1:?OUT_DIR}; bh=${2:-tt-bh-main}; mkdir -p "$out"
P=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality
python3 "$P/arms.py" ~/mol1-fair-gpu-ref arms
arms=(--arm gpu=arms/gpu --arm gpu_self=arms/gpu_self --arm tt_bh=$bh)
[ -n "${3:-}" ] && arms+=(--seeds tt_bh=$3)
[ -d arms/tt_wh ] && arms+=(--arm tt_wh=arms/tt_wh --seeds tt_wh=${4:-42,66})
{ PATH=$PWD/env/fb/bin:$PATH nice -n 19 env/fb/bin/python "$P/grade.py" --inputs inputs --cases CASES.json --work work \
    --out "$out/QUALITY.json" "${arms[@]}" --headline arms/headline --ref-split "$P/ref_split.json"
  echo "GRADE-EXIT rc=$?"; } > "$out/grade.log" 2>&1
