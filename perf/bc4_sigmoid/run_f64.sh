#!/bin/bash
# Step 2 of bc4-bet-sigmoid on the leased chip: OUT=<dir> TT_VISIBLE_DEVICES=<chip> bash perf/bc4_sigmoid/run_f64.sh
set -u
OUT=${OUT:?}; : "${TT_VISIBLE_DEVICES:?lease a chip first}"; mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
PY=${PY:-/home/ttuser/bcx_e2e_venv/bin/python}
echo "chip $TT_VISIBLE_DEVICES at $(date -u +%FT%TZ), sha $(git rev-parse HEAD) dirty $(git status --porcelain -uno | wc -l)" >> "$OUT/run.log"
PYTHONPATH=$PWD timeout 1500 "$PY" perf/bc4_sigmoid/f64_check.py --out "$OUT/f64.jsonl" >> "$OUT/run.log" 2>&1
echo "f64 rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
