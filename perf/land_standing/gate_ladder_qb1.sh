#!/bin/bash
# D10+D24 gate: the size-ladder arm, run on qb1 p150a because that is the board class its
# recorded baseline was taken on. Running it on qb2's p300c would score a p300c ladder against
# a p150a baseline, which is a different cell, not a stricter check.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_d10d24
mkdir -p "$OUT"
cd "$WT" || exit 1
echo "size-ladder arm, qb1 p150a card 0, tip $(git rev-parse --short HEAD), started $(date -u +%FT%TZ)" >> "$OUT/LADDER.txt"
t0=$(date +%s)
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:land-standing \
  PYTHONPATH=$WT RELEASE_GATE_CENSUS_PYTHONPATH=$WT \
  "$PY" scripts/release_gate.py --model size-ladder > "$OUT/size-ladder.log" 2>&1
rc=$?
t1=$(date +%s)
v=$(grep -Eo "\b(PASS|FAIL|SKIP)\b" "$OUT/size-ladder.log" | tail -1)
echo "size-ladder rc=$rc verdict=${v:-none} wall=$((t1-t0))s $(date -u +%FT%TZ)" >> "$OUT/LADDER.txt"
