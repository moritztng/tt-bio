#!/bin/bash
# D10+D24 gate: the size-ladder arm, on qb1 p150a where its recorded baseline lives. Scoring a
# p300c ladder against a p150a baseline would be a different cell, not a stricter check.
#
# --resume, because the first attempt wedged: boltz2-768-rep2 sat at "trunk 3/4" for ten
# minutes at 2.1% CPU while rep0 and rep1 at the same rung each finished inside a minute. The
# chain was killed by pid and card 0 reset with tt-smi -r 0. A wedge costs the whole arm
# otherwise, and this arm is nine per-model ladders over eleven rungs, not the single-model
# run pass 15 recorded at 4274 s.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_d10d24
mkdir -p "$OUT"
cd "$WT" || exit 1
echo "size-ladder arm (resume), qb1 p150a card 0, tip $(git rev-parse --short HEAD), started $(date -u +%FT%TZ)" >> "$OUT/LADDER.txt"
t0=$(date +%s)
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:land-standing \
  PYTHONPATH=$WT RELEASE_GATE_CENSUS_PYTHONPATH=$WT \
  "$PY" scripts/release_gate.py --model size-ladder --resume > "$OUT/size-ladder.log" 2>&1
rc=$?
t1=$(date +%s)
v=$(grep -Eo "\b(PASS|FAIL|SKIP)\b" "$OUT/size-ladder.log" | tail -1)
echo "size-ladder rc=$rc verdict=${v:-none} wall=$((t1-t0))s $(date -u +%FT%TZ)" >> "$OUT/LADDER.txt"
