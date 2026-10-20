#!/bin/bash
# Takes the first qb2 chip whose flock (~/spd_qb2_chip<N>.lock, the box convention) is free, then runs phase_bound.py
# and composed.py under it. Non-blocking tries every 60 s, so blocked waiters (release gates) are woken first.
# Usage: OUT=<dir> [CARDS="2 3 0 1"] bash perf/bc4_fusedtri/run_bound.sh
set -u
OUT=${OUT:?}; CARDS=${CARDS:-2 3 0 1}; mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
PY=${PY:-/home/ttuser/bcx_e2e_venv/bin/python}; export PY PYTHONPATH=$PWD
run() {  # runs inside the held lock
  local c=$1
  echo "chip $c lock taken at $(date -u +%FT%TZ), sha $(git rev-parse HEAD)" >> "$OUT/run.log"
  export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:bc4-bet-fusedtri
  timeout 1800 "$PY" perf/bc4_fusedtri/phase_bound.py --out "$OUT/phase.jsonl" >> "$OUT/run.log" 2>&1
  echo "phase rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
  timeout 1200 "$PY" perf/bc4_fusedtri/composed.py --out "$OUT/composed.jsonl" >> "$OUT/run.log" 2>&1
  echo "composed rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
}
export -f run; export OUT
while :; do
  for c in $CARDS; do
    flock -n -E 75 "/home/ttuser/spd_qb2_chip$c.lock" bash -c "run $c"
    [ $? -ne 75 ] && exit 0
  done
  sleep 60
done
