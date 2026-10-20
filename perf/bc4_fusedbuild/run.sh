#!/bin/bash
# Runs the bc4-bet-fusedbuild instruments on the leased qb2 chip ($TT_VISIBLE_DEVICES) or, without a lease, the
# first qb2 chip whose box flock (~/spd_qb2_chip<N>.lock) is free.
# Usage: OUT=<dir> bash perf/bc4_fusedbuild/run.sh <script.py> [args...]
set -u
OUT=${OUT:?}; mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
PY=${PY:-/home/ttuser/bcx_e2e_venv/bin/python}; export PY PYTHONPATH=$PWD OUT
export TT_BIO_LEASE_HOLDER=worker:bc4-bet-fusedbuild
run() {
  local c=$1; shift
  echo "chip $c at $(date -u +%FT%TZ), sha $(git rev-parse HEAD) dirty=$(git status --porcelain -uno | wc -l): $*" >> "$OUT/run.log"
  TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c timeout 1800 "$PY" "$@" >> "$OUT/run.log" 2>&1
  echo "rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
}
export -f run
CARDS=${CARDS:-${TT_VISIBLE_DEVICES:-2 3 0 1}}
while :; do
  for c in $CARDS; do
    flock -n -E 75 "/home/ttuser/spd_qb2_chip$c.lock" bash -c 'run "$@"' _ "$c" "$@"
    [ $? -ne 75 ] && exit 0
  done
  sleep 60
done
