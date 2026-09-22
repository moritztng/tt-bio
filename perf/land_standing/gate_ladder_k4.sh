#!/bin/bash
# The one gate arm K4 still owes: size-ladder, all nine models, at the wk/land-standing tip.
# Runs alone on card 0 under benchlock. Not splittable across passes -- gate_journal.resumable()
# keeps only the LATEST record per arm, so a --size-ladder-models run replaces a partial record
# rather than accumulating with it. One ~2h45m process, one verdict at the end.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out
STAMP=$(date -u +%H%M)
LOG=$OUT/gate_k4_ladder_$STAMP.log
cd "$WT" || exit 1
mkdir -p "$OUT"

{
  echo "size-ladder arm, tip $(git rev-parse --short HEAD), started $(date -u +%FT%TZ)"
  echo "host $(hostname) loadavg $(cut -d' ' -f1-3 /proc/loadavg) card 0 p300c"
} > "$LOG"

# benchlock: a measurement arm must not share the box with another row's fold.
exec 9>/home/ttuser/.coworker/state/benchlock.flock
if ! flock -n 9; then
  echo "benchlock held by another row; not starting" >> "$LOG"
  exit 3
fi
echo "benchlock acquired $(date -u +%FT%TZ)" >> "$LOG"

export PYTHONPATH="$WT"
export RELEASE_GATE_CENSUS_PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:land-standing

TT_BIO_SDPA_BAND_DIV_K=1 "$PY" scripts/release_gate.py --model size-ladder --keep >> "$LOG" 2>&1
rc=$?
echo "EXIT rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
echo "$rc" > "$OUT/gate_k4_ladder_$STAMP.rc"
