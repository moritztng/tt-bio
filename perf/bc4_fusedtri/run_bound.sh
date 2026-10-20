#!/bin/bash
# Waits until no other process holds or is pinned to CARD (two clean checks 60 s apart), then runs phase_bound.py.
# Usage: CARD=1 OUT=<dir> bash perf/bc4_fusedtri/run_bound.sh
set -u
CARD=${CARD:-1}; OUT=${OUT:?}; mkdir -p "$OUT"
busy() {
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    ls -l "$p/fd" 2>/dev/null | grep -q "/dev/tenstorrent/$CARD$" && return 0
    tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -qx "TT_VISIBLE_DEVICES=$CARD" && return 0
  done
  return 1
}
clean=0
while [ $clean -lt 2 ]; do
  if busy; then clean=0; else clean=$((clean + 1)); fi
  [ $clean -lt 2 ] && sleep 60
done
echo "card $CARD free at $(date -u +%FT%TZ), sha $(git rev-parse HEAD)" >> "$OUT/run.log"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fusedtri \
  timeout 1800 python3 perf/bc4_fusedtri/phase_bound.py --out "$OUT/phase.jsonl" >> "$OUT/run.log" 2>&1
echo "rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fusedtri \
  timeout 1200 python3 perf/bc4_fusedtri/composed.py --out "$OUT/composed.jsonl" >> "$OUT/run.log" 2>&1
echo "composed rc=$? at $(date -u +%FT%TZ)" >> "$OUT/run.log"
