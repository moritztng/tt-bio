#!/bin/bash
# Record every pick in order on one chip; one log per run under runs/.
#   bash demo/booth/gallery/record_all.sh <chip> <pick>...
cd "$(dirname "$0")"
CHIP=$1; shift
mkdir -p runs
for p in "$@"; do
  echo "$(date -u +%FT%TZ) start $p" >> runs/record_all.log
  ~/tt-bio-dev/env/bin/python3 record.py "$p" --chip "$CHIP" >> runs/record_all.log 2>&1
  rc=$?   # before the $(date) below, which would reset it
  echo "$(date -u +%FT%TZ) end $p rc=$rc" >> runs/record_all.log
done
echo "$(date -u +%FT%TZ) ALL DONE" >> runs/record_all.log
