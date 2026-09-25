#!/bin/bash
# size-ladder for the narrow-q tip, resumed after the 03:54Z opendde-1024 wedge left card 0 dirty
# (its reset would take down card 1, held by another row). Runs on the idle 2/3 pair, card 2, in
# slices with their own journals: rf3 first (the one model the lever moves), then the models the
# wedged run never reached. boltz2..openfold3 finished their folds before the wedge but the arm
# prints no verdict until it ends, so they are a third slice for a later pass.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_narrowq
cd "$WT" || exit 1
exec 9>/home/ttuser/.coworker/state/benchlock.flock
flock -n 9 || { echo "benchlock held; not starting" >> "$OUT/card2_slices.log"; exit 3; }
export PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 TT_BIO_LEASE_HOLDER=worker:land-standing
for slice in rf3 opendde,nesso1,openbind; do
  tag=${slice//,/_}
  echo "######## slice $slice tip $(git rev-parse --short HEAD) $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) ########" >> "$OUT/card2_slices.log"
  "$PY" scripts/release_gate.py --model size-ladder --size-ladder-models "$slice" --keep \
      --journal "$OUT/journal_sl_$tag.json" > "$OUT/sl_$tag.log" 2>&1
  rc=$?
  echo "slice $slice rc=$rc $(date -u +%FT%TZ)" >> "$OUT/card2_slices.log"
  [ $rc -eq 0 ] || grep -q "could not be reset" "$OUT/sl_$tag.log" && break
done
