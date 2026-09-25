#!/bin/bash
# The four remaining matched seeds, one at a time, banked as each lands.
# Order is by how much of the reference arm is already in hand: 400 and 200 have a complete
# reference trajectory 1 (rejected at screen and at refine), 300 is in anneal, 0 runs on qb2.
# A seed with a done-marker is never re-run, so a relaunch resumes instead of repeating.
#
# The lock matters as much as the markers. This row gets relaunched while its own detached
# driver is still on card 3, and two drivers on one card would interleave two campaigns on one
# device context. flock makes a second launch exit instead of competing.
set -u
WT=/home/ttuser/.coworker/wt/bcx-seeds
D="$WT/perf/bcx_seeds"
mkdir -p "$D/runs"

exec 9>"$D/.drive.lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another driver holds the lock, exiting" >>"$D/drive.log"
  exit 0
fi
echo $$ >&9

for SEED in 400 200 300 0; do
  MARK="$D/runs/device_qb1_seed${SEED}/.done"
  if [ -f "$MARK" ]; then
    echo "$(date -u +%FT%TZ) seed $SEED already banked, skipping" >>"$D/drive.log"
    continue
  fi
  echo "$(date -u +%FT%TZ) seed $SEED starting" >>"$D/drive.log"
  bash "$D/run_seed.sh" "$SEED"
  RC=$?
  echo "$(date -u +%FT%TZ) seed $SEED finished rc=$RC" >>"$D/drive.log"
  [ $RC -eq 0 ] && date -u +%FT%TZ >"$MARK"
  sleep 20
done
echo "$(date -u +%FT%TZ) drive complete" >>"$D/drive.log"
