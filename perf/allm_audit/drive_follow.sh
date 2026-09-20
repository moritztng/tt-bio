#!/bin/bash
# allm-audit: a second and third attempt at whatever the first driver pass could not get onto the
# card. The box runs one benchmark at a time behind a host-wide benchlock, and an arm that waits
# out BENCHLOCK_WAIT_S leaves no output and is skipped by the pass it was in. Each pass here
# blocks on the same flock the live driver holds, so it starts only when that driver is finished,
# and every arm inside it skips itself if its output already exists. The loop exits as soon as
# all six outputs are present, so a finished campaign costs one cheap pass and stops.
set -u
OUT=/home/ttuser/allm_audit/out
LOG=/home/ttuser/allm_audit/drive.log
need() {
  for f in odd_new_s1.json bg_old_s1.jsonl bg_new_s1.jsonl rfd3_old_s1.jsonl rfd3_new_s1.jsonl; do
    [ -s "$OUT/$f" ] || return 0
  done
  return 1
}
for pass in 1 2 3; do
  need || { echo "############ follower: all arms present, stopping at pass $pass" >>"$LOG"; exit 0; }
  echo "############ follower pass $pass waiting for the driver lock $(date -u +%FT%TZ)" >>"$LOG"
  flock /home/ttuser/allm_audit/driver.lock bash /home/ttuser/.coworker/wt/allm-audit/perf/allm_audit/drive_rest.sh
done
echo "############ follower: gave up after 3 passes $(date -u +%FT%TZ)" >>"$LOG"
