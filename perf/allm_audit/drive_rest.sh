#!/bin/bash
# allm-audit: the three unfinished pairs, in one resumable sequence. Arms alternate old/new per
# model, every arm skips itself if its summary already exists, and one flock means a relaunch of
# this row cannot put a second driver on the same card.
set -u
R=/home/ttuser/.coworker/wt/allm-audit/perf/allm_audit/run_qb2c1.sh
LOG=/home/ttuser/allm_audit/drive.log
exec >>"$LOG" 2>&1
echo "############ driver start $(date -u +%FT%TZ) pid=$$"
for arm in odd_old odd_new bg_old bg_new rfd3_old rfd3_new; do
  echo "############ $arm $(date -u +%FT%TZ)"
  bash "$R" "$arm" s1
done
echo "############ driver done $(date -u +%FT%TZ)"
