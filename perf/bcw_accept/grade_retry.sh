#!/bin/bash
# Get a SECOND axis for the device/bf16 trend. n=352 graded clean at ratio 1.028; 800 OOMs the box
# outright (187.5 GB) and 512 and 640 both hit a 140 GB cgroup cap. 178 GB is free, so retry 512
# with the cap raised to 170G, and if it still does not fit drop to 448, which is still 1.27x the
# graded axis and enough to say whether the ratio moves with n.
#
# Stop at the FIRST axis that succeeds: one extra point is what the trend needs, and every attempt
# holds a card. MemoryMax confines any kill to this scope, so armB1 and armB4 are never at risk.
set -u
cd /home/ttuser/.coworker/wt/bcw-accept || exit 1
LOG=perf/bcw_accept/out/grade_axes.log
{
  echo "== retry sitting started $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) MemoryMax=170G"
  for N in 512 448; do
    echo "===== n=$N begin $(date -u +%FT%TZ)"
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-accept \
      systemd-run --user --scope -p MemoryMax=170G --quiet \
      timeout 3600 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
      perf/bcw_land/stack_grade.py "bcwaccept_n${N}" --n "$N" --evo 8 --blocks 0,3,7
    rc=$?
    echo "===== n=$N rc=$rc end $(date -u +%FT%TZ)"
    [ $rc -eq 0 ] && { echo "== got a second axis at n=$N, stopping"; break; }
  done
  echo "== retry sitting finished $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
