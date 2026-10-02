#!/bin/bash
# n=800 cannot be graded on this box: it reached 187.5 GB anon-RSS and the kernel OOM-killed it
# (rc=137, 09:28:45Z) on a 249 GB box with the two live arms holding ~73 GB. So grade the axes
# that DO fit and read the TREND of the device/bf16 ratio across them instead.
#
# ulimit -v caps this process so it dies on its own rather than inviting the global OOM killer to
# choose a victim. It chose this process last time, but armB1 and armB4 were the alternatives and
# losing one of those costs a trajectory of real campaign data.
set -u
cd /home/ttuser/.coworker/wt/bcw-accept || exit 1
ulimit -v 125829120   # 120 GiB of address space
LOG=perf/bcw_accept/out/grade_axes.log
{
  echo "== mid-axis sitting started $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) ulimit -v $(ulimit -v)"
  for N in 512 640; do
    echo "===== n=$N begin $(date -u +%FT%TZ)"
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-accept \
      timeout 3600 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
      perf/bcw_land/stack_grade.py "bcwaccept_n${N}" --n "$N" --evo 8 --blocks 0,3,7
    echo "===== n=$N rc=$? end $(date -u +%FT%TZ)"
  done
  echo "== mid-axis sitting finished $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
