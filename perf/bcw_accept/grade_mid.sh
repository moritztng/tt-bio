#!/bin/bash
# Grade the axes that fit and read the TREND, because n=800 does not fit: it reached 187.5 GB
# anon-RSS and the kernel OOM-killed it at 09:28:45Z on a 249 GB box.
#
# The cap is a cgroup MemoryMax, NOT ulimit -v. The first attempt used `ulimit -v 120 GiB` and
# both axes died in under a minute with "can't allocate 8388608000 bytes" while their real
# footprint was far smaller: torch and ttnn reserve enormous VIRTUAL address space, so -v measures
# the wrong thing and bites long before resident memory is the problem. MemoryMax bounds RSS,
# which is what the OOM killer actually counts, and it confines the kill to this scope so armB1
# and armB4 can never be chosen as the victim.
set -u
cd /home/ttuser/.coworker/wt/bcw-accept || exit 1
LOG=perf/bcw_accept/out/grade_axes.log
{
  echo "== cgroup-capped sitting started $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) MemoryMax=140G"
  for N in 512 640; do
    echo "===== n=$N begin $(date -u +%FT%TZ)"
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-accept \
      systemd-run --user --scope -p MemoryMax=140G --quiet \
      timeout 3600 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
      perf/bcw_land/stack_grade.py "bcwaccept_n${N}" --n "$N" --evo 8 --blocks 0,3,7
    echo "===== n=$N rc=$? end $(date -u +%FT%TZ)"
  done
  echo "== cgroup-capped sitting finished $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
