#!/bin/bash
# Grade the Evoformer VJP against float64 at BOTH arms' token axes, same tree, same card,
# same sitting. bcw-land graded this tree at 288 only; bcw-accept's arm B runs at 800 and
# 60% of its screen-gate deficit is optimisation gaining less, which is the shape a degraded
# gradient would make. Nothing had graded the gradient above 288 before this.
set -u
cd /home/ttuser/.coworker/wt/bcw-accept || exit 1
LOG=perf/bcw_accept/out/grade_axes.log
{
  echo "== started $(date -u +%FT%TZ) head $(git rev-parse --short HEAD)"
  echo "== script origin/main:perf/bcw_land/stack_grade.py, unmodified"
  for N in 352 800; do
    echo "===== n=$N begin $(date -u +%FT%TZ)"
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-accept \
      timeout 5400 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
      perf/bcw_land/stack_grade.py "bcwaccept_n${N}" --n "$N" --evo 8 --blocks 0,3,7
    echo "===== n=$N rc=$? end $(date -u +%FT%TZ)"
  done
  echo "== finished $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
