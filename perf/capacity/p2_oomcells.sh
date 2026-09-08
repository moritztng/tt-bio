#!/bin/bash
# The four cells the committed baseline records as HOST_OOM, re-measured one at a time.
#
# These cells die by exhausting HOST RAM, which is a real verdict the gate records -- but the kill
# is delivered by the kernel's host-wide OOM killer, and pc also runs the fleet control plane, the
# JapanFold tunnel and the sentinel cron. Two things follow.
#
# A cgroup MemoryMax does NOT work here, measured 2026-09-08: the gate learns the verdict by
# watching its own worker subprocess die, so capping the whole tree at 24 GiB terminated the GATE
# at the same ~85 s the worker used to die at, and four cells ran with nothing recorded (rc=143,
# empty table). The cap moved the kill from the child to the parent and destroyed the measurement.
#
# What does work is leaving the limit alone and biasing the kernel's victim choice: oom_score_adj
# 1000 on this shell is inherited by the gate and its worker, so the OOM killer picks inside this
# tree, and within the tree it still picks the worker, which is the largest by RSS by far. Same
# verdict the committed baseline was recorded with, control plane no longer a candidate.
set -u
WT=/home/moritz/.coworker/wt/wh-transition-wchunk-hang-fix-p2
PY=/home/moritz/tt-bio/env/bin/python3
cd $WT
export PYTHONPATH=$WT PYTHONNOUSERSITE=1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=3,0
export TT_BIO_LEASE_HOLDER=worker:wh-transition-wchunk-hang-fix-p2
for m in esmfold2 esmfold2-fast opendde opendde-abag; do
  echo "===== cell $m start $(date -u +%FT%TZ)"
  ( echo 1000 > /proc/self/oom_score_adj
    exec timeout 900 $PY -u scripts/capacity_gate.py --models $m --record --no-card-reset --no-bisect \
        --report $WT/perf/capacity/p2_oom_$m.json )
  echo "===== cell $m rc=$? end $(date -u +%FT%TZ)"
done
echo "===== oom cells done $(date -u +%FT%TZ)"
# --no-bisect: the committed baseline's four HOST_OOM cells all carry ceiling_tokens=null,
# alloc_ceiling_tokens=null and decided_by="screen", so the downward bisect these cells pay for
# (~20 min each, measured) produces nothing that reaches the record. The screen at the bar is the
# whole measurement.
