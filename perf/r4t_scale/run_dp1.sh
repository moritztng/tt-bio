#!/bin/bash
# 1-chip baseline of the DP pair-stack step on card 0 (the curve s denominator). Waits for the node.
set -u
WT=$HOME/.coworker/wt/r4t-scale; V=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
O=$HOME/r4t/scale/dp1; mkdir -p $O; cd $WT; export PYTHONPATH=$WT
busy() { for f in /proc/[0-9]*/fd/*; do [ "$(readlink $f 2>/dev/null)" = /dev/tenstorrent/0 ] && return 0; done; return 1; }
t0=$(date +%s)
for try in $(seq 1 30); do
  while busy; do [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "node 0 never freed" >> $O/driver.log; exit 3; }; sleep 20; done
  echo "try $try $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" >> $O/driver.log
  TT_BIO_LEASE_HOLDER=worker:r4t-scale timeout 2400 $V perf/r4t_scale/dp_pairstack.py --cards 0 --steps 6 --out $O >> $O/driver.log 2>&1 && exit 0
  grep -q DeviceInUseError $O/rank0.log || exit 1
done
