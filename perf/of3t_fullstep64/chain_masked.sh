#!/usr/bin/env bash
# of3t-fullstep64: the 384-wide full step with the trunk masks diagnostic, ON, OFF, then ON again (A/A).
cd /home/ttuser/.coworker/wt/of3t-fullstep64
export DEVSTEP_EXTRA=--trunk-masks
for t in "M384ON on" "M384OFF off" "M384ON2 on"; do
  set -- $t
  bash perf/of3t_fullstep64/devarm.sh "$1" "$2" > /home/ttuser/of3t_fullstep64/devarm_$1.log 2>&1
done
echo "=== chain done $(date -u +%FT%TZ)" >> /home/ttuser/of3t_fullstep64/chain_masked.log
