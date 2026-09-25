#!/usr/bin/env bash
# of3t-fullstep64: exact ops ON through the fixed trunk call (77d0ec8aa), scored against the
# regenerated float64 at 384. M384ON used the --trunk-masks diagnostic; this is the shipped path.
cd /home/ttuser/.coworker/wt/of3t-fullstep64
bash perf/of3t_fullstep64/devarm.sh FIXON on > /home/ttuser/of3t_fullstep64/devarm_FIXON.log 2>&1
echo "=== chain done $(date -u +%FT%TZ)" >> /home/ttuser/of3t_fullstep64/chain_fixon.log
