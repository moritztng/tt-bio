#!/usr/bin/env bash
set -u
O=/home/ttuser/.coworker/wt/boltz2-affinity-device-only/perf/boltz2-affinity-device-only
bash /home/ttuser/.coworker/scripts/benchlock.sh boltz2-affinity-device-only -- \
  bash -c "$O/wall_ab.sh /home/ttuser/.coworker/wt/boltz2-affinity-device-only/.before-tree before 4; \
           $O/wall_ab.sh /home/ttuser/.coworker/wt/boltz2-affinity-device-only after 4" \
  >> $O/walls.log 2>&1
echo "WALLS_EXIT=$? $(date -u +%FT%TZ)" >> $O/walls.log
