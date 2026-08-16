#!/usr/bin/env bash
set -u
O=/home/ttuser/.coworker/wt/boltz2-affinity-device-only/perf/boltz2-affinity-device-only
until grep -q WALLS_EXIT $O/walls.log 2>/dev/null; do sleep 10; done
bash /home/ttuser/.coworker/scripts/benchlock.sh boltz2-affinity-device-only -- bash $O/run_digest.sh >> $O/digest.log 2>&1
