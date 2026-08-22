#!/usr/bin/env bash
# s2_1024 exists to test one pre-registered prediction (l1_refused 0 at 96 cores) and give one A/B
# pair. Two processes are enough for that; the clean benchlocked apo-512 pair the brief owes is
# queued behind it and is worth more than reps 2. Retire the sweep after B1 lands.
set -u
D=/home/ttuser/.coworker/wt/openbind-perf-p3/perf/openbind/tt_results/ab/s2_1024
for i in $(seq 1 200); do
  grep -q device_s_median "$D/ob_apo_1024_B1.json" 2>/dev/null && break
  sleep 20
done
echo "TRIM: B1 landed @ $(date -u +%H:%M:%SZ)"
for p in $(pgrep -f "ob_ab.sh 1 ob_apo_1024 s2_1024"); do echo "kill sweep $p"; kill "$p"; done
sleep 2
for p in $(pgrep -f "label s2_1024"); do echo "kill run $p"; kill "$p"; done
sleep 3
for p in $(pgrep -f "queue_p3c.sh"); do echo "kill queue $p"; kill "$p"; done
echo "TRIM DONE @ $(date -u +%H:%M:%SZ)"
