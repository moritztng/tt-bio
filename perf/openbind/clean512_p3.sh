#!/usr/bin/env bash
# The clean apo-512 cell the 4.083x row needs: sole tenant, benchlock held, two processes per arm
# (perf-page protocol). Chained after queue_p3c so it cannot race it: that script is running NOW,
# so this guard is not vacuous the way queue part 2's cell-specific one was.
set -u
WT=/home/ttuser/.coworker/wt/openbind-perf-p3
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
OUT=$WT/perf/openbind/tt_results/clean512
LOG=$WT/perf/openbind/tt_results/ab/logs
mkdir -p "$OUT"
while pgrep -f "queue_p3c.sh" >/dev/null 2>&1; do sleep 30; done
echo "CLEAN: queue clear @ $(date -u +%H:%M:%SZ)"
# wait for the box to be genuinely quiet: no device process on any card but ours
for i in $(seq 1 240); do
  busy=0
  for d in 0 1 2 3; do fuser /dev/tenstorrent/$d >/dev/null 2>&1 && busy=$((busy+1)); done
  [ "$busy" -eq 0 ] && break
  sleep 30
done
echo "CLEAN: cards quiet=$busy loadavg=$(cut -d' ' -f1 /proc/loadavg) @ $(date -u +%H:%M:%SZ)"
exec 9>"$HOME/.coworker/state/benchlock.flock"
flock -w 3600 9 || { echo "CLEAN: benchlock timeout"; exit 1; }
echo "openbind-perf-p3 pid=$$ since=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" > "$HOME/.coworker/state/benchlock"
echo "CLEAN: benchlock held @ $(date -u +%H:%M:%SZ)"
for p in 1 2; do
  f=$OUT/ob_apo_512_clean_p$p.json
  grep -q device_s_median "$f" 2>/dev/null && { echo "SKIP p$p"; continue; }
  echo "=== clean apo-512 process $p @ $(date -u +%H:%M:%SZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
  env PYTHONPATH=$WT TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:openbind-perf-p3 \
      "$PY" perf/openbind/tt_ob_run.py --model openbind \
      --input perf/openbind/inputs/ob_apo_512.tt.yaml --repeat 3 \
      --label clean512_p$p --out "$f" 2>&1 | tail -20
  echo "=== p$p rc=$? @ $(date -u +%H:%M:%SZ) ==="
done
echo "CLEAN COMPLETE @ $(date -u +%H:%M:%SZ)"
