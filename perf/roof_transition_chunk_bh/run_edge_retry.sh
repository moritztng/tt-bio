#!/bin/bash
# qb2 is running four workers at loadavg ~37 and the granted card is rarely idle. Retry ONLY on
# the lease refusal, so a real throw is reported once and not re-run twenty times.
set -u
cd "$(dirname "$0")/../.."
LOG=perf/roof_transition_chunk_bh/out/edge_c3.log
for i in $(seq 1 40); do
  echo "=== attempt $i $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$LOG"
  ./perf/roof_transition_chunk_bh/run_edge.sh 3 >> "$LOG" 2>&1 && exit 0
  grep -q DeviceInUseError "$LOG" || exit 1
  tail -c 200000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  sleep 20
done
exit 1
