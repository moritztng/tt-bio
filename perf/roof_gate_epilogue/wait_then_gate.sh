#!/usr/bin/env bash
# The gate needs a card and qb2 is carrying two other workers' device jobs, one of which is an
# explicitly quiet timing run. Rather than pile on (its 768 aa arm already timed out at 900 s when
# load hit 18), wait for the box to go idle and take card 0, which is this worker's grant.
# Idle means: no process holds any /dev/tenstorrent node, three checks in a row, 60 s apart.
set -u
WT=/home/ttuser/.coworker/wt/roof-gate-epilogue-conflict-remerge
cd "$WT" || exit 1
LOG=perf/roof_gate_epilogue/out/wait_then_gate.log
deadline=$(( $(date +%s) + 6*3600 ))
quiet=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  busy=$(for d in /dev/tenstorrent/[0-9]; do lsof "$d" 2>/dev/null | tail -n +2 | awk '{print $2}'; done | sort -u | tr '\n' ' ')
  if [ -z "$busy" ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date -u +%H:%M:%SZ) busy='$busy' quiet=$quiet load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
  if [ "$quiet" -ge 3 ]; then
    echo "$(date -u +%H:%M:%SZ) box idle, launching gate on card 0" >> "$LOG"
    sed -i 's/TT_BIO_LEASE_CARDS=2/TT_BIO_LEASE_CARDS=0/; s/--workers tt-quietbox2:2/--workers tt-quietbox2:0/' perf/roof_gate_epilogue/run_gate_fd5c536.sh
    exec bash perf/roof_gate_epilogue/run_gate_fd5c536.sh >> perf/roof_gate_epilogue/out/gate_fd5c536.log 2>&1
  fi
  sleep 60
done
echo "$(date -u +%H:%M:%SZ) gave up: box never went idle inside 6 h" >> "$LOG"
