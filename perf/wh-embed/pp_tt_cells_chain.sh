#!/usr/bin/env bash
# Run the n=25 re-run after the main run finishes, without needing a second launch.
# Both touch card 0, so they must not overlap: wait for the main script to leave the process table
# (not for a log marker, which can be written before the last row releases the card).
set -u
WT=/home/ttuser/.coworker/wt/perf-page-tt-cells
cd "$WT" || exit 1
LOG=$WT/perf/wh-embed/results/pp_tt_cells_chain.log
{
  echo "chain: waiting for pp_tt_cells.sh to exit $(date -u +%H:%M:%SZ)"
  for i in $(seq 1 720); do
    pgrep -f "bash perf/wh-embed/pp_tt_cells.sh" >/dev/null 2>&1 || break
    sleep 10
  done
  echo "chain: main run gone $(date -u +%H:%M:%SZ), starting n25"
  bash perf/wh-embed/pp_tt_cells_hin.sh
  echo "chain: done $(date -u +%H:%M:%SZ)"
} >> "$LOG" 2>&1
