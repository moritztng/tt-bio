#!/usr/bin/env bash
# Chip 2 soak on the PR #56057 screen: 100k iterations at the upstream grid, then 100k on the full
# compute grid. Stops at the first failure. Never resets, never kills.  soak_screen.sh RUN CHIP
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); run=$here/runs/$1; chip=$2; mkdir -p "$run"
for grid in 10x10 11x10; do
  MATMUL_GRID=$grid "$here/screen_window.sh" "$1/$grid" "$chip" || { echo "$(date -u +%T) STOP grid $grid" >> "$run/soak.log"; exit 1; }
  echo "$(date -u +%T) grid $grid passed" >> "$run/soak.log"
done
