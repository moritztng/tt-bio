#!/usr/bin/env bash
# Wait for the in-flight fold, then fold the two commits flanking the prime suspect 1ea1e6f3.
set -u
WT=/home/moritz/.coworker/wt/opendde-512aa-numerics-drift-bisect
cd "$WT" || exit 1
while pgrep -f "tt_baseline.py --model opendde" >/dev/null 2>&1; do sleep 20; done
sleep 5
REPEAT=1 ./.bisect-out/fold_at.sh PRE_52a584a9 52a584a9 >> .bisect-out/PRE_52a584a9.log 2>&1
sleep 5
REPEAT=1 ./.bisect-out/fold_at.sh FIX_1ea1e6f3 1ea1e6f3 >> .bisect-out/FIX_1ea1e6f3.log 2>&1
sleep 5
git checkout -q wk/opendde-512aa-numerics-drift-bisect
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
