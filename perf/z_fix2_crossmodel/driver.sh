#!/bin/sh
# Sequential driver for the four arms. A first (cheapest, establishes the throw), then B (the
# headline), then C and D. Per-arm .done markers inside run_arm.sh make this resumable.
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-fix2-crossmodel
D=$WT/perf/z_fix2_crossmodel
cd "$WT" || exit 1
for arm in A B C D; do
  if [ -f "$D/.armdone_$arm" ]; then echo "=== skip arm $arm ==="; continue; fi
  echo "=== START arm $arm $(date -u +%H:%M:%S) ==="
  sh "$D/run_arm.sh" "$arm" && touch "$D/.armdone_$arm"
  echo "=== END arm $arm $(date -u +%H:%M:%S) rc=$? ==="
done
echo "=== DRIVER COMPLETE $(date -u +%H:%M:%S) ==="
