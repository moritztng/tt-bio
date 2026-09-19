#!/bin/bash
# The interleaved cadence A/B: host-built features against card-built, each arm from its own
# cold start, arms alternating so neither owns the warm page cache.
set -u
W=/home/ttuser/.coworker/wt/train-u-relpos-ondevice/perf/train_u_relpos
STEPS=${STEPS:-20}
CHIPS=${CHIPS:-1,2}
for round in 1 2; do
  for arm in A B; do
    name="${arm}${round}"
    [ "$arm" = A ] && extra="--host-features" || extra=""
    echo "=== arm $name start $(date -u +%FT%TZ) ==="
    python3 $W/aiclk_sample.py 60 6 > $W/clock_$name.txt 2>&1 &
    clk=$!
    bash $W/run_arm.sh "$name" "$STEPS" "$CHIPS" $extra
    kill $clk 2>/dev/null
    wait $clk 2>/dev/null
    echo "=== arm $name done $(date -u +%FT%TZ) ==="
  done
done
echo "CHAIN COMPLETE $(date -u +%FT%TZ)"
