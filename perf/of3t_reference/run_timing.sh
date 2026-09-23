#!/usr/bin/env bash
# The 100-step s/step arm: upstream's own LightningModule, their initial_training stage config,
# their frozen batch stream, per-rank batch 1, token_budget 384.
set -euo pipefail
cd /root/of3t
export PYTHONPATH=/root/of3t/openfold-3
( while true; do
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader
    sleep 5
  done > clock_timing.txt ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
python gpu_reference.py --mode timing --batches /root/of3t/batches \
    --out /root/of3t/timing_32true.json --steps 105 --warmup 5 --precision 32-true \
    2>&1 | tail -40
kill $SAMPLER 2>/dev/null || true
echo "CLOCK: $(grep -c MHz clock_timing.txt) samples"
