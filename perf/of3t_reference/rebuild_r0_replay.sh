#!/usr/bin/env bash
# The publication run: the r = 0 gradient on the draws every consumer already holds.
#
# rebuild_r0.sh lets the step draw its own randomness, which is self-consistent and NOT
# comparable with anything captured earlier: putting Dropout in eval removes 61 consumers of the
# CUDA generator, so the diffusion noise changes (measured, noise levels 23.705... -> 2.612...).
# PROTOCOL 4a makes the draws inputs to the update rule. This run replays draws_recycles0.pt, the
# draws the published bundle already ships and `of3t-gradients` captured its boundary against.
#
# Runs C (with the finite-difference validation) and D (a second production in a fresh process),
# then PROTOCOL A13: are the two bit-identical?
set -euo pipefail
ROOT=${ROOT:-/root/of3t}
cd "$ROOT"
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
DRAWS_SHA=36c8136c7c52cd55e7b15a2d1d627666094c64095e85ff7c9d38e05b66beacc5
export PYTHONPATH="$ROOT/openfold-3"

echo "$BATCH_SHA  batch_step003.pt" | sha256sum -c -
echo "$DRAWS_SHA  draws_recycles0.pt" | sha256sum -c -

for run in C D; do
  extra=""
  [ "$run" = D ] && extra="--fd-samples 0"
  echo "=== run $run === $(date -u +%FT%TZ)"
  ( while true; do
      nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
      sleep 5
    done > "clock_${run}.txt" ) & SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null || true' EXIT
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python bundle_min.py --batch batch_step003.pt \
      --batch-sha256 "$BATCH_SHA" --replay-draws draws_recycles0.pt \
      --out "out_$run" --dtype float64 --num-recycles 0 --checkpoint of3-p2-155k.pt \
      --fd-h 1e-4 --fd-samples 16 $extra 2>&1 | tee "run_${run}.log" | tail -30
  kill $SAMPLER 2>/dev/null || true
  echo "clock samples: $(grep -c MHz "clock_${run}.txt") range $(grep MHz "clock_${run}.txt" | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
done

echo "=== PROTOCOL A13: is it reproduced? === $(date -u +%FT%TZ)"
python compare_grads.py out_C/grads_f64.pt out_D/grads_f64.pt --json-out reproduction_A13_replay.json
