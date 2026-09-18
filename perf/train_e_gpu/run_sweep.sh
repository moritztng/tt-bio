#!/usr/bin/env bash
# Run the measurement sweep on one rented GPU box. TAG names the box in every output file.
# SUITE=full     everything, including the two runner-vs-upstream controls (primary A100)
# SUITE=confirm  stage 1 only: the second-rental agreement test
# SUITE=modern   stage 1 and stage 2, no controls (H100/H200 datapoint)
set -euo pipefail

TAG=${TAG:?set TAG, e.g. a100-1}
SUITE=${SUITE:-full}
ROOT=${ROOT:-/root/abb3/ABodyBuilder3}
OUT=${OUT:-/root/abb3/out}
STEPS=${STEPS:-100}
WARMUP=${WARMUP:-15}

cd "$ROOT"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src"
H=/root/abb3/gpu_abb3_step.py

{
  echo "=== $(date -u +%FT%TZ) idle box, TAG=$TAG SUITE=$SUITE ==="
  nvidia-smi
  echo "--- compute apps (co-tenancy: any pid here that is not ours means a shared card) ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "--- cpu ---"; lscpu | head -22
  echo "--- mem ---"; free -g
  echo "--- load ---"; cat /proc/loadavg
  echo "--- disk ---"; df -h /root | tail -1
  echo "--- pip ---"; pip list 2>/dev/null | grep -iE "^(torch|lightning|numpy|scipy|pandas|dvc|dvclive|ml-collections) "
} > "$OUT/box_${TAG}.txt" 2>&1

# Clock, power and co-tenancy sampled DURING every run, not before.
bash /root/abb3/clockwatch.sh "$OUT/clock_${TAG}.csv" &
CW=$!
trap 'kill $CW 2>/dev/null || true' EXIT

run() { echo ">>> $(date -u +%FT%TZ) $*"; python "$H" --tag "$TAG" "$@"; }

run --mode real     --stage 1 --steps "$STEPS" --warmup "$WARMUP" --out "$OUT/real_${TAG}_s1.json"
run --mode dataonly --stage 1 --steps "$STEPS" --warmup "$WARMUP" --out "$OUT/data_${TAG}_s1.json"
run --mode profile  --stage 1 --steps 10 --warmup 10 --out "$OUT/prof_${TAG}_s1.json"

if [ "$SUITE" != "confirm" ]; then
  run --mode real    --stage 2 --steps 40 --warmup 10 --out "$OUT/real_${TAG}_s2.json"
  run --mode profile --stage 2 --steps 10 --warmup 10 --out "$OUT/prof_${TAG}_s2.json"
fi

if [ "$SUITE" = "full" ]; then
  # The two places this runner departs from upstream's train.py, priced rather than assumed.
  run --mode real --stage 1 --steps 25 --warmup 10 --ddp     --out "$OUT/ctrl_${TAG}_s1_ddp.json"
  run --mode real --stage 1 --steps 25 --warmup 10 --dvclive --out "$OUT/ctrl_${TAG}_s1_dvclive.json"
fi

{
  echo "=== $(date -u +%FT%TZ) after sweep, TAG=$TAG ==="
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >> "$OUT/box_${TAG}.txt" 2>&1

echo "=== SWEEP DONE $TAG $(date -u +%FT%TZ) ==="
ls -la "$OUT"
