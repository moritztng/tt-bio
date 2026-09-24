#!/usr/bin/env bash
# Install BindCraft 2 on a rented GPU box and measure its gradient step. One command, because the
# box bills by the hour and debugging on it is the expensive way to find a typo.
#
#   scp -r perf/bcx_gpuref root@<box>:/root/ && ssh root@<box> 'bash /root/bcx_gpuref/run_on_box.sh'
#
# Everything it produces lands in /root/bcx_out, which is what gets copied back.
set -euo pipefail

COMMIT=5342aefa18dedad653f7a5f6dbee1e566ca24d8f
BINDER_LENGTH=${BINDER_LENGTH:-96}
FINAL_DESIGNS=${FINAL_DESIGNS:-4}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-12}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-/root/bcx_out}
SRC=${SRC:-/root/BindCraft2}

mkdir -p "$OUT"

# The stamp is part of the measurement's identity: a step time without the card, the driver and
# the clock it ran at is not reproducible by anyone, including us.
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "bc2_commit=$COMMIT"
  echo "binder_length=$BINDER_LENGTH"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv
} > "$OUT/stamp.txt"
cat "$OUT/stamp.txt"

if [[ ! -d $SRC ]]; then
  git clone https://github.com/PacesaLab/BindCraft2.git "$SRC"
fi
git -C "$SRC" checkout --quiet "$COMMIT"
git -C "$SRC" rev-parse HEAD > "$OUT/bc2_commit.txt"

# install.sh reads the accelerator off the driver and downloads the 5.3 GB AlphaFold parameters.
cd "$SRC"
bash install.sh 2>&1 | tee "$OUT/install.log"
PY="$SRC/.venv/bin/python"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
"$PY" -c 'import jax; print("jax", jax.__version__); print("devices", jax.devices())' | tee "$OUT/jax_devices.txt"

# The clock the steps actually ran at, sampled during the run and not before it.
( while true; do
    echo "$(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader)"
    sleep 5
  done ) > "$OUT/clocks.txt" &
CLOCK_WATCH=$!
trap 'kill $CLOCK_WATCH 2>/dev/null || true' EXIT

# One campaign, benchmark core (seed 0, no autotune, no desperation) so it is reproducible, and one
# pinned binder length so every trajectory after the first folds a shape already compiled. That is
# what makes the compile split a measurement instead of an estimate.
"$PY" "$HERE/bc2_step_timing.py" --jsonl "$OUT/steps_warm.jsonl" -- \
  design "$SRC/examples/pdl1.json" --core benchmark \
  --set "binder_lengths=[$BINDER_LENGTH]" \
  --set "number_of_final_designs=$FINAL_DESIGNS" \
  --set "max_trajectories=$MAX_TRAJECTORIES" \
  --set "project_folder=$OUT/pdl1_warm" \
  2>&1 | tee "$OUT/campaign_warm.log"

kill $CLOCK_WATCH 2>/dev/null || true

"$PY" "$HERE/analyze_steps.py" "$OUT/steps_warm.jsonl" --report "$OUT/denominator.json"
"$PY" "$HERE/validate_designs.py" "$OUT/pdl1_warm" --binder-length "$BINDER_LENGTH" \
  --target-length 115 --report "$OUT/designs.json" || echo "validation reported problems, see designs.json"

find "$OUT" -type f -name '*.csv' -o -type f -name '*.json' | sort > "$OUT/manifest.txt"
sha256sum $(find "$OUT" -maxdepth 2 -type f | sort) > "$OUT/sha256.txt" 2>/dev/null || true
echo "done; copy back $OUT"
