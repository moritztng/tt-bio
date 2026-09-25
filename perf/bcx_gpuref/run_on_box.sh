#!/usr/bin/env bash
# Install BindCraft 2 on a rented GPU box and measure it. One command, because the box bills by the
# hour and debugging on it is the expensive way to find a typo.
#
#   ssh root@<box> 'bash /root/bcx_gpuref/run_on_box.sh'
#
# Everything it produces lands in /root/bcx_out, which is what gets copied back.
#
# Two numbers come out of one run, and they answer two different questions:
#
#   * the per-gradient-step cost, compile timed apart from compute, which is the denominator the
#     campaign divides by today and which was measured on FreeBindCraft, a different program;
#   * the PHASE SPLIT -- the 125-step gradient phase against the whole cycle (gradient design +
#     ProteinMPNN redesign + validation). Our 8,069.9 chip-s is the gradient phase and BindCraft 2's
#     published 90.5 s/trajectory on a GH200 is the whole cycle, so the campaign's headline ratio
#     compares our part to their whole. This is what turns that lower bound into a measurement.
set -euo pipefail

# The commit the campaign's DEVICE arms run on qb1 and qb2 (state/bcx/UPSTREAM.md). A GPU reference
# measured against a different BindCraft 2 than the arm it is the denominator for would repeat this
# row's founding mistake one level up.
COMMIT=${COMMIT:-7a2dfdb8a285232a6f881899fe135c6dc48679f1}
# 146 is the binder our device arms drew (pdl1_denovo_l146_3150367da865d53b). With BC2's 32-residue
# bucket that is 160 padded binder + 128 padded target = 288 residues, the same shape the 64.6 s/step
# was measured on. Pinning it is the one deliberate departure from stock settings, and it is also
# what makes the compile split measurable: every trajectory after the first folds a compiled shape.
BINDER_LENGTH=${BINDER_LENGTH:-146}
TARGET_LENGTH=${TARGET_LENGTH:-115}
FINAL_DESIGNS=${FINAL_DESIGNS:-4}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-4}
CORE=${CORE:-benchmark}
ARM=${ARM:-warm}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-/root/bcx_out}
SRC=${SRC:-/root/BindCraft2}
# One GPU and one process. The wrapper only instruments the process it lives in, and BC2 fans out
# into subprocess design workers unless auto_multi_gpu is off (design_workers.py:239), so a fan-out
# would leave the harness measuring nothing. One worker is also the configuration BC2's own 90.5 s
# figure was taken at, and the configuration our device arm runs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUT"

# The stamp is part of the measurement's identity: a step time without the card, the driver and the
# clock it ran at is not reproducible by anyone, including us.
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "bc2_commit=$COMMIT"
  echo "arm=$ARM"
  echo "binder_length=$BINDER_LENGTH target_length=$TARGET_LENGTH core=$CORE"
  echo "max_trajectories=$MAX_TRAJECTORIES number_of_final_designs=$FINAL_DESIGNS"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv
} > "$OUT/stamp_$ARM.txt"
cat "$OUT/stamp_$ARM.txt"

if [[ ! -d $SRC ]]; then
  git clone https://github.com/PacesaLab/BindCraft2.git "$SRC"
fi
git -C "$SRC" fetch --quiet origin || true
git -C "$SRC" checkout --quiet "$COMMIT"
git -C "$SRC" rev-parse HEAD > "$OUT/bc2_commit.txt"

# Does this clone still have everything the harness patches -- the gradient step AND the four phase
# hooks? Checked before the install, because a moved function makes the harness measure nothing and
# the install is the slow part.
BC2_SRC="$SRC" python3 "$HERE/test_bc2_contract.py" 2>&1 | tee "$OUT/contract.log"

# install.sh reads the accelerator off the driver and downloads the 5.3 GB AlphaFold parameters.
# The image's own conda/venv is python 3.11 against BC2's 3.12 minimum, and install.sh installs into
# an ACTIVE environment when it finds one, so the two variables that advertise one are cleared and it
# builds its own .venv.
cd "$SRC"
if [[ ! -x $SRC/.venv/bin/python ]]; then
  env -u VIRTUAL_ENV -u CONDA_PREFIX bash install.sh 2>&1 | tee "$OUT/install.log"
fi
PY="$SRC/.venv/bin/python"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
"$PY" -c 'import jax; print("jax", jax.__version__); print("devices", jax.devices())' | tee "$OUT/jax_devices_$ARM.txt"

# BC2 keeps a persistent on-disk compile cache keyed by GPU name (cli.py use_campaign_compile_cache),
# so a second run would read every shape warm and the cold reading would be unrepeatable. Cleared
# here so "cold" is measured rather than remembered; the cache is a real property of the program and
# is reported as such, not quietly enjoyed.
if [[ ${KEEP_COMPILE_CACHE:-0} != 1 ]]; then
  rm -rf "$SRC"/.bindcraft-compile-cache* "$HOME"/.cache/bindcraft* 2>/dev/null || true
fi

# The clock the steps actually ran at, sampled during the run and not before it.
( while true; do
    echo "$(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader)"
    sleep 5
  done ) > "$OUT/clocks_$ARM.txt" &
CLOCK_WATCH=$!
trap 'kill $CLOCK_WATCH 2>/dev/null || true' EXIT

PROJECT="$OUT/pdl1_$ARM"
CAMPAIGN_START=$(date +%s.%N)
set +e
"$PY" "$HERE/bc2_step_timing.py" --jsonl "$OUT/steps_$ARM.jsonl" -- \
  design "$SRC/examples/pdl1.json" --core "$CORE" \
  ${BINDER_LENGTH:+--set "binder_lengths=[$BINDER_LENGTH]"} \
  --set "number_of_final_designs=$FINAL_DESIGNS" \
  --set "max_trajectories=$MAX_TRAJECTORIES" \
  --set "auto_multi_gpu=false" \
  --set "workers_per_gpu=1" \
  --set "project_folder=$PROJECT" \
  2>&1 | tee "$OUT/campaign_$ARM.log"
CAMPAIGN_STATUS=${PIPESTATUS[0]}
set -e
CAMPAIGN_END=$(date +%s.%N)
kill $CLOCK_WATCH 2>/dev/null || true

# The campaign's own process wall, which is the cross-check on the phase split: if the per-trajectory
# design + mpnn_validation walls do not add up to this minus startup, a phase is missing its tail.
{
  echo "arm=$ARM status=$CAMPAIGN_STATUS"
  echo "campaign_wall_s=$(python3 -c "print(round($CAMPAIGN_END - $CAMPAIGN_START, 3))")"
} > "$OUT/campaign_wall_$ARM.txt"
cat "$OUT/campaign_wall_$ARM.txt"

"$PY" "$HERE/analyze_steps.py" "$OUT/steps_$ARM.jsonl" --report "$OUT/denominator_$ARM.json" > /dev/null
# Only the ACCEPTED designs. BC2 writes them to <project>/3_Ranked (campaign_output.py:30); the
# trajectory folders under 1_Trajectories hold intermediate and rejected structures, and validating
# those would report failures that are not failures.
"$PY" "$HERE/validate_designs.py" "$PROJECT/3_Ranked" --binder-length "$BINDER_LENGTH" \
  --target-length "$TARGET_LENGTH" --report "$OUT/designs_$ARM.json" \
  || echo "validation reported problems (or no accepted design), see designs_$ARM.json"

# BC2's own per-trajectory Timing column, for corroboration of the design wall we measured.
find "$PROJECT" -maxdepth 2 -name '*.csv' -exec cp {} "$OUT/" \; 2>/dev/null || true
sha256sum $(find "$OUT" -maxdepth 2 -type f | sort) > "$OUT/sha256.txt" 2>/dev/null || true
echo "done; copy back $OUT"
