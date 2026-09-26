#!/usr/bin/env bash
# Re-derive the PERF10 denominator on a rented H200: an OpenFold3 training step at crop 384 with
# the recycle count PINNED to what our TT step pins (`--cycles 4`), and again at upstream's own
# U{0..3} draw, at 48 and at 4 diffusion samples.
#
# The record's GPU reference is n=2 steady samples over an unpinned draw. Four arms replace it.
# Everything banks under $ROOT, never /tmp (K61). Provisioning follows instrument_d_gpu.sh,
# which is the proven recipe on this image.
set -x
ROOT=${ROOT:-/root/of3t_p10gpuref}
TAG=v0.5.0
COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
STEPS_PIN=${STEPS_PIN:-10}
STEPS_DRAW=${STEPS_DRAW:-14}
mkdir -p "$ROOT"; cd "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1
echo "=== START $(date -u +%FT%TZ) ==="

echo "=== LINK CHECK FIRST: a box can install fine and still be unable to download ==="
_out=$(curl -sL --max-time 20 -o /dev/null -w "%{size_download} %{speed_download} %{http_code}" \
  "https://openfold3-data.s3.amazonaws.com/pdb_training_set/dataset_caches/validation_cache_with_templates.json" 2>/dev/null)
set -- $_out; _sz=${1:-0}; _lk=${2:-0}; _code=${3:-000}; _lk=${_lk%%.*}
echo "LINK: ${_lk} B/s, ${_sz} bytes, HTTP ${_code}"
if [ "${_code}" != "200" ] || [ "${_sz:-0}" -lt 1000000 ]; then
  echo "FATAL: the probe itself failed (http ${_code}, ${_sz} bytes). Fix the probe, not the box."; exit 4
fi
if [ "${_lk:-0}" -lt 2000000 ]; then
  echo "FATAL: link ${_lk} B/s is under the 2 MB/s floor. Destroy this box and take another."; exit 3
fi

echo "=== box stamp ==="
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
lscpu | head -12; free -g; df -h "$ROOT" | tail -1

echo "=== deps ==="
apt-get update -qq && apt-get install -y -qq git curl libxrender1 libxext6 libsm6 >/dev/null
pip install -q --no-input "pytorch-lightning>=2.1" ml-collections biotite "rdkit<2026" \
  pdbeccdutils kalign-python ijson dm-tree einops torchmetrics deepspeed boto3 pytest \
  2>&1 | tail -3

echo "=== upstream $TAG ==="
[ -d openfold-3 ] || git clone -q --branch "$TAG" https://github.com/aqlaboratory/openfold-3.git
cd openfold-3
git log -1 --format='commit %H %ci %s'
test "$(git rev-parse HEAD)" = "$COMMIT" || { echo "FATAL: not the pinned commit"; exit 2; }
pip install -q -e . 2>&1 | tail -5
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'dev',torch.cuda.get_device_name(0))"

echo "=== the local PDB subset ==="
python scripts/datasets/generate_subset_cache.py 2>&1 | tail -20
python scripts/datasets/download_subset.py 2>&1 | tail -20
ls -la datasets/ | head; du -sh datasets
RUNNER=$ROOT/openfold-3/datasets/train_pdb_subset.yaml
test -f "$RUNNER" || { echo "FATAL: no runner yaml at $RUNNER"; exit 5; }
echo "--- runner yaml as upstream generated it ---"; cat "$RUNNER"

# arm  <label> <pin_recycles|-1> <no_samples> <steps>
# A1 is the matched number: num_cycles 4, same as our `--cycles 4`, at upstream's 48 samples.
# A2 is upstream's own draw, unchanged, so the record carries both axes.
# A3 matches our scope on BOTH axes (4 cycles, 4 samples) -- the conversion factor p10axis wants.
# A4 gives the 48->4 sample factor at their draw.
run_arm () {
  label=$1; pin=$2; samples=$3; steps=$4
  echo "=== ARM $label  pin_recycles=$pin  no_samples=$samples  steps=$steps  $(date -u +%FT%TZ) ==="
  python "$ROOT/matched_step.py" \
    --runner-yaml "$RUNNER" \
    --out "$ROOT/out/${label}.json" \
    --output-dir "$ROOT/train_out/${label}" \
    --steps "$steps" --pin-recycles "$pin" --samples "$samples" --label "$label" 2>&1 | tail -80
  echo "ARM_EXIT[$label]=${PIPESTATUS[0]}"
}

mkdir -p "$ROOT/out" "$ROOT/train_out"
run_arm a1_pin4_s48  3 48 "$STEPS_PIN"
run_arm a2_draw_s48 -1 48 "$STEPS_DRAW"
run_arm a3_pin4_s4   3  4 "$STEPS_PIN"
run_arm a4_draw_s4  -1  4 "$STEPS_DRAW"

echo "=== ALL ARMS DONE $(date -u +%FT%TZ) ==="
ls -la "$ROOT/out"
