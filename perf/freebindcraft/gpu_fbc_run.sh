#!/usr/bin/env bash
# One bounded FreeBindCraft run with the stage timers on, plus GPU memory sampling.
#
#   bash gpu_fbc_run.sh /work 6 /work/out
#
# Args: <work dir> <max trajectories> <output dir>. Six trajectories on the PDL1 example is enough
# for a stable split and costs about an hour: upstream's own PDL1 run averaged 415 s of
# non-validation work per trajectory on a B200. The point of this run is the SHAPE of the wall, not
# a design campaign, so do not raise the trajectory count to "get better designs".
#
# PDL1 is the fixture because upstream published a full run on it (12.25 h, 91 trajectories, 101
# accepted, B200) in performance_data/pdl1_miniprotein, so our split has something to be checked
# against. It is a 115-residue target with a 65-150 residue binder, which does NOT saturate an
# H200 - that is fine here, this is a feasibility gate and not a perf bar. Do not publish anything
# from this run as a GPU reference cell.
set -euo pipefail

WORK="${1:-/work}"
MAX_TRAJ="${2:-6}"
OUT="${3:-$WORK/out}"
mkdir -p "$OUT"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda's activate.d hooks are not `set -u` clean (the cuda-nvcc hook reads an unset
# NVCC_PREPEND_FLAGS), so drop -u across the activation and restore it after.
set +u
conda activate BindCraft
set -u
cd "$WORK/FreeBindCraft"

DESIGN_PATH="$OUT/pdl1"
mkdir -p "$DESIGN_PATH"

python - "$DESIGN_PATH" "$MAX_TRAJ" <<'PY'
import json, sys, pathlib
design_path, max_traj = sys.argv[1], int(sys.argv[2])
t = json.load(open("settings_target/PDL1.json"))
t["design_path"] = design_path
t["number_of_final_designs"] = 1000          # never the stopping condition; max_trajectories is
pathlib.Path("settings_target/PDL1_probe.json").write_text(json.dumps(t, indent=4))
a = json.load(open("settings_advanced/default_4stage_multimer.json"))
a["max_trajectories"] = max_traj
a["save_design_animations"] = False          # matplotlib animation writing is pure host noise
a["save_design_trajectory_plots"] = False
pathlib.Path("settings_advanced/probe_4stage_multimer.json").write_text(json.dumps(a, indent=4))
print("wrote PDL1_probe.json and probe_4stage_multimer.json")
PY

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,power.draw --format=csv -l 5 > "$OUT/gpumem.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
export FBC_TIMING_LOG="$OUT/stages.jsonl"
: > "$FBC_TIMING_LOG"

time python -u ./bindcraft.py \
  --settings ./settings_target/PDL1_probe.json \
  --filters ./settings_filters/default_filters.json \
  --advanced ./settings_advanced/probe_4stage_multimer.json \
  --no-pyrosetta --verbose 2>&1 | tee "$OUT/run.log"

kill $SMI_PID 2>/dev/null || true
HERE="$(dirname "$(readlink -f "$0")")"
python "$HERE/parse_fbc_run.py" \
  --run-dir "$OUT" --design-path "$DESIGN_PATH" --report "$OUT/split.json"
# The split to actually quote. parse_fbc_run.py sums stages independently, which double-counts the
# relaxes that run nested inside predict_binder_complex; this one rebuilds it from the interval
# timestamps and separates XLA compile from compute.
python "$HERE/analyze_measured_split.py" \
  --run-dir "$OUT" --report "$OUT/measured_split.json"

# gpu-benchmark-harness-transfer-must-include-hashed-files: hash before the instance dies.
( cd "$OUT" && find . -type f -name '*.json' -o -name '*.log' -o -name '*.csv' -o -name '*.jsonl' ) | sort > "$OUT/manifest.txt"
( cd "$OUT" && xargs -a manifest.txt sha256sum ) > "$OUT/sha256sums.txt"
echo "== done. copy $OUT back, then verify sha256sums.txt on the receiving side BEFORE destroying the instance."
