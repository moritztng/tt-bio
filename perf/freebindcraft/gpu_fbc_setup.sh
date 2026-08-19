#!/usr/bin/env bash
# Bring a rented NVIDIA box up to a runnable FreeBindCraft, PyRosetta-free, with stage timers.
#
# Run this ON the rented instance, from a directory with >=60 GB free. It is idempotent enough to
# re-run after a failed step. Everything it needs is public; no token, no licence.
#
#   bash gpu_fbc_setup.sh /work
#
# The one acceptance check that matters is the last line it prints: jax must report a cuda device.
# A CPU-only jaxlib installs cleanly and then runs the whole benchmark 100x too slow, which is the
# expensive way to find out.
set -euo pipefail

WORK="${1:-/work}"
mkdir -p "$WORK"
cd "$WORK"

if ! command -v conda >/dev/null 2>&1 && [ ! -d "$WORK/miniforge" ]; then
  echo "== installing miniforge"
  wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$WORK/miniforge"
fi
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$WORK/miniforge")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

[ -d FreeBindCraft ] || git clone --depth 1 https://github.com/cytokineking/FreeBindCraft.git
cd FreeBindCraft

# --no-pyrosetta is the whole point: PyRosetta needs a commercial licence and FreeBindCraft exists
# to not need it. Do not add it back to "make the comparison fair".
# CONDA_OVERRIDE_CUDA (-c) only tells the solver which cuda-variant jaxlib to pick; the driver on
# the box decides what actually runs, so the jax.devices() check below is the real verification.
if [ -d "$CONDA_BASE/envs/BindCraft" ] && [ -s params/params_model_1_multimer_v3.npz ]; then
  echo "== BindCraft env and AF2 params already present, skipping install"
else
  bash ./install_bindcraft.sh --pkg_manager mamba --cuda '12.6' --no-pyrosetta
fi

# conda's activate.d hooks are not `set -u` clean (the cuda-nvcc hook reads an unset
# NVCC_PREPEND_FLAGS), so drop -u across the activation and restore it after.
set +u
conda activate BindCraft
set -u
python "$(dirname "$(readlink -f "$0")")/fbc_stage_timing.py" --repo "$PWD"

echo "== versions"
python - <<'PY'
import jax, jaxlib, sys
print("python", sys.version.split()[0], "jax", jax.__version__, "jaxlib", jaxlib.__version__)
import colabdesign, optax, haiku
print("colabdesign", getattr(colabdesign, "__version__", "n/a"), "optax", optax.__version__)
devs = jax.devices()
print("jax.devices():", devs)
assert any(d.platform == "gpu" for d in devs), "jax has no GPU device: the cuda jaxlib did not take"
print("OK: jax is on the GPU")
PY
