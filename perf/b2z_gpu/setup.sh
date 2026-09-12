#!/bin/bash
# Set up a rented NVIDIA box to run the Boltz-2 512 aa fold through tt-bio's own torch
# reference modules -- the same module tree the ttnn port replaces layer for layer
# (tt_bio/reference.py PairformerLayer / MSALayer / DiffusionModule), so the GPU
# decomposition lands on the same three units as perf/b2x_op_cost/device_floor.py.
#
# torch is installed from the cu124 index on purpose: this box's driver is 555.x
# (CUDA 12.5), and pinning the wheel to a toolkit the driver actually carries avoids
# leaning on minor-version compatibility for a measurement. tt-bio is installed
# after torch and declares a bare "torch", so it does not pull a second one.
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git curl python3-pip || FAIL "apt"
echo "--- gcc: $(gcc --version | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

pip3 install -q --break-system-packages uv 2>/dev/null || pip3 install -q uv || FAIL "uv"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
$UV venv --python 3.12 /work/venv || FAIL "venv"
PY=/work/venv/bin/python

$UV pip install -q --python $PY torch --index-url https://download.pytorch.org/whl/cu124 || FAIL "torch"
$UV pip install -q --python $PY -e /work/tt-bio || FAIL "tt-bio"

$PY - <<'PY' || FAIL "import check"
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import tt_bio.reference as R
print("  reference modules:", R.PairformerLayer, R.MSALayer, R.DiffusionModule)
PY

# boltz2 checkpoint + the CCD molecule library, from tt-bio's own weight registry.
/work/venv/bin/tt-bio weights --models boltz2 --download || FAIL "weights"
/work/venv/bin/tt-bio weights --models boltz2 || FAIL "weights verify"

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
