#!/usr/bin/env bash
# Unattended setup of the CUDA side of the Protenix-v2 / OpenDDE benchmark on a rented GPU box.
#
# Runs start-to-finish without a human so paid GPU time is spent measuring, not installing.
# Every version here is pinned to what the upstream repos pin (Protenix requirements.txt,
# OpenDDE pyproject.toml) -- floating them is how a benchmark stops being reproducible.
#
# Recommended base image: nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 (OpenDDE's own base, public
# on Docker Hub). Do NOT use Protenix's Dockerfile base -- it lives in a ByteDance registry that is
# slow or unreachable from a rented box -- and avoid the ~10GB pytorch devel image, which has
# stalled >15 min on a slow-disk vast.ai host before.
#
#   bash gpu_setup.sh 2>&1 | tee setup.log
#
set -euo pipefail

WORK="${WORK:-/workspace}"
PROTENIX_REF="${PROTENIX_REF:-main}"
OPENDDE_REF="${OPENDDE_REF:-main}"
CUTLASS_TAG="v3.5.1"

echo "=== apt ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    git g++ gcc make libc6-dev ca-certificates curl \
    hmmer kalign libglib2.0-0 libgl1 python3-pip python3-venv
apt-get clean

mkdir -p "$WORK"
cd "$WORK"

echo "=== venv ==="
python3 -m venv "$WORK/env"
# shellcheck disable=SC1091
source "$WORK/env/bin/activate"
pip install --upgrade pip wheel setuptools

echo "=== torch (cu126, pinned to what both upstreams pin) ==="
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

echo "=== NVIDIA cuEquivariance triangle kernels ==="
# This is NVIDIA's own kernel library for triangle attention / triangle multiplicative update and
# is the DEFAULT triangle backend in Protenix (configs/configs_base.py). Wheels, no compilation.
pip install cuequivariance-torch==0.8.0 cuequivariance-ops-torch-cu12==0.8.0

echo "=== DeepSpeed (for the DS4Sci_EvoformerAttention ladder rung) ==="
pip install deepspeed==0.17.5 triton==3.3.1
git clone --depth 1 -b "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git /opt/cutlass
export CUTLASS_PATH=/opt/cutlass
echo 'export CUTLASS_PATH=/opt/cutlass' >> "$WORK/env/bin/activate"

echo "=== Protenix ==="
git clone https://github.com/bytedance/Protenix "$WORK/Protenix"
git -C "$WORK/Protenix" checkout "$PROTENIX_REF"
pip install -r "$WORK/Protenix/requirements.txt"
pip install -e "$WORK/Protenix"

echo "=== OpenDDE ==="
git clone https://github.com/aurekaresearch/OpenDDE "$WORK/OpenDDE"
git -C "$WORK/OpenDDE" checkout "$OPENDDE_REF"
pip install -e "$WORK/OpenDDE[gpu]"

echo "=== weights ==="
mkdir -p "$WORK/ckpt"
curl -L --fail -o "$WORK/ckpt/protenix-v2.pt" \
    https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt
# Fairness precondition: this must equal the checkpoint tt-bio folds with on our side.
EXPECT_PROTENIX_SHA=8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
GOT=$(sha256sum "$WORK/ckpt/protenix-v2.pt" | cut -d' ' -f1)
echo "protenix-v2.pt sha256: $GOT"
if [ "$GOT" != "$EXPECT_PROTENIX_SHA" ]; then
    echo "WARNING: checkpoint sha256 differs from the tt-bio copy ($EXPECT_PROTENIX_SHA)."
    echo "Do not proceed with a head-to-head claim until this is explained. Compare state-dict"
    echo "key/shape/dtype signatures and record the diff in the state doc."
fi
pip install "huggingface_hub>=0.34"
hf download aurekaresearch/OpenDDE --local-dir "$WORK/ckpt/opendde"

echo "=== provenance (paste into the results JSON) ==="
{
  echo "date_utc=$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
  echo "cuda_runtime=$(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
  python3 -c "import torch;print(f'torch={torch.__version__} cuda={torch.version.cuda} capability={torch.cuda.get_device_capability()}')"
  python3 -c "import triton;print(f'triton={triton.__version__}')"
  echo "protenix_sha=$(git -C "$WORK/Protenix" rev-parse HEAD)"
  echo "opendde_sha=$(git -C "$WORK/OpenDDE" rev-parse HEAD)"
} | tee "$WORK/provenance.txt"

echo "=== done ==="
