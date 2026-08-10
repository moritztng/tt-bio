#!/usr/bin/env bash
# GPU-box setup for the GPU-vs-TT head-to-head. Runs ONCE at the start of the
# paid session. Target image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# (torch 2.7.1+cu128 already inside == protenix 2.0.0's pinned torch, so no
# multi-GB torch download on the clock).
set -euo pipefail
cd "$(dirname "$0")"

echo "== gpu_setup: $(date -u +%FT%TZ) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# CUDA toolchain. The runtime image has no nvcc/CUDA_HOME, but both vendors
# JIT-compile a fused layernorm CUDA extension at import (their own dockers use
# devel images). It needs more than nvcc: torch's cpp_extension pulls in
# cuda_runtime_api.h and, through ATen's CUDAContextLight.h, cusparse.h. The
# cuda-compiler metapackage ships neither, so a 2026-08-10 session died at the
# first fold with "fatal error: cuda_runtime_api.h: No such file or directory".
# cuda-toolkit is the metapackage that carries the headers. Gate on a header,
# not on nvcc, or a box that already has nvcc skips the fix.
if [ ! -f /opt/conda/include/cuda_runtime_api.h ] || [ ! -f /opt/conda/include/cusparse.h ]; then
  apt-get update -qq && apt-get install -y -qq build-essential
  /opt/conda/bin/conda install -y -n base -c nvidia "cuda-toolkit=12.8"
fi
# conda's nvidia channel lays CUDA out as targets/x86_64-linux/{include,lib},
# not as CUDA_HOME/{include,lib64}, which is the only layout torch's
# cpp_extension knows. Link the two views together; skip names that already
# exist under include/ (CL/ is a real directory there and ln refuses it).
if [ -d /opt/conda/targets/x86_64-linux/include ]; then
  for f in /opt/conda/targets/x86_64-linux/include/*; do
    b=$(basename "$f"); [ -e "/opt/conda/include/$b" ] || ln -sfn "$f" "/opt/conda/include/$b"
  done
  [ -e /opt/conda/lib64 ] || ln -sfn /opt/conda/targets/x86_64-linux/lib /opt/conda/lib64
fi
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH

# Two venvs: opendde[gpu] pins cuequivariance 0.10.0 while protenix 2.0.0 pins
# cuequivariance 0.8.0 -- one env cannot serve both. --system-site-packages
# reuses the image's torch 2.7.1+cu128 (== both packages' pinned torch), so no
# multi-GB torch download on the clock.
# SETUP_MODELS trims the install to what the session will actually run. The throughput
# pass rents ~40 min of H200 and only needs protenix-v2, so building the opendde venv and
# pulling its checkpoint would spend paid minutes on an unused environment.
SETUP_MODELS=${SETUP_MODELS:-"protenix opendde"}
for V in $SETUP_MODELS; do
  python3 -m venv --system-site-packages /root/venv-$V
  /root/venv-$V/bin/pip install --no-cache-dir --upgrade pip -q
done
case " $SETUP_MODELS " in *" protenix "*)
  /root/venv-protenix/bin/pip install --no-cache-dir -q protenix==2.0.0 huggingface_hub==0.34.4 ;;
esac
case " $SETUP_MODELS " in *" opendde "*)
  /root/venv-opendde/bin/pip install --no-cache-dir -q "opendde[gpu]==1.0.3" huggingface_hub==0.34.4 ;;
esac
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# Weights. Protenix-v2's official checkpoint URL is gated (403); the public HF
# mirror TMF001/protenix-v2-weights carries the SAME protenix-v2.pt the TT side
# runs (tt_bio/main.py PROTENIX_REPO). OpenDDE's opendde.pt is public on HF.
mkdir -p /root/ckpt
if [ ! -s /root/ckpt/protenix-v2.pt ] && [[ " $SETUP_MODELS " == *" protenix "* ]]; then
  /root/venv-protenix/bin/python3 - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("TMF001/protenix-v2-weights", "protenix-v2.pt",
                    local_dir="/root/ckpt")
print("protenix ckpt:", p)
EOF
fi
if [ ! -s /root/ckpt/opendde.pt ] && [[ " $SETUP_MODELS " == *" opendde "* ]]; then
  /root/venv-opendde/bin/python3 - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("aurekaresearch/OpenDDE", "opendde.pt",
                    local_dir="/root/ckpt")
print("opendde ckpt:", p)
EOF
fi

# Smoke: imports resolve, CLIs answer, GPU visible to torch.
if [[ " $SETUP_MODELS " == *" protenix "* ]]; then
  /root/venv-protenix/bin/protenix --help >/dev/null && echo "protenix CLI ok"
  /root/venv-protenix/bin/python3 -c "
import runner.inference as ri
print('protenix runner import ok:', ri.InferenceRunner.__name__)" || \
  /root/venv-protenix/bin/python3 -c "
import protenix.runner.inference as ri
print('protenix runner import ok (pkg path):', ri.InferenceRunner.__name__)"
fi
if [[ " $SETUP_MODELS " == *" opendde "* ]]; then
  /root/venv-opendde/bin/opendde --help >/dev/null && echo "opendde CLI ok"
  /root/venv-opendde/bin/python3 -c "
import runner.inference as ri
print('opendde runner import ok:', ri.InferenceRunner.__name__)" || \
  /root/venv-opendde/bin/python3 -c "
import opendde.runner.inference as ri
print('opendde runner import ok (pkg path):', ri.InferenceRunner.__name__)" || true
fi

# GPU-sharing facilities, recorded before anything is measured. MPS decides whether the
# mps leg can run at all; the MIG lines are the "what state is this card in" half of the
# MIG question (the "can we change it" half is attempted at the very END of the session,
# because a successful MIG toggle needs a GPU reset and would end the run).
echo "== gpu sharing probe =="
command -v nvidia-cuda-mps-control && echo "MPS control binary PRESENT" || echo "MPS control binary ABSENT"
nvidia-smi --query-gpu=mig.mode.current,mig.mode.pending,compute_mode --format=csv || true

echo "== gpu_setup done: $(date -u +%FT%TZ) =="
