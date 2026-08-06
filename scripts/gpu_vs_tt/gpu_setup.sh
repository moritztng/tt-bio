#!/usr/bin/env bash
# GPU-box setup for the GPU-vs-TT head-to-head. Runs ONCE at the start of the
# paid session. Target image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# (torch 2.7.1+cu128 already inside == protenix 2.0.0's pinned torch, so no
# multi-GB torch download on the clock).
set -euo pipefail
cd "$(dirname "$0")"

echo "== gpu_setup: $(date -u +%FT%TZ) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Two venvs: opendde[gpu] pins cuequivariance 0.10.0 while protenix 2.0.0 pins
# cuequivariance 0.8.0 -- one env cannot serve both. --system-site-packages
# reuses the image's torch 2.7.1+cu128 (== both packages' pinned torch), so no
# multi-GB torch download on the clock.
for V in protenix opendde; do
  python3 -m venv --system-site-packages /root/venv-$V
  /root/venv-$V/bin/pip install --no-cache-dir --upgrade pip -q
done
/root/venv-protenix/bin/pip install --no-cache-dir -q protenix==2.0.0 huggingface_hub==0.34.4
/root/venv-opendde/bin/pip install --no-cache-dir -q "opendde[gpu]==1.0.3" huggingface_hub==0.34.4
/root/venv-protenix/bin/python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# Weights. Protenix-v2's official checkpoint URL is gated (403); the public HF
# mirror TMF001/protenix-v2-weights carries the SAME protenix-v2.pt the TT side
# runs (tt_bio/main.py PROTENIX_REPO). OpenDDE's opendde.pt is public on HF.
mkdir -p /root/ckpt
if [ ! -s /root/ckpt/protenix-v2.pt ]; then
  /root/venv-protenix/bin/python3 - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("TMF001/protenix-v2-weights", "protenix-v2.pt",
                    local_dir="/root/ckpt")
print("protenix ckpt:", p)
EOF
fi
if [ ! -s /root/ckpt/opendde.pt ]; then
  /root/venv-opendde/bin/python3 - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("aurekaresearch/OpenDDE", "opendde.pt",
                    local_dir="/root/ckpt")
print("opendde ckpt:", p)
EOF
fi

# Smoke: imports resolve, CLIs answer, GPU visible to torch.
/root/venv-protenix/bin/protenix --help >/dev/null && echo "protenix CLI ok"
/root/venv-opendde/bin/opendde --help >/dev/null && echo "opendde CLI ok"
/root/venv-protenix/bin/python3 -c "
import runner.inference as ri
print('protenix runner import ok:', ri.InferenceRunner.__name__)" || \
/root/venv-protenix/bin/python3 -c "
import protenix.runner.inference as ri
print('protenix runner import ok (pkg path):', ri.InferenceRunner.__name__)"
/root/venv-opendde/bin/python3 -c "
import runner.inference as ri
print('opendde runner import ok:', ri.InferenceRunner.__name__)" || \
/root/venv-opendde/bin/python3 -c "
import opendde.runner.inference as ri
print('opendde runner import ok (pkg path):', ri.InferenceRunner.__name__)" || true

echo "== gpu_setup done: $(date -u +%FT%TZ) =="
