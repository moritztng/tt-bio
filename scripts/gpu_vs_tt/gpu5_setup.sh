#!/usr/bin/env bash
# Setup for the five-model / two-GPU 512 aa benchmark. Staged, because the protenix
# control has to run before anything else is installed (if the control is off, no other
# install is worth paying for) and because ESMC-6B is a 24 GB download that should be
# pulling while the small venvs build.
#
# Usage on the box:
#   bash gpu5_setup.sh base                # CUDA toolchain fix, once
#   bash gpu5_setup.sh protenix            # venv + weights for the control
#   bash gpu5_setup.sh fetch &             # background: the two big downloads
#   bash gpu5_setup.sh boltz esm of3 opendde
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime. torch 2.7.1+cu128 is protenix
# 2.0.0's exact pin and the first torch line with Blackwell/sm_100, so ONE image serves
# both the H200 and the B200 and torch stops being a variable between them.
set -uo pipefail   # NOT -e: one model failing to install must not kill the rest
cd "$(dirname "$0")"
HERE=$(pwd)
LOG=/root/results/setup.log
mkdir -p /root/results /root/ckpt /root/common
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0

say() { echo "== $* : $(date -u +%FT%TZ) ==" | tee -a "$LOG"; }

stage_base() {
  say "stage base"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$LOG"
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
'cap',torch.cuda.get_device_capability(),torch.cuda.get_device_name(0))" | tee -a "$LOG"
  # Both protenix-family vendors JIT-compile a fused layernorm CUDA extension at import.
  # The runtime image has no nvcc, and torch's cpp_extension needs more than nvcc:
  # cuda_runtime_api.h and, through ATen's CUDAContextLight.h, cusparse.h. cuda-compiler
  # ships neither; cuda-toolkit is the metapackage that carries the headers. Gate on a
  # header, not on nvcc, so a box that already has nvcc still gets the fix.
  if [ ! -f /opt/conda/include/cuda_runtime_api.h ] || [ ! -f /opt/conda/include/cusparse.h ]; then
    apt-get update -qq && apt-get install -y -qq build-essential git curl
    /opt/conda/bin/conda install -y -n base -c nvidia "cuda-toolkit=12.8" 2>&1 | tail -3
  else
    apt-get update -qq && apt-get install -y -qq build-essential git curl
  fi
  # conda's nvidia channel lays CUDA out as targets/x86_64-linux/{include,lib}, not as
  # CUDA_HOME/{include,lib64}, which is the only layout cpp_extension knows.
  if [ -d /opt/conda/targets/x86_64-linux/include ]; then
    for f in /opt/conda/targets/x86_64-linux/include/*; do
      b=$(basename "$f"); [ -e "/opt/conda/include/$b" ] || ln -sfn "$f" "/opt/conda/include/$b"
    done
    [ -e /opt/conda/lib64 ] || ln -sfn /opt/conda/targets/x86_64-linux/lib /opt/conda/lib64
  fi
  say "stage base done"
}

# One venv per model. --system-site-packages reuses the image's torch 2.7.1+cu128 so no
# multi-GB torch download lands on the clock. The venvs are separate because the
# cuequivariance pins genuinely conflict: protenix 2.0.0 wants 0.8.0, opendde 1.0.3 wants
# 0.10.0, boltz/of3 take >=0.8 unpinned.
mkvenv() {
  [ -x "/root/venv-$1/bin/pip" ] || python3 -m venv --system-site-packages "/root/venv-$1"
  "/root/venv-$1/bin/pip" install --no-cache-dir --upgrade pip -q
}

stage_protenix() {
  say "stage protenix"
  mkvenv protenix
  /root/venv-protenix/bin/pip install --no-cache-dir -q protenix==2.0.0 huggingface_hub==0.34.4 \
    2>&1 | tail -5 | tee -a "$LOG"
  if [ ! -s /root/ckpt/protenix-v2.pt ]; then
    # The official checkpoint URL is gated (403); TMF001/protenix-v2-weights is the public
    # mirror of the same protenix-v2.pt the TT side runs (tt_bio/main.py PROTENIX_REPO).
    /root/venv-protenix/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('TMF001/protenix-v2-weights','protenix-v2.pt',local_dir='/root/ckpt'))
" 2>&1 | tail -3 | tee -a "$LOG"
  fi
  /root/venv-protenix/bin/protenix --help >/dev/null && echo "protenix CLI ok" | tee -a "$LOG"
  say "stage protenix done"
}

stage_opendde() {
  say "stage opendde"
  mkvenv opendde
  /root/venv-opendde/bin/pip install --no-cache-dir -q "opendde[gpu]==1.0.3" huggingface_hub==0.34.4 \
    2>&1 | tail -5 | tee -a "$LOG"
  if [ ! -s /root/ckpt/opendde.pt ]; then
    /root/venv-opendde/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('aurekaresearch/OpenDDE','opendde.pt',local_dir='/root/ckpt'))
" 2>&1 | tail -3 | tee -a "$LOG"
  fi
  say "stage opendde done"
}

stage_boltz() {
  say "stage boltz"
  mkvenv boltz
  # cuEquivariance kernels are ON by default in boltz 2.2.1 (--no_kernels defaults False),
  # so the [cuda] extra is the fast path and no flag is needed to get it.
  /root/venv-boltz/bin/pip install --no-cache-dir -q "boltz[cuda]==2.2.1" 2>&1 | tail -5 | tee -a "$LOG"
  /root/venv-boltz/bin/boltz predict --help >/dev/null 2>&1 && echo "boltz CLI ok" | tee -a "$LOG"
  say "stage boltz done"
}

stage_of3() {
  say "stage of3"
  mkvenv of3
  /root/venv-of3/bin/pip install --no-cache-dir -q "openfold3[cuequivariance]==0.4.4" \
    2>&1 | tail -8 | tee -a "$LOG"
  /root/venv-of3/bin/setup_openfold 2>&1 | tail -5 | tee -a "$LOG"
  say "stage of3 done"
}

stage_esm() {
  say "stage esm"
  mkvenv esm
  /root/venv-esm/bin/pip install --no-cache-dir -q "esm@git+https://github.com/Biohub/esm.git@main" \
    2>&1 | tail -8 | tee -a "$LOG"
  /root/venv-esm/bin/pip install --no-cache-dir -q huggingface_hub==0.34.4 2>&1 | tail -2
  say "stage esm done"
}

# The two big pulls, kicked off in the background while the small venvs build.
stage_fetch() {
  say "stage fetch"
  if [ ! -s /root/ckpt/of3-p2-155k.pt ]; then
    curl -sSL -o /root/ckpt/of3-p2-155k.pt \
      https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-155k.pt \
      && ls -l /root/ckpt/of3-p2-155k.pt | tee -a "$LOG"
  fi
  say "stage fetch done"
}

stage_esmweights() {
  say "stage esmweights"
  /root/venv-esm/bin/python3 -c "
from huggingface_hub import snapshot_download
for r in ('biohub/ESMFold2','biohub/ESMC-6B'):
    print(r, snapshot_download(r))
" 2>&1 | tail -4 | tee -a "$LOG"
  say "stage esmweights done"
}

for s in "$@"; do
  case "$s" in
    base) stage_base ;;
    protenix) stage_protenix ;;
    opendde) stage_opendde ;;
    boltz) stage_boltz ;;
    of3) stage_of3 ;;
    esm) stage_esm ;;
    esmweights) stage_esmweights ;;
    fetch) stage_fetch ;;
    *) echo "unknown stage: $s" >&2 ;;
  esac
done
say "gpu5_setup finished stages: $*"
