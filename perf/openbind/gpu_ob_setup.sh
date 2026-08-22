#!/usr/bin/env bash
# Stage a rented GPU box for the OpenBind-0 reference measurement.
#
#   bash gpu_ob_setup.sh base ckpt ob p2
#
# Two venvs, not one, because upstream made the two checkpoints mutually exclusive:
# openbind-2025-06-30-174k needs openfold3 >=0.5.0 and of3-p2-155k needs >=0.4,<0.5
# (openfold3/entry_points/parameters.py:38-52). Measuring both arms therefore means two
# installs, and the OB0-vs-preview2 delta is a weights AND code delta that upstream does not
# let anyone separate.
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime, the one gpu5_setup.sh proved on both
# H200 and B200. Both venvs are ISOLATED: openfold3[cuequivariance] pulls
# cuequivariance-ops-torch-cu12, which requires a newer torch than the image ships, and pip then
# installs that torch INTO the venv where it ABI-breaks against the image's torchvision if
# system site-packages are visible.
set -uo pipefail   # NOT -e: one stage failing must not kill the rest
LOG=/root/results/setup.log
mkdir -p /root/results /root/ckpt
export PATH=/opt/conda/bin:$PATH

S3=https://openfold3-data.s3.amazonaws.com/openfold3-parameters
OB_CKPT=of3-ob-2025-06-30-174k.pt
P2_CKPT=of3-p2-155k.pt

say() { echo "== $* : $(date -u +%FT%TZ) ==" | tee -a "$LOG"; }

stage_base() {
  say "stage base"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv | tee -a "$LOG"
  nproc | tee -a "$LOG"
  # A vast.ai container reports the host's core count, not its own cgroup quota.
  cat /sys/fs/cgroup/cpu.max 2>/dev/null | tee -a "$LOG"
  apt-get update -qq
  # rdkit's Chem.Draw is imported transitively by pdbeccdutils on the OF3 import path and dies
  # on libXrender.so.1, which a runtime image does not ship. Without these the whole import
  # fails long before the model is reached.
  apt-get install -y -qq build-essential git curl libxrender1 libxext6 libsm6 libxi6 \
    2>&1 | tail -2 | tee -a "$LOG"
  say "stage base done"
}

stage_ckpt() {
  say "stage ckpt"
  for f in "$OB_CKPT" "$P2_CKPT"; do
    if [ ! -s "/root/ckpt/$f" ]; then
      curl -fsSL -o "/root/ckpt/$f" "$S3/$f" || echo "FETCH FAILED $f" | tee -a "$LOG"
    fi
    ls -l "/root/ckpt/$f" 2>/dev/null | tee -a "$LOG"
    sha256sum "/root/ckpt/$f" 2>/dev/null | tee -a "$LOG"
  done
  say "stage ckpt done"
}

# $1 venv name, $2 pip spec
mk() {
  local name=$1 spec=$2
  python3 -m venv "/root/venv-$name" || return 1
  "/root/venv-$name/bin/pip" install -q --no-cache-dir -U pip wheel 2>&1 | tail -2
  "/root/venv-$name/bin/pip" install --no-cache-dir -q "$spec" 2>&1 | tail -8 | tee -a "$LOG"
  # setup_openfold is INTERACTIVE (it prompts for a cache directory) and nothing here needs the
  # cache it would create: the checkpoint is passed with --inference-ckpt-path.
  "/root/venv-$name/bin/python" -c "
import torch, openfold3
from importlib.metadata import version
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
from openfold3.core.kernels.cueq_utils import is_cuequivariance_available
print('openfold3', version('openfold3'), 'torch', torch.__version__, torch.version.cuda)
print('cueq available:', is_cuequivariance_available())
print('predict_step:', hasattr(OpenFold3AllAtom, 'predict_step'))
" 2>&1 | tail -6 | tee -a "$LOG"
}

stage_ob() { say "stage ob";  mk ob "openfold3[cuequivariance]==0.5.0"; say "stage ob done"; }
stage_p2() { say "stage p2";  mk p2 "openfold3[cuequivariance]==0.4.5"; say "stage p2 done"; }

for s in "$@"; do
  case "$s" in
    base) stage_base ;;
    ckpt) stage_ckpt ;;
    ob) stage_ob ;;
    p2) stage_p2 ;;
    *) echo "unknown stage: $s" >&2 ;;
  esac
done
say "gpu_ob_setup finished stages: $*"
