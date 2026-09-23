#!/usr/bin/env bash
# Build every upstream the MGX references need on a fresh vast.ai box
# (image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime), from the repo root:
#   bash perf/mgx/ref/ref_setup.sh > /root/results/ref_setup.log 2>&1
# The shared venvs come from scripts/gpu_vs_tt/gpu5_setup.sh; this adds what only the reference
# set needs: the protenix v0.5.0 and opendde-abag checkpoints, and rf3 (rc-foundry).
set -uo pipefail
cd "$(dirname "$0")/../../.."
S=scripts/gpu_vs_tt/gpu5_setup.sh
mkdir -p /root/results /root/ckpt
bash $S base
bash $S fetch obfetch > /root/results/fetch.log 2>&1 &
bash $S protenix opendde boltz of3 ob esm
bash $S esmweights embedweights

/root/venv-protenix/bin/python3 - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("moritztng/protenix-v0.5.0", "model_v0.5.0.pt", local_dir="/root/ckpt"))
print(hf_hub_download("aurekaresearch/OpenDDE", "opendde_abag.pt", local_dir="/root/ckpt"))
PY

# rf3: rc-foundry wants python 3.12, same as esm. The checkpoint is the one tt-bio's rf3 port
# loads (tt_bio/weights.py "rf3").
export PATH=/root/.local/bin:$PATH
uv venv --python 3.12 /root/venv-rf3
uv pip install --python /root/venv-rf3/bin/python "rc-foundry[rf3]" 2>&1 | tail -3
[ -s /root/ckpt/rf3_foundry_01_24_latest_remapped.ckpt ] || (cd /root/ckpt &&
  aria2c -x16 -s16 -k1M --file-allocation=none --console-log-level=warn \
    https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt)
wait
ls -l /root/ckpt
echo REF_SETUP_DONE "$(date -u +%FT%TZ)"
