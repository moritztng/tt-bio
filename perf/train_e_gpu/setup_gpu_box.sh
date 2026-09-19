#!/usr/bin/env bash
# Provision a rented GPU box to run ABodyBuilder3's OWN training step.
#
# Upstream: https://github.com/Exscientia/ABodyBuilder3 at 18e4058015a39c5405c08a0d5629cf302627b253
# Data:     Zenodo 10.5281/zenodo.11354577, data.tar.gz (3.02 GB). structures_plm.tar.gz is NOT
#           needed: params.yaml sets language.model: null, so the run uses the one-hot single
#           feature, not PLM embeddings.
# Image assumed: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime (torch 2.1.2 is upstream's pin).
set -euo pipefail

ROOT=${ROOT:-/root/abb3}
COMMIT=18e4058015a39c5405c08a0d5629cf302627b253

mkdir -p "$ROOT"
cd "$ROOT"

echo "=== deps ==="
apt-get update -qq && apt-get install -y -qq git curl aria2 >/dev/null
# Upstream's pins from pinned-versions.txt / setup.cfg, minus the inference-only and PLM extras.
# numpy is pinned 1.21.6 upstream, which has no wheel for py3.10; 1.26.4 is the nearest that does
# and torch 2.1.2 is built against the 1.x ABI, so the arrays behave identically.
pip install -q --no-input \
  "lightning==2.1.2" "ml_collections==0.1.1" "python-box==7.1.1" \
  "loguru==0.7.2" "einops" "dm-tree" "tqdm" \
  "numpy==1.26.4" "scipy==1.10.1" "pandas==1.5.3" \
  "biopython==1.81" "levenshtein"
# dvc and dvclive at upstream's pins: stages/finetune.py imports dvc.api at module scope, and the
# DVCLiveLogger control run needs dvclive. openmm/pdbfixer are pulled in by the same import chain
# (openfold/np/relax/cleanup.py), from PyPI rather than conda-forge.
pip install -q --no-input "dvc==2.58.2" "dvclive==2.11.0" "openmm" "pdbfixer"

echo "=== clone ==="
if [ ! -d ABodyBuilder3 ]; then
  git clone -q https://github.com/Exscientia/ABodyBuilder3.git
fi
cd ABodyBuilder3
git checkout -q "$COMMIT"
git log -1 --format='commit %H %ci %s'

echo "=== data ==="
if [ ! -d data/structures/structures ]; then
  mkdir -p zenodo
  [ -s zenodo/data.tar.gz ] || aria2c -x8 -s8 -d zenodo -o data.tar.gz \
    https://zenodo.org/records/11354577/files/data.tar.gz
  tar -xzf zenodo/data.tar.gz
fi
echo "structures: $(ls data/structures/structures | wc -l) .pt files"
python -c "
import pandas as pd
df = pd.read_csv('data/split.csv', index_col=0)
print(df['split'].value_counts().to_dict())
"

echo "=== torch ==="
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('bf16 supported', torch.cuda.is_bf16_supported())
"
echo "=== SETUP OK ==="
