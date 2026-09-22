#!/usr/bin/env bash
# Provision a rented GPU box for the OF3 training reference.
#
# Upstream: github.com/aqlaboratory/openfold-3 tag v0.5.0 = c4771653c5d0a3ebb0b3af71b05efd64bc44ee86.
# The reference stack targets 0.5.0 (LEDGER R1-AMENDED); tt-bio's production pin stays 0.4.3 and
# the update-rule constants are identical between the two, which is checked below rather than
# assumed.
set -euo pipefail

ROOT=${ROOT:-/root/of3t}
TAG=v0.5.0
COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86

mkdir -p "$ROOT"
cd "$ROOT"

echo "=== box ==="
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
lscpu | head -18
free -g
cat /proc/loadavg
df -h "$ROOT" | tail -1

echo "=== deps ==="
# libxrender1/libxext6 are rdkit.Chem.Draw's, and pdbeccdutils imports Draw at module scope, so
# without them upstream's own conftest fails to import. The pip list is upstream's declared
# dependency set from pyproject.toml plus what the reference harness needs; installing it in one
# go is worth doing because chasing ModuleNotFoundError one package at a time cost a pass.
apt-get update -qq && apt-get install -y -qq git curl libxrender1 libxext6 >/dev/null
pip install -q --no-input "pytorch-lightning>=2.1" ml-collections biotite "rdkit<2026" \
  pdbeccdutils kalign-python ijson dm-tree einops torchmetrics deepspeed \
  numpy scipy pandas requests awscli awscrt boto3 PyYAML wandb func_timeout tqdm \
  typing-extensions lmdb memory_profiler pytest 2>&1 | tail -3 || true

echo "=== upstream ==="
[ -d openfold-3 ] || git clone -q --branch "$TAG" https://github.com/aqlaboratory/openfold-3.git
cd openfold-3 && git log -1 --format='commit %H %ci %s' && test "$(git rev-parse HEAD)" = "$COMMIT"
cd "$ROOT"

echo "=== torch ==="
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('bf16', torch.cuda.is_bf16_supported())
"

echo "=== update-rule constants, read off the config we will actually run ==="
PYTHONPATH="$ROOT/openfold-3" python -c "
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
s = OF3ProjectEntry().get_model_config_with_presets(presets=['train']).settings
print('optimizer', dict(s.optimizer))
print('lr_scheduler', dict(s.lr_scheduler))
print('gradient_clipping', dict(s.gradient_clipping))
print('ema', dict(s.ema))
"
echo "=== SETUP DONE ==="
