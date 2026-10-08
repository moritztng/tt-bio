#!/bin/bash
# Runs ON the rented GPU box. Kept as its own file rather than inlined into ssh so the quoting
# stays readable and so it can be re-run by hand if a step fails.
set -eu
echo "=== remote setup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc; df -h / | tail -1

# Pinned to exactly what pc's working host arm ran. Not a fresh resolve: the comparator arm has to
# be the same software as the host arm bci-accept already measured, or a difference in its count
# could be a dependency bump rather than the trunk.
echo "--- bc2 deps (pinned) ---"
pip install -q --no-input -r /root/requirements_host.txt 2>&1 | tail -5 || true

# Installed LAST so the CUDA build wins the numpy / ml-dtypes pins above.
echo "--- jax[cuda12] ---"
pip install -q --no-input "jax[cuda12]" 2>&1 | tail -5 || true

echo "--- GPU check ---"
JAX_PLATFORMS=cuda python -c '
import jax
d = jax.devices()
print("jax", jax.__version__, d)
assert d and d[0].platform == "gpu", f"NOT on the GPU: {d}"
print("GPU OK")
'

echo "--- harness imports ---"
cd /root/tt-bio
JAX_PLATFORMS=cuda PYTHONPATH=/root/tt-bio:/root/bcx_shipped/bc2 \
  python perf/bci_accept/capture_logits.py --help > /dev/null && echo "harness imports OK"

echo "--- bindcraft + mpnn weights resolve ---"
PYTHONPATH=/root/tt-bio:/root/bcx_shipped/bc2 python -c '
import os, bindcraft
w = os.path.join(os.path.dirname(bindcraft.__file__), "weights", "proteinmpnn", "weights_neutral")
print("mpnn weights:", w, "exists:", os.path.isdir(w))
assert os.path.isdir(w), "ProteinMPNN neutral weights missing; --full cannot redesign"
p = "/root/bcx_shipped/af2_params"
n = len([f for f in os.listdir(p) if f.endswith(".npz")])
print("af2 param files:", n)
assert n, "no af2 params"
t = "/root/bcx_shipped/bc2/settings/target/structures/hPDL1.pdb"
print("target:", t, "exists:", os.path.isfile(t))
assert os.path.isfile(t), "target pdb missing"
'
echo "=== remote setup OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
