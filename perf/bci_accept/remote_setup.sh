#!/bin/bash
# Runs ON the rented GPU box. Kept as its own file rather than inlined into ssh so the quoting
# stays readable and so it can be re-run by hand if a step fails.
set -eu
echo "=== remote setup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc; df -h / | tail -1

# Pinned to exactly what pc's working host arm ran, because the comparator arm should be the same
# software as the host arm bci-accept already measured: a difference in its count should be the
# trunk, not a dependency bump.
#
# But the pins came from pc's PYTHON 3.12 venv and these boxes are commonly 3.11, where some of
# them do not exist at all. `pip install -r` is all-or-nothing, so ONE unsatisfiable pin aborts the
# whole file: on 2026-10-08 `numpy==2.5.2` (3.12-only) took all 104 packages down with it, optax
# included, and the arm died on `ModuleNotFoundError: optax` two seconds after launch.
#
# So: try the exact pins first, and if that fails fall back to the same package NAMES unpinned,
# constrained to the numpy already present so the resolve cannot move numpy under the jax/torch
# CUDA stack. A fallback install is a DISCLOSED deviation, not a silent one -- it says so.
echo "--- bc2 deps (pinned) ---"
if pip install -q --no-input -r /root/requirements_host.txt 2>&1 | tail -5; then
  echo "exact pins installed"
else
  echo "!! exact pins did not resolve on $(python3 -V 2>&1); falling back to UNPINNED names"
  echo "!! versions below are a fresh resolve, not pc's set -- report this with the numbers"
  python3 -c 'import numpy;print("numpy=="+numpy.__version__)' > /root/constraints.txt
  sed -E 's/[=<>!~].*$//' /root/requirements_host.txt | grep -vE '^[[:space:]]*$|^#' | sort -u \
    > /root/req_unpinned.txt
  # PER-PACKAGE, not `-r`: the list was frozen from a venv that also had the LOCAL packages
  # installed (tt-bio, bindcraft), and those are not on PyPI at all. With `-r`, the first such
  # name aborts the whole retry -- which is exactly how the first repair attempt died, on
  # "No matching distribution found for tt-bio". Per package, a bad name costs its own line.
  FAILED=""
  while read -r pkg; do
    [ -z "$pkg" ] && continue
    case "$pkg" in tt-bio|tt_bio|bindcraft*|bcx*) continue;; esac
    pip install -q --no-input --no-cache-dir -c /root/constraints.txt "$pkg" >/dev/null 2>&1 \
      || FAILED="$FAILED $pkg"
  done < /root/req_unpinned.txt
  echo "unresolved packages:${FAILED:- none}"
  # Not fatal on its own: the deep import check below is what decides whether the arm may run.
fi

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
# NOT `capture_logits.py --help`: that exits inside argparse, and the harness's real imports
# (bindcraft.sequence_optimization -> optax, tt_bio.bindcraft2) are AFTER parse_args(), so --help
# is structurally incapable of reaching them. On 2026-10-08 it reported "harness imports OK" on a
# box with no optax at all, and the arm died 2 s after launch having burned the provisioning.
# Import exactly what main() imports, in the same order, under the same env the arm will use.
JAX_PLATFORMS=cuda PYTHONPATH=/root/tt-bio:/root/bcx_shipped/bc2 python -c '
import jax.numpy as jnp
from bindcraft.sequence_optimization import GradientSequenceOptimizer
from tt_bio import bindcraft2
from bindcraft.campaign import run_campaign
import optax
print("harness imports OK (deep: optax, sequence_optimization, bindcraft2, run_campaign)")
' || { echo "ABORT: the harness cannot import; the arm would die seconds after launch"; exit 4; }

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
