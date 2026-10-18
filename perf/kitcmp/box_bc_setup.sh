#!/bin/bash
# Runs ON the rented box: the two BindCraft stacks for the spd-kitcmp BindCraft section, each in its own environment.
#   bash box_bc_setup.sh colabdesign     marker /root/kc/SETUP-OK-colabdesign
# 1. stock BindCraft 2 (PacesaLab/BindCraft2 at the pin this fleet benches, its own [cuda13] extra, python 3.12) in
#    /opt/bc2-venv, checkout /root/bc2, AF2 + ProteinMPNN checkpoints fetched by `bindcraft fetch-weights`.
# 2. the Anthropic kit's colabdesign kit (BindCraft 1 / ColabDesign 1.1.3), route C of its README: micromamba from
#    environment/conda-linux-64.lock, pip --no-deps environment/requirements.lock, run.sh install --weights /weights/af2.
set -eux
M=$1; [ "$M" = colabdesign ] || { echo "box_bc_setup.sh takes colabdesign only"; exit 2; }
[ -f /root/kc/SETUP-OK-$M ] && exit 0
export CC=gcc CXX=g++ DEBIAN_FRONTEND=noninteractive
BC2_PIN=${BC2_PIN:-7a2dfdb8a285232a6f881899fe135c6dc48679f1}
apt-get update -q
apt-get install -y -q build-essential ca-certificates git wget bzip2 curl ffmpeg

# --- 1. stock BindCraft 2 -------------------------------------------------------------------------------------------
PY=/opt/py3.12.1
if [ ! -x $PY/bin/python3 ]; then
  curl -fsSL -o /tmp/py.tgz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz"
  echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/py.tgz" | sha256sum -c -
  mkdir -p $PY && tar -xzf /tmp/py.tgz -C /tmp && cp -rL /tmp/python/. $PY/ && rm -rf /tmp/python /tmp/py.tgz
fi
[ -d /root/bc2 ] || git clone -q https://github.com/PacesaLab/BindCraft2.git /root/bc2
git -C /root/bc2 checkout -q $BC2_PIN
[ -d /opt/bc2-venv ] || $PY/bin/python3 -m venv /opt/bc2-venv
/opt/bc2-venv/bin/pip install -q --upgrade pip
(cd /root/bc2 && /opt/bc2-venv/bin/pip install -q -e ".[cuda13]")
/opt/bc2-venv/bin/bindcraft fetch-weights
/opt/bc2-venv/bin/python -c "import jax; print('bc2 jax', jax.__version__, jax.devices())"

# --- the burn gate's own venv: box_chain.sh runs burn_gate.py (torch) and neither BindCraft stack carries torch
[ -x /opt/burn-venv/bin/python ] || { $PY/bin/python3 -m venv /opt/burn-venv
  /opt/burn-venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu130; }

# --- 2. the kit's colabdesign (BindCraft 1 / ColabDesign) -----------------------------------------------------------
# BC_STOCK_ONLY=1: stock BindCraft 2 alone (box_bc_arms.sh then skips the kit arms)
[ -n "${BC_STOCK_ONLY:-}" ] && { touch /root/kc/SETUP-OK-$M; exit 0; }
K=/root/kit/colabdesign
if [ ! -x /opt/micromamba ]; then
  wget -qO /tmp/mm https://github.com/mamba-org/micromamba-releases/releases/download/2.1.1-0/micromamba-linux-64
  echo "0d2dd49fb0171f27a63175e2c21941aafabdd6e46f521c498620b4b14ff604d8  /tmp/mm" | sha256sum -c -
  install -m 755 /tmp/mm /opt/micromamba && rm -f /tmp/mm
fi
export MAMBA_ROOT_PREFIX=/opt/mamba
[ -d /opt/mamba/envs/cdkit ] ||
  /opt/micromamba create -y -n cdkit -f $K/environment/conda-linux-64.lock
E=/opt/mamba/envs/cdkit
$E/bin/python -m pip install -q --no-deps -r $K/environment/requirements.lock
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
(cd $K && bash run.sh install --weights /weights/af2)
(cd $K && COLABDESIGN_PARAMS_DIR=/weights/af2 bash run.sh check --mode fast)
touch /root/kc/SETUP-OK-$M
