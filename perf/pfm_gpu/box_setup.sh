#!/bin/bash
# Runs ON the rented box (image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04). Kit stack per protenix_v2/STOCK.md
# 'Stack' and the kit Dockerfile's Python 3.11.5. Expects /root/kit (repo: protenix_v2 + common) and
# /weights/protenix/checkpoint/protenix-v2.pt already copied in. Log: /root/pfm/setup.log, marker SETUP-OK.
set -eux
export DEBIAN_FRONTEND=noninteractive
# No apt: some hosts cannot reach security.ubuntu.com (DE 54695728 hung on it). curl ships in the image;
# kalign is only needed for templates, which these runs do not use.
if [ ! -x /usr/local/bin/python3.11 ]; then
  curl -fsSL -o /tmp/py.tgz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
  echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/py.tgz" | sha256sum -c -
  tar -xzf /tmp/py.tgz -C /tmp && cp -rL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/py.tgz
fi
cd /root/kit/protenix_v2
[ -d venv ] || /usr/local/bin/python3.11 -m venv venv
. venv/bin/activate
export CUDA_HOME=/usr/local/cuda PROTENIX_ROOT_DIR=/weights/protenix PYTHONHASHSEED=0
pip install -q $(grep -E '^pip==' environment/requirements.lock)
grep -v -E '^(#|protenix==)' environment/requirements.lock > /tmp/stack.txt
pip install -q --no-deps -r /tmp/stack.txt
pip install -q --no-deps stock/protenix-2.0.0-py3-none-any.whl
python -c "import protenix.model.layer_norm.layer_norm"
# session.sh pushes the checkpoint in parallel and marks it verified with CKPT-OK.
while [ ! -f /root/pfm/CKPT-OK ]; do sleep 10; done
bash run.sh install --weights /weights/protenix
bash run.sh check --config a100
echo SETUP-OK
