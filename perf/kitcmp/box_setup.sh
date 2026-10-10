#!/bin/bash
# Runs ON the rented box (image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04): one kit's stack in its own venv, the way
# its environment/Dockerfile does it (requirements.lock with --no-deps, then the pinned stock wheel, then run.sh install).
#   bash box_setup.sh MODEL     MODEL = boltz2 | openfold3 | opendde | colabdesign; marker /root/kc/SETUP-OK-<MODEL>
# Weights are fetched by the kit's own `install --weights` into /weights/<model>.
set -eux
M=$1; K=/root/kit/$M
if [ ! -x /usr/local/bin/python3.11 ]; then
  curl -fsSL -o /tmp/py.tgz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
  echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/py.tgz" | sha256sum -c -
  tar -xzf /tmp/py.tgz -C /tmp && cp -rL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/py.tgz
fi
cd $K; [ -d venv ] || /usr/local/bin/python3.11 -m venv venv; . venv/bin/activate
export CUDA_HOME=/usr/local/cuda PYTHONHASHSEED=0 CFLAGS=-g0
L=environment/requirements.lock
pip install -q $(grep -E '^(pip|wheel)==' $L)
case $M in
  openfold3)   X='openfold3|deepspeed';;
  opendde)     X='opendde';;
  boltz2)      X='boltz';;
  colabdesign) X='colabdesign|bindcraft';;
esac
grep -v -E "^(#|pip==|wheel==|($X)==)" $L > /tmp/stack-$M.txt
pip install -q --no-deps -r /tmp/stack-$M.txt
case $M in
  openfold3)  # DS4Sci evoformer kernel built for the A100 (sm 8.0), as the Dockerfile does for sm 9.0
    pip install -q --no-deps stock/openfold3-0.4.1-py3-none-any.whl
    DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=$(nproc) \
      pip install -q --no-deps --no-build-isolation --no-binary deepspeed $(grep -E '^deepspeed==' $L)
    export OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt; W=/weights/openfold3;;
  opendde)    pip install -q --no-deps stock/opendde-1.1.1-py3-none-any.whl; export OPENDDE_ROOT_DIR=/weights/opendde; W=$OPENDDE_ROOT_DIR;;
  boltz2)     pip install -q --no-deps stock/boltz-2.2.1-py3-none-any.whl; export BOLTZ_CACHE=/weights/boltz2; W=$BOLTZ_CACHE;;
  colabdesign) export COLABDESIGN_PARAMS_DIR=/weights/af2; W=$COLABDESIGN_PARAMS_DIR;;   # its stock tarballs install via run.sh
esac
mkdir -p $W
bash run.sh install --weights $W
if [ $M = colabdesign ]; then bash run.sh check --mode fast; else bash run.sh check --config a100 --mode fast; fi
echo SETUP-OK > /root/kc/SETUP-OK-$M
