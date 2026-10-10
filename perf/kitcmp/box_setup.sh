#!/bin/bash
# Runs ON the rented box (image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04): one kit's stack in its own venv, the way
# its environment/Dockerfile does it (requirements.lock with --no-deps, then the pinned stock wheel, then run.sh install).
#   bash box_setup.sh MODEL     MODEL = protenix_v2 | boltz2 | openfold3 | opendde; marker /root/kc/SETUP-OK-<MODEL>
# Also drops kctime.py + kctime.pth into the venv (the synced per-fold timer, inert unless KC_WRAP is set).
# Weights are fetched by the kit's own `install --weights` into /weights/<model>; session.sh pre-places the files whose
# upstream URL fails from vast (Protenix-v2 checkpoint: 403; components.cif.rdkit_mol.pkl: a newer file than the pin).
set -eux
M=$1; K=/root/kit/$M
if [ ! -x /usr/local/bin/python3.11 ]; then
  curl -fsSL -o /tmp/py.tgz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
  echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/py.tgz" | sha256sum -c -
  tar -xzf /tmp/py.tgz -C /tmp && cp -rL /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/py.tgz
fi
cd $K; [ -d venv ] || /usr/local/bin/python3.11 -m venv venv; . venv/bin/activate
# each kit's Dockerfile pins its own nvcc (openfold3 12.8, opendde 12.6, protenix 13.0 = the image's, boltz2 none): install it
# beside the image's toolkit; box_arms.sh points CUDA_HOME at the same one
case $M in openfold3) CV=12.8;; opendde) CV=12.6;; *) CV=;; esac
if [ -n "$CV" ] && [ ! -x /usr/local/cuda-$CV/bin/nvcc ]; then
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends cuda-toolkit-${CV/./-}
fi
export CUDA_HOME=/usr/local/cuda${CV:+-$CV} PYTHONHASHSEED=0 CFLAGS=-g0; export PATH=$CUDA_HOME/bin:$PATH
L=environment/requirements.lock
P=$(grep -E '^(pip|wheel)==' $L || true); [ -z "$P" ] || pip install -q $P
case $M in
  openfold3)   X='openfold3|deepspeed';;
  opendde)     X='opendde';;
  boltz2)      X='boltz';;
  protenix_v2) X='protenix';;
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
  protenix_v2) pip install -q --no-deps stock/protenix-2.0.0-py3-none-any.whl; export PROTENIX_ROOT_DIR=/weights/protenix; W=$PROTENIX_ROOT_DIR;;
esac
SP=$(python -c 'import site; print(site.getsitepackages()[0])'); cp /root/kc/kctime.py $SP/ && echo "import kctime" > $SP/kctime.pth
mkdir -p $W
bash run.sh install --weights $W
bash run.sh check --config a100 --mode fast
echo SETUP-OK > /root/kc/SETUP-OK-$M
