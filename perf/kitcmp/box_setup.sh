#!/bin/bash
# Runs ON the rented box (image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04): one kit's stack in its own venv, the way
# its environment/Dockerfile does it (requirements.lock with --no-deps, then the pinned stock wheel, then run.sh install).
#   bash box_setup.sh MODEL     MODEL = protenix_v2 | boltz2 | openfold3 | opendde; marker /root/kc/SETUP-OK-<MODEL>
# Also drops kctime.py + kctime.pth into the venv (the synced per-fold timer, inert unless KC_WRAP is set).
# Weights are fetched by the kit's own `install --weights` into /weights/<model>; session.sh pre-places the files whose
# upstream URL fails from vast (Protenix-v2 checkpoint: 403; components.cif.rdkit_mol.pkl: a newer file than the pin).
set -eux
M=$1; K=/root/kit/$M; [ -f /root/kc/SETUP-OK-$M ] && exit 0
# python-build-standalone's sysconfig names clang as CC, which the CUDA image lacks: every native build (DeepSpeed's
# evoformer op, apex, the LayerNorm extension) goes through the image's gcc instead
export CC=gcc CXX=g++
py(){ # <dir> <url> <sha256>: python-build-standalone, the interpreter each kit's Dockerfile pins
  [ -x $1/bin/python3 ] && return; curl -fsSL -o /tmp/py.tgz "$2"; echo "$3  /tmp/py.tgz" | sha256sum -c -
  mkdir -p $1 && tar -xzf /tmp/py.tgz -C /tmp && cp -rL /tmp/python/. $1/ && rm -rf /tmp/python /tmp/py.tgz; }
G=https://github.com/indygreg/python-build-standalone/releases/download
case $M in
  esmfold2) PY=/opt/py3.12.10; py $PY "$G/20250409/cpython-3.12.10+20250409-x86_64-unknown-linux-gnu-install_only.tar.gz" \
              e9cf6f7da499a4400ba30ae1da8f7ef25ce97827bd8c1084717aa05438035186
            command -v git >/dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q git; };;
  rosettafold3|rfdiffusion3) PY=/opt/py3.12.1; py $PY "$G/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
              a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338;;
  *) PY=/usr/local; py $PY "$G/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
              fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1;;
esac
cd $K; [ -d venv ] || $PY/bin/python3 -m venv venv; . venv/bin/activate
# each kit's Dockerfile pins its own nvcc (openfold3 12.8, opendde 12.6, protenix 13.0 = the image's, boltz2 none): install it
# beside the image's toolkit; box_arms.sh points CUDA_HOME at the same one
case $M in openfold3|openfold3_ob0) CV=12.8;; opendde) CV=12.6;; *) CV=;; esac
if [ -n "$CV" ] && [ ! -x /usr/local/cuda-$CV/bin/nvcc ]; then
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends cuda-toolkit-${CV/./-}
fi
if [ $M = pxdesign ] && [ ! -x /usr/local/cuda-12.1/bin/nvcc ]; then  # CUDA 12.1 has no ubuntu24.04 package: the runfile, toolkit
  # only; nvcc 12.1 takes gcc <= 12 (box_arms.sh passes -ccbin gcc-12); git for the lock's editable checkouts
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends gcc-12 g++-12 git
  curl -fsSL -o /tmp/cu121.run https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
  sh /tmp/cu121.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.1 --override --no-man-page && rm -f /tmp/cu121.run
fi
[ $M = pxdesign ] && CV=12.1 && export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-12" CC=gcc-12 CXX=g++-12
export CUDA_HOME=/usr/local/cuda${CV:+-$CV} PYTHONHASHSEED=0 CFLAGS=-g0; export PATH=$CUDA_HOME/bin:$PATH
L=environment/requirements.lock
P=$(grep -E '^(pip|wheel)==' $L || true); [ -z "$P" ] || pip install -q $P
case $M in
  openfold3|openfold3_ob0) X='openfold3|deepspeed';;
  opendde)     X='opendde';;
  boltz2)      X='boltz';;
  protenix_v2) X='protenix';;
  esmfold2)    X='flash_attn|transformer_engine|xformers';;
  rosettafold3) X='-';;
  rfdiffusion3) X='apex|rc-foundry';;
  pxdesign)    X='-';;
esac
grep -v -E "^(#|pip==|wheel==|($X)==)" $L > /tmp/stack-$M.txt
[ $M = rosettafold3 ] && printf 'hatchling==1.31.0\nhatch-vcs==0.4.0\n' > /tmp/bp.txt && export PIP_CONSTRAINT=/tmp/bp.txt
pip install -q --no-deps $([ $M = pxdesign ] && echo --src /opt/pxsrc) -r /tmp/stack-$M.txt; unset PIP_CONSTRAINT
case $M in
  openfold3|openfold3_ob0)  # DS4Sci evoformer kernel built for the A100 (sm 8.0), as the Dockerfile does for sm 9.0
    pip install -q --no-deps stock/openfold3-*-py3-none-any.whl
    DS_ACCELERATOR=cuda DS_BUILD_EVOFORMER_ATTN=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=$(nproc) \
      pip install -q --no-deps --no-build-isolation --no-binary deepspeed $(grep -E '^deepspeed==' $L)
    export OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt OPENFOLD3_OB0_CKPT=/weights/openfold3_ob0/of3-ob-2025-06-30-174k.pt; W=/weights/$M;;
  opendde)    pip install -q --no-deps stock/opendde-1.1.1-py3-none-any.whl; export OPENDDE_ROOT_DIR=/weights/opendde; W=$OPENDDE_ROOT_DIR;;
  boltz2)     pip install -q --no-deps stock/boltz-2.2.1-py3-none-any.whl; export BOLTZ_CACHE=/weights/boltz2; W=$BOLTZ_CACHE;;
  esmfold2)   # xformers, transformer_engine, flash-attn have no index wheel for torch 2.13+cu130: the kit's own build, sm 8.0 alone
    export NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 HF_HOME=/weights/esmfold2
    TORCH_CUDA_ARCH_LIST=8.0 bash environment/build_wheels.sh --stack img_esmfold2_a100; W=$HF_HOME;;
  rosettafold3)  # the stock interpreter is this venv; install --make-venv builds the kit's opt/venv from it
    pip install -q --no-deps --force-reinstall nvidia-cudnn-cu13==9.20.0.48 nvidia-nccl-cu13==2.29.7
    export ROSETTAFOLD3_OPT_STOCK_PYTHON=$K/venv/bin/python; W=/weights/rosettafold3;;
  rfdiffusion3)  # two interpreters from one lock (pristine stock + the kit's patched one), foundry at the pin and apex
    # built here from its pinned commit for sm 8.0 with the image's nvcc 13.0, installed into both (environment/Dockerfile)
    SV=/opt/rfd3-stock; [ -d $SV ] || $PY/bin/python3 -m venv $SV
    $SV/bin/pip install -q --no-deps $(grep -E '^pip==' $L); $SV/bin/pip install -q --no-deps -r /tmp/stack-$M.txt
    printf 'hatchling==1.32.0\nhatch-vcs==0.4.0\nsetuptools-scm==10.2.1\nvcs-versioning==2.3.1\n' > /tmp/bp.txt
    for v in $K/venv $SV; do SETUPTOOLS_SCM_PRETEND_VERSION=0.2.1.dev13+g4010e3e2e PIP_CONSTRAINT=/tmp/bp.txt \
      $v/bin/pip install -q --no-deps stock/foundry-4010e3e.tar.gz; done
    AC=a1d527a857e8da64c4e7237ca89ec699fb4d9eaf
    if ! ls /tmp/apexw/apex-0.1-*.whl >/dev/null 2>&1; then
      curl -fsSL -o /tmp/apex.tar.gz https://github.com/NVIDIA/apex/archive/$AC.tar.gz
      echo "f3cc6e36d357f5c47e39fe6f7ab7282ec7fc19af0b4a8425758ec2f2c6cc038e  /tmp/apex.tar.gz" | sha256sum -c -
      mkdir -p /tmp/src && tar -xzf /tmp/apex.tar.gz -C /tmp/src && $SV/bin/pip install -q ninja==1.13.0
      ( cd /tmp/src/apex-$AC && PATH=$SV/bin:$PATH APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 MAX_JOBS=4 \
          NVCC_APPEND_FLAGS="--threads 4" TORCH_CUDA_ARCH_LIST=8.0 $SV/bin/python -m pip wheel --no-deps --no-build-isolation -w /tmp/apexw . )
    fi
    for v in $K/venv $SV; do $v/bin/pip install -q --no-deps /tmp/apexw/apex-0.1-*.whl
      $v/bin/python -c "import rfd3, apex, importlib; importlib.import_module('apex.normalization.fused_layer_norm'); import fused_layer_norm_cuda"; done
    PATH=$SV/bin:$PATH bash run.sh install
    S3=$($SV/bin/python -c 'import site; print(site.getsitepackages()[0])'); cp /root/kc/kctime.py $S3/
    export RFDIFFUSION3_STOCK_PYTHON=$SV/bin/python RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt; W=/weights/rfd3;;
  pxdesign)   # stock's LayerNorm extension compiles here once, as the Dockerfile does
    python -c "import os, protenix.model.layer_norm.layer_norm as m; p = os.path.join(os.path.dirname(m.__file__), 'fastfold_layer_norm_cuda.so'); assert os.path.isfile(p), p"
    export PXDESIGN_CKPT_DIR=/weights/pxdesign/checkpoint PROTENIX_DATA_ROOT_DIR=/weights/pxdesign/ccd_cache; W=/weights/pxdesign;;
  protenix_v2) pip install -q --no-deps stock/protenix-2.0.0-py3-none-any.whl; export PROTENIX_ROOT_DIR=/weights/protenix; W=$PROTENIX_ROOT_DIR;;
esac
SP=$(python -c 'import site; print(site.getsitepackages()[0])'); cp /root/kc/kctime.py $SP/ && echo "import kctime" > $SP/kctime.pth
mkdir -p $W
bash run.sh install $([ $M = rosettafold3 ] && echo --make-venv) --weights $W
[ $M = rfdiffusion3 ] && python -m rfdiffusion3_opt.install apply && python -m rfdiffusion3_opt.install check-only
[ -d opt/venv ] && { S2=$(opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])'); cp /root/kc/kctime.py $S2/ && echo 'import kctime' > $S2/kctime.pth; }
bash run.sh check --config a100 --mode fast $([ $M = esmfold2 ] && echo --variant fast)
echo SETUP-OK > /root/kc/SETUP-OK-$M
