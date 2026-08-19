#!/bin/bash
# Build PXDesign on a rented GPU box, detached. Writes /work/SETUP_OK or /work/SETUP_FAIL last;
# poll for the marker. Supersedes the five phase scripts this was reduced from.
#
#   image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel   (devel: DeepSpeed JIT-compiles the
#          Evoformer attention kernel against CUTLASS, so nvcc must exist)
#   usage: scp this to /work/setup.sh; nohup setsid bash /work/setup.sh >/dev/null 2>&1 </dev/null &
#
# Upstream's install.sh does not survive on a clean box. Five things it gets wrong, each a SILENT
# failure that leaves the pipeline importable and wrong rather than crashing:
#
#  1. `jax[cuda]==0.4.29 -f jax_cuda_releases` resolves to jaxlib 0.4.29+cuda12.cudnn91, a
#     local-CUDA build wanting system cuDNN 9.1 and cuSPARSE >= 12.1. The torch-2.3.1/cu121 image
#     has cuDNN 8 and cuSPARSE 12.0.2, so jax.devices() returns [CpuDevice(id=0)] with only a log
#     line and AF2-IG silently runs on the CPU. Use jax[cuda12], and assert the platform.
#  2. protenix requires deepspeed>=0.15.1 with no ceiling, so pip takes 0.19.5, whose
#     compile/custom_ops calls torch.library.custom_op (torch 2.4+). protenix's primitives then
#     fail to import at all. 0.15.4 imports against torch 2.3.1.
#  3. jax 0.4.29's nvidia-* requirements have no ceiling either, so pip installs the CUDA 12.9
#     line. An isolated jnp.linalg.svd works, but AF2-IG dies inside colabdesign's Kabsch SVD with
#     gpusolverDnCreate "cuSolver internal error". Pin the CUDA 12.5 wheels jaxlib 0.4.29 shipped
#     against.
#  4. download_inference_cache prefetches only the checkpoint for configs.model_name; the three
#     Protenix eval checkpoints download lazily inside the eval stage.
#  5. the volces cn-beijing origin served those checkpoints at ~20 KB/s from the US east coast --
#     14 hours for the 1.47 GB base model. aria2c -x16 finishes all three in about two minutes.
#
# One more, not fixable here and deliberately left alone so the measurement matches what a user
# gets: `--use_fast_ln True` never reaches FusedLayerNorm, because protenix snapshots
# LAYERNORM_TYPE at import time and pxdesign sets it after the import. Export
# LAYERNORM_TYPE=fast_layernorm before python starts if you want the fused kernel.
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git wget curl aria2 || FAIL "apt"
echo "--- gcc $(gcc --version | head -1)  nvcc $(nvcc --version | tail -1)"

PY=$(command -v python3)
$PY -m pip install -q --upgrade pip || FAIL "pip"

export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
mkdir -p $TOOL_WEIGHTS_ROOT/af2 $TOOL_WEIGHTS_ROOT/mpnn $PROTENIX_DATA_ROOT_DIR /work/results

# --- the big downloads, in parallel with the pip work -------------------------------------------
( cd $TOOL_WEIGHTS_ROOT && aria2c -q -x8 -s8 --file-allocation=none \
    -o af2.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar \
  && tar -xf af2.tar -C af2 && rm -f af2.tar && echo ok > /work/.af2_ok ) >/work/dl_af2.log 2>&1 &
( cd $PROTENIX_DATA_ROOT_DIR
  B=https://pxdesign.tos-cn-beijing.volces.com/release_data
  for f in components.v20240608.cif components.v20240608.cif.rdkit_mol.pkl \
           clusters-by-entity-40.txt; do
    aria2c -q -x16 -s16 -k1M --file-allocation=none -o $f "$B/$f" || exit 1
  done
  echo ok > /work/.ccd_ok ) >/work/dl_ccd.log 2>&1 &
( git clone -q --depth 1 https://github.com/dauparas/ProteinMPNN.git /work/_mpnn
  for d in ca_model_weights soluble_model_weights vanilla_model_weights; do
    cp -r /work/_mpnn/$d $TOOL_WEIGHTS_ROOT/mpnn/$d
  done
  rm -rf /work/_mpnn && echo ok > /work/.mpnn_ok ) >/work/dl_mpnn.log 2>&1 &
( CK=/work/PXDesign_ckpt; mkdir -p $CK
  B=https://pxdesign.tos-cn-beijing.volces.com/release_model
  for m in pxdesign_v0.1.0 protenix_base_default_v0.5.0 protenix_mini_default_v0.5.0 \
           protenix_mini_tmpl_v0.5.0; do
    aria2c -q -x16 -s16 -k1M --file-allocation=none -d $CK -o $m.pt "$B/$m.pt" || exit 1
  done
  echo ok > /work/.ckpt_ok ) >/work/dl_ckpt.log 2>&1 &

# --- python stack --------------------------------------------------------------------------------
$PY -m pip install -q --no-cache-dir "git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd" \
  || FAIL "protenix"
$PY -m pip install -q --no-cache-dir einops natsort dm-tree posix_ipc "transformers==4.51.3" \
  "dm-haiku==0.0.13" "optax==0.2.5" immutabledict || FAIL "pxdbench deps"
$PY -m pip install -q --no-cache-dir git+https://github.com/sokrypton/ColabDesign.git --no-deps \
  || FAIL "colabdesign"
$PY -m pip install -q --no-cache-dir "jax[cuda12]==0.4.29" || FAIL "jax"          # (1)
$PY -m pip install -q --no-cache-dir git+https://github.com/bytedance/PXDesignBench.git@v0.1.2 \
  --no-deps || FAIL "pxdbench"
git clone -q --depth 1 https://github.com/bytedance/PXDesign.git /work/PXDesign || FAIL "clone"
cd /work/PXDesign && $PY -m pip install -q -e . || FAIL "pxdesign"
$PY -m pip install -q --no-cache-dir "deepspeed==0.15.4" || FAIL "deepspeed pin"  # (2)
$PY -m pip install -q --no-cache-dir \
  "nvidia-cublas-cu12==12.5.3.2" "nvidia-cusolver-cu12==11.6.3.83" \
  "nvidia-cusparse-cu12==12.5.1.3" "nvidia-cufft-cu12==11.2.3.61" \
  "nvidia-cuda-cupti-cu12==12.5.82" "nvidia-cuda-nvcc-cu12==12.5.82" \
  "nvidia-cuda-runtime-cu12==12.5.82" "nvidia-cudnn-cu12==9.2.1.18" \
  "nvidia-nvjitlink-cu12==12.5.82" || FAIL "cuda 12.5 pins"                        # (3)
$PY -m pip install -q --no-cache-dir "numpy==1.26.3" || FAIL "numpy pin"
[ -d $CUTLASS_PATH ] || git clone -q -b v3.5.1 --depth 1 \
  https://github.com/NVIDIA/cutlass.git $CUTLASS_PATH || FAIL "cutlass"

# --- assert, do not hope --------------------------------------------------------------------------
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY - <<'PYEOF'
from importlib.metadata import PackageNotFoundError, version
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
for p in ("torch","protenix","pxdbench","pxdesign","jax","jaxlib","deepspeed","numpy"):
    print("  %-12s %s" % (p, v(p)))
import torch
print("  gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import jax, jax.numpy as jnp, numpy as np
d = jax.devices(); print("  jax devices", d)
assert d[0].platform == "gpu", "JAX ON CPU -- AF2-IG would be measured on the CPU"
print("  jax svd", jnp.linalg.svd(jnp.array(np.eye(128, dtype=np.float32)))[1][:2])
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention  # noqa: F401
import protenix.openfold_local.model.primitives as prim
assert hasattr(prim, "DS4Sci_EvoformerAttention"), "protenix cannot see the DS evo kernel"
import colabdesign, pxdbench, pxdesign  # noqa: F401
print("  all imports OK")
PYEOF
[ $? -eq 0 ] || FAIL "import/assert check"

wait
for m in af2 ccd mpnn ckpt; do [ -s /work/.${m}_ok ] || FAIL "download $m"; done
mkdir -p /work/PXDesign/release_data/checkpoint
mv /work/PXDesign_ckpt/*.pt /work/PXDesign/release_data/checkpoint/
$PY - <<'PYEOF'
import glob, torch
for f in sorted(glob.glob("/work/PXDesign/release_data/checkpoint/*.pt")):
    ck = torch.load(f, map_location="cpu")
    print("  %s loads, %d entries" % (f.split("/")[-1], len(ck) if isinstance(ck, dict) else -1))
PYEOF
[ $? -eq 0 ] || FAIL "checkpoint verify"
du -sh $TOOL_WEIGHTS_ROOT/af2 $TOOL_WEIGHTS_ROOT/mpnn $PROTENIX_DATA_ROOT_DIR
sha256sum /work/PXDesign/release_data/checkpoint/*.pt

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
