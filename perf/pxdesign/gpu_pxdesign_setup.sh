#!/bin/bash
# Install PXDesign on a rented GPU box, detached. Writes /work/SETUP_OK or /work/SETUP_FAIL last;
# poll for the marker.
#
# PXDesign pins torch 2.3.1+cu121 and jax[cuda] 0.4.29, so the base image is the matching
# pytorch devel image: nvcc is required because DeepSpeed JIT-compiles the Evoformer attention
# kernel against CUTLASS on first call, and the runtime image has no compiler.
#
# Everything the pipeline would otherwise fetch lazily is fetched here: AF2 params, ProteinMPNN
# weights, the CCD cache, and the four checkpoints (pxdesign-d, protenix base/mini/mini_tmpl).
# A lazy CCD build inside a timed run is a 55-minute lie (protenix-lazy-ccd-cache-race).
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git wget curl || FAIL "apt"
echo "--- gcc $(gcc --version | head -1)  nvcc $(nvcc --version | tail -1)"

PY=$(command -v python3)
$PY -m pip install -q --upgrade pip || FAIL "pip"

# --- big downloads first, in parallel with the pip work -----------------------------------------
export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
mkdir -p $TOOL_WEIGHTS_ROOT/af2 $TOOL_WEIGHTS_ROOT/mpnn $PROTENIX_DATA_ROOT_DIR /work/results

( set -x
  cd $TOOL_WEIGHTS_ROOT
  curl -sSL -o af2.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar \
    && tar -xf af2.tar -C af2 && rm -f af2.tar && echo AF2_OK > /work/.af2_ok
) >/work/dl_af2.log 2>&1 &
DL_AF2=$!

( set -x
  cd $PROTENIX_DATA_ROOT_DIR
  B=https://pxdesign.tos-cn-beijing.volces.com/release_data
  curl -sSL -C - -o components.v20240608.cif              $B/components.v20240608.cif
  curl -sSL -C - -o components.v20240608.cif.rdkit_mol.pkl $B/components.v20240608.cif.rdkit_mol.pkl
  curl -sSL -C - -o clusters-by-entity-40.txt              $B/clusters-by-entity-40.txt
  echo CCD_OK > /work/.ccd_ok
) >/work/dl_ccd.log 2>&1 &
DL_CCD=$!

( set -x
  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /work/_mpnn
  for d in ca_model_weights soluble_model_weights vanilla_model_weights; do
    cp -r /work/_mpnn/$d $TOOL_WEIGHTS_ROOT/mpnn/$d
  done
  rm -rf /work/_mpnn && echo MPNN_OK > /work/.mpnn_ok
) >/work/dl_mpnn.log 2>&1 &
DL_MPNN=$!

# --- python stack, exactly the versions install.sh pins ------------------------------------------
$PY -m pip install -q --no-cache-dir "git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd" \
  || FAIL "protenix"
$PY -m pip install -q --no-cache-dir einops natsort dm-tree posix_ipc "transformers==4.51.3" \
  "dm-haiku==0.0.13" "optax==0.2.5" || FAIL "pxdbench deps"
$PY -m pip install -q --no-cache-dir git+https://github.com/sokrypton/ColabDesign.git --no-deps \
  || FAIL "colabdesign"
$PY -m pip install -q --no-cache-dir "jax[cuda]==0.4.29" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || FAIL "jax"
$PY -m pip install -q --no-cache-dir "numpy==1.26.3" || FAIL "numpy pin"
$PY -m pip install -q --no-cache-dir git+https://github.com/bytedance/PXDesignBench.git@v0.1.2 \
  --no-deps || FAIL "pxdbench"

git clone -q --depth 1 https://github.com/bytedance/PXDesign.git /work/PXDesign || FAIL "clone pxd"
cd /work/PXDesign && $PY -m pip install -q -e . || FAIL "pxdesign"

export CUTLASS_PATH=/work/cutlass
[ -d $CUTLASS_PATH ] || git clone -q -b v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git \
  $CUTLASS_PATH || FAIL "cutlass"

# --- sanity: the same import set install.sh checks, plus the device -------------------------------
$PY - <<PYEOF
import sys
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
for p in ("torch","protenix","pxdbench","pxdesign","jax","jaxlib","deepspeed","numpy",
          "transformers","dm-haiku","optax","biotite","rdkit"):
    print("  %-16s %s" % (p, v(p)))
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "dev", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
import jax, colabdesign, protenix, pxdbench, pxdesign
print("  jax devices", jax.devices())
PYEOF
[ $? -eq 0 ] || FAIL "import check"

wait $DL_AF2 $DL_CCD $DL_MPNN
for m in af2 ccd mpnn; do [ -s /work/.${m}_ok ] || FAIL "download $m"; done
echo "--- weights"
du -sh $TOOL_WEIGHTS_ROOT/af2 $TOOL_WEIGHTS_ROOT/mpnn $PROTENIX_DATA_ROOT_DIR

# --- checkpoints: fetch now, not inside a timed run ----------------------------------------------
cd /work/PXDesign
$PY - <<PYEOF
import sys
sys.argv = ["x"]
from pxdesign.utils.infer import download_inference_cache, get_configs
c = get_configs(["--dump_dir", "/work/_ckpt_probe", "--input_json_path",
                 "/work/PXDesign/examples/PDL1_quick_start.yaml"])
download_inference_cache(c)
print("  ckpt dir:", c.load_checkpoint_dir)
PYEOF
[ $? -eq 0 ] || FAIL "checkpoint fetch"
ls -la /work/PXDesign/release_data/checkpoint/ 2>/dev/null

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
