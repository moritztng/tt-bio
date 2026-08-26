#!/bin/bash
# Build PXDesign on a rented GPU box with the reference stack's pins EXCEPT torch, CUDA and triton.
# Writes /work/SETUP_MODERN_OK or /work/SETUP_MODERN_FAIL last; poll for the marker.
#
#   image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel   (any devel image with a >=525 driver works;
#          the cu128 wheels bring their own CUDA runtime, so the image's CUDA is not the constraint)
#   usage: scp this to /work/setup_modern.sh, then, with a venv python so the pinned stack on the
#          same box survives:
#            python3 -m venv /work/venv_modern
#            PY=/work/venv_modern/bin/python nohup setsid bash /work/setup_modern.sh \
#              >/dev/null 2>&1 </dev/null &
#
# Why this file exists. The pinned reference stack is torch 2.3.1 on CUDA 12.1, and CUDA 12.1 emits
# no sm_100 device code, so a B200 cannot run the thing the H200 cell measured. torch 2.7.0 is the
# first stable release whose cu128 wheels carry sm_100. What makes raising it legitimate rather than
# a different measurement is that the perf-page cell is the generator stage alone -- featurise,
# generate, write -- and neither DeepSpeed's Evoformer kernel nor JAX executes inside it. That is
# measured, not assumed: see state/pxdesign-perf-page-honest.md section 2, and the per-stage counts
# gpu_pxdesign_run.py records.
#
# The diff from gpu_pxdesign_setup.sh is deliberately three packages and nothing else, so an A/B
# between the two stacks on one box is a one-variable experiment:
#
#   torch    2.3.1+cu121 -> 2.7.1+cu128   the whole point: sm_100
#   triton   3.3.1       -> 3.4.0         torch 2.7.1 pins 3.3.1, whose CUDA launcher hangs on
#                                         sm_100 (memory gpu-b200-cuequivariance-sm100-hang,
#                                         CORRECTION 2026-08-18). PXDesign-d probably never enters
#                                         a triton kernel; this is one line against a known hang.
#   jax      0.4.29+cuda -> absent        jaxlib's CUDA build wants the 12.5 nvidia-* line and torch
#                                         cu128 wants 12.8. AF2-IG is not in the measured stage and
#                                         is not measured on this stack, so there is nothing to
#                                         reconcile. Set WANT_JAX=1 for a CPU-only jax if some
#                                         import in the pipeline demands the module.
#
# Everything else -- protenix 0.5.0+pxd, pxdesign 0.1.0, pxdbench 0.1.2, numpy 1.26.3, transformers
# 4.51.3, the checkpoints, the CCD cache -- is byte for byte what the published cells ran.
#
# Order matters: protenix's setup.py reads its requirements.txt into install_requires and that file
# pins torch==2.3.1, so torch has to be overridden AFTER protenix, not before. pip will print a
# dependency-conflict line about it. That line is expected and gets recorded, not hidden.
set -u
exec >>/work/setup_modern.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_MODERN_FAIL; exit 1; }

# This stack shares a box with the pinned one (that is what makes the A/B one-variable), so every
# path it writes is distinct and every shared artifact -- the checkpoints, the AF2 params, the CCD
# cache, the PXDesign clone -- is reused rather than re-fetched. Run it against a venv python:
#
#   python3 -m venv /work/venv_modern
#   PY=/work/venv_modern/bin/python nohup setsid bash /work/setup_modern.sh >/dev/null 2>&1 </dev/null &
#
# On a box where the pinned setup has not run, PY defaults to the system python and the guards are
# all no-ops, so it still works standalone.
WANT_JAX=${WANT_JAX:-0}
export DEBIAN_FRONTEND=noninteractive
# build-essential is not optional: triton JIT-compiles a driver shim with the system gcc on first
# call, and a -runtime image has no C compiler. Installed here rather than after the first hang.
apt-get update -qq && apt-get install -y -qq build-essential git wget curl aria2 || FAIL "apt"
echo "--- gcc $(gcc --version | head -1)  nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

PY=${PY:-$(command -v python3)}
echo "--- python: $PY ($($PY -V 2>&1))"
$PY -m pip install -q --upgrade pip || FAIL "pip"

export TOOL_WEIGHTS_ROOT=/work/tool_weights
export PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache
export CUTLASS_PATH=/work/cutlass
mkdir -p $TOOL_WEIGHTS_ROOT/af2 $TOOL_WEIGHTS_ROOT/mpnn $PROTENIX_DATA_ROOT_DIR /work/results

# --- the big downloads, in parallel with the pip work -------------------------------------------
# Kept in full even though the eval half is never timed on this stack: check_tool_weights is a prep
# stage and refusing it costs more than the bandwidth does.
[ -s /work/.af2_ok ] || ( cd $TOOL_WEIGHTS_ROOT && aria2c -q -x8 -s8 --file-allocation=none \
    -o af2.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar \
  && tar -xf af2.tar -C af2 && rm -f af2.tar && echo ok > /work/.af2_ok ) >/work/dl_af2.log 2>&1 &
[ -s /work/.ccd_ok ] || ( cd $PROTENIX_DATA_ROOT_DIR
  B=https://pxdesign.tos-cn-beijing.volces.com/release_data
  for f in components.v20240608.cif components.v20240608.cif.rdkit_mol.pkl \
           clusters-by-entity-40.txt; do
    aria2c -q -x16 -s16 -k1M --file-allocation=none -o $f "$B/$f" || exit 1
  done
  echo ok > /work/.ccd_ok ) >/work/dl_ccd.log 2>&1 &
[ -s /work/.mpnn_ok ] || ( git clone -q --depth 1 https://github.com/dauparas/ProteinMPNN.git /work/_mpnn
  for d in ca_model_weights soluble_model_weights vanilla_model_weights; do
    cp -r /work/_mpnn/$d $TOOL_WEIGHTS_ROOT/mpnn/$d
  done
  rm -rf /work/_mpnn && echo ok > /work/.mpnn_ok ) >/work/dl_mpnn.log 2>&1 &
[ -s /work/.ckpt_ok ] || ( CK=/work/PXDesign_ckpt; mkdir -p $CK
  B=https://pxdesign.tos-cn-beijing.volces.com/release_model
  for m in pxdesign_v0.1.0 protenix_base_default_v0.5.0 protenix_mini_default_v0.5.0 \
           protenix_mini_tmpl_v0.5.0; do
    aria2c -q -x16 -s16 -k1M --file-allocation=none -d $CK -o $m.pt "$B/$m.pt" || exit 1
  done
  echo ok > /work/.ckpt_ok ) >/work/dl_ckpt.log 2>&1 &

# --- python stack, pinned half first -------------------------------------------------------------
$PY -m pip install -q --no-cache-dir "git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd" \
  || FAIL "protenix"
$PY -m pip install -q --no-cache-dir einops natsort dm-tree posix_ipc "transformers==4.51.3" \
  immutabledict || FAIL "pxdbench deps"
$PY -m pip install -q --no-cache-dir git+https://github.com/bytedance/PXDesignBench.git@v0.1.2 \
  --no-deps || FAIL "pxdbench"
[ -d /work/PXDesign ] || git clone -q --depth 1 \
  https://github.com/bytedance/PXDesign.git /work/PXDesign || FAIL "clone"
cd /work/PXDesign && $PY -m pip install -q -e . || FAIL "pxdesign"   # setup.py has no deps

# --- the three packages this file exists to change -----------------------------------------------
echo "--- torch override (expect a protenix-wants-2.3.1 conflict line, it is recorded not hidden)"
$PY -m pip install --no-cache-dir "torch==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -20
$PY -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__" \
  || FAIL "torch 2.7.1 did not take"
$PY -m pip install -q --no-cache-dir "triton==3.4.0" || FAIL "triton 3.4.0"
$PY -m pip install -q --no-cache-dir "numpy==1.26.3" || FAIL "numpy pin"

# deepspeed only has to IMPORT: its Evoformer kernel is never called inside the measured stage.
# 0.15.4 first, because it keeps the diff from the published stack to torch+CUDA+triton alone.
DS_WANT=0.15.4
$PY -m pip install -q --no-cache-dir "deepspeed==$DS_WANT" || FAIL "deepspeed $DS_WANT"
if ! $PY -c "from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention" 2>/dev/null; then
  echo "--- deepspeed $DS_WANT will not import on torch 2.7.1; falling back to 0.17.5 (Protenix@main's pin)"
  DS_WANT=0.17.5
  $PY -m pip install -q --no-cache-dir "deepspeed==$DS_WANT" || FAIL "deepspeed 0.17.5"
fi
echo "--- deepspeed resolved to $DS_WANT"

if [ "$WANT_JAX" = "1" ]; then
  # CPU only, deliberately: AF2-IG is not measured on this stack and a CUDA jaxlib fights torch's
  # nvidia-* wheels. If this is on, say so in the cell's ref.
  $PY -m pip install -q --no-cache-dir "jax==0.4.29" "jaxlib==0.4.29" \
    "dm-haiku==0.0.13" "optax==0.2.5" || FAIL "jax cpu"
  $PY -m pip install -q --no-cache-dir git+https://github.com/sokrypton/ColabDesign.git --no-deps \
    || FAIL "colabdesign"
fi

[ -d $CUTLASS_PATH ] || git clone -q -b v3.5.1 --depth 1 \
  https://github.com/NVIDIA/cutlass.git $CUTLASS_PATH || FAIL "cutlass"

# --- assert, do not hope --------------------------------------------------------------------------
$PY - <<'PYEOF' || FAIL "import/assert check"
import json, os, subprocess
from importlib.metadata import PackageNotFoundError, version
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
stack = {p: v(p) for p in ("torch","triton","protenix","pxdbench","pxdesign","jax","jaxlib",
                           "deepspeed","numpy","transformers","biotite","rdkit","colabdesign")}
import torch
stack.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
             gpu_capability=list(torch.cuda.get_device_capability(0)))
cap = tuple(stack["gpu_capability"])
print(json.dumps(stack, indent=1))
# A cu128 wheel on a Blackwell card is the whole point; fail here rather than six frames deep.
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(x @ x).all(), "a bf16 matmul on this card is not finite"
torch.cuda.synchronize()
print("  bf16 matmul on %s sm_%d%d: ok" % (stack["gpu"], cap[0], cap[1]))
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention  # noqa: F401
import protenix.openfold_local.model.primitives as prim
assert hasattr(prim, "DS4Sci_EvoformerAttention"), "protenix cannot see the DS evo kernel"
import pxdbench, pxdesign  # noqa: F401
stack["cpu"] = subprocess.run(["bash","-lc","lscpu | grep -m1 'Model name'"],
                              capture_output=True, text=True).stdout.strip()
try:
    stack["cgroup_cpu_max"] = open("/sys/fs/cgroup/cpu.max").read().strip()
except OSError:
    stack["cgroup_cpu_max"] = None
stack["nproc_visible"] = os.cpu_count()
open("/work/STACK_modern.json","w").write(json.dumps(stack, indent=1) + "\n")
print("  all imports OK; stack written to /work/STACK_modern.json")
PYEOF
$PY -m pip check 2>&1 | tee /work/pip_check_modern.txt | tail -10
$PY -m pip freeze > /work/pip_freeze_modern.txt

wait
for m in af2 ccd mpnn ckpt; do [ -s /work/.${m}_ok ] || FAIL "download $m"; done
mkdir -p /work/PXDesign/release_data/checkpoint
# already in place if the pinned setup ran on this box first; that is the normal case here
ls /work/PXDesign_ckpt/*.pt >/dev/null 2>&1 && mv /work/PXDesign_ckpt/*.pt /work/PXDesign/release_data/checkpoint/
$PY - <<'PYEOF' || FAIL "checkpoint verify"
import glob, torch
for f in sorted(glob.glob("/work/PXDesign/release_data/checkpoint/*.pt")):
    ck = torch.load(f, map_location="cpu")
    print("  %s loads, %d entries" % (f.split("/")[-1], len(ck) if isinstance(ck, dict) else -1))
PYEOF
sha256sum /work/PXDesign/release_data/checkpoint/*.pt

# The target fixture's yaml carries the absolute path /work/targets2/laczc_512.cif, so the CIF has
# to sit at exactly that path or the yaml bytes stop matching the sha the published cells recorded.
mkdir -p /work/targets2
echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_MODERN_OK
