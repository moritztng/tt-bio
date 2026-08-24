#!/bin/bash
# Install Nesso-1 on a rented GPU box, detached. Writes /work/SETUP_OK or /work/SETUP_FAIL last;
# poll for the marker.
#
# Nesso-1 has no PyPI release yet, so this installs from the pinned git commit. `nesso[kernels]`
# pulls cuequivariance-torch + cuequivariance-ops-torch-cu12, which the triangle attention and
# triangle multiplication route to *only* when `use_kernels` is true at call time -- a per-call
# decision read off the checkpoint's hparams and overridable with `--no_kernels`. Whether the
# kernels are actually REACHED is a question for the counters in gpu_nesso1_run.py, never for this
# script.
#
# build-essential is not optional: cuEquivariance's fused kernels are triton, and triton JIT-compiles
# a CUDA driver shim with the system gcc on first call. Without a C compiler that surfaces six frames
# inside the model as "Failed to find C compiler" (gpu-b200-cuequivariance-sm100-hang).
#
# Three assets are prefetched here so no timed run pays a lazy download:
#   ccd.pkl + model.safetensors + hparams.json   HF recursionpharma/nesso
#   facebook/esm2_t33_650M_UR50D                 the per-sequence embedding Nesso conditions on
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

NESSO_REV=${NESSO_REV:-f0156e9}
# The published h200 and a100 cells were both measured on torch 2.11.0+cu128 and say so, and
# comparing a new column against them means installing that, not whatever the cu128 index
# resolves to today. Overridable because a new GPU can need a newer wheel; when it is
# overridden the cell has to say which torch it ran.
TORCH_SPEC=${TORCH_SPEC:-torch==2.11.0}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git wget curl python3-pip || FAIL "apt"
echo "--- gcc: $(gcc --version | head -1)"
echo "--- cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)  nproc=$(nproc)"

PIP=$(command -v pip3 || command -v pip)
$PIP install -q --break-system-packages uv 2>/dev/null || $PIP install -q uv || FAIL "uv"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")

$UV venv --python 3.12 /work/v_nesso || FAIL "venv"
PY=/work/v_nesso/bin/python

# torch first, from the cu12 index: cuequivariance-ops-torch-cu12 is a cu12 build, and the torch
# wheel is also what puts libnvrtc.so.12 on the path that libcue_ops.so links against.
$UV pip install -q --python $PY "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cu128 || FAIL "torch"
$UV pip install -q --python $PY "nesso[kernels] @ git+https://github.com/recursionpharma/nesso.git@${NESSO_REV}" || FAIL "nesso[kernels]"

# Fail-fast cueq probe: load_library() swallows its own failure and then names the WRONG missing
# .so, so import it explicitly right after torch rather than trusting a later error message.
echo "--- versions"
$PY - <<'PY'
from importlib.metadata import version, PackageNotFoundError
import torch
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
for p in ("nesso","torch","triton","cuequivariance-torch","cuequivariance-ops-torch-cu12",
          "lightning","transformers","rdkit","numpy","safetensors","huggingface_hub"):
    print("  %-34s %s" % (p, v(p)))
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
import importlib.util
print("  cuequivariance_torch spec:", importlib.util.find_spec("cuequivariance_torch") is not None)
import cuequivariance_torch as cuet
print("  cuet", cuet.__file__)
print("  has triangle_attention", hasattr(cuet, "triangle_attention"),
      "has triangle_multiplicative_update", hasattr(cuet, "triangle_multiplicative_update"))
PY
[ $? -eq 0 ] || FAIL "import check"

# --- assets ------------------------------------------------------------------------------------
export NESSO_CACHE=/work/cache
mkdir -p "$NESSO_CACHE"
$PY - <<'PY' || FAIL "asset prefetch"
import os, json, pathlib
os.environ.setdefault("HF_HOME", "/work/cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/work/cache/huggingface")
from huggingface_hub import hf_hub_download
from importlib.metadata import version
rev = "v%s" % version("nesso")
try:
    ccd = hf_hub_download("recursionpharma/nesso", "ccd.pkl", revision=rev,
                          cache_dir="/work/cache/huggingface")
except Exception as e:
    print("  revision %s failed (%s), falling back to main" % (rev, type(e).__name__))
    rev = "main"
    ccd = hf_hub_download("recursionpharma/nesso", "ccd.pkl", revision=rev,
                          cache_dir="/work/cache/huggingface")
w = hf_hub_download("recursionpharma/nesso", "%s/model.safetensors" % rev, revision=rev,
                    cache_dir="/work/cache/huggingface")
h = hf_hub_download("recursionpharma/nesso", "%s/hparams.json" % rev, revision=rev,
                    cache_dir="/work/cache/huggingface")
print("  revision", rev)
print("  ccd   ", ccd, os.path.getsize(ccd))
print("  ckpt  ", w, os.path.getsize(w))
print("  hparams", h)
print("  hparams contents:", json.dumps(json.load(open(h)))[:2000])
pathlib.Path("/work/ASSETS.json").write_text(json.dumps(
    {"revision": rev, "ccd": ccd, "ckpt_dir": str(pathlib.Path(w).parent),
     "ckpt_bytes": os.path.getsize(w)}, indent=2))

from transformers import AutoModelForMaskedLM, AutoTokenizer
m = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                         cache_dir="/work/cache/huggingface")
t = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                  cache_dir="/work/cache/huggingface")
print("  esm2 params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
PY

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
