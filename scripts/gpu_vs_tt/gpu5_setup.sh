#!/usr/bin/env bash
# Setup for the five-model / two-GPU 512 aa benchmark. Staged, because the protenix
# control has to run before anything else is installed (if the control is off, no other
# install is worth paying for) and because ESMC-6B is a 24 GB download that should be
# pulling while the small venvs build.
#
# Usage on the box:
#   bash gpu5_setup.sh base                # CUDA toolchain fix, once
#   bash gpu5_setup.sh protenix            # venv + weights for the control
#   bash gpu5_setup.sh fetch &             # background: the two big downloads
#   bash gpu5_setup.sh boltz esm of3 opendde
#   bash gpu5_setup.sh ob obfetch     # OpenBind-0: openfold3 0.5.0 + its own checkpoint
#   bash gpu5_setup.sh esmweights          # after esm: ESMFold2 + ESMC-6B, ~25 GB
#   bash gpu5_setup.sh embedweights        # esmfold2-fast + 3x ESM-C + 3x SaProt, ~12 GB
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime. torch 2.7.1+cu128 is protenix
# 2.0.0's exact pin and the first torch line with Blackwell/sm_100, so ONE image serves
# both the H200 and the B200 and torch stops being a variable between them.
set -uo pipefail   # NOT -e: one model failing to install must not kill the rest
cd "$(dirname "$0")"
HERE=$(pwd)
LOG=/root/results/setup.log
mkdir -p /root/results /root/ckpt /root/common
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0

say() { echo "== $* : $(date -u +%FT%TZ) ==" | tee -a "$LOG"; }

stage_base() {
  say "stage base"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$LOG"
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
'cap',torch.cuda.get_device_capability(),torch.cuda.get_device_name(0))" | tee -a "$LOG"
  # Both protenix-family vendors JIT-compile a fused layernorm CUDA extension at import.
  # The runtime image has no nvcc, and torch's cpp_extension needs more than nvcc:
  # cuda_runtime_api.h and, through ATen's CUDAContextLight.h, cusparse.h. cuda-compiler
  # ships neither; cuda-toolkit is the metapackage that carries the headers. Gate on a
  # header, not on nvcc, so a box that already has nvcc still gets the fix.
  if [ ! -f /opt/conda/include/cuda_runtime_api.h ] || [ ! -f /opt/conda/include/cusparse.h ]; then
    apt-get update -qq && apt-get install -y -qq build-essential git curl
    /opt/conda/bin/conda install -y -n base -c nvidia "cuda-toolkit=12.8" 2>&1 | tail -3
  else
    apt-get update -qq && apt-get install -y -qq build-essential git curl
  fi
  # conda's nvidia channel lays CUDA out as targets/x86_64-linux/{include,lib}, not as
  # CUDA_HOME/{include,lib64}, which is the only layout cpp_extension knows.
  if [ -d /opt/conda/targets/x86_64-linux/include ]; then
    for f in /opt/conda/targets/x86_64-linux/include/*; do
      b=$(basename "$f"); [ -e "/opt/conda/include/$b" ] || ln -sfn "$f" "/opt/conda/include/$b"
    done
    [ -e /opt/conda/lib64 ] || ln -sfn /opt/conda/targets/x86_64-linux/lib /opt/conda/lib64
  fi
  say "stage base done"
}

# One venv per model, because the cuequivariance pins genuinely conflict: protenix 2.0.0
# wants 0.8.0, opendde 1.0.3 wants 0.10.0, boltz/of3 take >=0.8 unpinned.
#
# ISOLATION IS NOT OPTIONAL for boltz and of3. --system-site-packages reuses the image's
# torch 2.7.1+cu128 and saves a multi-GB download, which is right for the protenix family
# (protenix 2.0.0 pins torch==2.7.1, so nothing moves). But boltz and of3 pull
# cuequivariance-ops-torch 0.11.1, which requires torch>=2.11: pip then installs torch
# 2.13.0 INTO the venv, where it shadows the image's 2.7.1 while the image's torchvision
# 0.22.1 -- compiled against 2.7.1 -- stays visible on the path and ABI-breaks the moment
# anything imports it ("partially initialized module 'torchvision' has no attribute
# 'extension'", "operator torchvision::nms does not exist"). Three of the five models hit
# this in one session. Installing torchvision on top makes it worse: it drags torch back
# to 2.7.1 and yields libcusparseLt.so.0 errors. An isolated venv is also exactly what a
# researcher gets from a plain pip install, so it costs nothing in fairness.
#   mkvenv <name>            -> isolated
#   mkvenv <name> shared     -> --system-site-packages
mkvenv() {
  local sys=""
  [ "${2:-}" = "shared" ] && sys="--system-site-packages"
  [ -x "/root/venv-$1/bin/pip" ] || python3 -m venv $sys "/root/venv-$1"
  "/root/venv-$1/bin/pip" install --no-cache-dir --upgrade pip -q
}

stage_protenix() {
  say "stage protenix"
  mkvenv protenix shared
  /root/venv-protenix/bin/pip install --no-cache-dir -q protenix==2.0.0 huggingface_hub==0.34.4 \
    2>&1 | tail -5 | tee -a "$LOG"
  if [ ! -s /root/ckpt/protenix-v2.pt ]; then
    # The official checkpoint URL is gated (403); TMF001/protenix-v2-weights is the public
    # mirror of the same protenix-v2.pt the TT side runs (tt_bio/main.py PROTENIX_REPO).
    /root/venv-protenix/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('TMF001/protenix-v2-weights','protenix-v2.pt',local_dir='/root/ckpt'))
" 2>&1 | tail -3 | tee -a "$LOG"
  fi
  /root/venv-protenix/bin/protenix --help >/dev/null && echo "protenix CLI ok" | tee -a "$LOG"
  say "stage protenix done"
}

stage_opendde() {
  say "stage opendde"
  mkvenv opendde shared
  /root/venv-opendde/bin/pip install --no-cache-dir -q "opendde[gpu]==1.0.3" huggingface_hub==0.34.4 \
    2>&1 | tail -5 | tee -a "$LOG"
  if [ ! -s /root/ckpt/opendde.pt ]; then
    /root/venv-opendde/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('aurekaresearch/OpenDDE','opendde.pt',local_dir='/root/ckpt'))
" 2>&1 | tail -3 | tee -a "$LOG"
  fi
  say "stage opendde done"
}

stage_boltz() {
  say "stage boltz"
  mkvenv boltz
  # cuEquivariance kernels are ON by default in boltz 2.2.1 (--no_kernels defaults False),
  # so the [cuda] extra is the fast path and no flag is needed to get it.
  /root/venv-boltz/bin/pip install --no-cache-dir -q "boltz[cuda]==2.2.1" 2>&1 | tail -5 | tee -a "$LOG"
  /root/venv-boltz/bin/boltz predict --help >/dev/null 2>&1 && echo "boltz CLI ok" | tee -a "$LOG"
  say "stage boltz done"
}

stage_of3() {
  say "stage of3"
  # rdkit's Chem.Draw is imported transitively by pdbeccdutils on the OF3 import path and
  # dies on libXrender.so.1, which a runtime image does not ship. Without these four the
  # whole OF3 import fails long before the model is reached.
  apt-get install -y -qq libxrender1 libxext6 libsm6 libxi6 2>&1 | tail -2 | tee -a "$LOG"
  mkvenv of3
  /root/venv-of3/bin/pip install --no-cache-dir -q "openfold3[cuequivariance]==0.4.4" \
    2>&1 | tail -8 | tee -a "$LOG"
  # setup_openfold is INTERACTIVE -- it prompts for a cache directory and aborts under
  # nohup. Nothing in this benchmark needs the cache it would create: the inference
  # checkpoint is passed with --inference-ckpt-path and the MSA is precomputed.
  /root/venv-of3/bin/python -c "
import openfold3
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
print('of3 import OK', hasattr(OpenFold3AllAtom, 'predict_step'))
" 2>&1 | tail -3 | tee -a "$LOG"
  say "stage of3 done"
}

stage_esm() {
  say "stage esm"
  # Three things the obvious recipe gets wrong, all verified on the H200 box:
  #  1. upstream esm requires python >=3.12,<3.13 and the image is 3.11, so pip refuses
  #     outright. uv fetches a standalone 3.12; venv-esm312, not venv-esm.
  #  2. the ESMFold2 MODEL CLASS is not in esm at all -- esm.models.esmfold2 carries only
  #     input builders -- and biohub/ESMFold2 on HF holds config.json + model.safetensors
  #     + ccd.pkl with no .py, so trust_remote_code has nothing to fetch. The class is
  #     transformers.models.esmfold2.modeling_esmfold2.ESMFold2Model. transformers is a
  #     hard requirement here, not a convenience.
  #  3. set_kernel_backend("cuequivariance") raises unless cuequivariance-torch is present;
  #     the reference path is the default, so the fast path has to be installed for.
  [ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -2
  export PATH=/root/.local/bin:$PATH
  uv venv --python 3.12 /root/venv-esm312 2>&1 | tail -2 | tee -a "$LOG"
  uv pip install --python /root/venv-esm312/bin/python \
    "esm@git+https://github.com/Biohub/esm.git@main" transformers huggingface_hub \
    cuequivariance-torch cuequivariance-ops-torch-cu12 2>&1 | tail -4 | tee -a "$LOG"
  /root/venv-esm312/bin/python -c "
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
import transformers, torch
print('esmfold2 class OK; transformers', transformers.__version__, 'torch', torch.__version__)
" 2>&1 | tail -3 | tee -a "$LOG"
  #  4. xformers must not be in this venv, on any card. transformers' ESMC prefers
  #     xops.memory_efficient_attention over torch SDPA whenever xformers imports, which
  #     silently takes the ESM-C rows off the published torch-SDPA path -- the torch_sdpa
  #     counter then reads 0 and the row is void by the protocol. On the B200 it was worse:
  #     the wheel has no sm_100 op and was not built with CUDA support, so all three ESM-C
  #     rows died outright. That was fixed by hand on the box and never committed, so the
  #     fix is here now rather than rediscovered on the next paid rental.
  uv pip uninstall --python /root/venv-esm312/bin/python xformers 2>&1 | tail -2 | tee -a "$LOG" || true
  /root/venv-esm312/bin/python - <<'PYX' 2>&1 | tail -2 | tee -a "$LOG"
import importlib.util, sys
if importlib.util.find_spec("xformers") is not None:
    sys.exit("FATAL: xformers is still importable in venv-esm312; the ESM-C rows would leave "
             "the torch-SDPA path the published cells were measured on")
print("xformers absent: the ESM-C rows will run torch SDPA")
PYX
  # pipefail carries the python exit through tail/tee, so a still-importable xformers stops the
  # stage here instead of surfacing as a zero counter on a paid box.
  [ "${PIPESTATUS[0]}" = 0 ] || exit 3
  say "stage esm done"
}

# The harness control, and only it, runs against the packages the published cell names.
#
# stage_esm installs esm from git @main and lets torch resolve, both unpinned: the published
# ESMFold2 H200 cell (7.256 s) ran esm 3.3.0 @26b0bc2b with transformers 4.57.6 and torch
# 2.13.0+cu130, while the B200 box four days later resolved esm 3.4.0 and torch 2.11.0+cu130.
# So a control measured in the floating venv cannot tell a box or harness problem apart from
# package drift. This venv pins the two packages the published provenance actually names, and
# is built ONLY when control_verdict.py asks for it -- the new esmfold2-fast row keeps the
# floating venv on purpose, because its B200 cell was measured there and the two cells of one
# row have to share a stack.
#
# torch is deliberately NOT pinned here: over-constraining the resolution on a paid box risks an
# unsolvable install, and whatever torch resolves is recorded so the doc can name it as the one
# remaining unseparated variable rather than pretend it is not there.
stage_esmctl() {
  say "stage esmctl"
  export PATH=/root/.local/bin:$PATH
  uv venv --python 3.12 /root/venv-esm312ctl 2>&1 | tail -2 | tee -a "$LOG"
  uv pip install --python /root/venv-esm312ctl/bin/python \
    "esm@git+https://github.com/Biohub/esm.git@26b0bc2b" "transformers==4.57.6" huggingface_hub \
    "cuequivariance-torch==0.11.1" "cuequivariance-ops-torch-cu12==0.11.1" 2>&1 | tail -4 | tee -a "$LOG"
  uv pip uninstall --python /root/venv-esm312ctl/bin/python xformers 2>&1 | tail -2 | tee -a "$LOG" || true
  /root/venv-esm312ctl/bin/python -c "
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
import transformers, torch, esm
print('pinned control venv OK; esm', esm.__version__, 'transformers', transformers.__version__,
      'torch', torch.__version__)
" 2>&1 | tail -3 | tee -a "$LOG"
  say "stage esmctl done"
}

# The two big pulls, kicked off in the background while the small venvs build.
S3=https://openfold3-data.s3.amazonaws.com/openfold3-parameters

# The OpenFold3 checkpoints are 2.2 GB each and S3 serves a single curl stream at about
# 1.6 MB/s to a European box, so one checkpoint was 23 minutes of rental paid for nothing.
# aria2c -x16 pulls the same bytes in about one. Same lesson perf/pxdesign/gpu_pxdesign_setup.sh
# already recorded for its own weights. curl stays as the fallback for a box without aria2.
fetch_ckpt() {
  local f=$1
  [ -s "/root/ckpt/$f" ] && { say "have $f"; return 0; }
  if command -v aria2c >/dev/null 2>&1 || { apt-get install -y -qq aria2 >/dev/null 2>&1 &&
      command -v aria2c >/dev/null 2>&1; }; then
    (cd /root/ckpt && aria2c -x16 -s16 -k1M --file-allocation=none --console-log-level=warn \
      -o "$f" "$S3/$f") 2>&1 | tail -3 | tee -a "$LOG"
  else
    curl -sSL -o "/root/ckpt/$f" "$S3/$f"
  fi
  ls -l "/root/ckpt/$f" | tee -a "$LOG"
  sha256sum "/root/ckpt/$f" | tee -a "$LOG"
}

stage_fetch() {
  say "stage fetch"
  fetch_ckpt of3-p2-155k.pt
  say "stage fetch done"
}

# OpenBind-0 is the OpenFold3 stack on upstream's v0.5.0 checkpoint, and upstream made the two
# checkpoints mutually exclusive: of3-ob-2025-06-30-174k needs openfold3 >=0.5.0, of3-p2-155k
# needs >=0.4,<0.5 (openfold3/entry_points/parameters.py). So OpenBind gets its own venv beside
# venv-of3 rather than replacing it, and both arms can run in one session on one card.
#
# `pip install "openfold3[cuequivariance]==0.5.0"` produces an install that cannot import, and
# both halves of the fix are upstream's bug (perf/openbind/gpu_ob_setup.sh paid for finding
# them): the extra pins cuequivariance-ops-torch-cu12 while openfold3's own torch requirement
# resolves to 2.13.0+cu130, so libcue_ops.so wants libnvrtc.so.12 against a CUDA 13 runtime;
# and even the matching cu13 wheel does not import on its own, because nothing puts
# site-packages/cuequivariance_ops/lib on the loader path. gpu5_session.sh exports that path.
stage_ob() {
  say "stage ob"
  apt-get install -y -qq libxrender1 libxext6 libsm6 libxi6 2>&1 | tail -2 | tee -a "$LOG"
  mkvenv ob
  /root/venv-ob/bin/pip install --no-cache-dir -q "openfold3[cuequivariance]==0.5.0" \
    2>&1 | tail -8 | tee -a "$LOG"
  local sp
  sp=$(/root/venv-ob/bin/python -c "import site;print(site.getsitepackages()[0])")
  /root/venv-ob/bin/pip uninstall -y -q cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 \
    2>&1 | tail -1
  /root/venv-ob/bin/pip install -q --no-cache-dir --force-reinstall --no-deps \
    "cuequivariance-ops-cu13==0.11.1" "cuequivariance-ops-torch-cu13==0.11.1" 2>&1 | tail -2
  echo "$sp/cuequivariance_ops/lib:$sp/nvidia/cu13/lib" > /root/venv-ob/CUEQ_LD_PATH
  LD_LIBRARY_PATH="$(cat /root/venv-ob/CUEQ_LD_PATH)" /root/venv-ob/bin/python -c "
from importlib.metadata import version
from cuequivariance_ops_torch.triangle_attention import triangle_attention
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
from openfold3.core.kernels.cueq_utils import is_cuequivariance_available
print('ob import OK', version('openfold3'), 'predict_step', hasattr(OpenFold3AllAtom, 'predict_step'),
      'cueq available', is_cuequivariance_available())
" 2>&1 | tail -3 | tee -a "$LOG"
  say "stage ob done"
}

stage_obfetch() {
  say "stage obfetch"
  fetch_ckpt of3-ob-2025-06-30-174k.pt
  say "stage obfetch done"
}

stage_esmweights() {
  say "stage esmweights"
  /root/venv-esm312/bin/python -c "
from huggingface_hub import snapshot_download
for r in ('biohub/ESMFold2','biohub/ESMC-6B'):
    print(r, snapshot_download(r))
" 2>&1 | tail -4 | tee -a "$LOG"
  say "stage esmweights done"
}

# The seven rows the perf page gained after the A100 pass: esmfold2-fast plus the three ESM-C
# and three SaProt embedding rows. All seven live in venv-esm312 -- esmfold2-fast is the same
# transformers class as esmfold2, ESM-C is the same package, and SaProt is stock transformers
# EsmForMaskedLM -- so this stage installs nothing, it only pulls weights. ESMC-6B is not
# listed: stage_esmweights already pulled it for ESMFold2 and esmfold2-fast shares that exact
# backbone (both configs read esmc_id biohub/ESMC-6B), so the 25 GB is paid once.
stage_embedweights() {
  say "stage embedweights"
  /root/venv-esm312/bin/python -c "
from huggingface_hub import snapshot_download
for r in ('biohub/ESMFold2-Fast','biohub/ESMC-300M','biohub/ESMC-600M',
          'westlake-repl/SaProt_35M_AF2','westlake-repl/SaProt_650M_AF2',
          'westlake-repl/SaProt_1.3B_AF2'):
    print(r, snapshot_download(r))
" 2>&1 | tail -8 | tee -a "$LOG"
  # biohub/ESMFold2-Fast ships config.json + model.safetensors and NOTHING else; the ESMFold2
  # repo also carries ccd.pkl. If the model reaches for it the Fast run dies at the first fold,
  # on a paid box, so the file is staged next to the Fast weights up front. A no-op when the
  # code path never asks for it.
  /root/venv-esm312/bin/python -c "
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
src = Path(snapshot_download('biohub/ESMFold2')) / 'ccd.pkl'
dst = Path(snapshot_download('biohub/ESMFold2-Fast')) / 'ccd.pkl'
if src.exists() and not dst.exists():
    shutil.copyfile(src.resolve(), dst)
    print('staged ccd.pkl ->', dst)
else:
    print('ccd.pkl:', 'already present' if dst.exists() else 'not in ESMFold2 either')
" 2>&1 | tail -2 | tee -a "$LOG"
  # Preflight is torch-free and runs anywhere. Doing it here means a wrong repo name or a
  # checkpoint whose shape does not match its row fails before any measurement is attempted.
  python3 "$HERE/embed_preflight.py" 2>&1 | tail -12 | tee -a "$LOG"
  say "stage embedweights done"
}

for s in "$@"; do
  case "$s" in
    base) stage_base ;;
    protenix) stage_protenix ;;
    opendde) stage_opendde ;;
    boltz) stage_boltz ;;
    of3) stage_of3 ;;
    ob) stage_ob ;;
    obfetch) stage_obfetch ;;
    esm) stage_esm ;;
    esmctl) stage_esmctl ;;
    esmweights) stage_esmweights ;;
    embedweights) stage_embedweights ;;
    fetch) stage_fetch ;;
    *) echo "unknown stage: $s" >&2 ;;
  esac
done
say "gpu5_setup finished stages: $*"
