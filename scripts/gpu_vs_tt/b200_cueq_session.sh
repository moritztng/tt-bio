#!/usr/bin/env bash
# Protenix-v2 + OpenDDE at 512 aa on a B200. The sm_100 blocker is triton, not cuEquivariance.
#
# Both models hang on sm_100 in their shipped configuration. The blocker is ONE package and
# it is not cuEquivariance: cuEquivariance's fused_sigmoid_gated_dual_gemm is a triton kernel
# behind an autotuner, and triton 3.3.1 -- the version torch 2.7.1 pins, and which protenix
# 2.0.0 pins explicitly -- hangs in its own CUDA launcher on sm_100 when launching it.
# triton 3.4.0 runs it. Measured on a B200 2026-08-18: with both models held at every shipped
# pin (torch 2.7.1+cu128, cueq-ops 0.8.0 cu12 for protenix / 0.10.0 cu12 for opendde) and
# triton alone raised to 3.4.0, all three triangle primitives complete.
#
# How that was localised, because the obvious readings are all wrong:
#   * triangle_attention (a precompiled cubin in libcue_ops.so) never hung -- 25 ms at 512 aa
#     on the shipped cu12 build. Only triangle_multiplicative_update hung.
#   * WITHOUT CUDA_LAUNCH_BLOCKING the faulthandler frame is torch.functional.einsum, which is
#     a red herring: the triton launch had already wedged asynchronously and the CPU raced on
#     to the next synchronising call. Set CUDA_LAUNCH_BLOCKING=1 before believing a frame.
#   * The cu13 route (below) is NOT the fix and is not needed. protenix's own source says the
#     Blackwell-optimized kernels ship only in cu13 builds, which is true but irrelevant: the
#     kernel that hangs is triton-JIT from Python source, so the cu12/cu13 split -- which only
#     decides which cubins libcue_ops.so carries -- cannot reach it. cu13 also drags in
#     libcublas.so.13 / libcublasLt.so.13 / libnvrtc.so.13, absent from a torch 2.7.1+cu128
#     install, for nothing.
#   * cueq 0.11.1 is not needed either. It requires torch>=2.11 against both models'
#     torch==2.7.1, so reaching it overrides two pins per model; triton overrides one.
#
# Everything else is the shipped default: dtype, kernel selectors, fusion/cache, TF32
# untouched, 10 cycles / 200 steps / 1 sample / seed 0, the pinned cdk2x2_512 fixture with its
# 35-row alignment.
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel. protenix 2.0.0 pins torch==2.7.1, so
# the image IS the pin, and the devel variant already carries nvcc plus the CUDA headers
# both vendors need to JIT their fused-layernorm extension (the runtime image needs a
# multi-GB conda cuda-toolkit install for that).
#
#   bash b200_cueq_session.sh setup        # venvs + weights + the shipped cueq
#   bash b200_cueq_session.sh triton       # THE FIX: triton 3.3.1 -> 3.4.0, all else shipped
#   bash b200_cueq_session.sh cu13         # rejected route A: same cueq version, CUDA-13 build
#   bash b200_cueq_session.sh torchup      # route B: torch 2.13.0 + cueq 0.11.1 cu13
#   bash b200_cueq_session.sh run          # both cells + gate
set -uo pipefail   # NOT -e: one model failing must leave the other measured
cd "$(dirname "$0")"
HERE=$(pwd)
R=/root/results
mkdir -p "$R" /root/ckpt
LOG=$R/session.log
CUEQ_TARGET=${CUEQ_TARGET:-0.11.1}
REPEAT=${REPEAT:-4}
PER_MODEL_S=${PER_MODEL_S:-1200}
TORCH_TARGET=${TORCH_TARGET:-2.13.0}
TRITON_TARGET=${TRITON_TARGET:-3.4.0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export HF_HUB_ENABLE_HF_TRANSFER=0
# Which venv each cell runs in. protenix/opendde are the shipped-pin venvs (torch 2.7.1);
# prot211/odde211 are the torch>=2.11 ones cueq 0.11.1 actually requires. TAG suffixes the
# output filenames so a second configuration cannot overwrite the first one's evidence.
PROT_VENV=${PROT_VENV:-protenix}
ODDE_VENV=${ODDE_VENV:-opendde}
TAG=${TAG:-}

say() { echo "== $* : $(date -u +%FT%TZ) ==" | tee -a "$LOG"; }

# Which cuEquivariance distributions a venv actually has, plus a fail-fast import probe.
# Written per stage so the shipped-vs-measured comparison is read off recorded evidence,
# not off the install command that was typed.
versions_json() {  # versions_json <venv> <outfile>
  /root/venv-$1/bin/python3 - "$2" <<'EOF'
import importlib.metadata as md, json, sys
names = ["torch", "protenix", "opendde", "cuequivariance", "cuequivariance-torch",
         "cuequivariance-ops-cu12", "cuequivariance-ops-torch-cu12",
         "cuequivariance-ops-cu13", "cuequivariance-ops-torch-cu13",
         "cuequivariance-ops-torch", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-nvrtc-cu13"]
out = {}
for n in names:
    try:
        out[n] = md.version(n)
    except Exception:
        pass
# Fail-fast import probe. cuequivariance_ops.load_library() swallows its own failure and
# then names the WRONG missing .so, so "does it import" has to be asked directly, before
# any model is loaded, or a later ImportError sends you chasing a file that is present.
import torch  # noqa: F401  (must precede cueq: libcue_ops.so needs torch's nvrtc)
out["torch_runtime"] = torch.__version__
try:
    import cuequivariance_ops_torch as c
    out["cueq_ops_torch_import"] = "ok"
    out["cueq_ops_torch_triangle_syms"] = sorted(
        n for n in dir(c) if "triangle" in n.lower())
except Exception as e:
    out["cueq_ops_torch_import"] = f"{type(e).__name__}: {e}"
try:
    import cuequivariance_torch  # noqa: F401
    out["cueq_torch_import"] = "ok"
except Exception as e:
    out["cueq_torch_import"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, indent=2))
open(sys.argv[1], "w").write(json.dumps(out, indent=2) + "\n")
EOF
}

mkvenv() {  # shared site-packages: protenix 2.0.0 pins torch==2.7.1, which the image is
  [ -x "/root/venv-$1/bin/pip" ] || python3 -m venv --system-site-packages "/root/venv-$1"
  "/root/venv-$1/bin/pip" install --no-cache-dir --upgrade pip -q
}

stage_setup() {
  say "stage setup"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv | tee -a "$LOG"
  python3 -c "import torch;print('image torch',torch.__version__,'cuda',torch.version.cuda,
'cap',torch.cuda.get_device_capability(),torch.cuda.get_device_name(0))" | tee -a "$LOG"
  ls "$CUDA_HOME/include/cuda_runtime_api.h" "$CUDA_HOME/include/cusparse.h" | tee -a "$LOG"
  apt-get update -qq && apt-get install -y -qq build-essential git curl 2>&1 | tail -2

  mkvenv protenix
  /root/venv-protenix/bin/pip install --no-cache-dir -q protenix==2.0.0 \
    huggingface_hub==0.34.4 2>&1 | tail -5 | tee -a "$LOG"
  mkvenv opendde
  /root/venv-opendde/bin/pip install --no-cache-dir -q "opendde[gpu]==1.0.3" \
    huggingface_hub==0.34.4 2>&1 | tail -5 | tee -a "$LOG"

  # The official protenix checkpoint URL is gated (403); TMF001/protenix-v2-weights is the
  # public mirror of the same protenix-v2.pt the TT side runs.
  [ -s /root/ckpt/protenix-v2.pt ] || /root/venv-protenix/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('TMF001/protenix-v2-weights','protenix-v2.pt',local_dir='/root/ckpt'))" \
    2>&1 | tail -2 | tee -a "$LOG"
  [ -s /root/ckpt/opendde.pt ] || /root/venv-opendde/bin/python3 -c "
from huggingface_hub import hf_hub_download
print('ckpt:', hf_hub_download('aurekaresearch/OpenDDE','opendde.pt',local_dir='/root/ckpt'))" \
    2>&1 | tail -2 | tee -a "$LOG"
  ls -l /root/ckpt | tee -a "$LOG"

  say "SHIPPED cueq versions (what the models pin)"
  versions_json protenix "$R/cueq_shipped_protenix.json" | tee -a "$LOG"
  versions_json opendde  "$R/cueq_shipped_opendde.json"  | tee -a "$LOG"
  say "stage setup done"
}

# Route A: keep the torch AND the cuEquivariance version each model pins, and change only
# the CUDA build of the cuEquivariance binaries, cu12 -> cu13.
#
# protenix 2.0.0 says why in its own source, at the cuequivariance branch of
# protenix/model/triangular/layers.py: "Blackwell-optimized kernels (for compute
# capabilities 10.0 and 10.3) provide superior performance ... Currently, this feature is
# supported only for cu13 builds." Both models pin cuequivariance-ops-torch-CU12, so the
# configuration they ship cannot reach a Blackwell kernel at all. That is a sharper
# mechanism for the sm_100 hang than "version incompatibility", and it points at a fix that
# moves no version pin: same cuEquivariance release, CUDA-13 build.
#
# cuequivariance-ops-torch-cu13 0.8.0 and 0.10.0 list torch only under their `test` extra
# (torch>=2.11 becomes a hard requirement only at 0.11.x), so this route holds torch at the
# 2.7.1 both models pin. cu12 must be uninstalled first: both wheels install the same
# top-level cuequivariance_ops_torch package.
stage_cu13() {
  say "stage cu13: same cueq version, CUDA-13 build"
  for spec in "protenix:0.8.0" "opendde:0.10.0"; do
    V=${spec%%:*}; CV=${spec#*:}
    P=/root/venv-$V/bin/pip
    say "cu13 $V: cueq $CV, cu12 -> cu13"
    $P uninstall -y -q cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 \
      2>&1 | tail -2 | tee -a "$LOG"
    $P install --no-cache-dir -q "cuequivariance-ops-cu13==$CV" \
      "cuequivariance-ops-torch-cu13==$CV" 2>&1 | tail -4 | tee -a "$LOG"
    versions_json $V "$R/cueq_cu13_$V.json" | tee -a "$LOG"
    # A cu13 build wants libnvrtc.so.13 and a torch 2.7.1+cu128 install bundles 12.
    # cuequivariance_ops.load_library() swallows that failure and then names the WRONG
    # missing library, so react to this probe, not to a later ImportError's filename.
    grep -q '"cueq_ops_torch_import": "ok"' "$R/cueq_cu13_$V.json" || {
      say "cu13 $V: import failed, adding the CUDA-13 NVRTC wheel"
      $P install --no-cache-dir -q nvidia-cuda-nvrtc 2>&1 | tail -2 | tee -a "$LOG"
      versions_json $V "$R/cueq_cu13_${V}_nvrtc.json" | tee -a "$LOG"
    }
  done
  say "stage cu13 done"
}

# THE FIX. One package, applied to both shipped-pin venvs: triton 3.3.1 -> TRITON_TARGET.
# Nothing else moves -- torch stays at the 2.7.1 both models pin, cuEquivariance stays at the
# version and the cu12 build each model pins. pip prints a dependency-conflict warning for
# protenix's own triton==3.3.1 pin; that warning IS the deviation this stage makes, and it is
# the whole deviation, so it belongs in the page's ref string rather than being suppressed.
stage_triton() {
  say "stage triton: raise triton to $TRITON_TARGET, hold every other shipped pin"
  for V in protenix opendde; do
    say "triton $V"
    /root/venv-$V/bin/pip install --no-cache-dir -q "triton==$TRITON_TARGET" 2>&1 | tail -3 | tee -a "$LOG"
    versions_json $V "$R/cueq_triton_$V.json" | tee -a "$LOG"
  done
  # Smoke the two primitives directly before spending a fold: a hang costs 20 min at the fold
  # level and 60 s here. triangle_multiplicative_update is the one that hangs on sm_100.
  [ -s "$HERE/kernel_smoke.py" ] && {
    say "triton: direct sm_100 primitive smoke"
    timeout 300 /root/venv-protenix/bin/python3 "$HERE/kernel_smoke.py" 2>&1 \
      | tee "$R/kernel_smoke_triton.txt" | tail -6
  }
  say "stage triton done"
}

# REJECTED route B. Superseded by stage_triton, which fixes the hang moving one package
# instead of two pins. Kept only so the doc's claim that it was tested stays checkable.
#
# Original note: run only if route A does not fold. cueq 0.11.1 will NOT run against the torch
# both models pin: its attention_pair_bias imports
# torch.fx._symbolic_trace.is_fx_symbolic_tracing, which does not exist in torch 2.7.1, so a
# --no-deps install of 0.11.1 dies at import before a single kernel launches. That is what
# its torch>=2.11 requirement is protecting, and it is why route B has to raise torch too.
#
# The conflict is exact and two-sided: protenix 2.0.0 requires torch==2.7.1 AND
# cuequivariance-ops-torch-cu12==0.8.0; opendde 1.0.3 requires torch==2.7.1 AND
# cuequivariance-ops-torch-cu12==0.10.0. So reaching 0.11.1 means overriding two pins per
# model, not one, and no pip resolution satisfies both the model and 0.11.1.
#
# Fresh ISOLATED venvs, because the shared-site-packages venvs see the image's
# torchvision 0.22.1, compiled against torch 2.7.1: raising torch under it ABI-breaks on
# the first torchvision import. Install the model first so every other dependency
# resolves against its own pin, then force torch and cueq up on top -- the same two
# commands a user chasing the sm_100 hang would run.
stage_torchup() {
  say "stage torchup: isolated venvs, torch $TORCH_TARGET + cueq $CUEQ_TARGET"
  for spec in "prot211:protenix==2.0.0:venv-protenix" "odde211:opendde[gpu]==1.0.3:venv-opendde"; do
    V=${spec%%:*}; rest=${spec#*:}; PKG=${rest%%:*}
    [ -x /root/venv-$V/bin/pip ] || python3 -m venv /root/venv-$V
    P=/root/venv-$V/bin/pip
    $P install --no-cache-dir -q --upgrade pip
    say "torchup $V: install $PKG at its own pins"
    $P install --no-cache-dir -q "$PKG" huggingface_hub==0.34.4 2>&1 | tail -3 | tee -a "$LOG"
    # PyPI's torch 2.13.0 is a CUDA 13 build (nvidia-cudnn-cu13, nvidia-nccl-cu13), which
    # is what the cu13 cuEquivariance binaries want, and 2.13.0 is the torch the three
    # models that already have B200 cells were measured on. The cu128 index tops out at
    # 2.11.0, so it cannot serve this. torchvision/torchaudio are left at protenix's pins:
    # neither model imports either one, so upgrading them buys nothing and can only fail
    # to resolve.
    say "torchup $V: raise torch to $TORCH_TARGET (PyPI build, CUDA 13)"
    $P install --no-cache-dir -q --upgrade "torch==$TORCH_TARGET" 2>&1 | tail -3 | tee -a "$LOG"
    say "torchup $V: cueq $CUEQ_TARGET, cu13 build"
    $P uninstall -y -q cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 2>&1 | tail -2
    $P install --no-cache-dir -q --upgrade \
      "cuequivariance==$CUEQ_TARGET" "cuequivariance-torch==$CUEQ_TARGET" \
      "cuequivariance-ops-cu13==$CUEQ_TARGET" "cuequivariance-ops-torch-cu13==$CUEQ_TARGET" \
      2>&1 | tail -3 | tee -a "$LOG"
    versions_json $V "$R/cueq_torchup_$V.json" | tee -a "$LOG"
  done
  say "stage torchup done"
}

gate() {  # gate <structure> <expect-plddt>
  [ -s "$1" ] || { echo "GATE: no structure at $1"; return 1; }
  python3 "$HERE/gpu5_accuracy_gate.py" "$1" --expect-residues 512 ${2:+--expect-plddt "$2"}
}

stage_run() {
  say "stage run"
  for M in protenix-v2 opendde; do
    if [ "$M" = "protenix-v2" ]; then
      PY=/root/venv-$PROT_VENV/bin/python3; CK=/root/ckpt/protenix-v2.pt
      # A100 read this fixture at 0.843 and the H200 at 0.853; the 0.824329 on
      # record is the Tenstorrent number (post 377976ab/c66baa63), which is the
      # wrong reference for a GPU arm.
      RUNG=LD-shipped-default; EXP=0.843
    else
      PY=/root/venv-$ODDE_VENV/bin/python3; CK=/root/ckpt/opendde.pt
      RUNG=L2-bf16-fusion-cache; EXP=0.823   # A100 read 0.823, H200 0.827
    fi
    OUT=$R/gpu_${M}_prot512_b200${TAG:+_$TAG}.json
    ST=$R/struct_${M}_b200${TAG:+_$TAG}
    mkdir -p "$ST"
    say "$M rung $RUNG"
    # Power draw in the background: a cuEquivariance hang on sm_100 sits at ~200 W on a
    # 1000 W part while reporting 100 % util, so util alone cannot tell a hang from work.
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used \
      --format=csv,noheader -l 20 > "$R/power_${M}${TAG:+_$TAG}.csv" 2>&1 &
    SMI=$!
    timeout "$PER_MODEL_S" $PY gpu_bench.py --model "$M" --repeat "$REPEAT" \
      --checkpoint "$CK" --msa-a3m fixtures/prot512.a3m --seq-file fixtures/prot512.seq \
      --label "cdk2x2_512 (512 aa)" --name prot512 --rungs "$RUNG" \
      --save-structure "$ST" --out "$OUT" > "$R/${M}_b200.log" 2>&1
    echo "$M rc=$?" | tee -a "$LOG"
    kill $SMI 2>/dev/null
    gate "$ST/$RUNG.pdb" "$EXP" | tee "$R/gate_${M}_b200${TAG:+_$TAG}.txt"
    tail -3 "$R/power_${M}${TAG:+_$TAG}.csv" | tee -a "$LOG"
  done
  say "stage run done"
  ls -l "$R"/*.json | tee -a "$LOG"
}

for s in "$@"; do
  case "$s" in
    setup)   stage_setup ;;
    triton)  stage_triton ;;
    cu13)    stage_cu13 ;;
    torchup) stage_torchup ;;
    run)     stage_run ;;
    *) echo "unknown stage: $s" >&2 ;;
  esac
done
say "b200_cueq_session finished stages: $*"
