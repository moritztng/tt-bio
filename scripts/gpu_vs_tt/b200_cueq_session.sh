#!/usr/bin/env bash
# Protenix-v2 + OpenDDE at 512 aa on a B200, with cuEquivariance upgraded to 0.11.1.
#
# Both models hang on sm_100 in their shipped cuEquivariance configuration (protenix 2.0.0
# pulls 0.8.0, opendde 1.0.3 pulls 0.10.0); 0.11.1 is the version that runs clean. A
# production user hitting the hang would upgrade, so 0.11.1 is the configuration measured
# here. Everything else is the shipped default: dtype, kernel selectors, fusion/cache,
# TF32 untouched, 10 cycles / 200 steps / 1 sample / seed 0, the pinned cdk2x2_512 fixture
# with its 35-row alignment.
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel. protenix 2.0.0 pins torch==2.7.1, so
# the image IS the pin, and the devel variant already carries nvcc plus the CUDA headers
# both vendors need to JIT their fused-layernorm extension (the runtime image needs a
# multi-GB conda cuda-toolkit install for that).
#
#   bash b200_cueq_session.sh setup        # venvs + weights + the shipped cueq
#   bash b200_cueq_session.sh upgrade      # cueq -> 0.11.1, recorded before and after
#   bash b200_cueq_session.sh run          # both cells + gate
set -uo pipefail   # NOT -e: one model failing must leave the other measured
cd "$(dirname "$0")"
HERE=$(pwd)
R=/root/results
mkdir -p "$R" /root/ckpt
LOG=$R/session.log
CUEQ_TARGET=${CUEQ_TARGET:-0.11.1}
REPEAT=${REPEAT:-4}
PER_MODEL_S=${PER_MODEL_S:-1500}
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

# Upgrade cueq without moving torch. 0.11.1's ONLY new hard requirement over 0.10.0 is
# torch>=2.11 (0.8.0/0.10.0 list torch only under the `test` extra), and protenix 2.0.0
# pins torch==2.7.1 -- so honouring 0.11.1's metadata would replace the torch the model
# pins. --no-deps installs 0.11.1 against the pinned torch; if the wheel's ABI does not
# hold, the import probe below says so immediately rather than at fold time.
stage_upgrade() {
  say "stage upgrade -> cueq $CUEQ_TARGET (--no-deps, torch pin held)"
  for V in protenix opendde; do
    /root/venv-$V/bin/pip install --no-cache-dir -q --no-deps --upgrade \
      "cuequivariance==$CUEQ_TARGET" "cuequivariance-torch==$CUEQ_TARGET" \
      "cuequivariance-ops-cu12==$CUEQ_TARGET" "cuequivariance-ops-torch-cu12==$CUEQ_TARGET" \
      2>&1 | tail -4 | tee -a "$LOG"
    say "UPGRADED cueq versions in venv-$V"
    versions_json $V "$R/cueq_upgraded_$V.json" | tee -a "$LOG"
  done
  say "stage upgrade done"
}

# cueq 0.11.1 will NOT run against the torch both models pin. 0.11.1's
# attention_pair_bias imports torch.fx._symbolic_trace.is_fx_symbolic_tracing (and
# is_fx_tracing_symbolic_tracing), neither of which exists in torch 2.7.1, so the
# --no-deps install of stage_upgrade dies at import before a single kernel launches --
# which is what its torch>=2.11 requirement is actually protecting.
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
  say "stage torchup: isolated venvs, torch >=2.11 + cueq $CUEQ_TARGET"
  for spec in "prot211:protenix==2.0.0:venv-protenix" "odde211:opendde[gpu]==1.0.3:venv-opendde"; do
    V=${spec%%:*}; rest=${spec#*:}; PKG=${rest%%:*}
    [ -x /root/venv-$V/bin/pip ] || python3 -m venv /root/venv-$V
    P=/root/venv-$V/bin/pip
    $P install --no-cache-dir -q --upgrade pip
    say "torchup $V: install $PKG at its own pins"
    $P install --no-cache-dir -q "$PKG" huggingface_hub==0.34.4 2>&1 | tail -3 | tee -a "$LOG"
    say "torchup $V: force torch up (cu128 index first, PyPI default as fallback)"
    $P install --no-cache-dir -q --upgrade --index-url https://download.pytorch.org/whl/cu128 \
      "torch>=2.11" torchvision torchaudio 2>&1 | tail -3 | tee -a "$LOG"
    /root/venv-$V/bin/python3 -c "import torch,sys;sys.exit(0 if torch.__version__>='2.11' else 1)" \
      || $P install --no-cache-dir -q --upgrade "torch>=2.11" torchvision torchaudio \
           2>&1 | tail -3 | tee -a "$LOG"
    say "torchup $V: force cueq to $CUEQ_TARGET"
    $P install --no-cache-dir -q --no-deps --upgrade \
      "cuequivariance==$CUEQ_TARGET" "cuequivariance-torch==$CUEQ_TARGET" \
      "cuequivariance-ops-cu12==$CUEQ_TARGET" "cuequivariance-ops-torch-cu12==$CUEQ_TARGET" \
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
      RUNG=LD-shipped-default; EXP=0.828628
    else
      PY=/root/venv-$ODDE_VENV/bin/python3; CK=/root/ckpt/opendde.pt
      RUNG=L2-bf16-fusion-cache; EXP=
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
    upgrade) stage_upgrade ;;
    torchup) stage_torchup ;;
    run)     stage_run ;;
    *) echo "unknown stage: $s" >&2 ;;
  esac
done
say "b200_cueq_session finished stages: $*"
