#!/usr/bin/env bash
# The one command the paid H200 session runs for the throughput-at-concurrency pass:
# setup -> plain concurrency sweep -> MPS sweep -> MIG probe -> results dump.
#
# The budget is ~40 minutes of H200 (2026-08-08: $2.68 credit at $3.9351/hr), so the
# point list below is in strict value order and every point is gated on elapsed wall
# clock. If the clock runs out the session stops with whatever is finished; a plain
# concurrency curve alone is already enough to correct the published headline.
#
# Usage on the box (after scp of this directory):
#   BUDGET_S=1500 bash gpu_conc_session.sh 2>&1 | tee -a /root/bench-results/conc.log
#
# Knobs (env):
#   BUDGET_S=1500   wall clock from session start after which no new point is started
#   MODEL           protenix-v2 (default) | opendde
#   FOLDS=4         timed folds per worker per point
#   NS="1 2 4 8"    concurrency values for the plain sweep
#   MPS_NS="4 8 2"  concurrency values for the MPS sweep (4 first: most informative)
#   TARGETS300=1    also run the 298-aa target at N=1,4 if the budget allows
#   SAMPLES_LEG=1   also sweep diffusion samples 2 and 4 at N=1 (see the state doc)
#   SKIP_SETUP=1    reuse an already-provisioned instance
#   RESULTS/RUNROOT/PY  overrides, used to smoke-test the driver off the rental
set -uo pipefail    # NOT -e: one failed point must not kill the session
cd "$(dirname "$0")"
RESULTS=${RESULTS:-/root/bench-results}
RUNROOT=${RUNROOT:-/root}
mkdir -p "$RESULTS"

MODEL=${MODEL:-protenix-v2}
CKPT=${CKPT:-/root/ckpt/protenix-v2.pt}
FOLDS=${FOLDS:-4}
NS=${NS:-"1 2 4 8"}
MPS_NS=${MPS_NS:-"4 8 2"}
BUDGET_S=${BUDGET_S:-1500}
TARGETS300=${TARGETS300:-0}
START=$(date +%s)

el(){ echo $(( $(date +%s) - START )); }
have_budget(){ [ "$(el)" -lt "$BUDGET_S" ]; }

echo "== conc session start: $(date -u +%FT%TZ) model=$MODEL budget=${BUDGET_S}s =="
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  SETUP_MODELS=protenix bash gpu_setup.sh 2>&1 | tee -a "$RESULTS/conc.log"
  echo "setup done at $(el)s" | tee -a "$RESULTS/conc.log"
fi
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH
PY=${PY:-/root/venv-protenix/bin/python3}
if [ "$MODEL" = "opendde" ]; then
  PY=${PY_OPENDDE:-/root/venv-opendde/bin/python3}; CKPT=${CKPT_OPENDDE:-/root/ckpt/opendde.pt}
fi

# CUDA header fix (root-caused 2026-08-08, instance 47202144): on the runtime image,
# conda cuda-compiler leaves cuda_runtime_api.h under targets/x86_64-linux/include and
# ships no cusparse/cublas headers at all, so protenix's fused-layer_norm JIT build dies
# at import. The torch wheel bundles every needed header/lib under site-packages/nvidia;
# symlink them (and the conda targets dirs) into the prefix. Idempotent, runs on every
# session including SKIP_SETUP=1.
if [ -d /opt/conda ]; then
  SP=$(/opt/conda/bin/python3 -c "import nvidia,os;print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
  for pkg in $SP/*/ /opt/conda/targets/x86_64-linux/; do
    for sub in include lib; do
      d=${pkg}${sub}; [ -d "$d" ] || continue
      for f in "$d"/*; do
        b=$(basename "$f"); tgt=/opt/conda/$sub/$b
        [ -e "$tgt" ] || ln -s "$f" "$tgt" 2>/dev/null
      done
    done
  done
fi
# Warm the JIT extension cache once in the launcher (~2 min build) so N workers at a
# point do not serialize against the same torch-extensions lock on paid time.
$PY -c "import runner.inference" >/dev/null 2>&1 || true

point(){    # point <mode> <n> <target> [samples]
  local mode=$1 n=$2 tgt=$3 s=${4:-1} label tag
  if ! have_budget; then
    echo "== SKIP $mode N=$n $tgt: $(el)s past BUDGET_S=$BUDGET_S ==" | tee -a "$RESULTS/conc.log"
    return
  fi
  case "$tgt" in
    prot117) label="prot.yaml sequence (117 aa)" ;;
    prot300) label="CDK2 / PDB 1HCL (298 aa)" ;;
    *) label="$tgt" ;;
  esac
  tag="n${n}"; [ "$s" != "1" ] && tag="n${n}_s${s}"
  echo "== $mode N=$n $tgt samples=$s at $(el)s ==" | tee -a "$RESULTS/conc.log"
  $PY gpu_concurrency.py --model "$MODEL" --n "$n" --folds "$FOLDS" --mode "$mode" \
      --checkpoint "$CKPT" --name "$tgt" --label "$label" --samples "$s" \
      --msa-a3m "$(pwd)/fixtures/${tgt}.a3m" --seq-file "$(pwd)/fixtures/${tgt}.seq" \
      --run-dir "$RUNROOT/run-${mode}-${tgt}-${tag}" \
      --out "$RESULTS/conc_${MODEL}_${tgt}_${mode}_${tag}.json" 2>&1 | tee -a "$RESULTS/conc.log"
}

# ---- plain: N processes time-slicing. The honest "customer just runs more jobs" case.
# N=1 first because it doubles as the harness check: it must reproduce the committed
# 6.147 s warm median for protenix-v2 L2 at 117 aa. If it does not, nothing after it
# means anything and the session should be stopped.
for n in $NS; do point plain "$n" prot117; done

# ---- MPS: same processes, kernels from different contexts run concurrently instead of
# context-switching. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is deliberately left unset: the
# default gives every client the whole GPU and lets the hardware share dynamically, which
# is the configuration a customer gets by just turning MPS on.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  nvidia-cuda-mps-control -d 2>&1 | tee -a "$RESULTS/conc.log"
  sleep 2
  echo get_server_list | nvidia-cuda-mps-control 2>&1 | tee -a "$RESULTS/mps_state.txt"
  for n in $MPS_NS; do point mps "$n" prot117; done
  echo quit | nvidia-cuda-mps-control 2>&1 | tee -a "$RESULTS/mps_state.txt"
else
  echo "MPS UNAVAILABLE: nvidia-cuda-mps-control not on PATH in this container" \
    | tee -a "$RESULTS/mps_state.txt"
fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

# ---- in-process batching, the third throughput lever. Off by default: it changes the
# unit from folds to samples of one target, and it is only publishable as a comparison if
# the TT side ran the same sweep (it is free there).
if [ "${SAMPLES_LEG:-0}" = "1" ]; then
  for s in 2 4; do point plain 1 prot117 "$s"; done
fi

# ---- 298-aa repeat, only if the clock allows. Two points is enough to say whether the
# concurrency headroom survives at a compute-heavier size.
if [ "$TARGETS300" = "1" ]; then
  for n in 1 4; do point plain "$n" prot300; done
fi

# ---- MIG, deliberately last: enabling it needs a GPU reset, so a success here would end
# the session. Run after every measurement is safely on disk.
{
  echo "== MIG probe $(date -u +%FT%TZ) =="
  nvidia-smi --query-gpu=name,driver_version,mig.mode.current,mig.mode.pending --format=csv
  echo "--- attempt to enable MIG (expected to be refused inside a rented container):"
  nvidia-smi -i 0 -mig 1; echo "rc=$?"
  nvidia-smi -L
} > "$RESULTS/mig_probe.txt" 2>&1
cat "$RESULTS/mig_probe.txt"

nvidia-smi --query-gpu=name,power.limit,power.draw,memory.total --format=csv \
  > "$RESULTS/nvidia_smi_conc.txt" 2>&1
echo "== conc session end: $(date -u +%FT%TZ) total=$(el)s ==" | tee -a "$RESULTS/conc.log"
echo "RESULTS DIR: $RESULTS"
