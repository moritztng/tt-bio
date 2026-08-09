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
#   NS="1"          concurrency values for the plain sweep (N=1 doubles as the control)
#   MPS_NS="8 4 2"  concurrency values for the MPS sweep (8 first: the disputed headline)
#   DISTINCT_LEG=1  also run MPS N=8 with 8 DIFFERENT targets (the same-input control)
#   TARGETS300=1    also run the 298-aa target (plain N=1 + MPS N=4,8) if budget allows
#   CCD_CACHE=path  tarball of /root/common (prebuilt protenix chemical-component cache);
#                   without it the 298-aa leg spends ~55 min building it on a fresh instance
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
NS=${NS:-"1"}
MPS_NS=${MPS_NS:-"8 4 2"}
BUDGET_S=${BUDGET_S:-1500}
DISTINCT_LEG=${DISTINCT_LEG:-1}
DISTINCT_N=${DISTINCT_N:-8}
DISTINCT_TARGETS=${DISTINCT_TARGETS:-bk6_104,cytc_104,fkbp_107,barn_110,mlta_112,mltb_116,prt7_117,rnsa_124}
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

# Protenix's chemical-component cache: a 490 MB components.cif download followed by an
# rdkit_mol.pkl build. prot117 never needs it; prot300 does, and it is built lazily inside
# the first worker that asks for it. That is what killed every 298 aa attempt so far -- run
# N workers at a 298 aa point and all N race to write the same path, and each of them exits
# rc=1 (root-caused 2026-08-09 on instance 47229750; the earlier failures reported rc=1 at
# ~68 s with no reason, because worker log tails were not being captured yet).
#
# The pickle build is the expensive part: measured at ~1.65%/min, so ~55 minutes on a fresh
# instance, which is far more than the 298 aa measurement itself costs. So: unpack a
# prebuilt cache if one was shipped, otherwise build it in ONE process before any point
# runs. Set CCD_CACHE to a tarball of /root/common to skip the build entirely.
CCD_PKL=${CCD_PKL:-/root/common/components.cif.rdkit_mol.pkl}
if [ "$TARGETS300" = "1" ] && [ ! -s "$CCD_PKL" ]; then
  if [ -n "${CCD_CACHE:-}" ] && [ -s "${CCD_CACHE}" ]; then
    echo "== unpacking prebuilt CCD cache from $CCD_CACHE ==" | tee -a "$RESULTS/conc.log"
    mkdir -p /root/common && tar xzf "$CCD_CACHE" -C / 2>&1 | tail -2 | tee -a "$RESULTS/conc.log"
  fi
  # protenix downloads components.cif and then builds the pickle locally, but it publishes
  # the finished pickle at the same prefix (142 MB, verified 2026-08-09). Fetching it takes
  # a couple of minutes against ~55 for the build, so always try that first.
  if [ ! -s "$CCD_PKL" ]; then
    echo "== fetching the prebuilt CCD cache at $(el)s ==" | tee -a "$RESULTS/conc.log"
    mkdir -p /root/common
    for f in components.cif components.cif.rdkit_mol.pkl; do
      [ -s "/root/common/$f" ] || curl -fsSL --retry 3 -o "/root/common/$f" \
        "https://protenix.tos-cn-beijing.volces.com/common/$f" \
        || echo "  fetch FAILED: $f" | tee -a "$RESULTS/conc.log"
    done
    ls -l /root/common | tee -a "$RESULTS/conc.log"
  fi
  # Last resort only: build it in ONE process. ~55 min, so expect to pay for it.
  if [ ! -s "$CCD_PKL" ]; then
    echo "== prebuilt fetch failed, building in one process, expect ~55 min at $(el)s ==" \
      | tee -a "$RESULTS/conc.log"
    $PY gpu_concurrency.py --model "$MODEL" --n 1 --folds 1 --mode plain \
        --checkpoint "$CKPT" --name prot300 --label "CCD cache warm" \
        --msa-a3m "$(pwd)/fixtures/prot300.a3m" --seq-file "$(pwd)/fixtures/prot300.seq" \
        --run-dir "$RUNROOT/run-ccd-warm" --out "$RESULTS/ccd_warm.json" \
        > "$RESULTS/ccd_warm.log" 2>&1
    echo "== CCD warm done at $(el)s, pkl $( [ -s "$CCD_PKL" ] && echo present || echo MISSING) ==" \
      | tee -a "$RESULTS/conc.log"
  fi
fi

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

# The same-input control. Worker i folds target i out of fixtures/distinct/ instead of
# all N workers folding one copy of prot117, so an N-way point measures N different
# proteins. If throughput holds against the same-target point at the same N, the MPS
# speedup is not an artefact of repeating one prediction. The eight targets are 103-124
# aa (mean 111.6, so ~5% shorter than prot117's 117) and all carry n_msa=35, matching
# every other point's MSA depth.
dpoint(){   # dpoint <mode> <n>
  local mode=$1 n=$2
  if ! have_budget; then
    echo "== SKIP $mode N=$n distinct: $(el)s past BUDGET_S=$BUDGET_S ==" | tee -a "$RESULTS/conc.log"
    return
  fi
  echo "== $mode N=$n DISTINCT targets at $(el)s ==" | tee -a "$RESULTS/conc.log"
  $PY gpu_concurrency.py --model "$MODEL" --n "$n" --folds "$FOLDS" --mode "$mode" \
      --checkpoint "$CKPT" --name distinct --label "8 distinct targets, 103-124 aa" \
      --targets "$DISTINCT_TARGETS" \
      --run-dir "$RUNROOT/run-${mode}-distinct-n${n}" \
      --out "$RESULTS/conc_${MODEL}_distinct_${mode}_n${n}.json" 2>&1 | tee -a "$RESULTS/conc.log"
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
  # N=8 first: it is the point the disputed headline quotes, and its distinct-targets
  # control runs immediately after it so the pair is measured back to back under the
  # same MPS daemon. If the budget dies early these are the two points worth having.
  for n in $MPS_NS; do
    point mps "$n" prot117
    [ "$DISTINCT_LEG" = "1" ] && [ "$n" = "$DISTINCT_N" ] && dpoint mps "$n"
  done
  # (c) the size the BD pitch quotes. Runs inside the MPS block because stopping and
  # restarting the daemon around it would cost a minute of paid time for nothing.
  if [ "$TARGETS300" = "1" ]; then
    for n in 4 8; do point mps "$n" prot300; done
  fi
  echo get_server_list | nvidia-cuda-mps-control 2>&1 | tee -a "$RESULTS/mps_state.txt"
  echo quit | nvidia-cuda-mps-control 2>&1 | tee -a "$RESULTS/mps_state.txt"
else
  echo "MPS UNAVAILABLE: nvidia-cuda-mps-control not on PATH in this container" \
    | tee -a "$RESULTS/mps_state.txt"
fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

# ---- 298-aa single-process baseline. The MPS points at this size already ran above; this
# is the N=1 denominator they are measured against.
if [ "$TARGETS300" = "1" ]; then
  point plain 1 prot300
fi

# ---- in-process batching, the third throughput lever, and the lowest-priority leg: it
# changes the unit from folds to samples of one target, and it is only publishable as a
# comparison if the TT side ran the same sweep (it is free there).
if [ "${SAMPLES_LEG:-0}" = "1" ]; then
  for s in 2 4; do point plain 1 prot117 "$s"; done
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
