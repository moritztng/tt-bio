#!/usr/bin/env bash
# The ONE command the paid H200 session runs: setup -> both models -> results
# dump. Designed to run start-to-finish unattended; every step logs to
# /root/bench-results/session.log.
#
# Usage on the box (after scp of this directory + fixtures):
#   bash gpu_session.sh 2>&1 | tee -a /root/bench-results/session.log
#
# Knobs (env, all optional):
#   TARGET=prot300     prot117 | prot300 (default prot300, the 298-aa CDK2 target)
#   RUNGS=...          comma-separated rung subset (default L0+L2; L1 was "mixed"
#                      at 117 aa and is dropped here to fit the rental budget)
#   REPEAT=3           timed warm folds per rung
#   MODELS="..."       default "protenix-v2 opendde", in that order on purpose:
#                      protenix finishes first, so a budget overrun still leaves
#                      one complete head-to-head rather than two half ones
#   BUDGET_S=2400      wall-clock guard; once exceeded, remaining models are
#                      skipped rather than silently eating the last of the credit
#   SKIP_SETUP=1       reuse an already-provisioned instance
set -uo pipefail   # NOT -e: a failed rung must not kill the session
cd "$(dirname "$0")"
RESULTS=/root/bench-results
mkdir -p "$RESULTS"

TARGETS=${TARGETS:-"prot300 prot512"}
RUNGS=${RUNGS:-L0-eager-fp32,L2-bf16-fusion-cache,LD-shipped-default}
REPEAT=${REPEAT:-3}
MODELS=${MODELS:-"protenix-v2"}
BUDGET_S=${BUDGET_S:-3600}
START=$(date +%s)

echo "== session start: $(date -u +%FT%TZ) targets=$TARGETS rungs=$RUNGS =="
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  bash gpu_setup.sh 2>&1 | tee -a "$RESULTS/session.log"
  echo "setup rc=$? elapsed=$(( $(date +%s) - START ))s" | tee -a "$RESULTS/session.log"
fi
# gpu_setup.sh installs the CUDA toolchain; the bench needs it in env even when
# setup is skipped (venvs + toolchain persist on the instance disk).
export CUDA_HOME=/opt/conda
export PATH=/opt/conda/bin:$PATH

# Model-major, and within a model the 300-aa target first and the 117-aa control
# second. The 117-aa leg is the point of the whole ordering: re-measuring the old
# target on THIS host turns the scaling ratio into a within-host number instead of
# one taken across two different rented machines. It is also cheap (~7 s/fold), so
# per model the control costs ~1.5 min. If the budget guard fires, it fires on a
# whole model, leaving the earlier model with both of its sizes intact.
for MODEL in $MODELS; do
  if [ "$MODEL" = "protenix-v2" ]; then
    PY=/root/venv-protenix/bin/python3
    CKPT_ARG="--checkpoint /root/ckpt/protenix-v2.pt"
  else
    PY=/root/venv-opendde/bin/python3
    CKPT_ARG="--checkpoint /root/ckpt/opendde.pt"
  fi
  for TARGET in $TARGETS; do
    case "$TARGET" in
      prot117) LABEL="prot.yaml sequence (117 aa)" ;;
      prot300) LABEL="CDK2 / PDB 1HCL (298 aa)" ;;
      prot512) LABEL="CDK2 tandem dimer, cut to 512 aa (TT size512 fixture)" ;;
      *) echo "unknown TARGET=$TARGET" >&2; continue ;;
    esac
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt "$BUDGET_S" ]; then
      echo "== SKIPPING $MODEL/$TARGET: ${ELAPSED}s elapsed exceeds BUDGET_S=${BUDGET_S}s ==" \
        | tee -a "$RESULTS/session.log"
      continue
    fi
    echo "== $MODEL $TARGET: $(date -u +%FT%TZ) (${ELAPSED}s in) ==" | tee -a "$RESULTS/session.log"
    $PY gpu_bench.py --model "$MODEL" --repeat "$REPEAT" $CKPT_ARG \
        --msa-a3m "$(pwd)/fixtures/${TARGET}.a3m" \
        --seq-file "$(pwd)/fixtures/${TARGET}.seq" \
        --label "$LABEL" --name "$TARGET" --rungs "$RUNGS" \
        --out "$RESULTS/gpu_${MODEL}_${TARGET}.json" 2>&1 | tee -a "$RESULTS/session.log"
    echo "$MODEL/$TARGET rc=$?" | tee -a "$RESULTS/session.log"
  done
done

nvidia-smi --query-gpu=name,power.limit,power.draw --format=csv > "$RESULTS/nvidia_smi.txt" 2>&1
echo "== session end: $(date -u +%FT%TZ) total=$(( $(date +%s) - START ))s ==" \
  | tee -a "$RESULTS/session.log"
echo "RESULTS DIR: $RESULTS"
