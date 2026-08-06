#!/usr/bin/env bash
# The ONE command the paid H200 session runs: setup -> both models, full
# optimization ladder -> results dump. Designed to run start-to-finish
# unattended; every step logs to /root/bench-results/session.log.
#
# Usage on the box (after scp of this directory + fixtures):
#   bash gpu_session.sh 2>&1 | tee -a /root/bench-results/session.log
set -uo pipefail   # NOT -e: a failed rung must not kill the session
cd "$(dirname "$0")"
RESULTS=/root/bench-results
mkdir -p "$RESULTS"

echo "== session start: $(date -u +%FT%TZ) =="
bash gpu_setup.sh 2>&1 | tee -a "$RESULTS/session.log"
SETUP_RC=$?
echo "setup rc=$SETUP_RC" | tee -a "$RESULTS/session.log"

for MODEL in protenix-v2 opendde; do
  if [ "$MODEL" = "protenix-v2" ]; then
    PY=/root/venv-protenix/bin/python3
    CKPT_ARG="--checkpoint /root/ckpt/protenix-v2.pt"
  else
    PY=/root/venv-opendde/bin/python3
    CKPT_ARG="--checkpoint /root/ckpt/opendde.pt"
  fi
  echo "== $MODEL: $(date -u +%FT%TZ) ==" | tee -a "$RESULTS/session.log"
  $PY gpu_bench.py --model "$MODEL" --repeat 3 $CKPT_ARG \
      --msa-a3m "$(pwd)/fixtures/prot117.a3m" \
      --out "$RESULTS/gpu_${MODEL}.json" 2>&1 | tee -a "$RESULTS/session.log"
  echo "$MODEL rc=$?" | tee -a "$RESULTS/session.log"
done

nvidia-smi --query-gpu=name,power.limit,power.draw --format=csv > "$RESULTS/nvidia_smi.txt" 2>&1
echo "== session end: $(date -u +%FT%TZ) ==" | tee -a "$RESULTS/session.log"
echo "RESULTS DIR: $RESULTS"
