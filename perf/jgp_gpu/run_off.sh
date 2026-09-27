#!/usr/bin/env bash
# Vanilla arm: stock protenix 2.0.0 through the kit's `--mode off` (clean subprocess, no kit code loaded).
# Settings = the PopVax/Anthropic chart settings and our BH run: 5 samples, 200 steps, 10 cycles, seed 101, bf16
# (upstream default), MSA on from the precomputed a3m files. nvidia-smi sampled at 1 Hz for the whole process.
#   MODE=off|exact|fast DTYPE=bf16|fp32 bash run_off.sh <tag> [extra protenix pred args...]   (off = vanilla)
set -uo pipefail
TAG=$1; shift
cd /root/kit/protenix_v2
. venv/bin/activate
export CUDA_HOME=/usr/local/cuda PROTENIX_ROOT_DIR=/weights/protenix
OUT=/root/jgp/out/$TAG; mkdir -p "$OUT"
cp /root/jgp/complex730.json "$OUT/in.json"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -lms 1000 > "$OUT/smi.csv" &
SMI=$!
date -u +%FT%T.%NZ > "$OUT/t_start"
bash run.sh pred --config a100 --mode "${MODE:-off}" --input "$OUT/in.json" --out_dir "$OUT/pred" --model_name protenix-v2 \
  --seeds 101 --cycle 10 --step 200 --sample 5 --dtype "${DTYPE:-bf16}" --use_msa true "$@" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
RC=$?
date -u +%FT%T.%NZ > "$OUT/t_end"
kill $SMI
echo "rc=$RC" > "$OUT/rc"
grep -h "^PHASE\|^PEAK" "$OUT/stdout.log"
echo "RUN-DONE $TAG rc=$RC"
