#!/bin/bash
# One census-config fold at one rung on one card, logged. Same config as the size-ladder arm.
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-below-bar-all-bhp150a
MODEL=$1; RUNG=$2; CARD=$3
OUT=$WT/perf/sizegate/probe/out_${MODEL}_${RUNG}
LOG=$WT/perf/sizegate/probe/${MODEL}_${RUNG}.log
rm -rf "$OUT"
cd "$WT"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:cov-ladder-below-bar-all-bhp150a \
  python3 -m tt_bio.main predict perf/size512/fixtures/cdk2x2_${RUNG}.yaml \
  --model "$MODEL" --single_sequence --sampling_steps 6 --diffusion_samples 1 \
  --seed 0 --out_dir "$OUT" > "$LOG" 2>&1
echo "rc=$? rung=$RUNG model=$MODEL" >> "$LOG"
