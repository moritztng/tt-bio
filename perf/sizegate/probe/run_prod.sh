#!/bin/bash
# One fold at the platform's own sampling steps, so the top rung gets a structural reading
# the 6-step ladder config cannot give. Card 1, widened grant (card 0 is running the
# ladder re-record).
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-below-bar-all-bhp150a
CARD=1
RUNG=${1:-1536}
cd "$WT" || exit 1
OUT=$WT/perf/sizegate/probe/prod_boltz2_${RUNG}
LOG=$WT/perf/sizegate/probe/prod_boltz2_${RUNG}.log
rm -rf "$OUT"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
TT_BIO_LEASE_HOLDER=worker:cov-ladder-below-bar-all-bhp150a \
  python3 -m tt_bio.main predict "perf/size512/fixtures/cdk2x2_${RUNG}.yaml" \
  --model boltz2 --single_sequence --diffusion_samples 1 --seed 0 \
  --out_dir "$OUT" > "$LOG" 2>&1
RC=$?
echo "rc=$RC rung=$RUNG steps=default(200)" >> "$LOG"
