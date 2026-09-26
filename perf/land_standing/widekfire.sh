#!/bin/bash
# Does TT_BIO_SDPA_WIDE_K_UP fire on a REAL fold, and at what pair?
#
# The host-side reach count says bf16 gets an upward rung at 8 of 48 tile-aligned lengths, 384
# among them. Boltz-2's trunk has tri_att_sdpa_hifi False, so it takes the STOCK ladder, which is
# the only consumer of this lever. This folds 384 tokens with the lever off and on and reads the
# fold's OWN counters, because a silently-declined config is indistinguishable from an absent one.
#
# A firing question is load-insensitive, so the box being loud does not block it and no timing
# claim is taken from these legs.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/widekfire
IN=$OUT/inputs/aa384/cdk2apo_384.yaml
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

run () {
  tag=$1
  up=$2
  dump=$OUT/stats_$tag.jsonl
  rm -f "$dump"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$OUT/hook" TT_STOCK_STATS_DUMP="$dump" TT_BIO_SDPA_WIDE_K_UP="$up" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model boltz2 \
        --single_sequence --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "LEG $tag wide_k_up=$up rc=$rc secs=$((t1 - t0))"
  grep -h "picks" "$dump" 2>/dev/null | tail -1
}

run off 0
run on  1
echo FIRE_DONE
