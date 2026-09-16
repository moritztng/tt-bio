#!/usr/bin/env bash
# At and below the 1024-token cap the above-cap fused route must not fire, so the two arms have
# to write the same bytes. One process per (size, arm) -- the flag is never flipped inside a live
# device context, which is what wedged the host four times in ttx-a3-1536-200step-fold.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship
cd "$WT" || exit 1
OUT=perf/ttx_a3/nochange
CARD=${CARD:-3}
STEPS=${STEPS:-200}
for size in 298 512 768 1024; do
  for arm in off on; do
    tag="s${size}_${arm}"
    [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; continue; }
    echo "=== $(date -u +%H:%M:%SZ) $tag ==="
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:ttx-a3-fused-sdpa-default-ship \
    timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/ttx_a3/fold_parity_a3.py \
      --dir "$OUT" --arm "$arm" --tag "$tag" --fixture "cdk2x2_${size}" --steps "$STEPS" \
      >> "$OUT/run.log" 2>&1
    echo "  rc=$? $( [ -f "$OUT/$tag.json" ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r[\"fold_s\"],r[\"cif_sha256\"][:16],r[\"triatt_served\"],r[\"triatt_declined\"])" "$OUT/$tag.json" )"
  done
done
echo "=== ALL DONE $(date -u +%H:%M:%SZ) ==="
